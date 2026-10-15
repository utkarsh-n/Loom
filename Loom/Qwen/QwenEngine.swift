import Foundation
import LoomEngine
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import OfflineIntelligenceMLX
import OSLog

nonisolated private let log = Logger(subsystem: "com.utkarsh.loom", category: "Qwen")

/// What one Qwen pass hands back to the interface: threads in the same shape the 1.4M model produces
/// (so every view draws both), the newest position's 151,936 scores, and the work it took.
nonisolated struct QwenPass: Sendable {
  /// First position the threads cover. 0 means the whole window was run (prefill); otherwise only the
  /// new positions were, reading the keys and values kept from before (decode).
  var start: Int
  var threads: [Thread]
  var lastLogits: [Float]
  var macs: Int
  var seconds: Double
}

nonisolated struct QwenLoaded: Sendable {
  let config: QwenConfig
  /// The final RMSNorm's gains, for the sphere's gaze.
  let finalGain: [Float]
}

/// Runs Qwen3 on this iPhone's GPU with MLX. Its own forward pass walks every layer explicitly, keeps its
/// own keys and values, computes attention in float32 so all 32 heads' weights can be read, and applies
/// Loom's knobs (heads off, layers skipped, early exit, steering, neuron clamps).
actor QwenWorker {
  /// How many numbers of each older position's vector the interface keeps (plus the last one).
  static let keptDims = 32

  private var model: QwenModel?
  private let residencyLease = MLXExternalModelLease()
  private var keys: [MLXArray?] = []
  private var values: [MLXArray?] = []
  /// Every position's vector after each stage ([layers + 1, d]), kept on the GPU for the logit lens.
  private var residuals: [MLXArray] = []
  /// The ids and knobs the kept keys, values and residuals belong to.
  private var cachedIDs: [Int] = []
  private var cachedKnobs: RunKnobs?

  func load(directory: URL) async throws -> QwenLoaded {
    try await MLXForegroundExecution.prepare()
    try checkGPUAdmission()
    if let model { return QwenLoaded(config: model.config, finalGain: model.model.norm.weight.asType(.float32).asArray(Float.self)) }
    try residencyLease.acquire()
    do {
    Memory.cacheLimit = 32 * 1024 * 1024
    let data = try Data(contentsOf: directory.appending(path: "config.json"))
    let config = try JSONDecoder().decode(QwenConfig.self, from: data)
    let base = try JSONDecoder().decode(BaseConfiguration.self, from: data)
    let candidate = withRandomState(MLXRandom.RandomState(seed: 2026)) { QwenModel(config) }
    try await loadWeights(modelDirectory: directory, model: candidate, perLayerQuantization: base.perLayerQuantization)
    try checkGPUAdmission()
    eval(candidate)
    model = candidate
    forget()
    log.info("Qwen loaded: \(config.hiddenLayers) layers")
      return QwenLoaded(config: config, finalGain: candidate.model.norm.weight.asType(.float32).asArray(Float.self))
    } catch {
      MLXForegroundExecution.synchronizeIfAllowed()
      Memory.clearCache()
      residencyLease.release()
      throw error
    }
  }

  func unload() {
    model = nil
    forget()
    MLXForegroundExecution.synchronizeIfAllowed()
    Memory.clearCache()
    residencyLease.release()
  }

  var isLoaded: Bool { model != nil }

  private func forget() {
    let n = model?.config.hiddenLayers ?? 0
    keys = Array(repeating: nil, count: n)
    values = Array(repeating: nil, count: n)
    residuals = []
    cachedIDs = []
    cachedKnobs = nil
  }

  private func checkGPUAdmission() throws {
    try Task.checkCancellation()
    guard MLXForegroundExecution.isAllowed else { throw CancellationError() }
  }

  // MARK: The pass

  /// What the walk captured for the positions it ran.
  nonisolated private struct Capture {
    var stages: [MLXArray] = []
    var attention: [MLXArray?] = []
    var lastActs: [MLXArray?] = []
    var ran: [Bool] = []
  }

  /// The layers, walked by hand for positions `start ..< ids.count`; appends their keys and values.
  private func walk(_ model: QwenModel, ids: [Int], start: Int, knobs: RunKnobs,
                    keys: inout [MLXArray?], values: inout [MLXArray?], capture: Bool) -> (MLXArray, Capture)
  {
    let c = model.config
    let layerCount = c.hiddenLayers, H = c.attentionHeads, KV = c.kvHeads, hd = c.headDim, d = c.hiddenSize, ff = c.intermediateSize
    let group = H / KV
    let n = ids.count - start
    let T = ids.count
    let inner = model.model
    var x = inner.embedTokens(MLXArray(ids[start...].map { Int32($0) }).reshaped(1, n))
    let dtype = x.dtype
    var cap = Capture()
    if capture { cap.stages.append(x[0]) }

    // Causal mask: position start+i sees keys 0 ... start+i.
    var maskValues = [Float](repeating: 0, count: n * T)
    for i in 0..<n { for j in 0..<T where j > start + i { maskValues[i * T + j] = -.infinity } }
    let mask = MLXArray(maskValues, [n, T])
    let depth = min(knobs.depth ?? layerCount, layerCount)

    for l in 0..<layerCount {
      if l >= depth || knobs.bypass.contains(l) {
        // A skipped layer leaves no keys or values. Kept keys are only reused with identical knobs, so
        // later positions skip it too.
        if capture {
          cap.stages.append(x[0])
          cap.attention.append(nil)
          cap.lastActs.append(nil)
          cap.ran.append(false)
        }
        continue
      }
      let block = inner.layers[l]
      let a = block.attention
      let h = block.inputNorm(x)
      var q = a.qNorm(a.wq(h).reshaped(1, n, H, hd)).transposed(0, 2, 1, 3)
      var k = a.kNorm(a.wk(h).reshaped(1, n, KV, hd)).transposed(0, 2, 1, 3)
      let v = a.wv(h).reshaped(1, n, KV, hd).transposed(0, 2, 1, 3)
      q = applyRotaryPosition(a.rope, to: q, offset: .scalar(start))
      k = applyRotaryPosition(a.rope, to: k, offset: .scalar(start))
      let K = keys[l].map { concatenated([$0, k], axis: 2) } ?? k
      let V = values[l].map { concatenated([$0, v], axis: 2) } ?? v
      keys[l] = K
      values[l] = V
      // Grouped-query attention: each key/value head serves `group` query heads.
      let qg = q.asType(.float32).reshaped(1, KV, group, n, hd)
      let kt = K.asType(.float32).expandedDimensions(axis: 2).transposed(0, 1, 2, 4, 3)
      let scores = matmul(qg, kt) * a.scale + mask
      let probs = softmax(scores, axis: -1)
      if capture { cap.attention.append(probs.reshaped(H, n, T)) }
      var out = matmul(probs, V.asType(.float32).expandedDimensions(axis: 2)).reshaped(1, H, n, hd)
      let off = (0..<H).filter { knobs.muted.contains(HeadID(layer: l, head: $0)) }
      if !off.isEmpty {
        var keep = [Float](repeating: 1, count: H)
        for head in off { keep[head] = 0 }
        out = out * MLXArray(keep, [1, H, 1, 1])
      }
      x = x + a.wo(out.transposed(0, 2, 1, 3).reshaped(1, n, H * hd).asType(dtype))
      let h2 = block.postAttentionNorm(x)
      var act = silu(block.mlp.gate(h2)) * block.mlp.up(h2)
      let clamps = knobs.clamp.filter { $0.layer == l && $0.neuron < ff }
      if !clamps.isEmpty {
        var on = [Float](repeating: 0, count: ff), value = on
        for cl in clamps { on[cl.neuron] = 1; value[cl.neuron] = cl.value }
        act = which(MLXArray(on, [ff]) .> 0, MLXArray(value, [ff]).asType(act.dtype), act)
      }
      if capture { cap.lastActs.append(act[0, n - 1]) }
      x = x + block.mlp.down(act)
      for s in knobs.steer where s.layer == l && s.vector.count == d {
        x = x + MLXArray(s.vector, [d]).asType(dtype) * s.strength
      }
      if capture {
        cap.stages.append(x[0])
        cap.ran.append(true)
      }
    }
    return (x, cap)
  }

  /// Runs `ids` with Loom's knobs. With `reuse`, positions already run with the same knobs are read from
  /// the kept keys and values and only the new ones are computed; otherwise the whole window is.
  func run(ids: [Int], knobs: RunKnobs, reuse: Bool) throws -> QwenPass {
    try checkGPUAdmission()
    guard let model else { throw QwenError.notLoaded }
    let clock = Date()
    let c = model.config
    let layerCount = c.hiddenLayers, H = c.attentionHeads, KV = c.kvHeads, hd = c.headDim, d = c.hiddenSize, ff = c.intermediateSize
    let canReuse = reuse && cachedKnobs == knobs && !cachedIDs.isEmpty && cachedIDs.count < ids.count
      && Array(ids.prefix(cachedIDs.count)) == cachedIDs
    if !canReuse { forget() }
    let start = canReuse ? cachedIDs.count : 0
    let n = ids.count - start
    let T = ids.count
    var ks = keys, vs = values
    let (x, cap) = walk(model, ids: ids, start: start, knobs: knobs, keys: &ks, values: &vs, capture: true)
    keys = ks
    values = vs

    let inner = model.model
    let stack = stacked(cap.stages, axis: 0)                    // [layers + 1, n, d]
    let final = inner.norm(x)
    let logits = inner.embedTokens.asLinear(final[0, n - 1]).asType(.float32)
    // Logit lens for the newest position: every stage through the final norm and the readout at once.
    let lensScores = inner.embedTokens.asLinear(inner.norm(stack[0..., n - 1])).asType(.float32)
    let lensTop = argMax(lensScores, axis: -1)
    let lensProb = exp(lensScores.max(axis: -1) - logSumExp(lensScores, axis: -1))
    // Older positions keep a few numbers of each vector (what the pipeline draws); the newest keeps all.
    let k = Self.keptDims
    let older = n > 1 ? concatenated([stack[0..., 0 ..< (n - 1), 0 ..< k], stack[0..., 0 ..< (n - 1), (d - 1)...]], axis: -1).asType(.float32) : nil
    let newest = stack[0..., n - 1].asType(.float32)
    let floatActs = cap.lastActs.map { $0?.asType(.float32) }
    var toEval: [MLXArray] = [stack, logits, lensTop, lensProb, newest]
    if let older { toEval.append(older) }
    toEval += cap.attention.compactMap { $0 }
    toEval += floatActs.compactMap { $0 }
    try checkGPUAdmission()
    eval(toEval)
    try checkGPUAdmission()
    for i in 0..<n { residuals.append(stack[0..., i]) }
    cachedIDs = ids
    cachedKnobs = knobs

    let olderValues = older?.asArray(Float.self) ?? []        // [(layers + 1) · (n − 1) · (k + 1)]
    let newestValues = newest.asArray(Float.self)             // [(layers + 1) · d]
    let topIDs = lensTop.asArray(Int32.self)
    let topP = lensProb.asArray(Float.self)
    let attentionValues = cap.attention.map { $0?.asArray(Float.self) }
    let actValues = floatActs.map { $0?.asArray(Float.self) }
    var threads: [Thread] = []
    threads.reserveCapacity(n)
    for i in 0..<n {
      let isNewest = i == n - 1
      let pos = start + i
      var residual: [[Float]] = []
      residual.reserveCapacity(layerCount + 1)
      for s in 0...layerCount {
        if isNewest {
          residual.append(Array(newestValues[(s * d)..<((s + 1) * d)]))
        } else {
          let base = (s * (n - 1) + i) * (k + 1)
          residual.append(Array(olderValues[base..<(base + k + 1)]))
        }
      }
      var att: [[[Float]]] = []
      att.reserveCapacity(layerCount)
      for l in 0..<layerCount {
        guard let values = attentionValues[l] else { att.append([]); continue }
        att.append((0..<H).map { h in
          let base = (h * n + i) * T
          return Array(values[base..<(base + pos + 1)])
        })
      }
      let neurons = isNewest ? actValues.map { $0 ?? [Float](repeating: 0, count: ff) } : []
      let lens: [(token: Int, p: Double)] = isNewest ? (1...layerCount).map { (Int(topIDs[$0]), Double(topP[$0])) } : []
      threads.append(Thread(token: ids[pos], residual: residual, attention: att, neurons: neurons, lens: lens, ran: cap.ran))
    }
    let perPosition = layerCount * (d * (H + 2 * KV) * hd + H * hd * d + 3 * d * ff)
    let macs = n * perPosition + layerCount * n * T * hd * H * 2 + (1 + layerCount) * d * c.vocabularySize
    Memory.clearCache()
    return QwenPass(start: start, threads: threads, lastLogits: logits.asArray(Float.self), macs: macs,
      seconds: Date().timeIntervalSince(clock))
  }

  /// The logit lens for an earlier position of the last pass: its vector after every layer, through the
  /// final norm and the readout. Nil if the kept pass no longer matches `ids`.
  func lens(position: Int, ids: [Int]) throws -> [(token: Int, p: Double)]? {
    try checkGPUAdmission()
    guard let model else { throw QwenError.notLoaded }
    guard ids == cachedIDs, position < residuals.count else { return nil }
    let inner = model.model
    let scores = inner.embedTokens.asLinear(inner.norm(residuals[position][1...])).asType(.float32)
    let top = argMax(scores, axis: -1)
    let prob = exp(scores.max(axis: -1) - logSumExp(scores, axis: -1))
    try checkGPUAdmission()
    eval(top, prob)
    let tops = top.asArray(Int32.self), ps = prob.asArray(Float.self)
    Memory.clearCache()
    return zip(tops, ps).map { (Int($0), Double($1)) }
  }

  /// What one feedforward neuron writes into the residual, read through the readout: the tokens it raises and lowers.
  func neuronVotes(layer: Int, neuron: Int) throws -> (raises: [Int], lowers: [Int]) {
    try checkGPUAdmission()
    guard let model else { throw QwenError.notLoaded }
    let c = model.config
    var oneHot = [Float](repeating: 0, count: c.intermediateSize)
    oneHot[neuron] = 1
    let inner = model.model
    let column = inner.layers[layer].mlp.down(MLXArray(oneHot, [1, c.intermediateSize]).asType(inner.norm.weight.dtype))
    let scores = inner.embedTokens.asLinear(column[0]).asType(.float32)
    let order = argSort(scores, axis: -1)
    try checkGPUAdmission()
    eval(order)
    let all = order.asArray(Int32.self).map(Int.init)
    Memory.clearCache()
    return (Array(all.suffix(6).reversed()), Array(all.prefix(4)))
  }

  /// Embedding rows for a set of token ids, as float32 (rows × d).
  func embeddingRows(_ ids: [Int]) throws -> [Float] {
    try checkGPUAdmission()
    guard let model else { throw QwenError.notLoaded }
    var out: [Float] = []
    out.reserveCapacity(ids.count * model.config.hiddenSize)
    // In slices, so the dequantized rows never sit on the GPU all at once.
    for chunk in stride(from: 0, to: ids.count, by: 1024) {
      try checkGPUAdmission()
      let part = Array(ids[chunk..<min(ids.count, chunk + 1024)])
      let rows = model.model.embedTokens(MLXArray(part.map { Int32($0) })).asType(.float32)
      try checkGPUAdmission()
      eval(rows)
      out += rows.asArray(Float.self)
    }
    Memory.clearCache()
    return out
  }

  /// Generates up to `count` tokens after `ids` with the knobs, for side-by-side comparisons. It keeps its
  /// own keys and values, so the instrumented pass is untouched.
  func generate(ids: [Int], count: Int, knobs: RunKnobs, settings: SamplerSettings, seed: UInt64, stops: Set<Int>) throws -> [Int] {
    try checkGPUAdmission()
    guard let model else { throw QwenError.notLoaded }
    var ks: [MLXArray?] = Array(repeating: nil, count: model.config.hiddenLayers)
    var vs = ks
    var seq = ids
    var rng = SplitMix64(seed: seed)
    var start = 0
    for _ in 0..<count {
      try checkGPUAdmission()
      let (x, _) = walk(model, ids: seq, start: start, knobs: knobs, keys: &ks, values: &vs, capture: false)
      let logits = model.model.embedTokens.asLinear(model.model.norm(x[0, seq.count - start - 1])).asType(.float32)
      try checkGPUAdmission()
      eval(logits)
      let row = logits.asArray(Float.self)
      let draw = Sampler.draw(logits: row[...], context: seq, settings: settings, rng: &rng)
      start = seq.count
      seq.append(draw.chosen)
      if stops.contains(draw.chosen) { break }
    }
    Memory.clearCache()
    return Array(seq[ids.count...])
  }

  /// The plain model forward (MLX LM's fused attention path), for checking the instrumented pass.
  func plainLogits(_ ids: [Int]) throws -> [Float] {
    try checkGPUAdmission()
    guard let model else { throw QwenError.notLoaded }
    let out = model(MLXArray(ids.map { Int32($0) }).reshaped(1, -1), cache: nil)[0, ids.count - 1].asType(.float32)
    try checkGPUAdmission()
    eval(out)
    Memory.clearCache()
    return out.asArray(Float.self)
  }
}

nonisolated enum QwenError: LocalizedError {
  case notLoaded, missingFiles, simulator
  var errorDescription: String? {
    switch self {
    case .notLoaded: "Qwen isn't loaded."
    case .missingFiles: "Qwen's weights aren't in this build."
    case .simulator: "Qwen runs on a physical iPhone; the simulator has no GPU for MLX."
    }
  }
}
