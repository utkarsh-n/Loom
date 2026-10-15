import Accelerate
import Foundation

/// The transformer's shape, read from a `.loom` header.
public struct ModelConfig: Sendable, Equatable {
  public var vocab = 4096, ctx = 128, d = 128, heads = 4, layers = 4, ff = 384
  public var ropeBase: Float = 10000
  public var headDim: Int { d / heads }

  public init() {}

  init(meta: [String: JSONValue]) {
    guard let c = meta["config"] else { return }
    vocab = Int(c["vocab"]?.number ?? 4096)
    ctx = Int(c["ctx"]?.number ?? 128)
    d = Int(c["d"]?.number ?? 128)
    heads = Int(c["heads"]?.number ?? 4)
    layers = Int(c["layers"]?.number ?? 4)
    ff = Int(c["ff"]?.number ?? 384)
    ropeBase = Float(c["rope_base"]?.number ?? 10000)
  }

  /// Multiply-adds for one position through every floor and the readout (attention over `keys` positions).
  public func macsPerPosition(keys: Int) -> Int {
    layers * (4 * d * d + 3 * d * ff + 2 * keys * d) + vocab * d
  }
}

/// One floor's weights. Matrices are row-major (in × out): y = x · W.
public struct BlockWeights: Sendable {
  public var g1: [Float], g2: [Float]
  public var wq: [Float], wk: [Float], wv: [Float], wo: [Float]
  public var wg: [Float], wu: [Float], wd: [Float]

  public subscript(name: String) -> [Float] {
    get {
      switch name {
      case "wq": wq
      case "wk": wk
      case "wv": wv
      case "wo": wo
      case "wg": wg
      case "wu": wu
      case "wd": wd
      case "g1": g1
      default: g2
      }
    }
    set {
      switch name {
      case "wq": wq = newValue
      case "wk": wk = newValue
      case "wv": wv = newValue
      case "wo": wo = newValue
      case "wg": wg = newValue
      case "wu": wu = newValue
      case "wd": wd = newValue
      case "g1": g1 = newValue
      default: g2 = newValue
      }
    }
  }
}

/// Every weight of the model. `wte` is both the card catalogue (input) and the readout (output).
public struct ModelWeights: Sendable {
  public static let matrices = ["wq", "wk", "wv", "wo", "wg", "wu", "wd"]
  public let config: ModelConfig
  public var wte: [Float]
  public var gf: [Float]
  public var blocks: [BlockWeights]
  public let meta: [String: JSONValue]

  public init(file: WeightFile) throws {
    let config = ModelConfig(meta: file.meta)
    self.config = config
    meta = file.meta
    wte = try file.tensor("wte").data
    gf = try file.tensor("gf").data
    blocks = try (0..<config.layers).map { i in
      func t(_ n: String) throws -> [Float] { try file.tensor("b\(i).\(n)").data }
      return BlockWeights(
        g1: try t("g1"), g2: try t("g2"), wq: try t("wq"), wk: try t("wk"), wv: try t("wv"),
        wo: try t("wo"), wg: try t("wg"), wu: try t("wu"), wd: try t("wd"))
    }
  }

  /// (rows, cols) of a named matrix.
  public func shape(of name: String) -> (Int, Int) {
    let c = config
    switch name {
    case "wg", "wu": return (c.d, c.ff)
    case "wd": return (c.ff, c.d)
    default: return (c.d, c.d)
    }
  }

  public var parameterCount: Int {
    wte.count + gf.count + blocks.reduce(0) { n, b in
      n + b.g1.count + b.g2.count + Self.matrices.reduce(0) { $0 + b[$1].count }
    }
  }
}

// MARK: - Knobs that act during the pass

public struct HeadID: Hashable, Sendable, Codable {
  public var layer: Int, head: Int
  public init(layer: Int, head: Int) { self.layer = layer; self.head = head }
}

public struct Steer: Sendable, Equatable {
  public var layer: Int
  public var vector: [Float]
  public var strength: Float
  public init(layer: Int, vector: [Float], strength: Float) { self.layer = layer; self.vector = vector; self.strength = strength }
}

public struct Clamp: Sendable, Equatable, Hashable {
  public var layer: Int, neuron: Int
  public var value: Float
  public init(layer: Int, neuron: Int, value: Float) { self.layer = layer; self.neuron = neuron; self.value = value }
}

public struct RunKnobs: Sendable, Equatable {
  public var muted: Set<HeadID> = []
  public var bypass: Set<Int> = []
  /// Floors to run before the readout (logit lens / early exit). nil runs them all.
  public var depth: Int?
  public var steer: [Steer] = []
  public var clamp: [Clamp] = []
  public init() {}
}

// MARK: - Traces

/// Everything one floor computed for the positions of one pass. Row-major, `rows` rows.
public struct LayerTrace: Sendable {
  public var skipped = false
  public var input: [Float] = [], h1: [Float] = [], q: [Float] = [], k: [Float] = [], v: [Float] = []
  /// Attention weights, heads × rows × keys (keys = every position so far).
  public var att: [Float] = []
  public var keys = 0
  public var heads: [Float] = [], attnOut: [Float] = [], mid: [Float] = []
  public var h2: [Float] = [], gate: [Float] = [], up: [Float] = [], act: [Float] = []
  public var mlpOut: [Float] = [], out: [Float] = []
}

/// One forward pass: rows `start ..< start + rows` of the sequence.
public struct PassTrace: Sendable {
  public var start: Int
  public var rows: Int
  public var x0: [Float] = []
  public var layers: [LayerTrace] = []
  public var final: [Float] = []
  public var logits: [Float] = []
  public var macs = 0
}

/// Keys and values of every position woven so far, per floor (keys already turned by RoPE).
public struct KVCache: Sendable {
  public var k: [[Float]]
  public var v: [[Float]]
  public var count = 0
  public init(layers: Int) {
    k = Array(repeating: [], count: layers)
    v = Array(repeating: [], count: layers)
  }
}

// MARK: - Forward pass

public enum Forward {
  /// Runs positions `start ..< ids.count`, reading earlier positions from `cache` and appending the new ones.
  /// With an empty cache and start 0 this is the prefill; with one new id it is a decode step.
  public static func run(_ w: ModelWeights, ids: [Int], start: Int, cache: inout KVCache, knobs: RunKnobs = RunKnobs()) -> PassTrace {
    let c = w.config
    let d = c.d, dh = c.headDim, ff = c.ff
    precondition(cache.count == start && start < ids.count && ids.count <= c.ctx)
    let n = ids.count - start
    let T = ids.count
    var trace = PassTrace(start: start, rows: n)
    var x = [Float](repeating: 0, count: n * d)
    for i in 0..<n {
      let t = ids[start + i]
      for j in 0..<d { x[i * d + j] = w.wte[t * d + j] }
    }
    trace.x0 = x
    let depth = min(knobs.depth ?? c.layers, c.layers)
    let scale = 1 / Float(dh).squareRoot()
    var macs = 0

    for l in 0..<c.layers {
      var L = LayerTrace()
      L.input = x
      if l >= depth || knobs.bypass.contains(l) {
        L.skipped = true
        L.out = x
        cache.k[l] += [Float](repeating: 0, count: n * d)
        cache.v[l] += [Float](repeating: 0, count: n * d)
        trace.layers.append(L)
        continue
      }
      let b = w.blocks[l]
      let h1 = rmsNorm(x, rows: n, d: d, gain: b.g1)
      var q = matmul(h1, b.wq, m: n, k: d, n: d)
      var k = matmul(h1, b.wk, m: n, k: d, n: d)
      let v = matmul(h1, b.wv, m: n, k: d, n: d)
      rope(&q, rows: n, start: start, d: d, dh: dh, base: c.ropeBase)
      rope(&k, rows: n, start: start, d: d, dh: dh, base: c.ropeBase)
      cache.k[l] += k
      cache.v[l] += v
      let K = cache.k[l], V = cache.v[l]
      var att = [Float](repeating: 0, count: c.heads * n * T)
      var heads = [Float](repeating: 0, count: n * d)
      for h in 0..<c.heads {
        // scores = q_h · K_hᵀ / √dh, masked to the past, softmaxed per row.
        q.withUnsafeBufferPointer { qp in
          K.withUnsafeBufferPointer { kp in
            att.withUnsafeMutableBufferPointer { ap in
              cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans, Int32(n), Int32(T), Int32(dh), scale,
                qp.baseAddress! + h * dh, Int32(d), kp.baseAddress! + h * dh, Int32(d), 0,
                ap.baseAddress! + h * n * T, Int32(T))
            }
          }
        }
        for i in 0..<n {
          let row = h * n * T + i * T
          let last = start + i
          var mx = -Float.infinity
          for j in 0...last { mx = max(mx, att[row + j]) }
          var sum: Float = 0
          for j in 0..<T {
            if j <= last {
              let e = expf(att[row + j] - mx)
              att[row + j] = e
              sum += e
            } else {
              att[row + j] = 0
            }
          }
          let inv = 1 / sum
          for j in 0...last { att[row + j] *= inv }
        }
        if knobs.muted.contains(HeadID(layer: l, head: h)) { continue }
        att.withUnsafeBufferPointer { ap in
          V.withUnsafeBufferPointer { vp in
            heads.withUnsafeMutableBufferPointer { hp in
              cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(n), Int32(dh), Int32(T), 1,
                ap.baseAddress! + h * n * T, Int32(T), vp.baseAddress! + h * dh, Int32(d), 0,
                hp.baseAddress! + h * dh, Int32(d))
            }
          }
        }
      }
      let attnOut = matmul(heads, b.wo, m: n, k: d, n: d)
      vDSP.add(x, attnOut, result: &x)
      let mid = x
      let h2 = rmsNorm(x, rows: n, d: d, gain: b.g2)
      let gate = matmul(h2, b.wg, m: n, k: d, n: ff)
      let up = matmul(h2, b.wu, m: n, k: d, n: ff)
      var act = [Float](repeating: 0, count: n * ff)
      for i in 0..<(n * ff) {
        let g = gate[i]
        act[i] = g / (1 + expf(-g)) * up[i]
      }
      for cl in knobs.clamp where cl.layer == l && cl.neuron < ff {
        for i in 0..<n { act[i * ff + cl.neuron] = cl.value }
      }
      let mlpOut = matmul(act, b.wd, m: n, k: ff, n: d)
      vDSP.add(x, mlpOut, result: &x)
      for s in knobs.steer where s.layer == l && s.vector.count == d {
        for i in 0..<n {
          for j in 0..<d { x[i * d + j] += s.strength * s.vector[j] }
        }
      }
      macs += n * (4 * d * d + 3 * d * ff + 2 * T * d)
      L.h1 = h1; L.q = q; L.k = k; L.v = v; L.att = att; L.keys = T; L.heads = heads
      L.attnOut = attnOut; L.mid = mid; L.h2 = h2; L.gate = gate; L.up = up; L.act = act
      L.mlpOut = mlpOut; L.out = x
      trace.layers.append(L)
    }
    cache.count = T
    trace.final = rmsNorm(x, rows: n, d: d, gain: w.gf)
    trace.logits = readout(w, trace.final, rows: n)
    trace.macs = macs + n * c.vocab * d
    return trace
  }

  /// A whole sequence from scratch.
  public static func prefill(_ w: ModelWeights, ids: [Int], knobs: RunKnobs = RunKnobs()) -> (PassTrace, KVCache) {
    var cache = KVCache(layers: w.config.layers)
    let t = run(w, ids: ids, start: 0, cache: &cache, knobs: knobs)
    return (t, cache)
  }

  /// Logits for `rows` final (already normalised) vectors: the dot product with every token row.
  public static func readout(_ w: ModelWeights, _ final: [Float], rows: Int) -> [Float] {
    let c = w.config
    var logits = [Float](repeating: 0, count: rows * c.vocab)
    final.withUnsafeBufferPointer { fp in
      w.wte.withUnsafeBufferPointer { ep in
        logits.withUnsafeMutableBufferPointer { lp in
          cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans, Int32(rows), Int32(c.vocab), Int32(c.d), 1,
            fp.baseAddress!, Int32(c.d), ep.baseAddress!, Int32(c.d), 0, lp.baseAddress!, Int32(c.vocab))
        }
      }
    }
    return logits
  }

  /// Logit lens: read any residual vector straight through the final norm and the readout.
  public static func lens(_ w: ModelWeights, residual: ArraySlice<Float>) -> [Float] {
    readout(w, rmsNorm(Array(residual), rows: 1, d: w.config.d, gain: w.gf), rows: 1)
  }

  // MARK: Kernels

  public static func matmul(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
    var c = [Float](repeating: 0, count: m * n)
    a.withUnsafeBufferPointer { ap in
      b.withUnsafeBufferPointer { bp in
        c.withUnsafeMutableBufferPointer { cp in
          cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(m), Int32(n), Int32(k), 1,
            ap.baseAddress!, Int32(k), bp.baseAddress!, Int32(n), 0, cp.baseAddress!, Int32(n))
        }
      }
    }
    return c
  }

  public static func rmsNorm(_ x: [Float], rows: Int, d: Int, gain: [Float], eps: Float = 1e-5) -> [Float] {
    var out = [Float](repeating: 0, count: rows * d)
    for i in 0..<rows {
      var ss: Float = 0
      for j in 0..<d { ss += x[i * d + j] * x[i * d + j] }
      let r = 1 / (ss / Float(d) + eps).squareRoot()
      for j in 0..<d { out[i * d + j] = x[i * d + j] * r * gain[j] }
    }
    return out
  }

  /// Turns each adjacent pair (2i, 2i+1) of every head by angle pos·θᵢ, θᵢ = base^(−2i/dh).
  /// `inverse` turns it back (the transpose, used by backprop).
  public static func rope(_ x: inout [Float], rows: Int, start: Int, d: Int, dh: Int, base: Float, inverse: Bool = false) {
    let half = dh / 2
    for i in 0..<rows {
      let pos = Double(start + i)
      for p in 0..<half {
        let theta = pow(Double(base), -Double(2 * p) / Double(dh))
        let c = Float(cos(pos * theta)), s = Float(sin(pos * theta)) * (inverse ? -1 : 1)
        var h = 0
        while h < d {
          let a = i * d + h + 2 * p
          let x0 = x[a], x1 = x[a + 1]
          x[a] = x0 * c - x1 * s
          x[a + 1] = x0 * s + x1 * c
          h += dh
        }
      }
    }
  }
}
