import Accelerate
import Foundation

/// The Teach bench's knobs.
public struct TeachSettings: Sendable, Equatable, Codable {
  public var learningRate: Double = 3e-3
  /// LoRA rank; ignored when `full` is on.
  public var rank = 8
  public var steps = 120
  /// Lines averaged per step.
  public var batch = 8
  public var weightDecay: Double = 0
  /// Move every weight instead of training a low-rank adapter.
  public var full = false
  public init() {}
}

public struct TeachProgress: Sendable, Equatable {
  public var step: Int
  public var trainLoss: Double
  /// Loss on lines held back from training (nil when there are too few lines to hold any back).
  public var heldOutLoss: Double?
  /// Passes over the training lines so far.
  public var epochs: Double
  public var trainedParameters: Int
  public var seconds: Double
}

/// A real fine-tune on the phone: AdamW on either a fresh LoRA adapter or on every weight.
/// Not thread-safe; drive it from one task.
public final class Teacher: @unchecked Sendable {
  public let base: ModelWeights
  public let settings: TeachSettings
  public private(set) var adapter: Adapter
  public private(set) var tuned: ModelWeights
  public private(set) var history: [TeachProgress] = []
  let train: [[Int]]
  let heldOut: [[Int]]
  var rng: SplitMix64
  var t = 0
  var m: [String: [Float]] = [:]
  var v: [String: [Float]] = [:]
  let started = Date()

  /// `lines` are token sequences that each start with the newline token.
  public init(base: ModelWeights, lines: [[Int]], settings: TeachSettings, seed: UInt64 = 7) {
    self.base = base
    self.settings = settings
    let usable = lines.filter { $0.count >= 2 }
    if usable.count >= 5 {
      heldOut = usable.enumerated().filter { $0.offset % 5 == 4 }.map(\.element)
      train = usable.enumerated().filter { $0.offset % 5 != 4 }.map(\.element)
    } else {
      heldOut = []
      train = usable
    }
    rng = SplitMix64(seed: seed)
    let c = base.config
    let r = max(1, settings.rank)
    var layers: [[String: Adapter.LowRank]] = []
    for _ in 0..<c.layers {
      var l: [String: Adapter.LowRank] = [:]
      for name in ModelWeights.matrices {
        let (rows, cols) = base.shape(of: name)
        var a = [Float](repeating: 0, count: rows * r)
        let sd = 1 / Float(rows).squareRoot()
        for i in a.indices { a[i] = Self.gaussian(&rng) * sd }
        l[name] = Adapter.LowRank(a: a, b: [Float](repeating: 0, count: r * cols))
      }
      layers.append(l)
    }
    adapter = Adapter(name: "Taught", rank: r, alpha: Float(2 * r), layers: layers)
    tuned = base
  }

  public var trainLineCount: Int { train.count }
  public var heldOutLineCount: Int { heldOut.count }

  public var trainedParameters: Int {
    settings.full ? base.parameterCount : adapter.parameterCount
  }

  /// One AdamW step on a random batch of training lines.
  @discardableResult
  public func step() -> TeachProgress {
    guard !train.isEmpty else {
      return TeachProgress(step: t, trainLoss: 0, heldOutLoss: nil, epochs: 0, trainedParameters: 0, seconds: 0)
    }
    t += 1
    let batch = (0..<settings.batch).map { _ in train[Int(rng.next() % UInt64(train.count))] }
    let g = Backprop.gradients(tuned, sequences: batch, wantEmbedding: settings.full)
    if settings.full {
      for l in 0..<base.config.layers {
        for name in ModelWeights.matrices { adam("b\(l).\(name)", &tuned.blocks[l][name], g.blocks[l][name]!) }
        adam("b\(l).g1", &tuned.blocks[l].g1, g.g1[l])
        adam("b\(l).g2", &tuned.blocks[l].g2, g.g2[l])
      }
      adam("gf", &tuned.gf, g.gf)
      if let gw = g.wte { adam("wte", &tuned.wte, gw) }
    } else {
      let r = adapter.rank
      let s = adapter.alpha / Float(r)
      var layers = adapter.layers
      for l in 0..<base.config.layers {
        for name in ModelWeights.matrices {
          let (rows, cols) = base.shape(of: name)
          var lr = layers[l][name]!
          let dW = g.blocks[l][name]!
          // W' = W + s·A·B  ⇒  dA = s·dW·Bᵀ,  dB = s·Aᵀ·dW
          var dA = Backprop.gemm(dW, lr.b, m: rows, n: r, k: cols, transB: true)
          var dB = Backprop.gemm(lr.a, dW, m: r, n: cols, k: rows, transA: true)
          vDSP.multiply(s, dA, result: &dA)
          vDSP.multiply(s, dB, result: &dB)
          adam("b\(l).\(name).a", &lr.a, dA)
          adam("b\(l).\(name).b", &lr.b, dB)
          layers[l][name] = lr
        }
      }
      adapter = Adapter(name: adapter.name, rank: r, alpha: adapter.alpha, layers: layers)
      tuned = Workshop.compose(base: base, bits: 16, adapters: [(adapter, 1)])
    }
    let held = heldOut.isEmpty || (t % 5 != 0 && t != 1 && t != settings.steps) ? nil : Backprop.loss(tuned, sequences: heldOut)
    let p = TeachProgress(
      step: t, trainLoss: g.loss, heldOutLoss: held,
      epochs: Double(t * settings.batch) / Double(train.count), trainedParameters: trainedParameters,
      seconds: Date().timeIntervalSince(started))
    history.append(p)
    return p
  }

  private func adam(_ key: String, _ p: inout [Float], _ g: [Float]) {
    let b1: Float = 0.9, b2: Float = 0.999, eps: Float = 1e-8
    var mm = m[key] ?? [Float](repeating: 0, count: p.count)
    var vv = v[key] ?? [Float](repeating: 0, count: p.count)
    let lr = Float(settings.learningRate), wd = Float(settings.weightDecay)
    let c1 = 1 - powf(b1, Float(t)), c2 = 1 - powf(b2, Float(t))
    for i in p.indices {
      mm[i] = b1 * mm[i] + (1 - b1) * g[i]
      vv[i] = b2 * vv[i] + (1 - b2) * g[i] * g[i]
      let update = (mm[i] / c1) / ((vv[i] / c2).squareRoot() + eps)
      p[i] -= lr * (update + wd * p[i])
    }
    m[key] = mm
    v[key] = vv
  }

  static func gaussian(_ rng: inout SplitMix64) -> Float {
    let u1 = max(rng.nextUnit(), 1e-12), u2 = rng.nextUnit()
    return Float((-2 * log(u1)).squareRoot() * cos(2 * .pi * u2))
  }
}
