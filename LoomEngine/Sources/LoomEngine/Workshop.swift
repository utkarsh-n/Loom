import Accelerate
import Foundation

/// A LoRA adapter: for each adapted matrix W (in × out), A (in × r) and B (r × out).
/// Mixed into the base as W + strength · (α / r) · A·B (`Tools/train/adapters.py` writes them).
public struct Adapter: Sendable {
  public let name: String
  public let rank: Int
  public let alpha: Float
  /// Per floor: matrix name → (A, B).
  public let layers: [[String: LowRank]]
  public let meta: [String: JSONValue]

  public struct LowRank: Sendable {
    public var a: [Float]
    public var b: [Float]
    public init(a: [Float], b: [Float]) { self.a = a; self.b = b }
  }

  public init(name: String, rank: Int, alpha: Float, layers: [[String: LowRank]], meta: [String: JSONValue] = [:]) {
    self.name = name; self.rank = rank; self.alpha = alpha; self.layers = layers; self.meta = meta
  }

  public init(file: WeightFile, layers count: Int) throws {
    meta = file.meta
    name = file.meta["name"]?.string ?? "adapter"
    rank = Int(file.meta["rank"]?.number ?? 8)
    alpha = Float(file.meta["alpha"]?.number ?? 16)
    layers = (0..<count).map { i in
      var out: [String: LowRank] = [:]
      for m in ModelWeights.matrices {
        if let a = file.tensors["b\(i).\(m)_a"], let b = file.tensors["b\(i).\(m)_b"] {
          out[m] = LowRank(a: a.data, b: b.data)
        }
      }
      return out
    }
  }

  public var parameterCount: Int {
    layers.reduce(0) { n, l in n + l.values.reduce(0) { $0 + $1.a.count + $1.b.count } }
  }
}

/// Weight precision: symmetric round-to-nearest per output column, the same rule as `ref.quantize`.
public enum Quantizer {
  public static let choices = [16, 8, 6, 4, 3, 2, 1]

  /// `w` is rows × cols, row-major; each column gets its own scale.
  public static func quantize(_ w: [Float], rows: Int, cols: Int, bits: Int) -> [Float] {
    guard bits < 16 else { return w }
    var out = w
    for j in 0..<cols {
      var absMax = 0.0, absSum = 0.0
      for i in 0..<rows {
        let v = Double(w[i * cols + j]).magnitude
        absMax = max(absMax, v)
        absSum += v
      }
      if bits == 1 {
        let mean = absSum / Double(rows)
        for i in 0..<rows {
          let v = w[i * cols + j]
          out[i * cols + j] = Float(v > 0 ? mean : v < 0 ? -mean : 0)
        }
        continue
      }
      let levels = Double((1 << (bits - 1)) - 1)
      let scale = absMax > 0 ? absMax / levels : 1
      for i in 0..<rows {
        let q = (Double(w[i * cols + j]) / scale).rounded(.toNearestOrEven)
        out[i * cols + j] = Float(min(max(q, -levels), levels) * scale)
      }
    }
    return out
  }

  /// Bytes needed to store `params` weights at `bits` (16 = half precision, the app's shipped form).
  public static func bytes(params: Int, bits: Int) -> Int { params * bits / 8 }
}

/// Builds the weights a pass actually uses from the base, the precision knob and the adapter faders.
/// The base is rounded first and the adapters are added at full precision, as QLoRA does.
public enum Workshop {
  public static func compose(base: ModelWeights, bits: Int, adapters: [(Adapter, Float)]) -> ModelWeights {
    var w = base
    let active = adapters.filter { $0.1 != 0 }
    guard bits < 16 || !active.isEmpty else { return base }
    let c = base.config
    if bits < 16 {
      // The card catalogue is rounded per token row (it is read row-wise and used as the readout).
      w.wte = transpose(Quantizer.quantize(transpose(base.wte, rows: c.vocab, cols: c.d), rows: c.d, cols: c.vocab, bits: bits), rows: c.d, cols: c.vocab)
    }
    for l in 0..<c.layers {
      for m in ModelWeights.matrices {
        let (rows, cols) = base.shape(of: m)
        var mat = Quantizer.quantize(base.blocks[l][m], rows: rows, cols: cols, bits: bits)
        for (adapter, strength) in active {
          guard l < adapter.layers.count, let lr = adapter.layers[l][m] else { continue }
          let s = strength * adapter.alpha / Float(adapter.rank)
          lr.a.withUnsafeBufferPointer { ap in
            lr.b.withUnsafeBufferPointer { bp in
              mat.withUnsafeMutableBufferPointer { mp in
                cblas_sgemm(
                  CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), Int32(cols), Int32(adapter.rank), s,
                  ap.baseAddress!, Int32(adapter.rank), bp.baseAddress!, Int32(cols), 1, mp.baseAddress!, Int32(cols))
              }
            }
          }
        }
        w.blocks[l][m] = mat
      }
    }
    return w
  }

  public static func transpose(_ x: [Float], rows: Int, cols: Int) -> [Float] {
    var out = [Float](repeating: 0, count: x.count)
    vDSP_mtrans(x, 1, &out, 1, vDSP_Length(cols), vDSP_Length(rows))
    return out
  }
}
