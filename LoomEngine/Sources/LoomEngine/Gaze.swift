import Accelerate
import Foundation

/// First-person geometry for the sphere: where every card sits as seen from a gaze direction in the
/// model's 128-dimensional readout space.
///
/// The angle between the gaze and a card is exact (it is the arccos of their cosine in all 128
/// dimensions), and because the readout is a dot product, cards near the centre of view are the ones the
/// next-card scores favour. The direction around the gaze is a projection: the card's component
/// orthogonal to the gaze, read along two reference axes that are carried smoothly as the gaze moves
/// (parallel transport, as Gyre carries its cage frame).
public enum Gaze {
  /// Every card row scaled to length 1 (rows × d), and each row's original length.
  public static func unitCatalogue(_ wte: [Float], rows: Int, d: Int) -> (unit: [Float], norms: [Float]) {
    var unit = wte
    var norms = [Float](repeating: 0, count: rows)
    for r in 0..<rows {
      var ss: Float = 0
      for j in 0..<d { ss += wte[r * d + j] * wte[r * d + j] }
      let n = ss.squareRoot()
      norms[r] = n
      let inv = n > 0 ? 1 / n : 0
      for j in 0..<d { unit[r * d + j] *= inv }
    }
    return (unit, norms)
  }

  /// The direction a residual vector points in readout space: RMSNorm with the final gains, then length 1.
  public static func heading(_ residual: ArraySlice<Float>, gain: [Float]) -> [Float] {
    let d = gain.count
    var ss: Float = 0
    for v in residual { ss += v * v }
    let r = 1 / (ss / Float(d) + 1e-5).squareRoot()
    var h = zip(residual, gain).map { $0 * r * $1 }
    normalize(&h)
    return h
  }

  /// For each card: cosine with the gaze, and its coordinates along the two reference axes.
  /// `unit` is rows × d; returns three arrays of `rows`.
  public static func view(unit: [Float], rows: Int, d: Int, gaze: [Float], right: [Float], up: [Float])
    -> (cos: [Float], x: [Float], y: [Float])
  {
    var axes = [Float](repeating: 0, count: d * 3)
    for j in 0..<d {
      axes[j * 3] = gaze[j]
      axes[j * 3 + 1] = right[j]
      axes[j * 3 + 2] = up[j]
    }
    var out = [Float](repeating: 0, count: rows * 3)
    unit.withUnsafeBufferPointer { u in
      axes.withUnsafeBufferPointer { a in
        out.withUnsafeMutableBufferPointer { o in
          cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(rows), 3, Int32(d), 1,
            u.baseAddress!, Int32(d), a.baseAddress!, 3, 0, o.baseAddress!, 3)
        }
      }
    }
    var c = [Float](repeating: 0, count: rows), x = c, y = c
    for r in 0..<rows {
      c[r] = out[r * 3]; x[r] = out[r * 3 + 1]; y[r] = out[r * 3 + 2]
    }
    return (c, x, y)
  }

  /// Spherical interpolation between two unit vectors (falls back to a straight blend when parallel).
  public static func slerp(_ a: [Float], _ b: [Float], _ t: Float) -> [Float] {
    var dot: Float = 0
    for i in a.indices { dot += a[i] * b[i] }
    dot = max(-1, min(1, dot))
    let theta = acos(dot)
    var out: [Float]
    if theta < 1e-4 {
      out = zip(a, b).map { $0 + t * ($1 - $0) }
    } else {
      let s = sin(theta)
      let wa = sin((1 - t) * theta) / s, wb = sin(t * theta) / s
      out = zip(a, b).map { wa * $0 + wb * $1 }
    }
    normalize(&out)
    return out
  }

  /// Carries the two reference axes along as the gaze moves: each is made orthogonal to the gaze (and
  /// to the other) with the smallest change, so cards don't swirl as you ride.
  public static func transport(right: inout [Float], up: inout [Float], gaze: [Float]) {
    orthogonalize(&right, against: [gaze])
    if norm(right) < 1e-4 { right = fallback(orthogonalTo: [gaze, up]) }
    normalize(&right)
    orthogonalize(&up, against: [gaze, right])
    if norm(up) < 1e-4 { up = fallback(orthogonalTo: [gaze, right]) }
    normalize(&up)
  }

  /// Angle in radians between two unit vectors.
  public static func angle(_ a: [Float], _ b: [Float]) -> Float {
    var dot: Float = 0
    for i in a.indices { dot += a[i] * b[i] }
    return acos(max(-1, min(1, dot)))
  }

  // MARK: Helpers

  public static func norm(_ v: [Float]) -> Float { v.reduce(0) { $0 + $1 * $1 }.squareRoot() }

  public static func normalize(_ v: inout [Float]) {
    let n = norm(v)
    if n > 0 { for i in v.indices { v[i] /= n } }
  }

  static func orthogonalize(_ v: inout [Float], against basis: [[Float]]) {
    for b in basis {
      var dot: Float = 0
      for i in v.indices { dot += v[i] * b[i] }
      for i in v.indices { v[i] -= dot * b[i] }
    }
  }

  /// A deterministic vector orthogonal to the given ones.
  public static func fallback(orthogonalTo basis: [[Float]]) -> [Float] {
    let d = basis[0].count
    for k in 0..<d {
      var v = [Float](repeating: 0, count: d)
      v[k] = 1
      orthogonalize(&v, against: basis)
      if norm(v) > 0.3 {
        normalize(&v)
        return v
      }
    }
    return [Float](repeating: 0, count: d)
  }
}
