import Accelerate
import Foundation

/// Gradients of the mean next-token cross-entropy with respect to every weight.
public struct Gradients: Sendable {
  /// Per floor: matrix name → dL/dW (same shape as W).
  public var blocks: [[String: [Float]]]
  public var g1: [[Float]], g2: [[Float]]
  public var gf: [Float]
  /// dL/d(card catalogue); nil unless asked for (only full fine-tuning moves it).
  public var wte: [Float]?
  /// Mean loss in nats per predicted token, and how many tokens it averaged.
  public var loss: Double
  public var tokens: Int
}

/// Hand-written reverse pass through the forward pass in `Model.swift`, checked against MLX autodiff
/// by `ParityTests.testGradientsMatchMLX`. Knobs that change the pass (muting, bypass, steering) are
/// not differentiated: training always runs the plain machine.
public enum Backprop {
  public static func gradients(_ w: ModelWeights, sequences: [[Int]], wantEmbedding: Bool = false) -> Gradients {
    let c = w.config
    let d = c.d, ff = c.ff, V = c.vocab, dh = c.headDim
    var g = Gradients(
      blocks: (0..<c.layers).map { _ in
        Dictionary(uniqueKeysWithValues: ModelWeights.matrices.map { m in
          let (r, cc) = w.shape(of: m)
          return (m, [Float](repeating: 0, count: r * cc))
        })
      },
      g1: Array(repeating: [Float](repeating: 0, count: d), count: c.layers),
      g2: Array(repeating: [Float](repeating: 0, count: d), count: c.layers),
      gf: [Float](repeating: 0, count: d),
      wte: wantEmbedding ? [Float](repeating: 0, count: V * d) : nil,
      loss: 0, tokens: 0)
    let total = sequences.reduce(0) { $0 + max(0, $1.count - 1) }
    guard total > 0 else { return g }
    let invTotal = 1 / Float(total)
    var lossSum = 0.0

    for seqFull in sequences where seqFull.count >= 2 {
      let seq = Array(seqFull.prefix(c.ctx + 1))
      let inputs = Array(seq.dropLast())
      let targets = Array(seq.dropFirst())
      let T = inputs.count
      let (trace, _) = Forward.prefill(w, ids: inputs)

      // Readout: dlogits = (softmax − onehot) / total.
      var dlogits = [Float](repeating: 0, count: T * V)
      for i in 0..<T {
        let row = i * V
        var mx = -Float.infinity
        for j in 0..<V { mx = max(mx, trace.logits[row + j]) }
        var sum: Float = 0
        for j in 0..<V {
          let e = expf(trace.logits[row + j] - mx)
          dlogits[row + j] = e
          sum += e
        }
        lossSum += Double(logf(sum) + mx - trace.logits[row + targets[i]])
        let inv = 1 / sum
        for j in 0..<V { dlogits[row + j] *= inv * invTotal }
        dlogits[row + targets[i]] -= invTotal
      }
      var dfinal = gemm(dlogits, w.wte, m: T, n: d, k: V)                       // T×V · V×d
      if wantEmbedding {
        gemmAccumulate(&g.wte!, dlogits, trace.final, m: V, n: d, k: T, transA: true)   // (T×V)ᵀ · T×d
      }
      let xL = trace.layers.last!.out
      var dx = rmsBackward(x: xL, gain: w.gf, dy: dfinal, rows: T, d: d, dGain: &g.gf)
      dfinal.removeAll()

      for l in stride(from: c.layers - 1, through: 0, by: -1) {
        let L = trace.layers[l]
        let b = w.blocks[l]
        // MLP: out = mid + (silu(G) ⊙ U) · Wd
        let dAct = gemm(dx, b.wd, m: T, n: ff, k: d, transB: true)               // T×d · (ff×d)ᵀ
        gemmAccumulate(&g.blocks[l]["wd"]!, L.act, dx, m: ff, n: d, k: T, transA: true)
        var dG = [Float](repeating: 0, count: T * ff)
        var dU = [Float](repeating: 0, count: T * ff)
        for i in 0..<(T * ff) {
          let gv = L.gate[i]
          let s = 1 / (1 + expf(-gv))
          let silu = gv * s
          dU[i] = dAct[i] * silu
          dG[i] = dAct[i] * L.up[i] * s * (1 + gv * (1 - s))
        }
        var dh2 = gemm(dG, b.wg, m: T, n: d, k: ff, transB: true)
        gemmAccumulate(&dh2, dU, b.wu, m: T, n: d, k: ff, transB: true)
        gemmAccumulate(&g.blocks[l]["wg"]!, L.h2, dG, m: d, n: ff, k: T, transA: true)
        gemmAccumulate(&g.blocks[l]["wu"]!, L.h2, dU, m: d, n: ff, k: T, transA: true)
        let dmidNorm = rmsBackward(x: L.mid, gain: b.g2, dy: dh2, rows: T, d: d, dGain: &g.g2[l])
        var dmid = dx
        vDSP.add(dmid, dmidNorm, result: &dmid)

        // Attention: mid = x_in + heads · Wo
        let dheads = gemm(dmid, b.wo, m: T, n: d, k: d, transB: true)
        gemmAccumulate(&g.blocks[l]["wo"]!, L.heads, dmid, m: d, n: d, k: T, transA: true)
        var dq = [Float](repeating: 0, count: T * d)
        var dk = [Float](repeating: 0, count: T * d)
        var dv = [Float](repeating: 0, count: T * d)
        let scale = 1 / Float(dh).squareRoot()
        for h in 0..<c.heads {
          let P = Array(L.att[(h * T * T)..<((h + 1) * T * T)])
          // dP = dO_h · v_hᵀ ;  dv_h = Pᵀ · dO_h
          var dP = [Float](repeating: 0, count: T * T)
          dheads.withUnsafeBufferPointer { op in
            L.v.withUnsafeBufferPointer { vp in
              dP.withUnsafeMutableBufferPointer { pp in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Int32(T), Int32(T), Int32(dh), 1,
                  op.baseAddress! + h * dh, Int32(d), vp.baseAddress! + h * dh, Int32(d), 0, pp.baseAddress!, Int32(T))
              }
            }
          }
          P.withUnsafeBufferPointer { pp in
            dheads.withUnsafeBufferPointer { op in
              dv.withUnsafeMutableBufferPointer { vp in
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Int32(T), Int32(dh), Int32(T), 1,
                  pp.baseAddress!, Int32(T), op.baseAddress! + h * dh, Int32(d), 0, vp.baseAddress! + h * dh, Int32(d))
              }
            }
          }
          // Softmax backward: dS = P ⊙ (dP − Σⱼ dP·P), then the 1/√dh scale.
          var dS = [Float](repeating: 0, count: T * T)
          for i in 0..<T {
            var dot: Float = 0
            for j in 0...i { dot += dP[i * T + j] * P[i * T + j] }
            for j in 0...i { dS[i * T + j] = P[i * T + j] * (dP[i * T + j] - dot) * scale }
          }
          // dq_h = dS · k_h ;  dk_h = dSᵀ · q_h
          dS.withUnsafeBufferPointer { sp in
            L.k.withUnsafeBufferPointer { kp in
              dq.withUnsafeMutableBufferPointer { qp in
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(T), Int32(dh), Int32(T), 1,
                  sp.baseAddress!, Int32(T), kp.baseAddress! + h * dh, Int32(d), 0, qp.baseAddress! + h * dh, Int32(d))
              }
            }
            L.q.withUnsafeBufferPointer { qp in
              dk.withUnsafeMutableBufferPointer { kp in
                cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, Int32(T), Int32(dh), Int32(T), 1,
                  sp.baseAddress!, Int32(T), qp.baseAddress! + h * dh, Int32(d), 0, kp.baseAddress! + h * dh, Int32(d))
              }
            }
          }
        }
        // RoPE is a rotation, so its transpose turns the gradient back by the same angle.
        Forward.rope(&dq, rows: T, start: 0, d: d, dh: dh, base: c.ropeBase, inverse: true)
        Forward.rope(&dk, rows: T, start: 0, d: d, dh: dh, base: c.ropeBase, inverse: true)
        var dh1 = gemm(dq, b.wq, m: T, n: d, k: d, transB: true)
        gemmAccumulate(&dh1, dk, b.wk, m: T, n: d, k: d, transB: true)
        gemmAccumulate(&dh1, dv, b.wv, m: T, n: d, k: d, transB: true)
        gemmAccumulate(&g.blocks[l]["wq"]!, L.h1, dq, m: d, n: d, k: T, transA: true)
        gemmAccumulate(&g.blocks[l]["wk"]!, L.h1, dk, m: d, n: d, k: T, transA: true)
        gemmAccumulate(&g.blocks[l]["wv"]!, L.h1, dv, m: d, n: d, k: T, transA: true)
        let dinNorm = rmsBackward(x: L.input, gain: b.g1, dy: dh1, rows: T, d: d, dGain: &g.g1[l])
        dx = dmid
        vDSP.add(dx, dinNorm, result: &dx)
      }
      if wantEmbedding {
        for i in 0..<T {
          let t = inputs[i]
          for j in 0..<d { g.wte![t * d + j] += dx[i * d + j] }
        }
      }
    }
    g.loss = lossSum / Double(total)
    g.tokens = total
    return g
  }

  /// Mean next-token loss without gradients (the held-out curve).
  public static func loss(_ w: ModelWeights, sequences: [[Int]]) -> Double {
    var sum = 0.0, n = 0
    for s in sequences where s.count >= 2 {
      let seq = Array(s.prefix(w.config.ctx + 1))
      let (t, _) = Forward.prefill(w, ids: Array(seq.dropLast()))
      let V = w.config.vocab
      for i in 0..<(seq.count - 1) {
        let row = t.logits[(i * V)..<((i + 1) * V)]
        let mx = row.max()!
        var z: Float = 0
        for v in row { z += expf(v - mx) }
        sum += Double(logf(z) + mx - row[row.startIndex + seq[i + 1]])
        n += 1
      }
    }
    return n > 0 ? sum / Double(n) : 0
  }

  // y = x · r · g with r = 1/√(mean x² + ε):  dx = r·(ĝ − x̂·mean(ĝ ⊙ x̂)),  ĝ = dy ⊙ g
  static func rmsBackward(x: [Float], gain: [Float], dy: [Float], rows: Int, d: Int, dGain: inout [Float], eps: Float = 1e-5) -> [Float] {
    var dx = [Float](repeating: 0, count: rows * d)
    for i in 0..<rows {
      var ss: Float = 0
      for j in 0..<d { ss += x[i * d + j] * x[i * d + j] }
      let r = 1 / (ss / Float(d) + eps).squareRoot()
      var dot: Float = 0
      for j in 0..<d {
        let xh = x[i * d + j] * r
        let gh = dy[i * d + j] * gain[j]
        dGain[j] += dy[i * d + j] * xh
        dot += gh * xh
      }
      dot /= Float(d)
      for j in 0..<d {
        let xh = x[i * d + j] * r
        dx[i * d + j] = r * (dy[i * d + j] * gain[j] - xh * dot)
      }
    }
    return dx
  }

  /// C = op(A) · op(B), with op = transpose when asked. A is m×k (or k×m), B is k×n (or n×k).
  static func gemm(_ a: [Float], _ b: [Float], m: Int, n: Int, k: Int, transA: Bool = false, transB: Bool = false) -> [Float] {
    var c = [Float](repeating: 0, count: m * n)
    gemmInto(&c, a, b, m: m, n: n, k: k, transA: transA, transB: transB, beta: 0)
    return c
  }

  static func gemmAccumulate(_ c: inout [Float], _ a: [Float], _ b: [Float], m: Int, n: Int, k: Int, transA: Bool = false, transB: Bool = false) {
    gemmInto(&c, a, b, m: m, n: n, k: k, transA: transA, transB: transB, beta: 1)
  }

  private static func gemmInto(_ c: inout [Float], _ a: [Float], _ b: [Float], m: Int, n: Int, k: Int, transA: Bool, transB: Bool, beta: Float) {
    a.withUnsafeBufferPointer { ap in
      b.withUnsafeBufferPointer { bp in
        c.withUnsafeMutableBufferPointer { cp in
          cblas_sgemm(
            CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
            Int32(m), Int32(n), Int32(k), 1,
            ap.baseAddress!, Int32(transA ? m : k), bp.baseAddress!, Int32(transB ? k : n), beta,
            cp.baseAddress!, Int32(n))
        }
      }
    }
  }
}
