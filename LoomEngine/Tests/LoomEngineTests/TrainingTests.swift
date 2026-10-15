import Foundation
import XCTest

@testable import LoomEngine

/// The Teach bench: the hand-written backward pass against MLX autodiff, and real fine-tunes on a few lines.
final class TrainingTests: XCTestCase {
  var w: ModelWeights { ParityTests.weights }
  var tok: Tokenizer { ParityTests.tokenizer }

  func testGradientsMatchMLX() {
    let fx = ParityTests.fixtures["grads"]!
    let seqs = fx["sequences"]!.array!.map { $0.doubles.map(Int.init) }
    let g = Backprop.gradients(w, sequences: seqs, wantEmbedding: true)
    XCTAssertEqual(g.loss, fx["loss"]!.number!, accuracy: 1e-4, "loss")
    for (name, value) in fx["slices"]!.object! {
      let want = value.floats
      var got: [Float]
      switch name {
      case "gf": got = g.gf
      case "b2.g1": got = g.g1[2]
      case "b0.g2": got = g.g2[0]
      case "wte":
        let rows = fx["wte_rows"]!.doubles.map(Int.init)
        got = rows.flatMap { r in Array(g.wte![(r * w.config.d)..<((r + 1) * w.config.d)]) }
      default:
        let l = Int(name.dropFirst().prefix(1))!
        let m = String(name.split(separator: ".")[1])
        let full = g.blocks[l][m]!
        let cols = w.shape(of: m).1
        got = (0..<6).flatMap { i in Array(full[(i * cols)..<(i * cols + 10)]) }
      }
      let scale = want.map(abs).max()! + 1e-6
      var worst: Float = 0
      for i in want.indices { worst = max(worst, abs(got[i] - want[i])) }
      XCTAssertLessThan(worst / scale, 2e-3, "\(name): worst relative error \(worst / scale)")
    }
  }

  func lines(_ texts: [String]) -> [[Int]] { texts.map { [Tokenizer.newline] + tok.encode($0) } }

  func testLoRATeachLowersLossOnItsLines() {
    let mine = lines([
      "spitting bars in the cipher till the sun comes up",
      "the cipher spins and the bars come round again",
      "sun comes up on the cipher and the bars still burn",
      "round again the cipher, round again the sun",
    ])
    var s = TeachSettings()
    s.rank = 4; s.learningRate = 1e-2; s.steps = 40; s.batch = 4
    let teacher = Teacher(base: w, lines: mine, settings: s)
    let before = Backprop.loss(w, sequences: mine)
    var last = teacher.step()
    for _ in 1..<s.steps { last = teacher.step() }
    let after = Backprop.loss(teacher.tuned, sequences: mine)
    XCTAssertLessThan(after, before * 0.6, "LoRA teach: \(before) → \(after)")
    XCTAssertEqual(last.trainedParameters, teacher.adapter.parameterCount)
    XCTAssertLessThan(last.trainedParameters, w.parameterCount / 10)
  }

  func testFullTeachMovesEveryWeight() {
    let mine = lines(["la luna sale y el gato duerme", "el gato duerme bajo la luna"])
    var s = TeachSettings()
    s.full = true; s.learningRate = 3e-3; s.steps = 10; s.batch = 2
    let teacher = Teacher(base: w, lines: mine, settings: s)
    for _ in 0..<s.steps { teacher.step() }
    XCTAssertNotEqual(teacher.tuned.wte, w.wte)
    XCTAssertNotEqual(teacher.tuned.blocks[0].g1, w.blocks[0].g1)
    XCTAssertLessThan(Backprop.loss(teacher.tuned, sequences: mine), Backprop.loss(w, sequences: mine))
  }
}

/// The sphere's geometry.
final class GazeTests: XCTestCase {
  func testViewGivesExactCosines() {
    let w = ParityTests.weights
    let d = w.config.d, V = w.config.vocab
    let (unit, norms) = Gaze.unitCatalogue(w.wte, rows: V, d: d)
    XCTAssertEqual(norms[479], Gaze.norm(Array(w.wte[(479 * d)..<(480 * d)])), accuracy: 1e-5)
    let gaze = Array(unit[(2095 * d)..<(2096 * d)])
    var right = [Float](repeating: 0, count: d), up = right
    right[0] = 1; up[1] = 1
    Gaze.transport(right: &right, up: &up, gaze: gaze)
    let (cos, x, y) = Gaze.view(unit: unit, rows: V, d: d, gaze: gaze, right: right, up: up)
    XCTAssertEqual(cos[2095], 1, accuracy: 1e-4, "a card seen from itself is dead centre")
    for t in [8, 446, 3000] {
      var c: Float = 0
      for j in 0..<d { c += unit[t * d + j] * gaze[j] }
      XCTAssertEqual(cos[t], c, accuracy: 1e-5)
      XCTAssertLessThanOrEqual(x[t] * x[t] + y[t] * y[t] + cos[t] * cos[t], 1.0001, "a projection never exceeds the whole")
    }
    var g = gaze
    var orth: Float = 0, rn: Float = 0
    for j in 0..<d { orth += g[j] * right[j] + right[j] * up[j]; rn += right[j] * right[j] }
    XCTAssertEqual(orth, 0, accuracy: 1e-4)
    XCTAssertEqual(rn, 1, accuracy: 1e-4)
    g = Gaze.slerp(gaze, Array(unit[(446 * d)..<(447 * d)]), 0.5)
    XCTAssertEqual(Gaze.angle(g, gaze), Gaze.angle(g, Array(unit[(446 * d)..<(447 * d)])), accuracy: 1e-4, "halfway is halfway")
  }

  func testTransportIsSmooth() {
    let d = 128
    var a = [Float](repeating: 0, count: d), b = a
    a[0] = 1; b[5] = 1
    var right = [Float](repeating: 0, count: d), up = right
    right[1] = 1; up[2] = 1
    var previous = right
    for k in 1...50 {
      let g = Gaze.slerp(a, b, Float(k) / 50)
      Gaze.transport(right: &right, up: &up, gaze: g)
      XCTAssertLessThan(Gaze.angle(previous, right), 0.1, "the reference axis turns a little each step, never jumps")
      previous = right
    }
  }
}
