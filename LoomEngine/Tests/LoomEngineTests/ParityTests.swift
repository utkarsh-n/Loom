import Foundation
import XCTest

@testable import LoomEngine

/// Compares the Swift engine with the float64 NumPy reference (`Tools/train/fixtures.py`).
final class ParityTests: XCTestCase {
  static let dir = Bundle.module.url(forResource: "Fixtures", withExtension: nil)!
  static let fixtures: JSONValue = {
    let data = try! Data(contentsOf: dir.appending(path: "fixtures.json"))
    return try! JSONDecoder().decode(JSONValue.self, from: data)
  }()
  static let weights = try! ModelWeights(file: WeightFile(url: dir.appending(path: "model.loom")))
  static let tokenizer = try! Tokenizer(url: dir.appending(path: "tokenizer.json"))
  static let adapter = try! Adapter(file: WeightFile(url: dir.appending(path: "adapter.loom")), layers: weights.config.layers)

  func assertClose(_ a: [Float], _ b: [Float], tol: Float, _ what: String, file: StaticString = #filePath, line: UInt = #line) {
    XCTAssertEqual(a.count, b.count, "\(what): count", file: file, line: line)
    var worst: Float = 0
    var at = 0
    for i in 0..<min(a.count, b.count) where abs(a[i] - b[i]) > worst { worst = abs(a[i] - b[i]); at = i }
    XCTAssertLessThanOrEqual(worst, tol, "\(what): max |Δ| \(worst) at \(at) (swift \(a[at]), ref \(b[at]))", file: file, line: line)
  }

  func testTokenizerMatchesPython() {
    for case let .object(c) in Self.fixtures["tokenizer"]!.array! {
      let text = c["text"]!.string!
      XCTAssertEqual(Tokenizer.pieces(text), c["pieces"]!.array!.compactMap(\.string), "pieces of \(text)")
      XCTAssertEqual(Self.tokenizer.encode(text), c["ids"]!.doubles.map(Int.init), "ids of \(text)")
    }
  }

  func testReplayEndsAtEncoding() {
    let tok = Self.tokenizer
    for piece in Tokenizer.pieces("The extraordinary night, El caballero") {
      XCTAssertEqual(tok.replay(piece).last!.symbols, tok.bpe(piece))
    }
  }

  func knobs(_ name: String) -> (RunKnobs, Int, [(Adapter, Float)]) {
    var k = RunKnobs()
    let steer = Self.fixtures["forward"]!["steer_vector"]!.floats
    switch name {
    case "muted": k.muted = [HeadID(layer: 0, head: 1), HeadID(layer: 2, head: 3)]
    case "bypass": k.bypass = [1]
    case "depth2": k.depth = 2
    case "steer": k.steer = [Steer(layer: 1, vector: steer, strength: 2.5)]
    case "clamp": k.clamp = [Clamp(layer: 2, neuron: 17, value: 6)]
    case "bits4": return (k, 4, [])
    case "bits2": return (k, 2, [])
    case "bits1": return (k, 1, [])
    case "adapter": return (k, 16, [(Self.adapter, 1.5)])
    case "adapter_bits4": return (k, 4, [(Self.adapter, -0.7)])
    default: break
    }
    return (k, 16, [])
  }

  func testForwardMatchesReference() {
    let f = Self.fixtures["forward"]!
    let ids = f["ids"]!.doubles.map(Int.init)
    for (name, entry) in f["cases"]!.object!.sorted(by: { $0.key < $1.key }) {
      let (k, bits, adapters) = knobs(name)
      let w = Workshop.compose(base: Self.weights, bits: bits, adapters: adapters)
      let (t, _) = Forward.prefill(w, ids: ids, knobs: k)
      let V = w.config.vocab, d = w.config.d
      let last = t.rows - 1
      // Low precision amplifies rounding-boundary flips between float32 and float64 scales.
      let tol: Float = bits <= 2 ? 0.05 : 2e-3
      assertClose(Array(t.logits[(last * V)..<((last + 1) * V)]), entry["logits_last"]!.floats, tol: tol, "\(name) logits")
      assertClose(Array(t.final[(last * d)..<((last + 1) * d)]), entry["final_last"]!.floats, tol: tol, "\(name) final")
      if name == "plain" {
        assertClose(Array(t.x0[(last * d)..<((last + 1) * d)]), entry["x0_last"]!.floats, tol: 1e-5, "x0")
        assertClose(t.layers[0].att, entry["att0"]!.floats, tol: 1e-4, "attention floor 1")
        // q of floor 2, heads × first 3 rows × 32: check RoPE turned the right pairs.
        let q = t.layers[1].q
        var q3: [Float] = []
        for h in 0..<w.config.heads { for i in 0..<3 { for j in 0..<w.config.headDim { q3.append(q[i * d + h * w.config.headDim + j]) } } }
        assertClose(q3, entry["q1_first_rows"]!.floats, tol: 1e-4, "RoPE q")
        let ff = w.config.ff
        assertClose(Array(t.layers[2].act[(last * ff)..<((last + 1) * ff)]), entry["act2_last"]!.floats, tol: 1e-3, "neurons floor 3")
        assertClose(t.layers[3].out, entry["out3_all"]!.floats, tol: 2e-3, "residual after floor 4")
        let argmax = (0..<t.rows).map { r in (0..<V).max { t.logits[r * V + $0] < t.logits[r * V + $1] }! }
        XCTAssertEqual(argmax, entry["logits_all_argmax"]!.doubles.map(Int.init))
        for (l, lensRow) in entry["lens"]!.array!.enumerated() {
          let lens = Forward.lens(w, residual: t.layers[l].out[(last * d)..<((last + 1) * d)])
          assertClose(Array(lens.prefix(64)), lensRow.floats, tol: 2e-3, "lens floor \(l + 1)")
        }
      }
    }
  }

  func testKVCacheDecodeEqualsPrefill() {
    let w = Self.weights
    let ids = [1] + Self.tokenizer.encode("The night was dark and the sea was calm")
    let (full, _) = Forward.prefill(w, ids: ids)
    var cache = KVCache(layers: w.config.layers)
    _ = Forward.run(w, ids: Array(ids.prefix(4)), start: 0, cache: &cache)
    var last: PassTrace?
    for n in 5...ids.count { last = Forward.run(w, ids: Array(ids.prefix(n)), start: n - 1, cache: &cache) }
    let V = w.config.vocab
    assertClose(last!.logits, Array(full.logits[((ids.count - 1) * V)...]), tol: 1e-4, "decode vs prefill")
    XCTAssertLessThan(last!.macs, full.macs / 5, "a decode step should cost a fraction of a full pass")
  }

  func testQuantizerMatchesReference() {
    let w = Self.weights
    for (key, value) in Self.fixtures["quantize"]!.object! {
      let bits = Int(key.dropFirst())!
      let q = Quantizer.quantize(w.blocks[0].wg, rows: w.config.d, cols: w.config.ff, bits: bits)
      var top: [Float] = []
      for i in 0..<8 { for j in 0..<8 { top.append(q[i * w.config.ff + j]) } }
      assertClose(top, value.floats, tol: 1e-6, "quantize \(bits) bits")
    }
  }

  func testSamplerMatchesReference() {
    let s = Self.fixtures["sampler"]!
    let logits = s["logits"]!.floats
    let context = s["context"]!.doubles.map(Int.init)
    for c in s["cases"]!.array! {
      let o = c["settings"]!
      var set = SamplerSettings()
      set.temperature = o["temperature"]?.number ?? 1
      set.topK = Int(o["top_k"]?.number ?? 0)
      set.topP = o["top_p"]?.number ?? 1
      set.minP = o["min_p"]?.number ?? 0
      set.repetition = o["repetition"]?.number ?? 1
      for (k, v) in o["bias"]?.object ?? [:] { set.bias[Int(k)!] = v.number! }
      let dist = Sampler.distribution(logits: logits[...], context: context, settings: set)
      XCTAssertEqual(dist.final.filter { $0 > 0 }.count, Int(c["kept"]!.number!), "kept \(o)")
      for pair in c["top"]!.array! {
        let id = Int(pair.array![0].number!), p = pair.array![1].number!
        XCTAssertEqual(dist.final[id], p, accuracy: 1e-6, "p[\(id)] \(o)")
      }
      for pair in c["heat_top"]!.array! {
        let id = Int(pair.array![0].number!), p = pair.array![1].number!
        XCTAssertEqual(dist.heat[id], p, accuracy: 1e-6, "heat[\(id)] \(o)")
      }
    }
  }

  func testSeededDrawsRepeat() {
    let logits = Self.fixtures["sampler"]!["logits"]!.floats
    var a = SplitMix64(seed: 42), b = SplitMix64(seed: 42)
    let s = SamplerSettings()
    for _ in 0..<20 {
      XCTAssertEqual(
        Sampler.draw(logits: logits[...], context: [], settings: s, rng: &a).chosen,
        Sampler.draw(logits: logits[...], context: [], settings: s, rng: &b).chosen)
    }
  }
}
