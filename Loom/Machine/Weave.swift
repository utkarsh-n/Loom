import Foundation
import LoomEngine

/// What the machine drawing needs to know about one thread (one position of the sequence),
/// read out of the pass that computed it.
nonisolated struct Thread: Sendable {
  let token: Int
  /// The residual stream: the token's card row, then the thread after each floor (count = floors + 1).
  var residual: [[Float]]
  /// Attention weights of this thread over every earlier thread, per floor, per head.
  var attention: [[[Float]]]
  /// SwiGLU neuron activations per floor.
  var neurons: [[Float]]
  /// Logit lens: the most likely next card if the machine stopped after each floor, with its probability.
  var lens: [(token: Int, p: Double)]
  /// Which floors actually ran (bypass and depth skip some).
  var ran: [Bool]

  /// Keeps the first `k` numbers and the last of each vector and drops the neurons (Qwen's older
  /// positions; the newest keeps everything).
  mutating func trim(keeping k: Int) {
    residual = residual.map { $0.count > k + 1 ? Array($0.prefix(k)) + [$0[$0.count - 1]] : $0 }
    neurons = []
  }
}

/// One generated token: where it landed and how likely the draw made it. (Not the whole draw: Qwen's
/// holds four 151,936-long distributions.)
nonisolated struct Stitch: Sendable {
  let position: Int
  let chosen: Int
  /// Its probability under the final distribution.
  let probability: Double
  /// Picked by hand (the sphere's latch) rather than drawn.
  let forced: Bool

  init(position: Int, draw: Draw) {
    self.position = position
    chosen = draw.chosen
    probability = draw.final[draw.chosen]
    forced = draw.forced
  }
}

/// The cloth so far: the token sequence, the threads of the window the shuttles can reach, and the draws.
nonisolated struct Weave: Sendable {
  /// Every token on the tape, starting with the newline that opens a line.
  var ids: [Int] = []
  /// Index of the first woven token (everything before came from the prompt and cartridge).
  var promptEnd = 0
  /// Threads for `ids[windowStart...]`.
  var threads: [Thread] = []
  var windowStart = 0
  var stitches: [Stitch] = []
  /// Logits of the newest thread.
  var lastLogits: [Float] = []
  var macsLastStep = 0
  var macsTotal = 0
  /// Wall-clock seconds the last pass took on this phone.
  var secondsLastStep = 0.0
  var finished = false

  var woven: ArraySlice<Int> { ids[promptEnd...] }
  var window: ArraySlice<Int> { ids[windowStart...] }
}

/// The knobs that change the pass (as opposed to the draw), bundled so a change can be compared.
nonisolated struct PassSettings: Sendable, Equatable {
  var knobs = RunKnobs()
  var context = 128
  var kvCache = true
}

nonisolated enum Loomwork {
  /// Reads every thread computed by a pass out of its trace.
  static func threads(from t: PassTrace, ids: [Int], weights w: ModelWeights) -> [Thread] {
    let c = w.config
    let d = c.d, ff = c.ff
    return (0..<t.rows).map { i in
      let pos = t.start + i
      var residual = [Array(t.x0[(i * d)..<((i + 1) * d)])]
      var attention: [[[Float]]] = []
      var neurons: [[Float]] = []
      var lens: [(Int, Double)] = []
      var ran: [Bool] = []
      for L in t.layers {
        residual.append(Array(L.out[(i * d)..<((i + 1) * d)]))
        ran.append(!L.skipped)
        if L.skipped {
          attention.append([])
          neurons.append([Float](repeating: 0, count: ff))
        } else {
          attention.append((0..<c.heads).map { h in
            let row = h * t.rows * L.keys + i * L.keys
            return Array(L.att[row..<(row + pos + 1)])
          })
          neurons.append(Array(L.act[(i * ff)..<((i + 1) * ff)]))
        }
        let lensLogits = Forward.lens(w, residual: L.out[(i * d)..<((i + 1) * d)])
        lens.append(top(lensLogits))
      }
      return Thread(token: ids[pos], residual: residual, attention: attention, neurons: neurons, lens: lens, ran: ran)
    }
  }

  static func top(_ logits: [Float]) -> (Int, Double) {
    var best = 0
    var mx = -Float.infinity
    for (i, v) in logits.enumerated() where v > mx { mx = v; best = i }
    var z = 0.0
    for v in logits { z += exp(Double(v - mx)) }
    return (best, 1 / z)
  }

  /// Top-n token ids and probabilities of a logit row.
  static func topN(_ logits: ArraySlice<Float>, n: Int) -> [(Int, Double)] {
    let p = Sampler.softmax(logits.map(Double.init))
    return Sampler.stableOrder(p).prefix(n).map { ($0, p[$0]) }
  }
}
