import Foundation

/// The decode knobs, applied in this order (the same as `ref.sampling_distribution`):
/// logit bias → repetition penalty → temperature → softmax → top-k → top-p → min-p → renormalise.
public struct SamplerSettings: Sendable, Equatable, Codable {
  public var temperature: Double = 0.8
  /// 0 = off.
  public var topK = 0
  /// 1 = off.
  public var topP: Double = 0.95
  /// 0 = off.
  public var minP: Double = 0
  /// 1 = off. Logits of tokens already in the context are divided by it (multiplied when negative).
  public var repetition: Double = 1.1
  /// Added to a token's logit. −∞ bans it.
  public var bias: [Int: Double] = [:]
  public init() {}
}

/// Every stage of one draw, for the sampler panel.
public struct Draw: Sendable {
  /// Token ids by descending final probability (ties keep vocabulary order).
  public let order: [Int]
  /// Logits after bias and repetition.
  public let adjusted: [Double]
  /// Softmax of the raw logits at temperature 1, before any knob ("as trained").
  public let natural: [Double]
  /// After temperature, before the filters.
  public let heat: [Double]
  /// Final distribution after top-k, top-p and min-p, renormalised.
  public let final: [Double]
  /// How many tokens survived the filters.
  public let kept: Int
  /// The uniform number the generator drew, and the token it landed on.
  public let u: Double
  public let chosen: Int
  /// True when a person picked the card instead of the generator (the sphere's latch).
  public var forced = false
  /// −log₂ p of the chosen token under the final distribution (bits of surprise).
  public var surprise: Double { -log2(max(final[chosen], 1e-12)) }

  /// The same draw, landing on a card someone chose by hand.
  public func choosing(_ token: Int) -> Draw {
    var d = Draw(order: order, adjusted: adjusted, natural: natural, heat: heat, final: final, kept: kept, u: u, chosen: token)
    d.forced = true
    return d
  }
}

public enum Sampler {
  public static func softmax(_ z: [Double]) -> [Double] {
    var mx = -Double.infinity
    for v in z where v.isFinite { mx = max(mx, v) }
    var e = z.map { $0.isFinite ? exp($0 - mx) : 0 }
    let s = e.reduce(0, +)
    for i in e.indices { e[i] /= s }
    return e
  }

  /// The distribution the knobs produce. `context` is every token id the repetition penalty can see.
  public static func distribution(logits: ArraySlice<Float>, context: [Int], settings s: SamplerSettings)
    -> (adjusted: [Double], natural: [Double], heat: [Double], final: [Double])
  {
    let raw = logits.map(Double.init)
    let natural = softmax(raw)
    var z = raw
    for (t, b) in s.bias where t >= 0 && t < z.count { z[t] += b }
    if s.repetition != 1 {
      for t in Set(context) where t >= 0 && t < z.count {
        z[t] = z[t] > 0 ? z[t] / s.repetition : z[t] * s.repetition
      }
    }
    if s.temperature <= 0 {
      var p = [Double](repeating: 0, count: z.count)
      p[argmax(z)] = 1
      return (z, natural, p, p)
    }
    let heat = softmax(z.map { $0 / s.temperature })
    // Top-k and top-p keep a prefix of the tokens sorted by probability. Only that prefix is sorted:
    // with 151,936 tokens a full sort on every knob turn is too slow, and the tail never matters.
    let usesK = s.topK > 0 && s.topK < heat.count
    let usesP = s.topP < 1
    var keep: [Bool]
    let first: Int
    if usesK || usesP {
      let head = leading(heat, count: usesK ? s.topK : Int.max, mass: usesP ? s.topP : .infinity)
      var n = head.count
      if usesK { n = min(n, s.topK) }
      if usesP {
        var cum = 0.0
        for (r, t) in head.enumerated() {
          cum += heat[t]
          if cum >= s.topP { n = min(n, r + 1); break }
        }
      }
      keep = [Bool](repeating: false, count: heat.count)
      for t in head.prefix(n) { keep[t] = true }
      first = head[0]
    } else {
      keep = [Bool](repeating: true, count: heat.count)
      first = argmax(heat)
    }
    if s.minP > 0 {
      let floor = s.minP * heat[first]
      for t in heat.indices where heat[t] < floor { keep[t] = false }
    }
    keep[first] = true
    var q = heat.indices.map { keep[$0] ? heat[$0] : 0 }
    let total = q.reduce(0, +)
    for i in q.indices { q[i] /= total }
    return (z, natural, heat, q)
  }

  /// Inverse-CDF draw: walk the cards from most to least likely until the running total passes u.
  public static func pick(_ p: [Double], order: [Int], u: Double) -> Int {
    var cum = 0.0
    var last = order[0]
    for t in order where p[t] > 0 {
      cum += p[t]
      last = t
      if u < cum { return t }
    }
    return last
  }

  public static func draw(logits: ArraySlice<Float>, context: [Int], settings: SamplerSettings, rng: inout SplitMix64) -> Draw {
    let (z, natural, heat, final) = distribution(logits: logits, context: context, settings: settings)
    let order = stableOrder(final)
    let u = rng.nextUnit()
    let chosen = pick(final, order: order, u: u)
    return Draw(
      order: order, adjusted: z, natural: natural, heat: heat, final: final,
      kept: final.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }, u: u, chosen: chosen)
  }

  public static func argmax(_ z: [Double]) -> Int {
    var best = 0
    for i in z.indices where z[i] > z[best] { best = i }
    return best
  }

  /// Indices by descending value; equal values keep ascending index order (numpy's stable argsort of −p).
  /// Zeros (every filtered-out token) are already in index order, so only the rest is sorted.
  public static func stableOrder(_ p: [Double]) -> [Int] {
    var positive: [Int] = [], rest: [Int] = []
    var restIsZero = true
    for i in p.indices {
      if p[i] > 0 {
        positive.append(i)
      } else {
        rest.append(i)
        if p[i] != 0 { restIsZero = false }
      }
    }
    positive.sort { p[$0] != p[$1] ? p[$0] > p[$1] : $0 < $1 }
    if !restIsZero { rest.sort { p[$0] != p[$1] ? p[$0] > p[$1] : $0 < $1 } }
    return positive + rest
  }

  /// The start of `stableOrder(p)` for non-negative `p`: at least `count` entries, or enough that they
  /// add up to `mass` (or everything). Entries below a threshold can't come before entries above it, so
  /// only those above are sorted, the threshold dropping until the prefix is long enough.
  public static func leading(_ p: [Double], count: Int, mass: Double) -> [Int] {
    var mx = 0.0
    for v in p where v > mx { mx = v }
    guard mx > 0 else { return stableOrder(p) }
    var tau = mx * 1e-3
    while true {
      var head: [Int] = []
      for i in p.indices where p[i] >= tau { head.append(i) }
      head.sort { p[$0] != p[$1] ? p[$0] > p[$1] : $0 < $1 }
      if head.count == p.count || head.count >= count { return head }
      var cum = 0.0
      for t in head { cum += p[t] }
      if cum >= mass { return head }
      tau = tau > 1e-280 ? tau * 1e-4 : -1
    }
  }
}

/// A small, fast, reproducible generator (Steele, Lea & Flood 2014). The seed is shown in the app.
public struct SplitMix64: Sendable, Equatable {
  public private(set) var state: UInt64
  public init(seed: UInt64) { state = seed }

  public mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }

  /// Uniform in [0, 1) with 53 bits.
  public mutating func nextUnit() -> Double { Double(next() >> 11) * 0x1.0p-53 }
}
