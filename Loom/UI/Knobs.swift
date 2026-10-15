import LoomEngine
import SwiftUI

enum KnobID: String, CaseIterable, Identifiable {
  case temperature, topK, topP, minP, repetition, length, seed, pace
  case precision, context, depth, age, kvCache
  case scripture, bard, gothic, twain, almanac, steer, steerFloor
  case taught, teachRate, teachRank, teachSteps, teachBatch
  var id: String { rawValue }
}

enum ConsolePage: String, CaseIterable, Identifiable {
  case decode = "Decode"
  case model = "Model"
  case style = "Style"
  var id: String { rawValue }
}

struct KnobDetent: Hashable {
  let value: Double
  let label: String
}

/// A knob's range, mapping, detents and words (Gyre's KnobSpec, pointed at LoomModel).
struct KnobSpec: Identifiable {
  let id: KnobID
  let title: String
  var glyph: String? = nil
  var symbol = "dial.medium"
  let tint: Color
  var range: ClosedRange<Double>
  var defaultValue: Double
  var logarithmic = false
  /// Where the value arc starts; bipolar knobs light up from their zero.
  var origin: Double? = nil
  var detents: [KnobDetent] = []
  /// Values snap to this step while dragging.
  var step: Double = 0
  var badge: Provenance
  var hint: String
  var detail: String
  let keyPath: ReferenceWritableKeyPath<LoomModel, Double>
  var format: (Double) -> String

  var isSwitch: Bool { step >= 1 }

  func position(of value: Double) -> Double {
    let v = clamped(value, range.lowerBound, range.upperBound)
    if logarithmic { return log(v / range.lowerBound) / log(range.upperBound / range.lowerBound) }
    return (v - range.lowerBound) / (range.upperBound - range.lowerBound)
  }

  func value(at position: Double) -> Double {
    let p = clamped(position, 0, 1)
    var v =
      logarithmic
      ? range.lowerBound * pow(range.upperBound / range.lowerBound, p)
      : range.lowerBound + p * (range.upperBound - range.lowerBound)
    if step > 0 { v = (v / step).rounded() * step }
    return clamped(v, range.lowerBound, range.upperBound)
  }

  var originPosition: Double { position(of: origin ?? range.lowerBound) }

  func detent(at value: Double) -> KnobDetent? {
    detents.first { abs(position(of: $0.value) - position(of: value)) < 0.004 }
  }

  static func label(_ names: [String]) -> (Double) -> String {
    { v in names[clamped(Int(v.rounded()), 0, names.count - 1)] }
  }

  static func spec(_ id: KnobID, for m: LoomModel) -> KnobSpec? { all(for: m).first { $0.id == id } }
  static func all(for m: LoomModel) -> [KnobSpec] { decode(for: m) + model(for: m) + style(for: m) }

  static func page(_ page: ConsolePage, for m: LoomModel) -> [KnobSpec] {
    switch page {
    case .decode: decode(for: m)
    case .model: model(for: m)
    case .style: style(for: m)
    }
  }

  // MARK: Qwen's knobs
  // The same knobs where they mean the same thing. Precision, checkpoints and the five fine-tunes belong
  // to the 1.4M model (Qwen ships at 4 bits, as one finished model).

  static func decode(for m: LoomModel) -> [KnobSpec] {
    guard m.isQwen else { return decode }
    let vocab = m.config.vocab.formatted()
    return decode.map { spec in
      var s = spec
      switch s.id {
      case .topK: s.detail = "Sort the tokens by probability, keep the first k, renormalise. Off keeps all \(vocab)."
      case .length:
        s.hint = "How many tokens to generate. Qwen also stops when it ends its reply (⟨im_end⟩), or, without the chat template, at a line break."
        s.detail = "Loom gives Qwen a 128-token context, template included, so every position's attention can be read back to the phone."
      case .pace:
        s.hint = "How fast to show generation. A Qwen token takes a fraction of a second on this phone's GPU; Slow replays its 36 layers one by one."
        s.detail = "Step waits for you. Slow replays each token's pass stage by stage. Fast and Instant just generate."
      default: break
      }
      return s
    }
  }

  static func model(for m: LoomModel) -> [KnobSpec] {
    guard m.isQwen else { return model }
    let layers = m.config.layers
    var context = spec(.context, in: model)
    context.detail = "The window the model runs over. Qwen was trained on far longer contexts; Loom stops at 128 tokens so every attention weight fits on the phone."
    var depth = spec(.depth, in: model)
    depth.range = 1...Double(layers)
    depth.defaultValue = Double(layers)
    depth.detents = [KnobDetent(value: Double(layers) / 2, label: "Halfway"), KnobDetent(value: Double(layers), label: "All")]
    depth.format = { "\(Int($0)) of \(layers)" }
    return [context, depth, spec(.kvCache, in: model)]
  }

  static func style(for m: LoomModel) -> [KnobSpec] {
    guard m.isQwen else { return style }
    let layers = m.config.layers
    var steer = spec(.steer, in: style)
    steer.range = -8...8
    steer.detail = "Activation steering (Turner et al. 2023): Qwen's mean vector over 200 Cipher lines of one group minus another's, measured after each of its 36 layers on a Mac, added α times. Nothing is retrained."
    var floor = spec(.steerFloor, in: style)
    floor.range = 0...Double(layers - 1)
    floor.defaultValue = 15
    floor.detents = [KnobDetent(value: 15, label: "Middle")]
    return [steer, floor]
  }

  private static func spec(_ id: KnobID, in list: [KnobSpec]) -> KnobSpec { list.first { $0.id == id }! }

  // MARK: Decode: how the next token is sampled

  static let decode: [KnobSpec] = [
    KnobSpec(
      id: .temperature, title: "Temperature", glyph: "τ", tint: Theme.orange, range: 0...2, defaultValue: 0.7,
      detents: [
        KnobDetent(value: 0, label: "Greedy"), KnobDetent(value: 0.7, label: "Focused"),
        KnobDetent(value: 1, label: "As trained"), KnobDetent(value: 1.6, label: "Fever"),
      ], step: 0.01, badge: .live,
      hint: "Divides the scores before softmax. Low temperature sharpens the favourite; high temperature flattens the field.",
      detail: "p = softmax(z / τ). At 0 the model always takes the top token (greedy). At 1 it samples from the odds it learned. Above 1 long shots get likelier. It reshapes the odds; it doesn't add ideas.",
      keyPath: \.temperature, format: { $0 < 0.005 ? "Greedy" : String(format: "%.2f", $0) }),
    KnobSpec(
      id: .topK, title: "Top-k", glyph: "k", tint: Theme.negative, range: 0...200, defaultValue: 0,
      detents: [KnobDetent(value: 0, label: "Off"), KnobDetent(value: 1, label: "Greedy"), KnobDetent(value: 40, label: "Common")],
      step: 1, badge: .live,
      hint: "Only the k likeliest tokens stay in the running.",
      detail: "Sort the tokens by probability, keep the first k, renormalise. Off keeps all 4,096.",
      keyPath: \.topK, format: { $0 < 0.5 ? "Off" : "\(Int($0))" }),
    KnobSpec(
      id: .topP, title: "Top-p", glyph: "p", tint: Color(red: 0.35, green: 0.94, blue: 0.69), range: 0.05...1, defaultValue: 0.9,
      detents: [KnobDetent(value: 0.9, label: "Nucleus 0.9"), KnobDetent(value: 1, label: "Off")], step: 0.01, badge: .live,
      hint: "Keeps the smallest set of tokens whose probabilities add up to p. The set shrinks when the model is sure.",
      detail: "Nucleus sampling (Holtzman et al. 2019): add up probabilities from the top until they reach p; everything after is out.",
      keyPath: \.topP, format: { $0 >= 0.999 ? "Off" : String(format: "%.2f", $0) }),
    KnobSpec(
      id: .minP, title: "Min-p", glyph: "m", tint: Theme.violet, range: 0...0.5, defaultValue: 0,
      detents: [KnobDetent(value: 0, label: "Off"), KnobDetent(value: 0.05, label: "Gentle")], step: 0.01, badge: .live,
      hint: "Drops any token with less than m × the top token's probability.",
      detail: "A threshold relative to the favourite: when the model is confident it removes more, when it is unsure it keeps more.",
      keyPath: \.minP, format: { $0 < 0.005 ? "Off" : String(format: "%.2f", $0) }),
    KnobSpec(
      id: .repetition, title: "Repetition", glyph: "ρ", tint: Theme.rose, range: 1...2, defaultValue: 1.1,
      detents: [KnobDetent(value: 1, label: "Off"), KnobDetent(value: 1.2, label: "CTRL")], step: 0.01, badge: .live,
      hint: "Tokens already in the context have their scores divided by ρ, so the model is less likely to repeat itself.",
      detail: "Keskar et al. 2019 (CTRL): a positive score is divided by ρ, a negative one multiplied. It applies to every token in the context, the prompt included.",
      keyPath: \.repetition, format: { $0 < 1.005 ? "Off" : String(format: "×%.2f", $0) }),
    KnobSpec(
      id: .length, title: "Max tokens", glyph: "n", tint: Color(red: 0.62, green: 0.69, blue: 0.80), range: 1...120, defaultValue: 24,
      step: 1, badge: .live,
      hint: "How many tokens to generate. The model also stops when it writes a line break.",
      detail: "The context holds 128 tokens; past that the window slides and the oldest tokens drop out of reach.",
      keyPath: \.length, format: { "\(Int($0))" }),
    KnobSpec(
      id: .seed, title: "Seed", glyph: "#", tint: .white, range: 1...999, defaultValue: 6, step: 1, badge: .live,
      hint: "The random number generator behind sampling. Same seed and knobs, same text.",
      detail: "SplitMix64, seeded with this number. Each sample takes one uniform number u and walks the tokens from likeliest to least until their probabilities add past u.",
      keyPath: \.seed, format: { "\(Int($0))" }),
    KnobSpec(
      id: .pace, title: "Speed", symbol: "metronome.fill", tint: Theme.phosphor, range: 0...3, defaultValue: 1,
      step: 1, badge: .metaphor,
      hint: "How fast to show generation. A token really takes about a millisecond; anything slower is for watching.",
      detail: "Step waits for you. Slow replays each token's pass stage by stage, slowed about 2,000 times. Fast and Instant just generate.",
      keyPath: \.pace, format: label(["Step", "Slow", "Fast", "Instant"])),
  ]

  // MARK: Model: change the model itself

  static let model: [KnobSpec] = [
    KnobSpec(
      id: .precision, title: "Precision", glyph: "b", tint: Color(red: 0.89, green: 0.83, blue: 1), range: 0...6, defaultValue: 0,
      step: 1, badge: .live,
      hint: "Quantization: rounds every weight to fewer levels. Watch the language fall apart around 3 bits.",
      detail: "Symmetric round-to-nearest with one scale per output column. 8 bits gives 255 levels, 4 bits 15, 2 bits 3; 1 bit keeps only each weight's sign times the column's mean size. Adapters stay full precision on top, as QLoRA does.",
      keyPath: \.bitsIndex, format: { v in
        let b = Quantizer.choices[clamped(Int(v.rounded()), 0, 6)]
        return b == 16 ? "16 bit" : "\(b) bit"
      }),
    KnobSpec(
      id: .context, title: "Context", glyph: "c", tint: Theme.ice, range: 1...128, defaultValue: 128, logarithmic: true,
      detents: [KnobDetent(value: 1, label: "Bigram"), KnobDetent(value: 128, label: "Full")], step: 1, badge: .live,
      hint: "How many tokens attention can see. At 1 the model sees only the current token.",
      detail: "The window the model runs over. Shorter than the trained 128 is real truncation; longer doesn't exist for this model.",
      keyPath: \.context, format: { "\(Int($0))" }),
    KnobSpec(
      id: .depth, title: "Layers used", glyph: "ℓ", tint: Theme.phosphor, range: 1...4, defaultValue: 4,
      step: 1, badge: .live,
      hint: "Stop after this layer and predict from there (logit lens, early exit).",
      detail: "nostalgebraist's logit lens: the vector after any layer can be read through the final norm and the output. Early layers guess crudely; later ones sharpen.",
      keyPath: \.depth, format: { "\(Int($0)) of 4" }),
    KnobSpec(
      id: .age, title: "Checkpoint", glyph: "t", tint: Theme.ice, range: 0...3, defaultValue: 3,
      step: 1, badge: .baked,
      hint: "Real checkpoints from training: babble, then words, then phrases, then the finished model.",
      detail: "Saved from the same training run at steps 40, 250 and 1,500, and at the end: the same model earlier in its optimization.",
      keyPath: \.ageIndex, format: label(["Step 40", "Step 250", "Step 1,500", "Finished"])),
    KnobSpec(
      id: .kvCache, title: "KV cache", glyph: "κ", tint: Theme.muted, range: 0...1, defaultValue: 1,
      step: 1, badge: .live,
      hint: "With the cache, a new token costs one position of work. Without it, the whole context is recomputed every step.",
      detail: "Every layer keeps the keys and values of earlier positions. The text is identical either way; only the work counter changes.",
      keyPath: \.kvCache, format: label(["Off", "On"])),
  ]

  // MARK: Style: five fine-tunes and a vector

  static func voice(_ m: Voice, _ kp: ReferenceWritableKeyPath<LoomModel, Double>) -> KnobSpec {
    KnobSpec(
      id: KnobID(rawValue: m.rawValue)!, title: m.title, glyph: String(m.title.prefix(1)), tint: Theme.voice(m),
      range: -1...2, defaultValue: 0, origin: 0,
      detents: [KnobDetent(value: 0, label: "Off"), KnobDetent(value: 1, label: "As trained")], step: 0.01, badge: .baked,
      hint: "A LoRA fine-tune on \(m.books), which the base model never saw.",
      detail: "Rank 8 on every matrix of every layer: W + λ·(α/r)·A·B. Trained on the Mac with early stopping on held-back lines. Negative λ pushes away from the style.",
      keyPath: kp, format: { abs($0) < 0.005 ? "Off" : String(format: "%+.2f", $0) })
  }

  static let style: [KnobSpec] = [
    voice(.scripture, \.scripture),
    voice(.bard, \.bard),
    voice(.gothic, \.gothic),
    voice(.twain, \.twain),
    voice(.almanac, \.almanac),
    KnobSpec(
      id: .steer, title: "Steer", glyph: "→", tint: Color(red: 0.35, green: 0.94, blue: 0.69), range: -6...6, defaultValue: 0,
      origin: 0, detents: [KnobDetent(value: 0, label: "Off")], step: 0.05, badge: .live,
      hint: "Adds a measured direction to every position after one layer, e.g. Spanish minus the other languages.",
      detail: "Activation steering (Turner et al. 2023): the mean residual of one group of lines minus another's, added α times. Nothing is retrained.",
      keyPath: \.steerStrength, format: { abs($0) < 0.01 ? "Off" : String(format: "%+.1f", $0) }),
    KnobSpec(
      id: .steerFloor, title: "After layer", glyph: "f", tint: Color(red: 0.35, green: 0.94, blue: 0.69), range: 0...3, defaultValue: 1,
      step: 1, badge: .live,
      hint: "Which layer the steering direction is added after.",
      detail: "Early layers carry spelling and script; later layers carry meaning and what comes next.",
      keyPath: \.steerFloor, format: { "Layer \(Int($0) + 1)" }),
  ]
}
