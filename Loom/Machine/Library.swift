import Foundation
import LoomEngine

/// Everything Loom ships, loaded once at launch from `Resources/Model`.
nonisolated struct Library: Sendable {
  let tokenizer: Tokenizer
  let base: ModelWeights
  /// Training-age checkpoints, earliest first; the last is the finished base.
  let ages: [Age]
  let adapters: [Adapter]
  let steering: Steering
  let neurons: NeuronCards
  let galaxy: Galaxy

  struct Age: Sendable, Identifiable {
    let step: Int
    let valLoss: Double
    let weights: ModelWeights
    var id: Int { step }
  }

  static func load(bundle: Bundle = .main) throws -> Library {
    guard let dir = bundle.url(forResource: "Model", withExtension: nil) else {
      throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "The model files are missing from the app."])
    }
    let tokenizer = try Tokenizer(url: dir.appending(path: "tokenizer.json"))
    let base = try ModelWeights(file: WeightFile(url: dir.appending(path: "loom-base.loom")))
    let finalLoss = base.meta["curve"]?.array?.last?.array?.last?.number ?? 0
    var ages: [Age] = []
    for step in [40, 250, 1500] {
      let file = try WeightFile(url: dir.appending(path: "loom-age-\(step).loom"))
      ages.append(Age(step: step, valLoss: file.meta["val_loss"]?.number ?? 0, weights: try ModelWeights(file: file)))
    }
    ages.append(Age(step: Int(base.meta["steps"]?.number ?? 12000), valLoss: finalLoss, weights: base))
    let adapters = try Voice.allCases.map {
      try Adapter(file: WeightFile(url: dir.appending(path: "adapter-\($0.rawValue).loom")), layers: base.config.layers)
    }
    let steering = try Steering(file: WeightFile(url: dir.appending(path: "steering.loom")), layers: base.config.layers)
    let neurons = try NeuronCards(url: dir.appending(path: "neurons.json"))
    let galaxy = try Galaxy(url: dir.appending(path: "galaxy.json"))
    return Library(
      tokenizer: tokenizer, base: base, ages: ages, adapters: adapters, steering: steering, neurons: neurons, galaxy: galaxy)
  }
}

/// The five voice adapters, each taught from text the base never read.
nonisolated enum Voice: String, CaseIterable, Identifiable, Sendable {
  case scripture, bard, gothic, twain, almanac
  var id: String { rawValue }
  var title: String { rawValue.capitalized }

  /// What the adapter was taught from (all from Cipher's shelves, all held out of the base).
  var books: String {
    switch self {
    case .scripture: "the Book of Mormon and The Woman's Bible"
    case .bard: "eight Shakespeare plays, Hamlet to The Tempest"
    case .gothic: "Dracula, Frankenstein, Dorian Gray, Jekyll and Hyde, Usher and The Raven"
    case .twain: "Tom Sawyer, Huckleberry Finn and A Connecticut Yankee"
    case .almanac: "Cipher's 22,254 fact cards about films, songs, books and shows"
    }
  }
}

/// Difference-of-means directions in the residual stream, one per floor (`Tools/train/extras.py`).
nonisolated struct Steering: Sendable {
  struct Direction: Sendable, Identifiable, Hashable {
    let key: String          // "lang:es", "mood:dread"
    var id: String { key }
    var title: String {
      let name = key.split(separator: ":").last.map(String.init) ?? key
      switch name {
      case "en": return "English"
      case "es": return "Spanish"
      case "fr": return "French"
      case "hi": return "Hindi"
      case "gu": return "Gujarati"
      default: return name.capitalized
      }
    }
    var isLanguage: Bool { key.hasPrefix("lang:") }
  }

  let directions: [Direction]
  /// key → floors × d
  let vectors: [String: [[Float]]]
  /// Typical length of the residual after each floor, for scale.
  let residualNorm: [Double]

  init(file: WeightFile, layers: Int) throws {
    var dirs: [Direction] = []
    var vecs: [String: [[Float]]] = [:]
    let order = ["lang:en", "lang:es", "lang:fr", "lang:hi", "lang:gu", "mood:dread", "mood:tender", "mood:wonder", "mood:rousing", "mood:wry"]
    for key in order {
      guard let t = file.tensors[key] else { continue }
      let d = t.cols
      dirs.append(Direction(key: key))
      vecs[key] = (0..<layers).map { Array(t.data[($0 * d)..<(($0 + 1) * d)]) }
    }
    directions = dirs
    vectors = vecs
    residualNorm = file.meta["residual_norm"]?.doubles ?? Array(repeating: 1, count: layers)
  }

  func vector(_ key: String, floor: Int) -> [Float] { vectors[key]?[floor] ?? [] }
}

/// For each neuron, the three corpus moments that light it up most.
nonisolated struct NeuronCards: Sendable {
  struct Moment: Sendable, Hashable {
    let activation: Double
    let before: String, token: String, after: String
  }
  let layers: [[[Moment]]]

  init(url: URL) throws {
    let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
    layers = (json["layers"]?.array ?? []).map { layer in
      (layer.array ?? []).map { neuron in
        (neuron.array ?? []).compactMap { m in
          guard let a = m.array, a.count == 4 else { return nil }
          return Moment(activation: a[0].number ?? 0, before: a[1].string ?? "", token: a[2].string ?? "", after: a[3].string ?? "")
        }
      }
    }
  }

  func moments(floor: Int, neuron: Int) -> [Moment] {
    guard floor < layers.count, neuron < layers[floor].count else { return [] }
    return layers[floor][neuron]
  }
}

/// The card catalogue projected onto its first three principal axes.
nonisolated struct Galaxy: Sendable {
  enum Script: String, Sendable, CaseIterable { case latin, devanagari, gujarati, digit, mark }
  let points: [SIMD3<Float>]
  let scripts: [Script]
  let variance: [Double]

  init(url: URL) throws {
    let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
    let flat = json["points"]?.floats ?? []
    var pts = stride(from: 0, to: flat.count - 2, by: 3).map { SIMD3(flat[$0], flat[$0 + 1], flat[$0 + 2]) }
    // A few outlying cards would otherwise set the scale and squash the cloud: scale each axis by its
    // 98th-percentile spread and pull the outliers in to the edge (the map says so).
    for axis in 0..<3 {
      let mags = pts.map { abs($0[axis]) }.sorted()
      let p98 = mags.isEmpty ? 1 : max(mags[Int(Double(mags.count - 1) * 0.98)], 1e-6)
      for i in pts.indices { pts[i][axis] = max(-1.3, min(1.3, pts[i][axis] / p98)) }
    }
    points = pts
    scripts = (json["script"]?.array ?? []).map { Script(rawValue: $0.string ?? "") ?? .mark }
    variance = json["variance"]?.doubles ?? []
  }
}
