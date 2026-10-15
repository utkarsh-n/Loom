import Foundation
import LoomEngine
import Observation
import OSLog
import Tokenizers

private let log = Logger(subsystem: "com.utkarsh.loom", category: "Qwen")

/// Qwen3-4B-Instruct-2507 (4-bit MLX), shipped inside the app: its weights load onto the GPU on demand,
/// its byte-level tokenizer, the vocabulary as the cards show it, and the Mac-side extras (the steering
/// directions and the token catalogue, from Tools/qwen/precompute.py).
@MainActor @Observable final class QwenRuntime {
  enum Status: Equatable {
    case unavailable(String), idle, loading, ready, failed(String)
  }

  nonisolated static let name = "Qwen3-4B-Instruct-2507"
  nonisolated static let endOfText = 151643, imStart = 151644, imEnd = 151645
  /// Generation ends at either; the chat template ends a reply with ⟨im_end⟩.
  nonisolated static let stops: Set<Int> = [endOfText, imEnd]
  /// Ids from here up are control tokens (and unused padding rows).
  nonisolated static let firstSpecial = 151643
  /// The longest context Loom runs Qwen over: every position's attention is read back to the phone.
  nonisolated static let contextCap = 128

  private(set) var status: Status
  /// The model's shape, from its config.json; known before the weights load.
  let shape: QwenConfig?
  /// Wrap the prompt in the chat template (as a user message, the reply following) or run it raw.
  var chat = UserDefaults.standard.object(forKey: "qwenChat") as? Bool ?? true {
    didSet { UserDefaults.standard.set(chat, forKey: "qwenChat") }
  }
  private(set) var loadSeconds: Double?

  @ObservationIgnored let worker = QwenWorker()
  @ObservationIgnored private var tokenizer: (any Tokenizers.Tokenizer)?
  /// Each token's raw bytes (empty for control tokens).
  @ObservationIgnored private(set) var bytes: [[UInt8]] = []
  /// Each token as a card shows it.
  @ObservationIgnored private(set) var cards: [String] = []
  /// Tokens that contain a line break (they end a line in raw mode).
  @ObservationIgnored private(set) var lineBreaks: Set<Int> = []
  @ObservationIgnored private(set) var finalGain: [Float] = []
  @ObservationIgnored private(set) var steering: Steering?
  @ObservationIgnored private(set) var catalogue: Catalogue?
  /// Unit embedding directions fetched for tokens outside the catalogue (the prompt, the candidates).
  @ObservationIgnored private var extraRows: [Int: [Float]] = [:]
  @ObservationIgnored private var fetching: Set<Int> = []
  @ObservationIgnored private var loadTask: Task<Void, Never>?
  @ObservationIgnored private var loadIdentity = UUID()
  @ObservationIgnored private var unloading = false
  @ObservationIgnored private var unloadWaiters: [CheckedContinuation<Void, Never>] = []

  static var directory: URL? {
    guard let dir = Bundle.main.url(forResource: "Qwen", withExtension: nil),
      FileManager.default.fileExists(atPath: dir.appending(path: "model.safetensors").path)
    else { return nil }
    return dir
  }

  init() {
    shape = Self.directory.flatMap { try? JSONDecoder().decode(QwenConfig.self, from: Data(contentsOf: $0.appending(path: "config.json"))) }
    #if targetEnvironment(simulator)
      status = .unavailable("Qwen runs on a physical iPhone: the simulator has no GPU for MLX.")
    #else
      status = Self.directory == nil ? .unavailable("Qwen's weights aren't in this build. Tools/build.sh adds them.") : .idle
    #endif
  }

  var isAvailable: Bool {
    if case .unavailable = status { return false }
    return true
  }

  var isReady: Bool { status == .ready }

  /// Qwen's shape in the 1.4M model's terms, so every view can read layers, heads and widths the same way.
  var config: ModelConfig {
    var c = ModelConfig()
    let s = shape
    c.vocab = s?.vocabularySize ?? 151_936
    c.ctx = Self.contextCap
    c.d = s?.hiddenSize ?? 2560
    c.heads = s?.attentionHeads ?? 32
    c.layers = s?.hiddenLayers ?? 36
    c.ff = s?.intermediateSize ?? 9728
    c.ropeBase = s?.ropeTheta ?? 5_000_000
    return c
  }

  var kvHeads: Int { shape?.kvHeads ?? 8 }
  var headDim: Int { shape?.headDim ?? 128 }

  /// Every weight: embedding (tied with the readout), and per layer q, k, v, o, the feedforward's three
  /// matrices and four norms, then the final norm. 4,022,468,096 for Qwen3-4B.
  var parameterCount: Int {
    let c = config, kv = kvHeads, hd = headDim
    let perLayer = c.d * c.heads * hd + 2 * c.d * kv * hd + c.heads * hd * c.d + 3 * c.d * c.ff + 2 * hd + 2 * c.d
    return c.vocab * c.d + c.layers * perLayer + c.d
  }

  // MARK: Loading

  /// Loads the weights onto the GPU, then the tokenizer and the extras. About 2.3 GB.
  func load() async {
    if unloading { await withCheckedContinuation { unloadWaiters.append($0) } }
    guard !Task.isCancelled else { return }
    switch status {
    case .unavailable, .ready: return
    case .loading:
      await loadTask?.value
      return
    case .idle, .failed: break
    }
    guard let dir = Self.directory else { return }
    status = .loading
    let identity = UUID()
    loadIdentity = identity
    let task = Task { await performLoad(directory: dir) }
    loadTask = task
    await task.value
    if loadIdentity == identity { loadTask = nil }
  }

  private func performLoad(directory dir: URL) async {
    let clock = Date()
    do {
      let loaded = try await worker.load(directory: dir)
      try Task.checkCancellation()
      finalGain = loaded.finalGain
      if tokenizer == nil {
        let tok = try await AutoTokenizer.from(modelFolder: dir)
        let V = loaded.config.vocabularySize
        let (b, c) = await Task.detached(priority: .userInitiated) { Self.vocabulary(tok, count: V) }.value
        try Task.checkCancellation()
        tokenizer = tok
        bytes = b
        cards = c
        lineBreaks = Set(b.indices.filter { b[$0].contains(10) })
      }
      if steering == nil, let file = try? WeightFile(url: dir.appending(path: "qwen-steering.loom")) {
        steering = try Steering(file: file, layers: loaded.config.hiddenLayers)
      }
      if catalogue == nil, let file = try? Catalogue.QwenFile(url: dir.appending(path: "qwen-catalogue.json")) {
        let rows = try await worker.embeddingRows(file.ids)
        try Task.checkCancellation()
        let d = loaded.config.hiddenSize
        catalogue = await Task.detached(priority: .userInitiated) {
          Catalogue(ids: file.ids, points: file.points, scripts: file.scripts, variance: file.variance, d: d, embeddings: rows,
            source: "the \((file.ids.count - 3).formatted()) tokens commonest in Cipher's lines, plus the 3 chat markers")
        }.value
      }
      try Task.checkCancellation()
      loadSeconds = Date().timeIntervalSince(clock)
      status = .ready
      log.info("Qwen ready in \(self.loadSeconds ?? 0) s")
    } catch is CancellationError {
      status = .idle
    } catch {
      log.error("Qwen load failed: \(error.localizedDescription)")
      status = .failed(error.localizedDescription)
    }
  }

  /// Frees the GPU memory (the app is going to the background, or the small model took over).
  func unload() async {
    if unloading {
      await withCheckedContinuation { unloadWaiters.append($0) }
      return
    }
    unloading = true
    let task = loadTask
    task?.cancel()
    status = isAvailable ? .idle : status
    await task?.value
    await worker.unload()
    loadTask = nil
    if isAvailable { status = .idle }
    unloading = false
    let waiters = unloadWaiters
    unloadWaiters.removeAll()
    for waiter in waiters { waiter.resume() }
  }

  // MARK: Text

  /// The prompt as Qwen reads it: in the chat template as a user turn, or raw.
  func encode(_ prompt: String) -> [Int] {
    guard let tokenizer else { return [] }
    let text = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    if chat {
      return tokenizer.encode(text: "<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n", addSpecialTokens: false)
    }
    return tokenizer.encode(text: text, addSpecialTokens: false)
  }

  /// Text as Qwen reads it with no template.
  func tokens(_ text: String) -> [Int] {
    tokenizer?.encode(text: text, addSpecialTokens: false) ?? []
  }

  /// Qwen's pre-tokenizer: the cuts tokens never cross (its tokenizer.json split pattern).
  nonisolated static let splitter = try? NSRegularExpression(
    pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#)

  nonisolated static func pretokenize(_ text: String) -> [String] {
    guard let splitter else { return [text] }
    let ns = text as NSString
    return splitter.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { ns.substring(with: $0.range) }
  }

  func isSpecial(_ id: Int) -> Bool { id >= Self.firstSpecial }

  /// The tokens that end generation: ⟨im_end⟩ and ⟨endoftext⟩, and in raw mode any line break.
  var stopIDs: Set<Int> { chat ? Self.stops : Self.stops.union(lineBreaks) }

  func ends(_ id: Int) -> Bool { Self.stops.contains(id) || (!chat && lineBreaks.contains(id)) }

  /// The token's text; a token that is part of a character shows as U+FFFD.
  func text(_ id: Int) -> String {
    guard id >= 0, id < bytes.count else { return "" }
    if isSpecial(id) { return cards[id] }
    return String(decoding: bytes[id], as: UTF8.self)
  }

  func card(_ id: Int) -> String { id >= 0 && id < cards.count ? cards[id] : "?" }

  /// Running text: bytes joined across tokens, so characters split between tokens come back whole.
  func decode(_ ids: some Sequence<Int>, keepSpecial: Bool = false) -> String {
    var out = ""
    var pending: [UInt8] = []
    for id in ids where id >= 0 && id < bytes.count {
      if isSpecial(id) {
        if !pending.isEmpty { out += String(decoding: pending, as: UTF8.self); pending = [] }
        if keepSpecial { out += cards[id] }
      } else {
        pending += bytes[id]
      }
    }
    if !pending.isEmpty { out += String(decoding: pending, as: UTF8.self) }
    return out
  }

  /// Every token's bytes and card text, read once from the tokenizer.
  nonisolated private static func vocabulary(_ tok: any Tokenizers.Tokenizer, count: Int) -> ([[UInt8]], [String]) {
    var bytes = [[UInt8]](repeating: [], count: count)
    var cards = [String](repeating: "⟨unused⟩", count: count)
    for id in 0..<count {
      guard let piece = tok.convertIdToToken(id) else { continue }
      if id >= firstSpecial {
        cards[id] = "⟨" + piece.trimmingCharacters(in: CharacterSet(charactersIn: "<|>")) + "⟩"
        continue
      }
      let b = ByteLevel.bytes(piece)
      bytes[id] = b
      cards[id] = card(for: b)
    }
    return (bytes, cards)
  }

  /// A token as a card shows it: ▁ for a leading space, ↵ for line breaks, and a token that is only part
  /// of a character as its bytes in hex.
  nonisolated static func card(for b: [UInt8]) -> String {
    guard let s = String(validating: b, as: UTF8.self) else {
      return "‹" + b.map { String(format: "%02x", $0) }.joined(separator: " ") + "›"
    }
    if s == " " { return "␣" }
    var t = s.replacingOccurrences(of: "\n", with: "↵").replacingOccurrences(of: "\t", with: "⇥")
    if t.hasPrefix(" ") { t = "▁" + t.dropFirst() }
    return t
  }

  // MARK: Directions, for the sphere

  /// A token's embedding direction (length 1), if it's in the catalogue or has been fetched.
  func direction(_ token: Int) -> ArraySlice<Float>? {
    if let c = catalogue, let r = c.row(of: token) { return c.direction(row: r) }
    return extraRows[token].map { $0[...] }
  }

  /// Fetches the directions of tokens outside the catalogue, in the background.
  func prefetch(_ tokens: some Sequence<Int>) {
    guard status == .ready else { return }
    let missing = Array(Set(tokens.filter { $0 >= 0 && $0 < config.vocab && direction($0) == nil && !fetching.contains($0) }))
    guard !missing.isEmpty else { return }
    fetching.formUnion(missing)
    let d = config.d
    Task {
      defer { fetching.subtract(missing) }
      guard let rows = try? await worker.embeddingRows(missing) else { return }
      for (i, t) in missing.enumerated() {
        var v = Array(rows[(i * d)..<((i + 1) * d)])
        Gaze.normalize(&v)
        extraRows[t] = v
      }
    }
  }
}

/// GPT-2's byte-level alphabet: every byte has a printable stand-in character in the vocabulary file.
nonisolated enum ByteLevel {
  static let decoder: [Unicode.Scalar: UInt8] = {
    var printable = Array(33...126) + Array(161...172) + Array(174...255)
    var chars = printable
    var n = 0
    for b in 0..<256 where !printable.contains(b) {
      printable.append(b)
      chars.append(256 + n)
      n += 1
    }
    var map: [Unicode.Scalar: UInt8] = [:]
    for (b, c) in zip(printable, chars) { map[Unicode.Scalar(UInt32(c))!] = UInt8(b) }
    return map
  }()

  static func bytes(_ piece: String) -> [UInt8] { piece.unicodeScalars.compactMap { decoder[$0] } }
}
