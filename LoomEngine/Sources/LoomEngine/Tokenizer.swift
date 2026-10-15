import Foundation

/// Loom's byte-pair tokenizer, a line-for-line twin of `Tools/train/bpe.py`.
///
/// Text is first split into pieces that merges never cross: "\n" alone; otherwise an optional single
/// leading space plus a run of one character class (L = letters, marks and the apostrophe; N = numbers;
/// P = everything else). Each piece starts as single characters (unknown ones become "<?>") and the
/// lowest-ranked adjacent pair is merged, everywhere in the piece, until no listed merge applies.
public final class Tokenizer: Sendable {
  public let vocab: [String]
  /// Token id for each vocabulary string (the last id when a string appears twice, as in Python).
  public let index: [String: Int]
  public static let unknown = 0
  public static let newline = 1
  public static let controls = ["⟨en⟩", "⟨es⟩", "⟨fr⟩", "⟨hi⟩", "⟨gu⟩", "⟨card⟩"]

  /// One merge, in training order: the pair it joins, its rank and the id it produces.
  public struct Merge: Sendable { public let left: Int, right: Int, rank: Int, result: Int }
  public let merges: [Merge]
  private let ranks: [UInt64: Merge]
  private let cache = Cache()

  public convenience init(url: URL) throws {
    try self.init(json: Data(contentsOf: url))
  }

  public init(json: Data) throws {
    struct File: Decodable { let vocab: [String]; let merges: [[String]] }
    let file = try JSONDecoder().decode(File.self, from: json)
    vocab = file.vocab
    var index: [String: Int] = [:]
    for (i, t) in file.vocab.enumerated() { index[t] = i }
    self.index = index
    var merges: [Merge] = []
    var ranks: [UInt64: Merge] = [:]
    for (r, pair) in file.merges.enumerated() where pair.count == 2 {
      guard let a = index[pair[0]], let b = index[pair[1]], let m = index[pair[0] + pair[1]] else { continue }
      let merge = Merge(left: a, right: b, rank: r, result: m)
      merges.append(merge)
      ranks[Self.key(a, b)] = merge
    }
    self.merges = merges
    self.ranks = ranks
  }

  @inline(__always) private static func key(_ a: Int, _ b: Int) -> UInt64 { UInt64(a) << 32 | UInt64(b) }

  public var count: Int { vocab.count }

  // MARK: Pre-tokenising

  public enum CharClass: Equatable { case newline, space, letter, number, punctuation }

  public static func charClass(_ s: Unicode.Scalar) -> CharClass {
    if s == "\n" { return .newline }
    if s == " " { return .space }
    if s == "'" { return .letter }
    switch s.properties.generalCategory {
    case .uppercaseLetter, .lowercaseLetter, .titlecaseLetter, .modifierLetter, .otherLetter,
      .nonspacingMark, .spacingMark, .enclosingMark:
      return .letter
    case .decimalNumber, .letterNumber, .otherNumber:
      return .number
    case .spaceSeparator, .lineSeparator, .paragraphSeparator:
      return .space
    default:
      return ["\t", "\r", "\u{0B}", "\u{0C}"].contains(s) ? .space : .punctuation
    }
  }

  /// The pieces BPE runs inside. Each piece is a string of Unicode scalars.
  public static func pieces(_ text: String) -> [String] {
    let s = Array(text.unicodeScalars)
    var out: [String] = []
    var i = 0
    while i < s.count {
      let c = charClass(s[i])
      if c == .newline { out.append("\n"); i += 1; continue }
      var lead = ""
      if c == .space {
        if i + 1 < s.count, [.letter, .number, .punctuation].contains(charClass(s[i + 1])) {
          lead = " "; i += 1
        } else {
          out.append(" "); i += 1; continue
        }
      }
      let k = charClass(s[i])
      var piece = String.UnicodeScalarView()
      while i < s.count, charClass(s[i]) == k { piece.append(s[i]); i += 1 }
      out.append(lead + String(piece))
    }
    return out
  }

  /// The cleaning the corpus went through, applied to typed text: NFC, straight quotes, single spaces.
  public static func normalize(_ text: String) -> String {
    var t = text.precomposedStringWithCanonicalMapping
    for (from, to) in [("\u{2019}", "'"), ("\u{2018}", "'"), ("\u{201C}", "\""), ("\u{201D}", "\""), ("\u{00A0}", " ")] {
      t = t.replacingOccurrences(of: from, with: to)
    }
    t = t.replacingOccurrences(of: #"\s*--\s*"#, with: " — ", options: .regularExpression)
    t = t.replacingOccurrences(of: #"[ \t]+"#, with: " ", options: .regularExpression)
    return t
  }

  // MARK: BPE

  /// One step of the merge replay, for the tokenizer inspection panel.
  public struct MergeStep: Sendable {
    public let symbols: [Int]
    public let merge: Merge?
  }

  /// Every intermediate state of one piece, from single characters to its final tokens.
  public func replay(_ piece: String) -> [MergeStep] {
    var syms = piece.unicodeScalars.map { index[String($0)] ?? Self.unknown }
    var steps = [MergeStep(symbols: syms, merge: nil)]
    while let m = bestMerge(syms) {
      syms = apply(m, to: syms)
      steps.append(MergeStep(symbols: syms, merge: m))
    }
    return steps
  }

  private func bestMerge(_ syms: [Int]) -> Merge? {
    guard syms.count > 1 else { return nil }
    var best: Merge?
    for j in 0..<(syms.count - 1) {
      if let m = ranks[Self.key(syms[j], syms[j + 1])], best == nil || m.rank < best!.rank { best = m }
    }
    return best
  }

  private func apply(_ m: Merge, to syms: [Int]) -> [Int] {
    var merged: [Int] = []
    merged.reserveCapacity(syms.count)
    var j = 0
    while j < syms.count {
      if j < syms.count - 1, syms[j] == m.left, syms[j + 1] == m.right {
        merged.append(m.result); j += 2
      } else {
        merged.append(syms[j]); j += 1
      }
    }
    return merged
  }

  public func bpe(_ piece: String) -> [Int] {
    if let hit = cache.get(piece) { return hit }
    var syms = piece.unicodeScalars.map { index[String($0)] ?? Self.unknown }
    while let m = bestMerge(syms) { syms = apply(m, to: syms) }
    cache.set(piece, syms)
    return syms
  }

  public func encode(_ text: String) -> [Int] {
    Self.pieces(text).flatMap(bpe)
  }

  public func decode(_ ids: [Int]) -> String {
    ids.map { vocab[$0] }.joined()
  }

  public func text(_ id: Int) -> String { id >= 0 && id < vocab.count ? vocab[id] : "<?>" }

  public func isControl(_ id: Int) -> Bool { id >= 2 && id < 2 + Self.controls.count }

  private final class Cache: @unchecked Sendable {
    private var store: [String: [Int]] = [:]
    private let lock = NSLock()
    func get(_ k: String) -> [Int]? { lock.withLock { store[k] } }
    func set(_ k: String, _ v: [Int]) { lock.withLock { if store.count < 50_000 { store[k] = v } } }
  }
}
