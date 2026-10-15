import LoomEngine
import SwiftUI

/// The math-engine palette: neon line work on near-black, one colour per pipeline stage (as in the
/// "LLMs explained" diagrams). Colour keeps one meaning each: a stage's colour marks that stage's data
/// everywhere, attention heads have their own four, and vector values are yellow (+) or blue (−).
enum Theme {
  static let background = Color(red: 0.008, green: 0.010, blue: 0.018)
  static let panel = Color(red: 0.035, green: 0.040, blue: 0.058)
  static let well = Color(red: 0.012, green: 0.014, blue: 0.024)
  static let muted = Color(red: 0.60, green: 0.64, blue: 0.72)

  // Neon hues, named for what they are.
  static let cyan = Color(red: 0.30, green: 0.86, blue: 1.00)
  static let green = Color(red: 0.42, green: 1.00, blue: 0.50)
  static let yellow = Color(red: 1.00, green: 0.86, blue: 0.28)
  static let violet = Color(red: 0.72, green: 0.50, blue: 1.00)
  static let orange = Color(red: 1.00, green: 0.58, blue: 0.24)
  static let pink = Color(red: 1.00, green: 0.38, blue: 0.74)
  static let blue = Color(red: 0.40, green: 0.62, blue: 1.00)

  // Older names, kept for the views that use them.
  static let phosphor = green
  static let amber = orange
  static let ice = cyan
  static let rose = pink
  static let flame = Color(red: 1.00, green: 0.45, blue: 0.25)

  /// One colour per attention head: the small model's four, then hues spread by the golden angle for
  /// the rest of Qwen's 32.
  static let heads: [Color] = [
    Color(red: 0.20, green: 0.92, blue: 1.00), Color(red: 1.00, green: 0.34, blue: 0.80),
    Color(red: 1.00, green: 0.88, blue: 0.22), Color(red: 0.42, green: 1.00, blue: 0.62),
  ] + (4..<32).map { i in
    Color(hue: (0.07 + Double(i) * 0.618_034).truncatingRemainder(dividingBy: 1), saturation: 0.62, brightness: 1)
  }
  /// Diverging scale for vector values: positive yellow (the embedding numbers), negative blue.
  static let positive = yellow
  static let negative = Color(red: 0.32, green: 0.62, blue: 1.00)

  static func value(_ v: Float, scale: Float) -> Color {
    let t = Double(min(1, abs(v) / max(scale, 1e-6)))
    return (v >= 0 ? positive : negative).opacity(0.08 + 0.92 * t)
  }

  static func script(_ s: Galaxy.Script) -> Color {
    switch s {
    case .latin: green
    case .devanagari: pink
    case .gujarati: violet
    case .digit: yellow
    case .mark: muted
    }
  }

  static func voice(_ v: Voice) -> Color {
    switch v {
    case .scripture: yellow
    case .bard: pink
    case .gothic: violet
    case .twain: green
    case .almanac: cyan
    }
  }

  static let mono = Font.system(.caption, design: .monospaced).weight(.semibold)
}

/// The seven stages of the pipeline, in order, each with its colour.
enum Stage: Int, CaseIterable, Identifiable {
  case input = 1, tokenize, embedding, transformer, predict, sample, loop
  var id: Int { rawValue }
  var title: String {
    switch self {
    case .input: "Input"
    case .tokenize: "Tokenize"
    case .embedding: "Embedding"
    case .transformer: "Transformer"
    case .predict: "Predict next token"
    case .sample: "Sample"
    case .loop: "Repeat"
    }
  }
  var color: Color {
    switch self {
    case .input: Theme.cyan
    case .tokenize: Theme.green
    case .embedding: Theme.yellow
    case .transformer: Theme.violet
    case .predict: Theme.orange
    case .sample: Theme.pink
    case .loop: Theme.blue
    }
  }
}

/// Where a number comes from (house rule: provenance on every number).
enum Provenance: String {
  case live = "LIVE", train = "TRAINED HERE", baked = "PRE-TRAINED", metaphor = "ILLUSTRATION", projection = "PROJECTION"
  var color: Color {
    switch self {
    case .live: Theme.green
    case .train: Theme.orange
    case .baked: Theme.cyan
    case .metaphor, .projection: Theme.muted
    }
  }
  var meaning: String {
    switch self {
    case .live: "Computed now, on this phone, by the real model."
    case .train: "A real training run on this phone."
    case .baked: "Trained before the app shipped: Loom 1.4M on a Mac, Qwen3 by Alibaba's Qwen team."
    case .metaphor: "A drawing, not a measurement."
    case .projection: "Many dimensions flattened to a few; distances are approximate."
    }
  }
}

struct Badge: View {
  let provenance: Provenance
  var body: some View {
    Text(provenance.rawValue)
      .font(.system(size: 9, weight: .bold, design: .monospaced))
      .tracking(1)
      .padding(.horizontal, 6)
      .padding(.vertical, 2)
      .foregroundStyle(provenance.color)
      .overlay(Capsule().strokeBorder(provenance.color.opacity(0.7), lineWidth: 0.8))
      .fixedSize()
      .accessibilityLabel(provenance.rawValue.lowercased())
      .accessibilityHint(provenance.meaning)
  }
}

/// A neon-outlined panel: a numbered title in its colour, an optional plain-language line, content.
struct InstrumentPanel<Content: View>: View {
  let number: String
  let title: String
  var badge: Provenance?
  var nameplate: String?
  var tint: Color = Theme.cyan
  var lit = false
  @ViewBuilder var content: Content

  var body: some View {
    VStack(alignment: .leading, spacing: 12) {
      HStack(spacing: 10) {
        Text(number)
          .font(.system(size: 12, weight: .bold, design: .monospaced))
          .foregroundStyle(tint)
          .padding(.horizontal, number.count > 2 ? 7 : 0)
          .frame(minWidth: 26, minHeight: 26)
          .overlay(Capsule().strokeBorder(tint, lineWidth: 1.2))
        Text(title.uppercased())
          .font(.system(.subheadline, design: .monospaced).weight(.bold))
          .tracking(1.2)
          .foregroundStyle(tint)
          .lineLimit(2)
          .minimumScaleFactor(0.8)
        Spacer(minLength: 0)
        if let badge { Badge(provenance: badge) }
      }
      if let nameplate {
        Text(nameplate)
          .font(.footnote)
          .foregroundStyle(.white.opacity(0.78))
          .fixedSize(horizontal: false, vertical: true)
      }
      content
    }
    .padding(16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(tint.opacity(lit ? 0.09 : 0.035), in: .rect(cornerRadius: 20))
    .overlay {
      RoundedRectangle(cornerRadius: 20)
        .strokeBorder(tint.opacity(lit ? 0.95 : 0.4), lineWidth: lit ? 1.6 : 1)
    }
    .shadow(color: tint.opacity(lit ? 0.45 : 0), radius: lit ? 14 : 0)
    .animation(.easeInOut(duration: 0.25), value: lit)
  }
}

/// A formula in mathematical type: "p = softmax(z / τ)".
struct Formula: View {
  let text: String
  var color: Color = .white
  init(_ text: String, color: Color = .white) { self.text = text; self.color = color }
  var body: some View {
    Text(text)
      .font(.system(.callout, design: .serif).italic())
      .foregroundStyle(color.opacity(0.92))
      .padding(.horizontal, 10)
      .padding(.vertical, 5)
      .background(color.opacity(0.07), in: .rect(cornerRadius: 8))
      .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(color.opacity(0.25), lineWidth: 0.8))
      .accessibilityLabel("Formula: \(text)")
  }
}

/// The arrow between two pipeline stages.
struct StageArrow: View {
  let from: Color
  let to: Color
  var body: some View {
    VStack(spacing: 0) {
      LinearGradient(colors: [from, to], startPoint: .top, endPoint: .bottom)
        .frame(width: 1.5, height: 16)
      Image(systemName: "chevron.down")
        .font(.system(size: 11, weight: .bold))
        .foregroundStyle(to)
    }
    .frame(maxWidth: .infinity)
    .accessibilityHidden(true)
  }
}

struct Readout: View {
  let title: String
  let value: String
  var color: Color = Theme.orange
  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Text(title.uppercased()).font(.system(.caption2, design: .monospaced).weight(.semibold)).foregroundStyle(Theme.muted)
      Text(value).font(.system(.callout, design: .monospaced).weight(.semibold)).monospacedDigit()
        .foregroundStyle(color)
        .contentTransition(.numericText())
        .lineLimit(1)
        .minimumScaleFactor(0.7)
    }
    .accessibilityElement(children: .combine)
  }
}

/// A vector drawn as a strip of cells, yellow for positive and blue for negative.
struct ValueStrip: View {
  let values: [Float]
  var columns = 32
  var scale: Float?
  var cell: CGFloat = 5

  var body: some View {
    let s = scale ?? max(values.map(abs).max() ?? 1, 1e-6)
    Canvas { ctx, size in
      let rows = Int(ceil(Double(values.count) / Double(columns)))
      let w = size.width / CGFloat(columns), h = size.height / CGFloat(max(rows, 1))
      for (i, v) in values.enumerated() {
        let r = CGRect(x: CGFloat(i % columns) * w, y: CGFloat(i / columns) * h, width: w - 0.6, height: h - 0.6)
        ctx.fill(Path(roundedRect: r, cornerRadius: 1), with: .color(Theme.value(v, scale: s)))
      }
    }
    .frame(height: cell * CGFloat(Int(ceil(Double(values.count) / Double(columns)))))
    .accessibilityElement()
    .accessibilityLabel("Vector of \(values.count) numbers")
    .accessibilityValue(String(format: "largest %.2f", values.map(abs).max() ?? 0))
  }
}

/// A vector written out the way the diagrams do: "0.21  −0.37  0.05 … 0.48".
struct VectorRow: View {
  let values: [Float]
  var color: Color = Theme.yellow
  var shown = 4

  var body: some View {
    let head = values.prefix(shown).map(Self.format)
    let tail = values.last.map(Self.format) ?? ""
    Text((head + ["…", tail]).joined(separator: "  "))
      .font(.system(size: 12, weight: .medium, design: .monospaced))
      .foregroundStyle(color)
      .lineLimit(1)
      .minimumScaleFactor(0.7)
      .accessibilityLabel("Vector starting \(head.joined(separator: ", "))")
  }

  static func format(_ v: Float) -> String {
    abs(v) < 0.005 ? " 0.00" : String(format: "%+.2f", v).replacingOccurrences(of: "-", with: "−")
  }
}

/// A token as the diagrams draw it: its text in an outlined box, its id underneath.
struct TokenChip: View {
  let token: Int
  let text: String
  var lit = false
  var dim = false
  var control = false
  var probability: Double?
  var tint: Color = Theme.green

  var body: some View {
    VStack(spacing: 5) {
      Text(text)
        .font(.system(size: 14, weight: .semibold, design: .monospaced))
        .lineLimit(1)
        .minimumScaleFactor(0.5)
        .foregroundStyle(control ? Theme.orange : .white)
        .padding(.horizontal, 8)
        .frame(minWidth: 40, minHeight: 30)
        .background(tint.opacity(lit ? 0.18 : 0.05), in: .rect(cornerRadius: 7))
        .overlay {
          RoundedRectangle(cornerRadius: 7)
            .strokeBorder(tint.opacity(lit ? 1 : 0.55), lineWidth: lit ? 1.6 : 1)
        }
        .shadow(color: tint.opacity(lit ? 0.6 : 0), radius: 6)
      Image(systemName: "arrow.down").font(.system(size: 8, weight: .bold)).foregroundStyle(tint.opacity(0.6))
      Text("\(token)")
        .font(.system(size: 10, weight: .semibold, design: .monospaced))
        .foregroundStyle(tint.opacity(0.9))
        .padding(.horizontal, 5)
        .padding(.vertical, 2)
        .overlay(RoundedRectangle(cornerRadius: 4).strokeBorder(tint.opacity(0.35), lineWidth: 0.8))
    }
    .opacity(dim ? 0.35 : (probability.map { 0.45 + 0.55 * min(1, $0 * 3) } ?? 1))
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("Token \(text), id \(token)")
  }
}

extension View {
  /// Small monospaced uppercase label.
  func eyebrow() -> some View {
    self.font(.system(.caption2, design: .monospaced).weight(.semibold)).tracking(1.2).foregroundStyle(Theme.muted)
  }
}

func percent(_ p: Double) -> String {
  if p >= 0.995 { return "100%" }
  if p >= 0.1 { return String(format: "%.0f%%", p * 100) }
  if p >= 0.001 { return String(format: "%.1f%%", p * 100) }
  return p > 0 ? "<0.1%" : "0%"
}
