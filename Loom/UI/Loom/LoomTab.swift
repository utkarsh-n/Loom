import LoomEngine
import SwiftUI

/// The pipeline, top to bottom, the way the diagrams draw it: input, tokenize, embedding, the
/// transformer's layers, the next-token prediction, the sample, and the loop that repeats it.
/// Each stage card lights up in turn while a token is generated.
struct LoomTab: View {
  @Bindable var model: LoomModel
  @FocusState private var typing: Bool

  var body: some View {
    ScrollView {
      VStack(spacing: 0) {
        header
        InputStage(model: model, typing: $typing)
        StageArrow(from: Stage.input.color, to: Stage.tokenize.color)
        TokenizeStage(model: model)
        StageArrow(from: Stage.tokenize.color, to: Stage.embedding.color)
        EmbeddingStage(model: model)
        StageArrow(from: Stage.embedding.color, to: Stage.transformer.color)
        TransformerStage(model: model)
        StageArrow(from: Stage.transformer.color, to: Stage.predict.color)
        PredictStage(model: model)
        StageArrow(from: Stage.predict.color, to: Stage.sample.color)
        SampleStage(model: model)
        StageArrow(from: Stage.sample.color, to: Stage.loop.color)
        RepeatStage(model: model)
        BigPicture(model: model)
          .padding(.top, 22)
      }
      .padding(.horizontal, 14)
      .padding(.bottom, 24)
    }
    .scrollDismissesKeyboard(.interactively)
    .safeAreaInset(edge: .bottom, spacing: 0) {
      KnobConsole(model: model)
        .padding(.horizontal, 14)
        .padding(.top, 10)
        .padding(.bottom, 6)
        .background {
          VStack(spacing: 0) {
            LinearGradient(colors: [.clear, Theme.background], startPoint: .top, endPoint: .bottom)
              .frame(height: model.consoleOpen ? 18 : 40)
            Theme.background
          }
          .ignoresSafeArea()
        }
    }
  }

  private var header: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text("\(Text("A REAL LANGUAGE MODEL. ").foregroundStyle(Theme.green))\(Text("\(model.parameterCount.formatted()) PARAMETERS. ").foregroundStyle(.white))\(Text("HERE'S WHAT'S HAPPENING.").foregroundStyle(Theme.cyan))")
        .font(.system(.footnote, design: .monospaced).weight(.bold))
        .tracking(0.8)
        .fixedSize(horizontal: false, vertical: true)
        .accessibilityElement(children: .combine)
      EngineSwitch(model: model)
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .padding(.vertical, 10)
  }
}

/// Which model runs the pipeline: Loom's own 1.4M, or Qwen3-4B on the GPU.
struct EngineSwitch: View {
  @Bindable var model: LoomModel

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 6) {
        ForEach(EngineKind.allCases) { kind in
          let on = model.engine == kind
          Button {
            guard !on else { return }
            withAnimation(.smooth) { model.engine = kind }
          } label: {
            VStack(spacing: 1) {
              Text(kind.title).font(.system(.subheadline, design: .monospaced).weight(.bold))
              Text(kind == .tiny ? "4 layers · trained on Cipher" : "36 layers · 4-bit · GPU")
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundStyle(on ? .white.opacity(0.8) : Theme.muted)
            }
            .frame(maxWidth: .infinity, minHeight: 40)
          }
          .buttonStyle(.glass(on ? .regular.tint((kind == .qwen ? Theme.violet : Theme.green).opacity(0.45)) : .regular))
          .disabled(kind == .qwen && !model.qwen.isAvailable)
          .accessibilityAddTraits(on ? .isSelected : [])
          .accessibilityIdentifier("engine_\(kind.rawValue)")
        }
      }
      if let line = statusLine {
        HStack(spacing: 6) {
          if model.qwen.status == .loading { ProgressView().controlSize(.mini) }
          Text(line).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
        }
      }
    }
  }

  private var statusLine: String? {
    switch model.qwen.status {
    case .unavailable(let why): return model.isQwen ? why : "Qwen3: \(why)"
    case .loading: return "Loading Qwen3-4B onto the GPU: 2.3 GB of 4-bit weights…"
    case .failed(let why): return "Qwen3 failed to load: \(why)"
    case .ready:
      guard model.isQwen else { return nil }
      return String(format: "Qwen3-4B-Instruct-2507 · Alibaba's Qwen team · loaded in %.1f s", model.qwen.loadSeconds ?? 0)
    case .idle: return model.isQwen ? "Qwen3 is not loaded." : nil
    }
  }
}

/// Which stage is running while a token is generated (the slowed replay of the real order).
extension LoomModel {
  var liveStage: Stage? {
    guard replaying else { return nil }
    let s = Int(stage)
    if s == 0 { return .tokenize }
    if s == 1 { return .embedding }
    if s <= config.layers + 1 { return .transformer }
    if s == config.layers + 2 { return .predict }
    return .sample
  }

  /// Multiply-adds, in millions or billions.
  func work(_ macs: Int) -> String {
    macs >= 1_000_000_000 ? String(format: "%.1f B mult-adds", Double(macs) / 1e9) : String(format: "%.2f M mult-adds", Double(macs) / 1e6)
  }
}

// MARK: - 1 Input

struct InputStage: View {
  @Bindable var model: LoomModel
  var typing: FocusState<Bool>.Binding

  var body: some View {
    InstrumentPanel(number: "1", title: "Input", nameplate: nameplate, tint: Stage.input.color) {
      HStack(spacing: 8) {
        if model.isQwen {
          Menu {
            Picker("Template", selection: Binding(get: { model.qwen.chat }, set: { model.qwen.chat = $0; model.restart() })) {
              Text("Chat: Qwen answers your message").tag(true)
              Text("Raw: Qwen continues your text").tag(false)
            }
          } label: {
            Text(model.qwen.chat ? "chat" : "raw")
              .font(.system(.subheadline, design: .monospaced).weight(.semibold))
              .foregroundStyle(model.qwen.chat ? Theme.orange : Theme.muted)
              .frame(minWidth: 44, minHeight: 36)
          }
          .accessibilityLabel("Chat template")
          .accessibilityValue(model.qwen.chat ? "on" : "off")
        } else {
          Menu {
            Button("No language tag") { model.cartridge = nil }
            Section("Tags the model was trained to follow") {
              ForEach(Array(Tokenizer.controls.enumerated()), id: \.offset) { i, name in
                Button(name) { model.cartridge = 2 + i }
              }
            }
          } label: {
            Text(model.cartridge.map { model.text($0) } ?? "tag")
              .font(.system(.subheadline, design: .monospaced).weight(.semibold))
              .foregroundStyle(model.cartridge == nil ? Theme.muted : Theme.orange)
              .frame(minWidth: 44, minHeight: 36)
          }
          .accessibilityLabel("Language tag")
          .accessibilityValue(model.cartridge.map { model.text($0) } ?? "none")
        }
        TextField("Type a message", text: $model.prompt)
          .font(.body)
          .textInputAutocapitalization(.sentences)
          .autocorrectionDisabled()
          .submitLabel(.go)
          .focused(typing)
          .onSubmit {
            model.restart()
            model.play()
          }
          .onChange(of: model.prompt) { _, _ in
            if !model.playing { model.restart() }
          }
          .accessibilityIdentifier("prompt")
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 4)
      .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 16))
      ContextText(model: model)
    }
  }

  private var nameplate: String {
    guard model.isQwen else {
      return "Type a message. Everything the model sees is the context: your words plus every token it has added so far."
    }
    return model.qwen.chat
      ? "Type a message. Qwen's chat template wraps it as a user turn and opens the assistant's; everything inside is the context the model sees, its reply included."
      : "Type the start of some text. Without the chat template Qwen simply continues it: the context is your words plus every token it has added so far."
  }
}

/// The context as the model sees it: the prompt in white, generated tokens in green, brighter when likelier.
struct ContextText: View {
  let model: LoomModel

  var body: some View {
    let w = model.weave
    VStack(alignment: .leading, spacing: 6) {
      Text("CONTEXT · \(w.ids.count) OF \(model.config.ctx) TOKENS").eyebrow()
      Text(context(w))
        .font(.system(.title3, design: .serif))
        .lineSpacing(3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }
    .accessibilityIdentifier("cloth")
  }

  private func context(_ w: Weave) -> AttributedString {
    if model.isQwen { return qwenContext(w) }
    var out = AttributedString()
    for (pos, id) in w.ids.enumerated() where pos > 0 {
      if model.isControl(id) { continue }
      var piece = AttributedString(model.text(id) == "\n" ? " ↵" : model.text(id))
      color(&piece, pos, w)
      out += piece
    }
    return out
  }

  /// Qwen's context, template markers included (dimmed): it's what the model reads. Tokens that carry
  /// only part of a character wait for the rest, so Hindi and Gujarati come out whole.
  private func qwenContext(_ w: Weave) -> AttributedString {
    var out = AttributedString()
    var pending: [UInt8] = []
    for (pos, id) in w.ids.enumerated() {
      if model.qwen.isSpecial(id) {
        var marker = AttributedString(model.qwen.card(id))
        marker.foregroundColor = Theme.cyan.opacity(0.55)
        marker.font = .system(.footnote, design: .monospaced)
        out += marker
        continue
      }
      pending += id < model.qwen.bytes.count ? model.qwen.bytes[id] : []
      let text: String
      if let whole = String(validating: pending, as: UTF8.self) {
        text = whole
      } else if pending.count < 8 {
        continue
      } else {
        text = String(decoding: pending, as: UTF8.self)
      }
      pending = []
      var piece = AttributedString(text.replacingOccurrences(of: "\n", with: " ↵\n"))
      color(&piece, pos, w)
      out += piece
    }
    return out
  }

  private func color(_ piece: inout AttributedString, _ pos: Int, _ w: Weave) {
    if pos < w.promptEnd {
      piece.foregroundColor = .white
    } else if let s = w.stitches.first(where: { $0.position == pos }) {
      let p = s.probability
      piece.foregroundColor = (s.forced ? Theme.pink : Theme.green).opacity(0.5 + 0.5 * min(1, p * 2.5))
    }
  }
}

// MARK: - 2 Tokenize

struct TokenizeStage: View {
  @Bindable var model: LoomModel

  var body: some View {
    let w = model.weave
    let chars = model.isQwen ? model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).unicodeScalars.count : Tokenizer.normalize(model.prompt).unicodeScalars.count
    InstrumentPanel(number: "2", title: "Tokenize", badge: .live,
      nameplate: model.isQwen
        ? "Text becomes UTF-8 bytes, and 151,387 learned merges join them into tokens from a vocabulary of \(model.config.vocab.formatted()). Each token is a number. ▁ marks a leading space; ‹hex› is a token holding only part of a character."
        : "Text is split into tokens from a vocabulary of 4,096. Each token is a number. ▁ marks a token that starts a word.",
      tint: Stage.tokenize.color, lit: model.liveStage == .tokenize) {
      ScrollViewReader { proxy in
        ScrollView(.horizontal, showsIndicators: false) {
          HStack(alignment: .top, spacing: 8) {
            ForEach(Array(w.ids.enumerated()), id: \.offset) { pos, id in
              let thread = pos - w.windowStart
              let stitch = w.stitches.first { $0.position == pos }
              Button {
                guard thread >= 0, thread < w.threads.count else { return }
                model.selectedThread = thread == w.threads.count - 1 ? nil : thread
                model.haptics.transient(intensity: 0.3, sharpness: 0.8)
              } label: {
                TokenChip(
                  token: id, text: model.cardText(id), lit: thread == model.focusThread, dim: pos < w.windowStart,
                  control: model.isControl(id), probability: stitch.map(\.probability))
              }
              .buttonStyle(.plain)
              .id(pos)
            }
          }
          .padding(.vertical, 4)
          .padding(.horizontal, 2)
        }
        .onChange(of: w.ids.count) { _, n in withAnimation(.smooth) { proxy.scrollTo(n - 1, anchor: .trailing) } }
      }
      .accessibilityIdentifier("tape")
      HStack {
        Text(tokenCount(chars, w)).font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.green)
        Spacer()
        Button(model.isQwen ? "See the bytes" : "See the merges") { model.inspector = .punch }
          .font(.caption.weight(.semibold))
          .buttonStyle(.glass)
      }
    }
  }
}

extension TokenizeStage {
  fileprivate func tokenCount(_ chars: Int, _ w: Weave) -> String {
    guard model.isQwen else { return "\(chars) characters → \(w.promptEnd - 1) tokens" }
    let own = model.qwen.tokens(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines)).count
    let template = w.promptEnd - own
    return template > 0 ? "\(chars) characters → \(own) tokens + \(template) of template" : "\(chars) characters → \(own) tokens"
  }
}

// MARK: - 3 Embedding

struct EmbeddingStage: View {
  @Bindable var model: LoomModel

  var body: some View {
    let threads = model.weave.threads
    let shown = Array(threads.suffix(4))
    InstrumentPanel(number: "3", title: "Embedding", badge: .live,
      nameplate: "Each token id picks one row of a \(model.config.vocab.formatted()) × \(model.config.d.formatted()) table: a vector of \(model.config.d.formatted()) numbers. Similar tokens have similar vectors.",
      tint: Stage.embedding.color, lit: model.liveStage == .embedding) {
      VStack(alignment: .leading, spacing: 8) {
        ForEach(Array(shown.enumerated()), id: \.offset) { _, t in
          HStack(spacing: 10) {
            Text(model.cardText(t.token))
              .font(.system(size: 13, weight: .semibold, design: .monospaced))
              .foregroundStyle(.white)
              .frame(width: 64, alignment: .leading)
              .lineLimit(1)
            Text("→").foregroundStyle(Theme.yellow.opacity(0.7))
            VectorRow(values: t.residual[0])
          }
        }
      }
      .padding(10)
      .overlay(alignment: .leading) { Bracket(color: Theme.yellow) }
      .overlay(alignment: .trailing) { Bracket(color: Theme.yellow).scaleEffect(x: -1) }
      HStack {
        Text("Vector size: \(model.config.d.formatted())").font(.system(.caption, design: .monospaced).weight(.semibold)).foregroundStyle(Theme.yellow)
        Spacer()
        Formula("x = E[id]", color: Theme.yellow)
      }
    }
  }
}

/// A matrix bracket, drawn as in the diagrams.
struct Bracket: View {
  let color: Color
  var body: some View {
    Canvas { ctx, size in
      var p = Path()
      p.move(to: CGPoint(x: 6, y: 1))
      p.addLine(to: CGPoint(x: 1, y: 1))
      p.addLine(to: CGPoint(x: 1, y: size.height - 1))
      p.addLine(to: CGPoint(x: 6, y: size.height - 1))
      ctx.stroke(p, with: .color(color.opacity(0.8)), lineWidth: 1.4)
    }
    .frame(width: 8)
    .accessibilityHidden(true)
  }
}

// MARK: - 4 Transformer

struct TransformerStage: View {
  @Bindable var model: LoomModel

  var body: some View {
    InstrumentPanel(number: "4", title: "Transformer × \(model.config.layers) layers", badge: .live,
      nameplate: model.isQwen
        ? "The vectors pass through \(model.config.layers) layers. In each, \(model.config.heads) attention heads (sharing \(model.qwen.kvHeads) sets of keys and values) let every position look at the ones before it, then a feedforward network of \(model.config.ff.formatted()) neurons transforms each position on its own. Tap a layer to look inside."
        : "The vectors pass through \(model.config.layers) layers. In each, attention lets every position look at the ones before it, then a feedforward network transforms each position on its own. Tap a layer to look inside.",
      tint: Stage.transformer.color, lit: model.liveStage == .transformer) {
      HStack(spacing: 6) {
        LayerBlock(title: "MULTI-HEAD ATTENTION")
        Image(systemName: "arrow.right").font(.caption2).foregroundStyle(Theme.violet)
        LayerBlock(title: "FEEDFORWARD")
        Image(systemName: "arrow.right").font(.caption2).foregroundStyle(Theme.violet)
        LayerBlock(title: "ADD & NORM")
      }
      MachineCanvas(model: model)
      StatusRow(model: model)
    }
  }
}

struct LayerBlock: View {
  let title: String
  var body: some View {
    Text(title)
      .font(.system(size: 9, weight: .bold, design: .monospaced))
      .multilineTextAlignment(.center)
      .foregroundStyle(Theme.violet)
      .frame(maxWidth: .infinity, minHeight: 30)
      .padding(.horizontal, 4)
      .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(Theme.violet.opacity(0.6), lineWidth: 1))
  }
}

/// What the computation cost and where it is: prefill or decode, how slowed, how much work.
struct StatusRow: View {
  @Bindable var model: LoomModel

  var body: some View {
    let w = model.weave
    let decode = !w.stitches.isEmpty
    HStack(spacing: 8) {
      chip(decode ? "DECODE · 1 POSITION" : "PREFILL · \(w.threads.count) POSITIONS AT ONCE", Theme.violet)
      chip(model.work(w.macsLastStep).uppercased(), Theme.cyan)
      if model.playing, (model.rideSeconds ?? model.paceValue.seconds) > 0, w.secondsLastStep > 0 {
        let factor = (model.rideSeconds ?? model.paceValue.seconds) / w.secondsLastStep
        chip(factor >= 1.5 ? "SLOWED ×\(Int(factor).formatted())" : String(format: "%.0f MS A TOKEN", w.secondsLastStep * 1000), Theme.orange)
      }
      Spacer(minLength: 0)
    }
    .lineLimit(1)
    .minimumScaleFactor(0.7)
  }

  private func chip(_ text: String, _ color: Color) -> some View {
    Text(text)
      .font(.system(size: 10, weight: .bold, design: .monospaced))
      .tracking(0.6)
      .padding(.horizontal, 7)
      .padding(.vertical, 3)
      .foregroundStyle(color)
      .overlay(Capsule().strokeBorder(color.opacity(0.5), lineWidth: 0.8))
      .contentTransition(.numericText())
  }
}

// MARK: - 5 Predict next token

struct PredictStage: View {
  @Bindable var model: LoomModel

  var body: some View {
    InstrumentPanel(number: "5", title: "Predict next token", badge: .live,
      nameplate: "The final vector is compared with all \(model.config.vocab.formatted()) token vectors (a dot product each), giving a score per token. Softmax turns the scores into probabilities.",
      tint: Stage.predict.color, lit: model.liveStage == .predict) {
      if let p = model.pending {
        VStack(spacing: 7) {
          ForEach(Array(p.order.prefix(6)), id: \.self) { t in
            CandidateBar(model: model, token: t, draw: p)
          }
        }
        HStack {
          Text("\(p.kept.formatted()) of \(model.config.vocab.formatted()) tokens in the running")
            .font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.orange)
            .contentTransition(.numericText())
          Spacer()
          Formula("p = softmax(z / τ)", color: Theme.orange)
        }
        Button("Every step of the prediction") { model.inspector = .readout }
          .font(.caption.weight(.semibold))
          .buttonStyle(.glass)
          .accessibilityIdentifier("readout")
      } else {
        Text("Type a message to start.").font(.footnote).foregroundStyle(.secondary)
      }
    }
  }
}

/// One token's probability: solid bar after the knobs, outline for the model's own odds (τ = 1, no cuts).
struct CandidateBar: View {
  let model: LoomModel
  let token: Int
  let draw: Draw
  var showControls = false

  var body: some View {
    let p = draw.final[token]
    let natural = draw.natural[token]
    let chosen = token == draw.chosen
    let banned = model.bias[token] != nil && model.bias[token]! < -1e6
    HStack(spacing: 8) {
      Text(model.cardText(token))
        .font(.system(size: 13, weight: chosen ? .bold : .medium, design: .monospaced))
        .foregroundStyle(chosen ? Theme.pink : .white)
        .strikethrough(banned)
        .frame(width: 70, alignment: .trailing)
        .lineLimit(1)
      GeometryReader { geo in
        let maxP = max(draw.final[draw.order[0]], draw.natural.max() ?? 0, 0.01)
        ZStack(alignment: .leading) {
          RoundedRectangle(cornerRadius: 2)
            .strokeBorder(Theme.orange.opacity(0.45), lineWidth: 0.8)
            .frame(width: max(2, geo.size.width * natural / maxP))
          RoundedRectangle(cornerRadius: 2)
            .fill(Theme.orange.opacity(p > 0 ? 0.9 : 0.1))
            .frame(width: max(p > 0 ? 2 : 0, geo.size.width * p / maxP))
            .shadow(color: Theme.orange.opacity(chosen ? 0.7 : 0), radius: 5)
        }
      }
      .frame(height: 11)
      Text(String(format: "%.2f", p))
        .font(.system(size: 11, weight: .semibold, design: .monospaced))
        .foregroundStyle(p > 0 ? Theme.orange : Theme.muted)
        .frame(width: 40, alignment: .trailing)
        .contentTransition(.numericText())
      if showControls {
        Button { model.toggleBan(token) } label: {
          Image(systemName: banned ? "nosign.app.fill" : "nosign").frame(width: 30, height: 30)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .accessibilityLabel(banned ? "Unban \(model.cardText(token))" : "Ban \(model.cardText(token))")
        Button { model.boost(token) } label: {
          Image(systemName: model.bias[token] == 2 ? "arrow.up.circle.fill" : "arrow.up.circle").frame(width: 30, height: 30)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .accessibilityLabel("Boost \(model.cardText(token))")
      }
    }
    .accessibilityElement(children: showControls ? .contain : .combine)
    .accessibilityLabel("\(model.cardText(token)): \(percent(p))\(chosen ? ", the sampled token" : "")")
  }
}

// MARK: - 6 Sample

struct SampleStage: View {
  @Bindable var model: LoomModel

  var body: some View {
    InstrumentPanel(number: "6", title: "Sample", badge: .live,
      nameplate: "One token is picked from the distribution, so likelier tokens win more often. The random number comes from a seeded generator: same seed, same text.",
      tint: Stage.sample.color, lit: model.liveStage == .sample) {
      if let p = model.pending, model.canWeave {
        HStack(spacing: 14) {
          Text(model.cardText(model.latchTarget ?? p.chosen))
            .font(.system(.title2, design: .monospaced).weight(.bold))
            .foregroundStyle(Theme.pink)
            .padding(.horizontal, 18)
            .padding(.vertical, 8)
            .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(Theme.pink, style: StrokeStyle(lineWidth: 1.4, dash: [5, 4])))
            .contentTransition(.opacity)
          VStack(alignment: .leading, spacing: 2) {
            Text("probability \(percent(p.final[model.latchTarget ?? p.chosen]))").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.pink)
            Text("Add it to the context.").font(.footnote).foregroundStyle(.secondary)
          }
        }
        DropBar(draw: p, model: model)
      } else {
        Text(endText)
          .font(.footnote).foregroundStyle(.secondary)
      }
    }
  }
}

/// Sampling drawn literally: probabilities laid end to end, likeliest first, and the generator's
/// uniform number u. The sampled token is the one whose stretch contains u.
struct DropBar: View {
  let draw: Draw
  let model: LoomModel

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      Canvas { ctx, size in
        var x: CGFloat = 0
        for (r, t) in draw.order.enumerated() {
          let p = draw.final[t]
          if p <= 0 { break }
          let w = size.width * p
          let rect = CGRect(x: x, y: 8, width: max(w - (w > 3 ? 1 : 0), 0.5), height: size.height - 14)
          let color = t == draw.chosen ? Theme.pink : (r % 2 == 0 ? Theme.orange.opacity(0.55) : Theme.orange.opacity(0.3))
          ctx.fill(Path(rect), with: .color(color))
          x += w
        }
        let ux = size.width * draw.u
        var stem = Path()
        stem.move(to: CGPoint(x: ux, y: 0))
        stem.addLine(to: CGPoint(x: ux, y: size.height))
        ctx.stroke(stem, with: .color(.white), lineWidth: 1.4)
        ctx.fill(Path(ellipseIn: CGRect(x: ux - 4, y: 0, width: 8, height: 8)), with: .color(.white))
      }
      .frame(height: 30)
      HStack {
        Text(String(format: "u = %.3f", draw.u))
        Spacer()
        Text("seed \(Int(model.seed)) · \(String(format: "%.1f", draw.surprise)) bits of surprise")
      }
      .font(.system(size: 10, design: .monospaced))
      .foregroundStyle(Theme.muted)
    }
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("Sampling")
    .accessibilityValue(String(format: "random number %.3f picks %@", draw.u, model.cardText(draw.chosen)))
  }
}

extension SampleStage {
  fileprivate var endText: String {
    let w = model.weave
    guard w.finished, let last = w.ids.last else { return model.engineReady ? "Stopped at the length limit." : "Waiting for the model." }
    if model.ends(last) {
      return model.isQwen && model.qwen.isSpecial(last)
        ? "The reply ended: Qwen picked \(model.cardText(last)), the token that closes its turn."
        : "The line ended: the model picked a line break."
    }
    return "The context is full."
  }
}

// MARK: - 7 Repeat

struct RepeatStage: View {
  @Bindable var model: LoomModel

  var body: some View {
    let w = model.weave
    let done = w.woven.count
    InstrumentPanel(number: "7", title: "Repeat", badge: .live,
      nameplate: model.isQwen && model.qwen.chat
        ? "The new token joins the context and the whole loop runs again, one token at a time, until Qwen closes its reply (⟨im_end⟩) or reaches the length limit."
        : "The new token joins the context and the whole loop runs again, one token at a time, until the model writes a line break or reaches the length limit.",
      tint: Stage.loop.color) {
      HStack(alignment: .center, spacing: 16) {
        LoopDial(done: done, total: Int(model.length), color: Theme.blue)
          .frame(width: 92, height: 92)
        VStack(alignment: .leading, spacing: 6) {
          Readout(title: "Generated", value: "\(done) of \(Int(model.length))", color: Theme.blue)
          Readout(title: "Work for that token", value: model.work(w.macsLastStep), color: Theme.cyan)
          if w.secondsLastStep > 0 {
            Readout(title: "Time on this phone", value: String(format: "%.0f ms", w.secondsLastStep * 1000), color: Theme.cyan)
          }
          if !w.stitches.isEmpty {
            let mean = w.stitches.map { -log2(max($0.probability, 1e-12)) }.reduce(0, +) / Double(w.stitches.count)
            Readout(title: "Average surprise", value: String(format: "%.1f bits per token", mean), color: Theme.pink)
          }
        }
      }
      if model.kvCache > 0.5 {
        Text("A KV cache keeps every earlier position's keys and values, so each new token runs just one position through the layers.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
  }
}

/// The loop drawn as a ring of steps, the latest lit.
struct LoopDial: View {
  let done: Int
  let total: Int
  let color: Color

  var body: some View {
    Canvas { ctx, size in
      let c = CGPoint(x: size.width / 2, y: size.height / 2)
      let r = min(size.width, size.height) / 2 - 6
      ctx.stroke(Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r)), with: .color(color.opacity(0.35)), lineWidth: 1)
      let n = max(1, min(total, 32))
      for k in 0..<n {
        let a = -Double.pi / 2 + Double(k) / Double(n) * 2 * .pi
        let p = CGPoint(x: c.x + r * CGFloat(cos(a)), y: c.y + r * CGFloat(sin(a)))
        let filled = Double(k) < Double(done) * Double(n) / Double(max(total, 1))
        ctx.fill(Path(ellipseIn: CGRect(x: p.x - 3, y: p.y - 3, width: 6, height: 6)), with: .color(filled ? color : color.opacity(0.2)))
      }
      ctx.draw(Text("\(done)").font(.system(.title2, design: .monospaced).weight(.bold)).foregroundStyle(color), at: c)
    }
    .accessibilityLabel("\(done) of \(total) tokens generated")
  }
}

// MARK: - The big picture

/// Where the predictions come from: data → optimization → weights → prediction → generation,
/// with this model's real numbers.
struct BigPicture: View {
  let model: LoomModel

  var body: some View {
    if model.isQwen { qwen } else { tiny }
  }

  /// Qwen3-4B's numbers: what it read, how it was optimized, what it is.
  private var qwen: some View {
    let params = model.parameterCount
    let ratio70 = 70e9 / Double(params)
    let ratioTiny = Double(params) / 1_377_408
    return VStack(alignment: .leading, spacing: 14) {
      HStack(spacing: 8) {
        Image(systemName: "lightbulb").foregroundStyle(Theme.green)
        Text("THE BIG PICTURE").font(.system(.subheadline, design: .monospaced).weight(.bold)).tracking(1.2).foregroundStyle(Theme.green)
        Spacer()
        Badge(provenance: .baked)
      }
      VStack(spacing: 6) {
        step("doc.text", "Data", "About 36 trillion tokens of text and code in 119 languages and dialects (Qwen3 technical report)", Theme.cyan)
        arrow
        step("chart.line.downtrend.xyaxis", "Optimization", "Pre-training by gradient descent on next-token loss, then instruction tuning and reinforcement learning, by Alibaba's Qwen team", Theme.yellow)
        arrow
        step("circle.grid.3x3", "Weights", "\(params.formatted()) numbers, stored at 4 bits each: 2.3 GB on this phone", Theme.violet)
        arrow
        step("chart.bar.xaxis", "Prediction", "A probability for each of \(model.config.vocab.formatted()) tokens", Theme.orange)
        arrow
        step("text.bubble", "Generation", "One token at a time, sampled from those probabilities", Theme.pink)
      }
      VStack(alignment: .leading, spacing: 6) {
        Text("NOT GUESSING. OPTIMIZATION, AT SCALE.")
          .font(.system(.footnote, design: .monospaced).weight(.bold))
          .foregroundStyle(Theme.cyan)
        Text(String(format: "The same loop as Loom 1.4M: the odds come from weights moved downhill on the loss, step after step. Qwen3-4B is about %.0f times bigger than Loom 1.4M and read millions of times more text, so its answers are usually sensible, though still about %.0f times smaller than a 70-billion-parameter model. It predicts likely text, not checked facts.", ratioTiny, ratio70))
          .font(.footnote)
          .foregroundStyle(.white.opacity(0.8))
          .fixedSize(horizontal: false, vertical: true)
      }
      .padding(12)
      .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Theme.cyan.opacity(0.5), style: StrokeStyle(lineWidth: 1, dash: [4, 4])))
    }
    .padding(16)
    .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(Theme.green.opacity(0.35), lineWidth: 1))
  }

  private var tiny: some View {
    let curve = model.library?.base.meta["curve"]?.array ?? []
    let first = curve.first?.array?.last?.number ?? 0
    let last = curve.last?.array?.last?.number ?? 0
    let params = model.weights?.parameterCount ?? 1_377_408
    return VStack(alignment: .leading, spacing: 14) {
      HStack(spacing: 8) {
        Image(systemName: "lightbulb").foregroundStyle(Theme.green)
        Text("THE BIG PICTURE").font(.system(.subheadline, design: .monospaced).weight(.bold)).tracking(1.2).foregroundStyle(Theme.green)
      }
      VStack(spacing: 6) {
        step("doc.text", "Data", "217,798 lines of public-domain text in 5 languages, from Cipher", Theme.cyan)
        arrow
        step("chart.line.downtrend.xyaxis", "Optimization", String(format: "12,000 steps of gradient descent: loss %.2f → %.2f", first, last), Theme.yellow)
        arrow
        step("circle.grid.3x3", "Weights", "\(params.formatted()) numbers, each set by that descent", Theme.violet)
        arrow
        step("chart.bar.xaxis", "Prediction", "A probability for each of 4,096 tokens", Theme.orange)
        arrow
        step("text.bubble", "Generation", "One token at a time, sampled from those probabilities", Theme.pink)
      }
      VStack(alignment: .leading, spacing: 6) {
        Text("NOT GUESSING. OPTIMIZATION, SCALED DOWN.")
          .font(.system(.footnote, design: .monospaced).weight(.bold))
          .foregroundStyle(Theme.cyan)
        Text("The output is sampled, so it varies from seed to seed. But the odds it samples from are not random: every weight was moved downhill on the loss, step after step, until \"Paris\"-style answers became likely where the training text made them likely. This model is tiny, about 50,000 times smaller than a 70-billion-parameter model, so its sentences are dreamy and it invents its facts.")
          .font(.footnote)
          .foregroundStyle(.white.opacity(0.8))
          .fixedSize(horizontal: false, vertical: true)
      }
      .padding(12)
      .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(Theme.cyan.opacity(0.5), style: StrokeStyle(lineWidth: 1, dash: [4, 4])))
    }
    .padding(16)
    .overlay(RoundedRectangle(cornerRadius: 20).strokeBorder(Theme.green.opacity(0.35), lineWidth: 1))
  }

  private var arrow: some View {
    Image(systemName: "arrow.down").font(.system(size: 10, weight: .bold)).foregroundStyle(Theme.muted).frame(maxWidth: .infinity)
  }

  private func step(_ symbol: String, _ title: String, _ detail: String, _ color: Color) -> some View {
    HStack(spacing: 12) {
      Image(systemName: symbol)
        .font(.system(size: 18, weight: .medium))
        .foregroundStyle(color)
        .frame(width: 30)
      VStack(alignment: .leading, spacing: 2) {
        Text(title).font(.system(.subheadline, design: .monospaced).weight(.bold)).foregroundStyle(color)
        Text(detail).font(.caption).foregroundStyle(.white.opacity(0.78))
      }
      Spacer(minLength: 0)
    }
    .accessibilityElement(children: .combine)
  }
}
