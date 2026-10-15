import LoomEngine
import SwiftUI

/// The sheet that opens one stage of the pipeline (Cipher's sticky sheet: medium/large, the pipeline stays live behind).
struct InspectorSheet: View {
  @Bindable var model: LoomModel
  let inspector: Inspector
  @Environment(\.dismiss) private var dismiss
  @State private var detent: PresentationDetent = .medium

  var body: some View {
    NavigationStack {
      ScrollView {
        VStack(spacing: 14) {
          switch inspector {
          case .punch: PunchInspector(model: model)
          case .floor(let f): FloorInspector(model: model, floor: f)
          case .readout: ReadoutInspector(model: model)
          }
        }
        .padding(14)
      }
      .background(Theme.background)
      .navigationTitle(title)
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .topBarLeading) {
          Button(detent == .large ? "Smaller" : "Full screen", systemImage: detent == .large ? "arrow.down.right.and.arrow.up.left" : "arrow.up.left.and.arrow.down.right") {
            detent = detent == .large ? .medium : .large
          }
          .labelStyle(.iconOnly)
        }
        ToolbarItem(placement: .confirmationAction) {
          Button("Done") { dismiss() }.fontWeight(.semibold)
        }
      }
    }
    .presentationDetents([.medium, .large], selection: $detent)
    .presentationBackgroundInteraction(.enabled(upThrough: .medium))
    .presentationDragIndicator(.visible)
  }

  private var title: String {
    switch inspector {
    case .punch: model.isQwen ? "Inside Qwen's tokenizer" : "Inside the tokenizer"
    case .floor(let f): "Inside layer \(f + 1)"
    case .readout: "Prediction and sampling"
    }
  }
}

// MARK: - Tokenizer

struct PunchInspector: View {
  @Bindable var model: LoomModel
  @State private var piece = 0

  var body: some View {
    if model.isQwen { QwenTokenizerInspector(model: model) } else { tiny }
  }

  @ViewBuilder private var tiny: some View {
    let tok = model.tokenizer
    let text = Tokenizer.normalize(model.prompt)
    let pieces = Tokenizer.pieces(text)
    let ids = tok?.encode(text) ?? []
    let chars = text.unicodeScalars.count

    InstrumentPanel(number: "1", title: "Pre-tokenize", badge: .live,
      nameplate: "First the text is cut at spaces and wherever letters, numbers and punctuation meet. Tokens never cross these cuts.", tint: Theme.green) {
      FlowChips(items: pieces.enumerated().map { ($0.offset, $0.element) }, selected: piece) { piece = $0 }
      HStack(spacing: 18) {
        Readout(title: "Characters", value: "\(chars)")
        Readout(title: "Tokens", value: "\(ids.count)", color: Theme.green)
        Readout(title: "Chars per token", value: ids.isEmpty ? "–" : String(format: "%.1f", Double(chars) / Double(ids.count)), color: Theme.green)
      }
    }

    if let tok, piece < pieces.count {
      let steps = tok.replay(pieces[piece])
      InstrumentPanel(number: "2", title: "Byte-pair merges", badge: .live,
        nameplate: "Start from single characters and keep joining the pair with the lowest merge rank. The merges were learned from the training text, most frequent pairs first.", tint: Theme.green) {
        VStack(alignment: .leading, spacing: 8) {
          ForEach(Array(steps.enumerated()), id: \.offset) { i, step in
            HStack(spacing: 6) {
              Text(step.merge.map { "#\($0.rank)" } ?? "start")
                .font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
                .frame(width: 46, alignment: .leading)
              ForEach(Array(step.symbols.enumerated()), id: \.offset) { _, s in
                Text(tok.text(s).replacingOccurrences(of: " ", with: "␣"))
                  .font(.system(size: 13, weight: .medium, design: .monospaced))
                  .padding(.horizontal, 5).padding(.vertical, 3)
                  .background(step.merge?.result == s ? Theme.green.opacity(0.3) : Color.white.opacity(0.07), in: .rect(cornerRadius: 5))
              }
            }
            .opacity(i == steps.count - 1 ? 1 : 0.75)
          }
        }
      }
    }

    InstrumentPanel(number: "3", title: "Token ids", badge: .live,
      nameplate: "Each token is a number from 0 to 4,095: its row in the vocabulary. Ids 2–7 are language tags.", tint: Theme.green) {
      FlowLayout(spacing: 8) {
        ForEach(Array(ids.enumerated()), id: \.offset) { _, id in
          TokenChip(token: id, text: model.cardText(id))
        }
      }
    }

    InstrumentPanel(number: "4", title: "Why some languages cost more", badge: .live,
      nameplate: "English, Spanish and French outnumber Hindi and Gujarati in the training text, so their words learned longer tokens. Indic words split into more pieces, which uses up the context faster.", tint: Theme.green) {
      ForEach(["The night was dark", "La noche era oscura", "La nuit était noire", "रात अंधेरी थी", "રાત અંધારી હતી"], id: \.self) { s in
        let n = tok?.encode(s).count ?? 0
        HStack {
          Text(s).font(.callout)
          Spacer()
          Text("\(n) tokens").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.green)
        }
      }
    }
  }
}

/// Qwen's byte-level BPE: the pre-tokenizer's cuts, each piece's UTF-8 bytes, and the tokens they
/// merge into.
struct QwenTokenizerInspector: View {
  @Bindable var model: LoomModel
  @State private var piece = 0

  var body: some View {
    let text = model.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    let pieces = QwenRuntime.pretokenize(text)
    let ids = model.qwen.tokens(text)
    let chars = text.unicodeScalars.count
    InstrumentPanel(number: "1", title: "Pre-tokenize", badge: .live,
      nameplate: "First the text is cut where letters, numbers, punctuation and spaces meet (a space stays with the word after it). Tokens never cross these cuts.", tint: Theme.green) {
      FlowChips(items: pieces.enumerated().map { ($0.offset, $0.element) }, selected: piece) { piece = $0 }
      HStack(spacing: 18) {
        Readout(title: "Characters", value: "\(chars)")
        Readout(title: "Bytes", value: "\(text.utf8.count)")
        Readout(title: "Tokens", value: "\(ids.count)", color: Theme.green)
      }
    }
    if piece < pieces.count {
      let p = pieces[piece]
      let pieceIDs = model.qwen.tokens(p)
      InstrumentPanel(number: "2", title: "Bytes, then merges", badge: .live,
        nameplate: "Every piece becomes its UTF-8 bytes, so any text in any script can be written. Then 151,387 merge rules learned from Qwen's training text join neighbouring bytes, most frequent pairs first, until no rule applies. This shows the start and the end; the merges in between aren't replayed here.", tint: Theme.green) {
        Text("BYTES").eyebrow()
        FlowLayout(spacing: 4) {
          ForEach(Array(p.utf8.enumerated()), id: \.offset) { _, b in
            Text(String(format: "%02x", b))
              .font(.system(size: 12, weight: .medium, design: .monospaced))
              .padding(.horizontal, 4).padding(.vertical, 3)
              .background(Color.white.opacity(0.07), in: .rect(cornerRadius: 4))
          }
        }
        Text("TOKENS").eyebrow()
        FlowLayout(spacing: 8) {
          ForEach(Array(pieceIDs.enumerated()), id: \.offset) { _, id in
            TokenChip(token: id, text: model.cardText(id))
          }
        }
      }
    }
    InstrumentPanel(number: "3", title: "Token ids", badge: .live,
      nameplate: "Each token is a number from 0 to \((model.config.vocab - 1).formatted()): its row in the vocabulary. Ids from 151,643 are control tokens such as ⟨im_start⟩, which the chat template uses.", tint: Theme.green) {
      FlowLayout(spacing: 8) {
        ForEach(Array(model.promptIDs().enumerated()), id: \.offset) { _, id in
          TokenChip(token: id, text: model.cardText(id), control: model.isControl(id))
        }
      }
    }
    InstrumentPanel(number: "4", title: "Why some languages cost more", badge: .live,
      nameplate: "Merges were learned from text that is mostly English and Chinese, so English words are often one token while Hindi and Gujarati split into more, sometimes mid-character. More tokens use up the context faster.", tint: Theme.green) {
      ForEach(["The night was dark", "La noche era oscura", "La nuit était noire", "रात अंधेरी थी", "રાત અંધારી હતી"], id: \.self) { s in
        HStack {
          Text(s).font(.callout)
          Spacer()
          Text("\(model.qwen.tokens(s).count) tokens").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.green)
        }
      }
    }
  }
}

// MARK: - A floor

struct FloorInspector: View {
  @Bindable var model: LoomModel
  let floor: Int
  @State private var head = 0
  @State private var neuron: Int?

  var body: some View {
    let threads = model.weave.threads
    let fi = model.focusThread ?? 0
    if threads.isEmpty || fi >= threads.count {
      Text("Type a message first.").foregroundStyle(.secondary)
    } else {
      let t = threads[fi]
      let ran = floor < t.ran.count && t.ran[floor]
      InstrumentPanel(number: "A", title: "Multi-head attention", badge: .live,
        nameplate: model.isQwen
          ? "Each of \(model.config.heads) heads compares this position's query with every earlier position's key, and blends their values by how well they match. Heads come in groups of \(model.config.heads / max(model.qwen.kvHeads, 1)) that share one set of keys and values (grouped-query attention): they ask different questions of the same memory. Rotary positions (RoPE) turn q and k first. The weights say how much was mixed in, not what the model \"meant\"."
          : "Each head compares this position's query with every earlier position's key, and blends their values by how well they match. Rotary positions (RoPE) turn q and k first, so distance matters. The weights say how much was mixed in, not what the model \"meant\".", tint: Theme.violet) {
        Formula("Attention = softmax(q·kᵀ / √\(model.isQwen ? model.qwen.headDim : model.config.headDim)) · v", color: Theme.violet)
        if model.config.heads <= 8 {
          Picker("Head", selection: $head) {
            ForEach(0..<model.config.heads, id: \.self) { h in Text("Head \(h + 1)").tag(h) }
          }
          .pickerStyle(.segmented)
        } else {
          HeadStrip(model: model, floor: floor, head: $head)
        }
        if ran, head < t.attention[floor].count {
          AttentionArcs(model: model, threads: threads, focus: fi, weights: t.attention[floor][head], color: Theme.heads[head % Theme.heads.count])
          AttentionMatrix(threads: threads, floor: floor, head: head, focus: fi)
        } else {
          Text(model.bypass.contains(floor) ? "This layer is skipped." : "This layer is past the Layers-used knob.")
            .font(.footnote).foregroundStyle(Theme.orange)
        }
        HStack {
          let id = HeadID(layer: floor, head: head)
          Button(model.muted.contains(id) ? "Turn head on" : "Turn head off", systemImage: "power") {
            model.toggleMute(id)
          }
          .buttonStyle(.glass)
          Button(model.bypass.contains(floor) ? "Run this layer" : "Skip layer", systemImage: "arrow.turn.down.right") {
            model.toggleBypass(floor)
          }
          .buttonStyle(.glass)
        }
        .font(.subheadline)
      }

      InstrumentPanel(number: "B", title: "Feedforward network · \(model.config.ff.formatted()) neurons", badge: .live,
        nameplate: model.isQwen
          ? "Then each position on its own passes through a neural network (SwiGLU) of \(model.config.ff.formatted()) neurons. Yellow fire positive, blue negative. Tap one, or one of the strongest, to see what it pushes the prediction toward."
          : "Then each position on its own passes through a small neural network (SwiGLU). Yellow neurons fire positive, blue negative. Tap one to see what it pushes the prediction toward, and what text lights it up.", tint: Theme.violet) {
        Formula("FFN(h) = (silu(h·W₁) ⊙ h·W₂) · W₃", color: Theme.violet)
        if ran, floor < t.neurons.count, !t.neurons[floor].isEmpty {
          let values = t.neurons[floor]
          let clamped = Set(model.clamps.filter { $0.layer == floor }.map(\.neuron))
          if values.count > 1000 {
            StrongestNeurons(values: values, selected: $neuron)
            NeuronBank(values: values, selected: $neuron, clamps: clamped, columns: 96)
          } else {
            NeuronBank(values: values, selected: $neuron, clamps: clamped)
          }
          if let n = neuron, n < values.count { NeuronCard(model: model, floor: floor, neuron: n, activation: values[n]) }
        } else if ran {
          Text("Qwen keeps the \(model.config.ff.formatted()) neuron values of every layer for the newest token only. Select the newest token to see them.")
            .font(.footnote).foregroundStyle(Theme.muted)
        }
      }

      let full = t.residual[floor].count == model.config.d
      let columns = model.config.d > 128 ? 64 : 32
      InstrumentPanel(number: "C", title: "Residual connection + norm", badge: .live,
        nameplate: full
          ? "Each block's output is added to its input, never replacing it, and each block normalises (RMSNorm) what it reads. Here are all \(model.config.d.formatted()) numbers of \"\(model.cardText(t.token))\" before and after this layer."
          : "Each block's output is added to its input, never replacing it. For earlier tokens Qwen keeps the first 32 of the \(model.config.d.formatted()) numbers and the last; select the newest token to see them all.", tint: Theme.violet) {
        Formula("x ← x + Attn(Norm(x));  x ← x + FFN(Norm(x))", color: Theme.violet)
        VStack(alignment: .leading, spacing: 6) {
          Text("BEFORE").eyebrow()
          ValueStrip(values: t.residual[floor], columns: columns, cell: full && columns > 32 ? 4 : 6)
          Text("AFTER").eyebrow()
          ValueStrip(values: t.residual[floor + 1], columns: columns, cell: full && columns > 32 ? 4 : 6)
          if floor < t.lens.count, ran {
            let l = t.lens[floor]
            Text("Logit lens: if the model stopped after this layer it would predict \"\(model.cardText(l.token))\" (\(percent(l.p))).")
              .font(.footnote).foregroundStyle(.white.opacity(0.85))
          }
        }
      }
    }
  }
}

/// The focused position's weights for one head, drawn as arcs over the tokens.
struct AttentionArcs: View {
  let model: LoomModel
  let threads: [Thread]
  let focus: Int
  let weights: [Float]
  let color: Color

  var body: some View {
    let visible = Array(max(0, focus - 7)...focus)
    VStack(spacing: 6) {
      Canvas { ctx, size in
        let n = visible.count
        let col = size.width / CGFloat(max(n, 1))
        func x(_ k: Int) -> CGFloat { CGFloat(k) * col + col / 2 }
        let base = size.height - 2
        for (k, j) in visible.enumerated() where j != focus {
          let w = j < weights.count ? weights[j] : 0
          guard w > 0.005 else { continue }
          var p = Path()
          p.move(to: CGPoint(x: x(n - 1), y: base))
          p.addQuadCurve(to: CGPoint(x: x(k), y: base), control: CGPoint(x: (x(n - 1) + x(k)) / 2, y: base - size.height * 0.9))
          ctx.stroke(p, with: .color(color.opacity(Double(0.2 + 0.8 * w))), lineWidth: CGFloat(0.5 + 6 * w))
        }
        if focus < weights.count {
          let self_ = weights[focus]
          ctx.fill(Path(ellipseIn: CGRect(x: x(n - 1) - CGFloat(4 + 10 * self_), y: base - CGFloat(4 + 10 * self_), width: CGFloat(8 + 20 * self_), height: CGFloat(8 + 20 * self_))), with: .color(color.opacity(0.4)))
        }
      }
      .frame(height: 70)
      HStack(spacing: 0) {
        ForEach(visible, id: \.self) { j in
          VStack(spacing: 2) {
            Text(model.cardText(threads[j].token)).font(.system(size: 11, weight: j == focus ? .bold : .regular, design: .monospaced)).lineLimit(1).minimumScaleFactor(0.6)
            Text(j < weights.count ? percent(Double(weights[j])) : "").font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
          }
          .frame(maxWidth: .infinity)
        }
      }
    }
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("Attention from \(model.cardText(threads[focus].token))")
    .accessibilityValue(visible.map { "\(model.cardText(threads[$0].token)) \(percent(Double($0 < weights.count ? weights[$0] : 0)))" }.joined(separator: ", "))
  }
}

/// Every position's weights for one head: rows attend to columns at or before them.
struct AttentionMatrix: View {
  let threads: [Thread]
  let floor: Int
  let head: Int
  let focus: Int

  var body: some View {
    let n = min(threads.count, 24)
    let offset = threads.count - n
    Canvas { ctx, size in
      let c = min(22, min(size.width, size.height) / CGFloat(max(n, 1)))
      ctx.translateBy(x: (size.width - c * CGFloat(n)) / 2, y: 0)
      for i in 0..<n {
        let t = threads[offset + i]
        guard floor < t.attention.count, head < t.attention[floor].count else { continue }
        let row = t.attention[floor][head]
        for j in 0..<n {
          let src = offset + j
          guard src < row.count else { continue }
          let w = Double(row[src])
          ctx.fill(Path(CGRect(x: CGFloat(j) * c, y: CGFloat(i) * c, width: c - 0.5, height: c - 0.5)),
            with: .color(Theme.heads[head].opacity(0.04 + 0.96 * w)))
        }
        if offset + i == focus {
          ctx.stroke(Path(CGRect(x: 0, y: CGFloat(i) * c, width: CGFloat(n) * c, height: c)), with: .color(.white.opacity(0.6)), lineWidth: 0.8)
        }
      }
    }
    .frame(height: min(150, CGFloat(min(threads.count, 24)) * 22))
    .accessibilityLabel("Attention matrix for the last \(n) threads")
  }
}

/// Qwen's attention heads for one layer: chips in groups that share keys and values.
struct HeadStrip: View {
  @Bindable var model: LoomModel
  let floor: Int
  @Binding var head: Int

  var body: some View {
    let heads = model.config.heads
    let group = max(1, heads / max(model.qwen.kvHeads, 1))
    VStack(alignment: .leading, spacing: 4) {
      ScrollView(.horizontal, showsIndicators: false) {
        HStack(spacing: 3) {
          ForEach(0..<heads, id: \.self) { h in
            let off = model.muted.contains(HeadID(layer: floor, head: h))
            Button { head = h } label: {
              Text("\(h + 1)")
                .font(.system(size: 11, weight: h == head ? .bold : .medium, design: .monospaced))
                .strikethrough(off)
                .frame(minWidth: 26, minHeight: 28)
                .foregroundStyle(h == head ? .black : .white)
                .background(Theme.heads[h % Theme.heads.count].opacity(h == head ? 0.95 : 0.22), in: .rect(cornerRadius: 6))
            }
            .buttonStyle(.plain)
            .padding(.trailing, h % group == group - 1 ? 5 : 0)
            .accessibilityLabel("Head \(h + 1)\(off ? ", off" : "")")
            .accessibilityAddTraits(h == head ? .isSelected : [])
          }
        }
      }
      Text("Head \(head + 1) reads key/value head \(head / group + 1), shared with heads \(head / group * group + 1)–\(head / group * group + group).")
        .font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
    }
  }
}

/// The neurons firing hardest, as chips (Qwen's bank is too fine to tap one by one).
struct StrongestNeurons: View {
  let values: [Float]
  @Binding var selected: Int?

  var body: some View {
    let top = values.indices.sorted { abs(values[$0]) > abs(values[$1]) }.prefix(12)
    VStack(alignment: .leading, spacing: 4) {
      Text("STRONGEST").eyebrow()
      FlowLayout(spacing: 4) {
        ForEach(Array(top), id: \.self) { i in
          Button { selected = i } label: {
            Text("\(i) \(VectorRow.format(values[i]))")
              .font(.system(size: 11, weight: i == selected ? .bold : .medium, design: .monospaced))
              .padding(.horizontal, 6).padding(.vertical, 3)
              .foregroundStyle(values[i] >= 0 ? Theme.positive : Theme.negative)
              .background(Color.white.opacity(i == selected ? 0.18 : 0.06), in: .rect(cornerRadius: 5))
          }
          .buttonStyle(.plain)
        }
      }
    }
  }
}

/// The neurons as bulbs: 384 in a 24 × 16 grid, or Qwen's 9,728 as a fine 96-wide field.
struct NeuronBank: View {
  let values: [Float]
  @Binding var selected: Int?
  let clamps: Set<Int>
  var columns = 24

  var body: some View {
    let cols = columns
    let scale = max(values.map(abs).max() ?? 1, 1e-6)
    Canvas { ctx, size in
      let c = size.width / CGFloat(cols)
      let inset: CGFloat = c > 6 ? 1 : 0.25
      for (i, v) in values.enumerated() {
        let r = CGRect(x: CGFloat(i % cols) * c + inset, y: CGFloat(i / cols) * c + inset, width: c - 2 * inset, height: c - 2 * inset)
        if c > 6 { ctx.fill(Path(ellipseIn: r), with: .color(Theme.value(v, scale: scale))) } else { ctx.fill(Path(r), with: .color(Theme.value(v, scale: scale))) }
        if clamps.contains(i) {
          ctx.stroke(Path(ellipseIn: r.insetBy(dx: -1, dy: -1)), with: .color(Theme.heads[2]), lineWidth: 1.2)
        }
        if i == selected {
          ctx.stroke(Path(ellipseIn: r.insetBy(dx: -1.5, dy: -1.5)), with: .color(.white), lineWidth: 1.5)
        }
      }
    }
    .aspectRatio(CGFloat(cols) / CGFloat((values.count + cols - 1) / cols), contentMode: .fit)
    .overlay {
      GeometryReader { geo in
        Color.clear.contentShape(.rect).gesture(SpatialTapGesture().onEnded { e in
          let c = geo.size.width / CGFloat(cols)
          let i = Int(e.location.y / c) * cols + Int(e.location.x / c)
          if i >= 0, i < values.count { selected = i }
        })
      }
    }
    .accessibilityElement()
    .accessibilityLabel("Neuron bank")
    .accessibilityValue("\(values.filter { abs($0) > scale * 0.5 }.count) neurons strongly active")
  }
}

/// A two-sided neuron card: what lights it up (training text) and what it pushes the next token toward.
struct NeuronCard: View {
  @Bindable var model: LoomModel
  let floor: Int
  let neuron: Int
  let activation: Float
  @State private var qwenVotes: (raises: [Int], lowers: [Int])?

  var body: some View {
    let (pushes, pulls) = model.isQwen ? (qwenVotes?.raises ?? [], qwenVotes?.lowers ?? []) : votes()
    let clamp = model.clamp(floor: floor, neuron: neuron)
    let reach = model.isQwen ? max(8, Double(abs(activation)) * 2) : 8
    VStack(alignment: .leading, spacing: 10) {
      HStack {
        Text("LAYER \(floor + 1) · NEURON \(neuron)").eyebrow()
        Spacer()
        Text(String(format: "%+.2f", activation)).font(.system(.callout, design: .monospaced)).foregroundStyle(activation >= 0 ? Theme.positive : Theme.negative)
      }
      HStack(alignment: .top) {
        VStack(alignment: .leading, spacing: 3) {
          Text("RAISES").eyebrow()
          Text(pushes.map { "\"\(model.cardText($0))\"" }.joined(separator: " ")).font(.system(.footnote, design: .monospaced))
        }
        Spacer()
        VStack(alignment: .leading, spacing: 3) {
          Text("LOWERS").eyebrow()
          Text(pulls.map { "\"\(model.cardText($0))\"" }.joined(separator: " ")).font(.system(.footnote, design: .monospaced))
        }
      }
      if !model.isQwen, let lib = model.library {
        let moments = lib.neurons.moments(floor: floor, neuron: neuron)
        if !moments.isEmpty {
          Text("FIRES MOST ON (FROM 2,500 TRAINING LINES)").eyebrow()
          ForEach(moments, id: \.self) { m in
            Text(moment(m))
              .font(.system(.footnote, design: .serif))
              .lineLimit(1)
          }
        }
      }
      VStack(alignment: .leading, spacing: 4) {
        HStack {
          Text("CLAMP").eyebrow()
          Spacer()
          Text(clamp.map { String(format: "held at %+.1f", $0) } ?? "free").font(.system(.caption, design: .monospaced)).foregroundStyle(clamp == nil ? Theme.muted : Theme.heads[2])
        }
        Slider(value: Binding(get: { clamp ?? Double(activation) }, set: { model.setClamp(floor: floor, neuron: neuron, value: $0) }), in: (model.isQwen ? -reach : -4)...reach)
          .tint(Theme.heads[2])
        if clamp != nil {
          Button("Let go", systemImage: "hand.raised.slash") { model.setClamp(floor: floor, neuron: neuron, value: nil) }
            .buttonStyle(.glass)
            .font(.subheadline)
        }
        Text("Holding one neuron fixed at every position is the Golden Gate Claude trick at toy scale. Watch the next-token probabilities.")
          .font(.caption).foregroundStyle(.secondary)
      }
    }
    .padding(14)
    .background(Theme.well, in: .rect(cornerRadius: 16))
    .task(id: "\(model.isQwen)|\(floor)|\(neuron)") {
      guard model.isQwen else { return }
      qwenVotes = nil
      qwenVotes = try? await model.qwen.worker.neuronVotes(layer: floor, neuron: neuron)
    }
  }

  private func moment(_ m: NeuronCards.Moment) -> AttributedString {
    var before = AttributedString(m.before)
    before.foregroundColor = .secondary
    var token = AttributedString(m.token)
    token.foregroundColor = Theme.amber
    token.inlinePresentationIntent = .stronglyEmphasized
    var after = AttributedString(m.after)
    after.foregroundColor = .secondary
    return before + token + after
  }

  /// The neuron's output row read through the readout: Wd[n] · wteᵀ.
  private func votes() -> ([Int], [Int]) {
    guard let w = model.weights else { return ([], []) }
    let d = w.config.d
    let row = Array(w.blocks[floor].wd[(neuron * d)..<((neuron + 1) * d)])
    let logits = Forward.readout(w, row, rows: 1)
    let order = logits.indices.sorted { logits[$0] > logits[$1] }
    return (Array(order.prefix(5)), Array(order.suffix(3).reversed()))
  }
}

// MARK: - Readout and draw

struct ReadoutInspector: View {
  @Bindable var model: LoomModel

  var body: some View {
    if let p = model.pending {
      InstrumentPanel(number: "1", title: "Scores (logits)", badge: .live,
        nameplate: "The last position's final vector is normalised and dotted with all \(model.config.vocab.formatted()) token vectors, the same table the embedding used. One score per token: its logit.", tint: Theme.orange) {
        Formula("zᵢ = Norm(x) · Eᵢ", color: Theme.orange)
        LogitHistogram(draw: p)
      }
      InstrumentPanel(number: "2", title: "Probabilities", badge: .live,
        nameplate: "In this order: bans and boosts, the repetition penalty, temperature τ, softmax, then top-k, top-p and min-p remove the long tail. Outlines are the model's own odds (τ = 1, nothing removed); solid bars are what the knobs leave.", tint: Theme.orange) {
        Formula("p = softmax(z / τ)", color: Theme.orange)
        ForEach(Array(p.order.prefix(12)), id: \.self) { t in
          CandidateBar(model: model, token: t, draw: p, showControls: true)
        }
        HStack(spacing: 18) {
          Readout(title: "Temperature", value: model.temperature < 0.005 ? "Greedy" : String(format: "%.2f", model.temperature), color: Theme.orange)
          Readout(title: "In the running", value: "\(p.kept.formatted())")
          Readout(title: "Banned", value: "\(model.bias.values.filter { $0 < -1e6 }.count)")
        }
      }
      InstrumentPanel(number: "3", title: "Sample", badge: .live,
        nameplate: "The seeded generator gives one number u between 0 and 1. Lay the remaining probabilities end to end, likeliest first: the token whose stretch contains u is picked. Turn a knob while paused and watch u stay put while the stretches move.", tint: Theme.pink) {
        DropBar(draw: p, model: model)
      }
    } else {
      Text("Type a message first.").foregroundStyle(.secondary)
    }
  }
}

/// Every logit as a histogram.
struct LogitHistogram: View {
  let draw: Draw

  var body: some View {
    let z = draw.adjusted.filter(\.isFinite)
    let lo = z.min() ?? 0, hi = z.max() ?? 1
    let bins = 48
    var counts = [Int](repeating: 0, count: bins)
    for v in z { counts[min(bins - 1, Int((v - lo) / max(hi - lo, 1e-9) * Double(bins)))] += 1 }
    let mx = Double(counts.max() ?? 1)
    return VStack(alignment: .leading, spacing: 4) {
      Canvas { ctx, size in
        let w = size.width / CGFloat(bins)
        for (i, c) in counts.enumerated() {
          let h = size.height * CGFloat(log(1 + Double(c)) / log(1 + mx))
          ctx.fill(Path(CGRect(x: CGFloat(i) * w, y: size.height - h, width: w - 1, height: h)), with: .color(Theme.amber.opacity(0.7)))
        }
      }
      .frame(height: 60)
      HStack {
        Text(String(format: "%.1f", lo))
        Spacer()
        Text("score (logit) · log count of tokens")
        Spacer()
        Text(String(format: "%.1f", hi))
      }
      .font(.system(size: 10, design: .monospaced))
      .foregroundStyle(Theme.muted)
    }
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("Histogram of \(draw.adjusted.count.formatted()) scores")
  }
}

// MARK: - Layout helpers

/// Wrapping row layout (Cipher's FlowLayout).
struct FlowLayout: Layout {
  var spacing: CGFloat = 6

  func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
    let width = proposal.width ?? 320
    var x: CGFloat = 0, y: CGFloat = 0, rowH: CGFloat = 0
    for s in subviews {
      let size = s.sizeThatFits(.unspecified)
      if x + size.width > width, x > 0 { x = 0; y += rowH + spacing; rowH = 0 }
      x += size.width + spacing
      rowH = max(rowH, size.height)
    }
    return CGSize(width: width, height: y + rowH)
  }

  func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
    var x = bounds.minX, y = bounds.minY, rowH: CGFloat = 0
    for s in subviews {
      let size = s.sizeThatFits(.unspecified)
      if x + size.width > bounds.maxX, x > bounds.minX { x = bounds.minX; y += rowH + spacing; rowH = 0 }
      s.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(size))
      x += size.width + spacing
      rowH = max(rowH, size.height)
    }
  }
}

struct FlowChips: View {
  let items: [(Int, String)]
  let selected: Int
  let pick: (Int) -> Void

  var body: some View {
    FlowLayout(spacing: 6) {
      ForEach(items, id: \.0) { i, s in
        Button { pick(i) } label: {
          Text(s.replacingOccurrences(of: " ", with: "␣"))
            .font(.system(size: 14, weight: .medium, design: .monospaced))
            .padding(.horizontal, 8).padding(.vertical, 5)
            .background(i == selected ? Theme.green.opacity(0.3) : Color.white.opacity(0.07), in: .rect(cornerRadius: 7))
        }
        .buttonStyle(.plain)
      }
    }
  }
}
