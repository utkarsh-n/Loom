import Foundation
import LoomEngine
import SwiftUI

/// "How it works": the pipeline, what is real, and why it isn't guessing, for the engine that is running.
struct FieldNotesView: View {
  let model: LoomModel
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    let qwen = model.isQwen
    NavigationStack {
      ScrollView {
        VStack(alignment: .leading, spacing: 14) {
          VStack(alignment: .leading, spacing: 8) {
            Text("HOW IT WORKS").font(.system(.subheadline, design: .monospaced).weight(.bold)).tracking(1.2).foregroundStyle(Theme.green)
            Text(qwen ? "A real language model, opened up on this phone." : "A real language model, small enough to see inside.")
              .font(.system(.title, design: .serif))
            Text(intro)
              .foregroundStyle(.secondary)
          }
          .padding(.horizontal, 4)

          InstrumentPanel(number: "→", title: "The pipeline", tint: Theme.cyan) {
            ForEach(Stage.allCases) { stage in
              HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text("\(stage.rawValue)").font(.system(.subheadline, design: .monospaced).weight(.bold)).foregroundStyle(stage.color).frame(width: 18)
                VStack(alignment: .leading, spacing: 2) {
                  Text(stage.title).font(.subheadline.weight(.semibold)).foregroundStyle(stage.color)
                  Text(explain(stage)).font(.callout).foregroundStyle(.white.opacity(0.85)).fixedSize(horizontal: false, vertical: true)
                }
              }
            }
          }

          InstrumentPanel(number: "Σ", title: "The model · \(model.engine.title)", tint: Theme.violet) {
            if qwen { qwenModelNotes } else { tinyModelNotes }
          }

          InstrumentPanel(number: "∇", title: "Not guessing: optimization", tint: Theme.yellow) {
            if qwen {
              note("Training", "Every weight started random. Training measured how wrong the model's next-token predictions were (the loss), worked out which way each weight should move to make it less wrong (the gradient), and moved them all a small step downhill, over and over, across trillions of tokens; more training then taught it to follow instructions. None of that ran here: switch to \(EngineKind.tiny.title) and Tune shows the same loop running on your phone.")
            } else {
              note("Training", "Every weight started random. Training measured how wrong the model's next-token predictions were (the loss), worked out which way each weight should move to make it less wrong (the gradient), and moved them all a small step downhill, 12,000 times. Fine-tune shows the same loop running on your phone.")
            }
            note("Generation", "The model computes a probability for every token, then samples one. Sampling varies; the probabilities don't. They are what the optimized weights compute from the context.")
            note("Scale", scale)
          }

          InstrumentPanel(number: "✓", title: "What is real", tint: Theme.green) {
            ForEach([Provenance.live, .train, .baked, .projection, .metaphor], id: \.self) { p in
              HStack(alignment: .firstTextBaseline, spacing: 10) {
                Badge(provenance: p).frame(width: 104, alignment: .leading)
                Text(meaning(p)).font(.callout).foregroundStyle(.white.opacity(0.85))
              }
            }
            note("Order", order)
            note("Limits", limits)
          }

          InstrumentPanel(number: "↗", title: "Read more", tint: Theme.blue) {
            link("Polo Club, Transformer Explainer", "https://poloclub.github.io/transformer-explainer/")
            link("Brendan Bycroft, LLM Visualization", "https://bbycroft.net/llm")
            link("3Blue1Brown, But what is a GPT?", "https://www.3blue1brown.com/lessons/gpt")
            link("nostalgebraist, the logit lens", "https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens")
            if qwen {
              link("Qwen team, Qwen3 technical report", "https://arxiv.org/abs/2505.09388")
              link("Ainslie et al., grouped-query attention", "https://arxiv.org/abs/2305.13245")
            }
            link("Hu et al., LoRA", "https://arxiv.org/abs/2106.09685")
            link("Turner et al., activation steering", "https://arxiv.org/abs/2308.10248")
            link("Holtzman et al., nucleus sampling", "https://arxiv.org/abs/1904.09751")
            link("Su et al., RoPE", "https://arxiv.org/abs/2104.09864")
          }
        }
        .padding(14)
      }
      .background(Theme.background)
      .navigationTitle("How it works")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() }.fontWeight(.semibold) }
      }
    }
  }

  // MARK: The model

  private var tinyModelNotes: some View {
    let c = model.config
    let params = model.parameterCount
    return VStack(alignment: .leading, spacing: 12) {
      note("Shape", "\(c.layers) layers, \(c.heads) attention heads per layer, vectors of \(c.d) numbers, \(c.ff) feedforward neurons per layer, a \(c.ctx)-token context. Rotary positions (RoPE), RMSNorm, SwiGLU, and an output tied to the embedding table: the design modern large models use.")
      note("Size", "\(params.formatted()) parameters, \(Quantizer.bytes(params: params, bits: 32) / 1_000_000) MB at full precision.")
      note("Training", "Gradient descent (AdamW) on a Mac: 12,000 steps of 64 windows of 128 tokens, about 8 minutes. The curve in Tune › Checkpoints is that run's log.")
      note("Text", "From Cipher: 195,544 public-domain lines in English, Spanish, French, Hindi and Gujarati, plus 22,254 fact cards Cipher wrote about films, songs, books and shows. Nothing was downloaded. The five fine-tunes' texts and all the fact cards were held out of the base model, so each fine-tune teaches it something new.")
      note("Vocabulary", "\(c.vocab.formatted()) tokens from byte-pair encoding trained on the same text, six of them language tags (⟨en⟩ ⟨es⟩ ⟨fr⟩ ⟨hi⟩ ⟨gu⟩ ⟨card⟩).")
    }
  }

  private var qwenModelNotes: some View {
    let c = model.config
    let params = model.parameterCount
    let shipped = qwenShippedSize(params: params)
    let tokens = model.qwen.catalogue.map { "the \($0.count.formatted()) tokens" } ?? "the tokens"
    let control = max(0, c.vocab - QwenRuntime.firstSpecial)
    return VStack(alignment: .leading, spacing: 12) {
      note("Model", "\(QwenRuntime.name), pre-trained and instruction-tuned by Alibaba's Qwen team and released under Apache-2.0. The Qwen3 technical report puts its pre-training at about 36 trillion tokens.")
      note("Shape", "\(c.layers) layers. Attention has \(c.heads) query heads per layer sharing \(model.qwen.kvHeads) key/value heads (grouped-query attention), each \(model.qwen.headDim) numbers wide. Vectors of \(c.d.formatted()) numbers, \(c.ff.formatted()) feedforward neurons per layer (SwiGLU). Rotary positions (RoPE, base \(Int(c.ropeBase).formatted())), RMSNorm, a norm on each head's queries and keys (QK-norm), and an output tied to the embedding table. Loom gives it a \(c.ctx)-token context, so every position's attention can be read back to the phone.")
      note("Size", "\(params.formatted()) parameters. They ship inside the app at 4 bits, each group of 64 weights sharing a scale and an offset: \(shipped) in all, loaded onto this iPhone's GPU when Qwen is chosen and given back when Loom goes to the background.")
      note("Cipher's part", "Loom doesn't change Qwen's weights. Cipher's lines were used only on a Mac: to measure the steering directions (200 lines per group, read after each of the \(c.layers) layers) and to pick \(tokens) the sphere and the embedding map draw, the commonest in those lines.")
      note("Vocabulary", "\(c.vocab.formatted()) tokens from byte-level BPE: text in any script splits into them, down to single bytes, so a token can be part of a character (its card then shows the bytes in hex). The top \(control.formatted()) ids are control tokens, like ⟨im_start⟩ and ⟨im_end⟩ that mark chat turns, and unused rows.")
    }
  }

  // MARK: Engine-aware text

  private var intro: String {
    guard model.isQwen else {
      return "Loom runs a genuine transformer on this phone. Every number on screen is read from that computation as it happens."
    }
    let billions = String(format: "%.0f", Double(model.parameterCount) / 1e9)
    return "Loom runs \(QwenRuntime.name), a \(billions)-billion-parameter transformer, on this phone's GPU. Every number on screen is read back from that computation as it happens. Settings › Engine switches to \(EngineKind.tiny.title), the small model that trains here."
  }

  private var scale: String {
    let tiny = Double(model.library?.base.parameterCount ?? 1_377_408)
    let large = 70e9
    guard model.isQwen else {
      return "Large models do the same with billions or trillions of weights and trillions of tokens. This one is about \(roughly(large / tiny)) times smaller than a 70-billion-parameter model, so its sentences are dreamy and it invents its facts."
    }
    let params = Double(model.parameterCount)
    return "Qwen3-4B has \(String(format: "%.1f", params / 1e9)) billion weights: about \(roughly(large / params)) times fewer than a 70-billion-parameter model, and about \(roughly(params / tiny)) times more than \(EngineKind.tiny.title). It writes fluent sentences and knows a great many facts, but at this size it still gets some of them wrong, just as confidently."
  }

  private var order: String {
    guard model.isQwen else {
      return "The animation follows the real order. A prompt is processed at all positions at once (prefill); each new token then runs one position through the layers, reusing the cached keys and values. Slow speed replays that about 2,000 times slower than it ran."
    }
    return "The animation follows the real order. A prompt is processed at all positions at once (prefill); each new token then runs one position through the \(model.config.layers) layers, reusing the cached keys and values. Slow speed spreads each token's pass over \(String(format: "%.1f", Pace.slow.seconds)) seconds so you can follow it."
  }

  private var limits: String {
    let d = model.config.d
    guard model.isQwen else {
      return "Attention weights show how much was mixed in, not what the model meant. Neuron examples are examples, not definitions. The embedding map and the sphere's directions are projections of \(d) dimensions."
    }
    return "Attention weights show how much was mixed in, not what the model meant. The embedding map and the sphere's directions are projections of \(d.formatted()) dimensions. To fit on the phone, older positions keep only the first \(QwenWorker.keptDims) numbers of each vector (and the last); the newest keeps all \(d.formatted())."
  }

  private func meaning(_ p: Provenance) -> String {
    guard model.isQwen else { return p.meaning }
    switch p {
    case .baked: return "Trained elsewhere and shipped with the app: Qwen by Alibaba's Qwen team; its steering directions and token map measured on a Mac."
    case .train: return "A real training run on this phone (\(EngineKind.tiny.title) only)."
    default: return p.meaning
    }
  }

  private func explain(_ stage: Stage) -> String {
    let c = model.config
    let qwen = model.isQwen
    let chat = model.qwen.chat
    switch stage {
    case .input:
      if !qwen { return "Your message, plus every token generated so far: the context." }
      if chat {
        return "Your message, wrapped as a chat turn (⟨im_start⟩user … ⟨im_end⟩ ⟨im_start⟩assistant), plus every token generated so far: the context, up to \(c.ctx) tokens."
      }
      return "Your text, plus every token generated so far: the context, up to \(c.ctx) tokens."
    case .tokenize:
      return qwen
        ? "Split by byte-level BPE into tokens from a \(c.vocab.formatted())-token vocabulary; each token is a number."
        : "Split into tokens from a \(c.vocab.formatted())-token vocabulary; each token is a number."
    case .embedding:
      return "Each number picks a row of \(c.d.formatted()) numbers: the token's vector."
    case .transformer:
      if qwen {
        return "\(c.layers) layers of attention (positions mix information; \(c.heads) query heads share \(model.qwen.kvHeads) key/value heads) and feedforward networks (\(c.ff.formatted()) neurons; each position transformed), each added to a running vector."
      }
      return "\(Self.word(c.layers)) layers of attention (positions mix information) and feedforward networks (each position transformed), each added to a running vector."
    case .predict:
      return qwen
        ? "The last vector is scored against all \(c.vocab.formatted()) rows of the embedding table (the output is tied to it); softmax turns the scores into probabilities."
        : "The last vector is scored against every token; softmax turns the scores into probabilities."
    case .sample:
      return "One token is picked, likelier ones more often, from a seeded random number."
    case .loop:
      if !qwen { return "It joins the context and everything runs again, one token at a time." }
      return chat
        ? "It joins the context and everything runs again, one token at a time, until Qwen ends its reply (⟨im_end⟩) or the context is full."
        : "It joins the context and everything runs again, one token at a time, until a line break or the context is full."
    }
  }

  /// Small counts as words ("Four layers"), larger ones as numerals.
  private static func word(_ n: Int) -> String {
    let words = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]
    return n >= 0 && n < words.count ? words[n] : n.formatted()
  }

  /// A ratio to two significant figures: 50,820 → "51,000", 17.4 → "17", 2,920 → "2,900".
  private func roughly(_ x: Double) -> String {
    guard x.isFinite, x > 0 else { return "0" }
    let digits = floor(log10(x)) + 1
    let unit = pow(10, max(0, digits - 2))
    return Int((x / unit).rounded() * unit).formatted()
  }

  private func note(_ title: String, _ body: String) -> some View {
    VStack(alignment: .leading, spacing: 3) {
      Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(.white)
      Text(body).font(.callout).foregroundStyle(.white.opacity(0.82)).fixedSize(horizontal: false, vertical: true)
    }
  }

  private func link(_ title: String, _ url: String) -> some View {
    Link(destination: URL(string: url)!) {
      HStack {
        Text(title).font(.callout)
        Spacer()
        Image(systemName: "arrow.up.right").font(.caption)
      }
    }
  }
}

/// Qwen's weights as shipped: the real file's size when it is in the app, else the arithmetic (4 bits a
/// weight plus a 16-bit scale and offset per group of 64: 4.5 bits a weight).
private func qwenShippedSize(params: Int) -> String {
  if let dir = QwenRuntime.directory,
    let attributes = try? FileManager.default.attributesOfItem(atPath: dir.appending(path: "model.safetensors").path),
    let size = attributes[.size] as? NSNumber
  {
    return String(format: "%.1f GB", size.doubleValue / 1e9)
  }
  return String(format: "about %.1f GB", Double(params) * 4.5 / 8 / 1e9)
}

/// Settings: the engine, feel, and what stays on the phone.
struct SettingsView: View {
  @Bindable var model: LoomModel
  @Environment(\.dismiss) private var dismiss
  @State private var haptics = true
  @State private var strength: Double = 0.85

  var body: some View {
    NavigationStack {
      Form {
        engineSection
        Section {
          Toggle("Tactile feedback", isOn: $haptics)
            .onChange(of: haptics) { _, v in model.haptics.enabled = v }
          VStack(alignment: .leading) {
            Text("Strength")
            Slider(value: $strength, in: 0.2...1)
              .onChange(of: strength) { _, v in
                model.haptics.strength = Float(v)
                model.haptics.transient(intensity: 0.8, sharpness: 0.5)
              }
          }
          .disabled(!haptics)
        } header: {
          Text("Feel")
        } footer: {
          Text("Knobs tick and click at their detents. Each generated token knocks, harder when it was a surprise (a low-probability sample).")
        }
        Section {
          Button(model.isQwen ? "Reset to the plain model" : "Reset to the base model", systemImage: "arrow.uturn.backward") {
            model.baseOnly()
            dismiss()
          }
        } footer: {
          if model.isQwen {
            Text("Turns off steering and every head, layer and neuron switch, and puts layers and context back. It also turns off \(EngineKind.tiny.title)'s fine-tunes and language tag and puts its precision and checkpoint back.")
          } else {
            Text("Turns off every fine-tune, steering, language tag, head and layer switch, and puts precision, checkpoint, layers and context back.")
          }
        }
        Section("Private by design") {
          Text("Loom runs entirely on this phone. It has no network code, asks for no permissions, and keeps your prompt and Teach lines only on this device." + (model.qwen.isAvailable ? " Qwen's weights ship inside the app, so it runs offline too." : ""))
            .font(.callout)
        }
        Section("About") {
          Text("Text from Cipher's culture index: public-domain works from Project Gutenberg and Wikisource, and fact cards written from Wikidata and Wikipedia facts. \(EngineKind.tiny.title) trained with Apple's MLX.")
            .font(.footnote)
            .foregroundStyle(.secondary)
          if model.qwen.isAvailable {
            Text("\(QwenRuntime.name) by Alibaba's Qwen team (Apache-2.0), 4-bit, run with Apple's MLX Swift; its tokenizer through Hugging Face's swift-transformers.")
              .font(.footnote)
              .foregroundStyle(.secondary)
          }
        }
      }
      .navigationTitle("Settings")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() }.fontWeight(.semibold) }
      }
      .onAppear {
        haptics = model.haptics.enabled
        strength = Double(model.haptics.strength)
      }
    }
  }

  // MARK: Engine

  private var engineSection: some View {
    Section {
      Picker("Engine", selection: $model.engine) {
        ForEach(EngineKind.allCases) { kind in
          Text(kind.title).tag(kind)
        }
      }
      .pickerStyle(.segmented)
      // Without Qwen only the 1.4M model can run; the footer says why.
      .disabled(!model.qwen.isAvailable)
      .accessibilityIdentifier("engine")
      if model.qwen.isAvailable {
        HStack(spacing: 8) {
          Text("Qwen")
          Spacer(minLength: 8)
          if model.qwen.status == .loading { ProgressView().controlSize(.small) }
          Text(qwenStatus)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.trailing)
        }
        .accessibilityElement(children: .combine)
        if model.isQwen, case .failed = model.qwen.status {
          Button("Try loading again", systemImage: "arrow.clockwise") {
            Task { await model.startQwen() }
          }
        }
        Toggle("Chat template", isOn: Binding(
          get: { model.qwen.chat },
          set: { on in
            model.qwen.chat = on
            if model.isQwen { model.restart() }
          }))
      }
    } header: {
      Text("Engine")
    } footer: {
      Text(engineFooter)
    }
  }

  private var qwenStatus: String {
    switch model.qwen.status {
    case .unavailable(let reason): return reason
    case .idle: return model.isQwen ? "Not loaded" : "Not loaded; loads when chosen"
    case .loading: return "Loading…"
    case .ready: return model.qwen.loadSeconds.map { String(format: "Ready, loaded in %.1f s", $0) } ?? "Ready"
    case .failed(let reason): return "Failed: \(reason)"
    }
  }

  private var engineFooter: String {
    if case .unavailable(let reason) = model.qwen.status {
      return "\(reason) \(EngineKind.tiny.title) runs everything here."
    }
    return "\(EngineKind.qwen.title) is \(QwenRuntime.name), \(qwenShippedSize(params: model.qwen.parameterCount)) at 4 bits, on this iPhone's GPU. \(EngineKind.tiny.title) is the small model trained from Cipher's lines; it keeps on-device training, the five fine-tunes and the checkpoints. The chat template wraps your prompt as a user message so Qwen answers it; off, Qwen continues the raw text."
  }
}
