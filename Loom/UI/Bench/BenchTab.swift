import LoomEngine
import SwiftUI

/// Change the model itself: five LoRA fine-tunes, language tags, steering, training checkpoints, and a fine-tune of your own.
/// Under Qwen only steering applies; the rest belongs to the 1.4M model.
struct BenchTab: View {
  @Bindable var model: LoomModel

  var body: some View {
    ScrollView {
      VStack(spacing: 14) {
        if model.isQwen {
          SteerPanel(model: model, number: "1")
          FineTuneEngineNotice(model: model, number: "2")
        } else {
          VoicesPanel(model: model)
          CartridgePanel(model: model)
          SteerPanel(model: model)
          AgePanel(model: model)
          TeachPanel(model: model)
        }
      }
      .padding(.horizontal, 14)
      .padding(.bottom, 24)
    }
    .scrollDismissesKeyboard(.interactively)
  }
}

/// Under Qwen: why the fine-tunes, checkpoints, language tags and Teach aren't here, with the real
/// memory arithmetic, and the way back to the model they run on.
struct FineTuneEngineNotice: View {
  @Bindable var model: LoomModel
  var number = "2"

  var body: some View {
    let tinyParams = model.library?.base.parameterCount ?? 1_377_408
    let qwenParams = model.qwen.parameterCount
    let phone = Double(ProcessInfo.processInfo.physicalMemory)
    InstrumentPanel(number: number, title: "Fine-tunes · Loom 1.4M only",
      nameplate: "The five fine-tunes, the training checkpoints, the language tags and training on your own text belong to Loom 1.4M. Qwen ships as one finished model rounded to 4 bits, and it is too big to fine-tune on a phone: Loom's trainer works in 32-bit numbers and keeps four of them per weight in memory (the weight, its gradient and AdamW's two running averages), 16 bytes a weight.",
      tint: Theme.violet) {
        Text("MEMORY TO TRAIN EVERY WEIGHT").eyebrow()
        HStack(alignment: .top, spacing: 16) {
          Readout(title: EngineKind.tiny.title, value: Self.bytes(Double(tinyParams) * 16), color: Theme.green)
          Readout(title: EngineKind.qwen.title, value: Self.bytes(Double(qwenParams) * 16), color: Theme.orange)
          Spacer(minLength: 0)
          Readout(title: "This iPhone has", value: Self.bytes(phone), color: Theme.cyan)
        }
        Text("\(tinyParams.formatted()) weights against \(qwenParams.formatted()). Even LoRA, which trains only two small matrices per weight, runs on top of the whole model held in 32-bit numbers: \(Self.bytes(Double(qwenParams) * 4)) for Qwen's weights alone.")
          .font(.caption)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
        if model.teaching {
          Label("Your fine-tune is still training on Loom 1.4M.", systemImage: "chart.line.downtrend.xyaxis")
            .font(.caption)
            .foregroundStyle(Theme.yellow)
        } else if model.taught != nil || model.fullTuned != nil {
          Label("Your kept fine-tune waits on Loom 1.4M.", systemImage: "checkmark.circle")
            .font(.caption)
            .foregroundStyle(Theme.green)
        }
        Button {
          model.engine = .tiny
        } label: {
          Label("Switch to \(EngineKind.tiny.title)", systemImage: "arrow.left.arrow.right")
            .font(.subheadline.weight(.semibold))
            .frame(minHeight: 36)
            .padding(.horizontal, 6)
        }
        .buttonStyle(.glassProminent)
        .tint(Theme.violet.opacity(0.6))
        .accessibilityIdentifier("switchToTiny")
      }
  }

  /// Bytes in decimal units, as storage is sold: "22 MB", "64.4 GB".
  static func bytes(_ n: Double) -> String {
    if n >= 1e9 { return String(format: "%.1f GB", n / 1e9) }
    if n >= 1e6 { return String(format: "%.0f MB", n / 1e6) }
    return String(format: "%.0f kB", n / 1e3)
  }
}

/// A knob bound to the model, for use outside the console.
struct BenchKnob: View {
  @Bindable var model: LoomModel
  let spec: KnobSpec
  var body: some View {
    GlassKnob(
      spec: spec,
      value: Binding(get: { model[keyPath: spec.keyPath] }, set: { model[keyPath: spec.keyPath] = $0 }),
      haptics: model.haptics, onActive: { _ in })
  }
}

/// The current prompt woven twice with the same seed: once by the plain base, once by the loom as set.
/// Under Qwen: once with no knobs, once with the pass knobs as set (steering included); each takes a few
/// seconds on the phone's GPU.
struct SideBySide: View {
  @Bindable var model: LoomModel
  let key: String
  @State private var base = ""
  @State private var tuned = ""
  @State private var busy = false
  /// Qwen only: what the rows on screen were generated from, so an unchanged row isn't generated again.
  @State private var shownBase: QwenPreviewInputs?
  @State private var shownTuned: QwenPreviewInputs?

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text("SAME PROMPT, SAME SEED").eyebrow()
        Spacer()
        if busy {
          if model.isQwen { Text("GENERATING").eyebrow() }
          ProgressView().controlSize(.mini)
        }
      }
      if model.isQwen {
        if model.qwen.isReady {
          if model.qwen.chat {
            row("YOU", model.prompt, Theme.cyan, whole: true)
          }
          row("BASE", base, Theme.muted)
          row(tunedLabel, tuned, Theme.green)
        } else {
          Text(Self.waiting(model.qwen.status))
            .font(.caption)
            .foregroundStyle(Theme.muted)
            .fixedSize(horizontal: false, vertical: true)
        }
      } else {
        row("BASE", base, Theme.muted)
        row("TUNED", tuned, Theme.green)
      }
    }
    .padding(12)
    .background(Theme.well, in: .rect(cornerRadius: 14))
    .task(id: taskID) {
      if model.isQwen {
        await runQwenPreviews()
        return
      }
      try? await Task.sleep(for: .milliseconds(250))
      guard !Task.isCancelled, let lib = model.library, let w = model.weights else { return }
      busy = true
      base = await model.preview(weights: lib.base)
      tuned = await model.preview(weights: w, knobs: model.passSettings.knobs)
      busy = false
    }
  }

  /// Restarts the comparison whenever something it depends on changes.
  private var taskID: PreviewTaskID {
    if model.isQwen {
      return PreviewTaskID(tiny: key, qwen: qwenInputs(knobs: model.passSettings.knobs), qwenReady: model.qwen.isReady)
    }
    return PreviewTaskID(tiny: "\(key)|\(model.prompt)|\(Int(model.seed))|\(model.weights?.wte.count ?? 0)|\(signature)")
  }

  /// Changes whenever a knob that the comparison depends on moves.
  private var signature: String {
    [model.scripture, model.bard, model.gothic, model.twain, model.almanac, model.steerStrength, model.steerFloor,
     model.bitsIndex, model.ageIndex, model.taughtStrength, model.temperature, model.topP]
      .map { String(format: "%.2f", $0) }.joined(separator: ",") + model.steerKey + "\(model.cartridge ?? -1)\(model.taught != nil)\(model.fullTuned != nil)"
  }

  /// Qwen's second row: "STEERED", or "AS SET" when heads, layers or neurons from the Model page also
  /// changed the pass it was generated with.
  private var tunedLabel: String {
    let k = shownTuned?.knobs ?? model.passSettings.knobs
    return k.muted.isEmpty && k.bypass.isEmpty && k.depth == nil && k.clamp.isEmpty ? "STEERED" : "AS SET"
  }

  private func qwenInputs(knobs: RunKnobs) -> QwenPreviewInputs {
    QwenPreviewInputs(prompt: model.prompt, seed: Int(model.seed), chat: model.qwen.chat, sampler: model.sampler, knobs: knobs)
  }

  /// Qwen's two generations, after the knobs have been still for 600 ms (each takes seconds, so a turning
  /// knob mustn't queue one per tick). The base row is only regenerated when its own inputs change.
  private func runQwenPreviews() async {
    try? await Task.sleep(for: .milliseconds(600))
    guard !Task.isCancelled else { return }
    guard model.isQwen, model.qwen.isReady else {
      busy = false
      return
    }
    let knobs = model.passSettings.knobs
    let baseInputs = qwenInputs(knobs: RunKnobs())
    let tunedInputs = qwenInputs(knobs: knobs)
    guard baseInputs != shownBase || tunedInputs != shownTuned else {
      busy = false
      return
    }
    busy = true
    if baseInputs != shownBase {
      let text = await model.previewQwen()
      // A newer task may have taken over, but if the inputs still match, the text is still right: keep it.
      if model.isQwen, baseInputs == qwenInputs(knobs: RunKnobs()) {
        base = text
        shownBase = text.isEmpty ? nil : baseInputs
      }
      guard !Task.isCancelled else { return }
    }
    if tunedInputs != shownTuned {
      let text: String
      if knobs == RunKnobs() {
        // Nothing set: the same generation as the base row.
        text = base
      } else {
        text = await model.previewQwen(knobs: knobs)
      }
      if model.isQwen, tunedInputs == qwenInputs(knobs: model.passSettings.knobs) {
        tuned = text
        shownTuned = text.isEmpty ? nil : tunedInputs
      }
      guard !Task.isCancelled else { return }
    }
    busy = false
  }

  static func waiting(_ status: QwenRuntime.Status) -> String {
    switch status {
    case .loading: "Qwen is loading onto the GPU. The comparison runs once it's ready."
    case .idle: "Qwen isn't loaded yet."
    case .ready: ""
    case .failed(let reason): "Qwen failed to load: \(reason)"
    case .unavailable(let reason): reason
    }
  }

  /// The prompt and the generated text as one passage. The 1.4M model continues the prompt; Qwen with
  /// the chat template answers it (the prompt is shown once, above), and raw Qwen continues it.
  private func passage(_ text: String) -> String {
    guard model.isQwen else { return model.prompt + text }
    if model.qwen.chat || text.isEmpty { return text }
    // Qwen's preview comes back trimmed, so put back the space a continuation almost always starts with.
    let joined = model.prompt.last?.isWhitespace == true || text.first.map { ",.;:!?'’)…".contains($0) } == true
    return model.prompt + (joined ? "" : " ") + text
  }

  private func row(_ label: String, _ text: String, _ color: Color, whole: Bool = false) -> some View {
    HStack(alignment: .top, spacing: 10) {
      Text(label).font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundStyle(color)
        .frame(width: model.isQwen ? 54 : 46, alignment: .leading)
      Text(whole ? text : passage(text))
        .font(.system(.callout, design: .serif))
        .foregroundStyle(.white.opacity(0.9))
        .fixedSize(horizontal: false, vertical: true)
        .contentTransition(.opacity)
    }
  }
}

/// Everything one Qwen preview depends on.
nonisolated struct QwenPreviewInputs: Equatable, Sendable {
  var prompt: String
  var seed: Int
  var chat: Bool
  var sampler: SamplerSettings
  var knobs: RunKnobs
}

/// The side-by-side's task identity: the 1.4M model's signature string, or Qwen's inputs.
nonisolated struct PreviewTaskID: Equatable, Sendable {
  var tiny: String
  var qwen: QwenPreviewInputs?
  var qwenReady = false
}

// MARK: - Voices

struct VoicesPanel: View {
  @Bindable var model: LoomModel

  var body: some View {
    InstrumentPanel(number: "1", title: "Fine-tunes · LoRA", badge: .baked,
      nameplate: "Five fine-tunes, each trained on text the base model never saw. Only two small matrices per weight were trained (rank 8: about 82,000 numbers, 6% of the model); the base stays untouched and the knobs mix them in. Negative pushes away.", tint: Theme.violet) {
      Formula("W′ = W + Σ λᵢ · (α/r) · Aᵢ Bᵢ", color: Theme.violet)
      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow { ForEach(KnobSpec.style.prefix(3)) { BenchKnob(model: model, spec: $0) } }
        GridRow {
          ForEach(KnobSpec.style[3..<5]) { BenchKnob(model: model, spec: $0) }
          taughtKnob
        }
      }
      if let lib = model.library {
        VStack(alignment: .leading, spacing: 6) {
          ForEach(Array(lib.adapters.enumerated()), id: \.offset) { i, a in
            let v = Voice.allCases[i]
            let before = a.meta["family_val_before"]?.number ?? 0
            let after = a.meta["family_val_after"]?.number ?? 0
            HStack(alignment: .firstTextBaseline, spacing: 8) {
              Circle().fill(Theme.voice(v)).frame(width: 7, height: 7)
              VStack(alignment: .leading, spacing: 1) {
                Text(v.title).font(.subheadline.weight(.semibold))
                Text("Trained on \(v.books).").font(.caption).foregroundStyle(.secondary)
              }
              Spacer(minLength: 4)
              VStack(alignment: .trailing, spacing: 1) {
                Text(String(format: "%.2f → %.2f", before, after)).font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.yellow)
                Text("loss on unseen lines").font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
              }
            }
          }
        }
      }
      SideBySide(model: model, key: "voices")
    }
  }

  @ViewBuilder private var taughtKnob: some View {
    if model.taught != nil {
      BenchKnob(model: model, spec: KnobSpec(
        id: .taught, title: "Yours", glyph: "✎", tint: Theme.green, range: -1...2, defaultValue: 1, origin: 0,
        detents: [KnobDetent(value: 0, label: "Off"), KnobDetent(value: 1, label: "As trained")], step: 0.01, badge: .train,
        hint: "The fine-tune you trained below, on your own lines.", detail: "", keyPath: \.taughtStrength,
        format: { abs($0) < 0.005 ? "Off" : String(format: "%+.2f", $0) }))
    } else {
      VStack(spacing: 6) {
        Image(systemName: "pencil.and.scribble").font(.title3).foregroundStyle(Theme.muted).frame(width: 84, height: 84)
        Text("YOURS").font(.caption2.weight(.semibold)).tracking(0.6).foregroundStyle(.white.opacity(0.62))
        Text("Train below").font(.caption).foregroundStyle(Theme.muted)
      }
      .frame(maxWidth: .infinity)
    }
  }
}

// MARK: - Cartridges

struct CartridgePanel: View {
  @Bindable var model: LoomModel

  var body: some View {
    InstrumentPanel(number: "2", title: "Language tags · control tokens", badge: .baked,
      nameplate: "Half of the training lines began with a tag for their language. A tag at the start of the context works like a system prompt, because the model was trained to follow it. ⟨card⟩ was only ever trained by the Almanac fine-tune.", tint: Theme.orange) {
      FlowLayout(spacing: 8) {
        chip(nil, "none")
        ForEach(Array(Tokenizer.controls.enumerated()), id: \.offset) { i, name in chip(2 + i, name) }
      }
      SideBySide(model: model, key: "cartridge")
    }
  }

  private func chip(_ id: Int?, _ name: String) -> some View {
    let on = model.cartridge == id
    return Button {
      model.cartridge = id
      model.haptics.transient(intensity: 0.5, sharpness: 0.6)
    } label: {
      Text(name)
        .font(.system(.subheadline, design: .monospaced).weight(.semibold))
        .padding(.horizontal, 12)
        .frame(minHeight: 36)
    }
    .buttonStyle(.glass(on ? .regular.tint(Theme.orange.opacity(0.35)) : .regular))
    .accessibilityAddTraits(on ? .isSelected : [])
  }
}

// MARK: - Steering

struct SteerPanel: View {
  @Bindable var model: LoomModel
  var number = "3"

  var body: some View {
    InstrumentPanel(number: number, title: "Steering · a measured direction", badge: .live, nameplate: nameplate, tint: Theme.cyan) {
      Formula("x ← x + α · v", color: Theme.cyan)
      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow {
          // The engine's own specs: Qwen's floor knob spans its 36 layers.
          if let steer = KnobSpec.spec(.steer, for: model) { BenchKnob(model: model, spec: steer) }
          if let floor = KnobSpec.spec(.steerFloor, for: model) { BenchKnob(model: model, spec: floor) }
          DirectionMenu(model: model)
        }
      }
      if let steering = model.steering, !steering.residualNorm.isEmpty {
        let floor = max(0, min(Int(model.steerFloor), model.config.layers - 1, steering.residualNorm.count - 1))
        let norm = steering.residualNorm[floor]
        let v = steering.vector(model.steerKey, floor: floor)
        let len = sqrt(v.reduce(0) { $0 + Double($1 * $1) })
        Text(String(format: "This direction is %.2f long; a vector after this layer is about %.1f long. At %+.1f it shifts each one by %.0f%% of its length.",
          len, norm, model.steerStrength, abs(model.steerStrength) * len / max(norm, 1e-6) * 100))
          .font(.caption)
          .foregroundStyle(.secondary)
      } else if model.isQwen {
        // Reading the status here redraws the panel when Qwen (and its directions) finish loading.
        Text(model.qwen.isReady ? "This build has no steering directions for Qwen (qwen-steering.loom)." : "Qwen's directions load with its weights.")
          .font(.caption)
          .foregroundStyle(Theme.muted)
      }
      SideBySide(model: model, key: "steer")
    }
  }

  private var nameplate: String {
    guard model.isQwen else {
      return "Nothing retrained: after one layer, every position gets a direction added to its vector. Each direction was measured from the training text as the average vector of one group of lines minus the others (Spanish minus the other languages, say)."
    }
    return "Nothing retrained: after one of Qwen's \(model.config.layers) layers, every position gets a direction added to its vector. Each direction was measured on a Mac from Qwen's own vectors after every layer, over 200 of Cipher's lines per group: one language minus the other four, or one mood minus 400 random lines."
  }
}

// MARK: - Age

struct AgePanel: View {
  @Bindable var model: LoomModel

  var body: some View {
    InstrumentPanel(number: "4", title: "Checkpoints · optimization over time", badge: .baked,
      nameplate: "The same model saved at four moments of its training. The curve is the real log of gradient descent: loss in nats per token (orange on training text, green on unseen text), lower is better.", tint: Theme.yellow) {
      if let lib = model.library {
        LossCurve(curve: lib.base.meta["curve"]?.array ?? [], marks: lib.ages.map(\.step), selected: lib.ages[min(Int(model.ageIndex), lib.ages.count - 1)].step)
        Picker("Age", selection: $model.ageIndex) {
          ForEach(Array(lib.ages.enumerated()), id: \.offset) { i, a in
            Text(i == lib.ages.count - 1 ? "Finished" : "Step \(a.step.formatted())").tag(Double(i))
          }
        }
        .pickerStyle(.segmented)
        let age = lib.ages[min(Int(model.ageIndex), lib.ages.count - 1)]
        Text(String(format: "Step %@ · loss on unseen text %.2f nats per token", age.step.formatted(), age.valLoss))
          .font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.yellow)
      }
      SideBySide(model: model, key: "age")
    }
  }
}

/// The base model's training curve from its log: training text (orange) and unseen text (green).
struct LossCurve: View {
  let curve: [JSONValue]
  var marks: [Int] = []
  var selected: Int?

  var body: some View {
    let pts = curve.compactMap { $0.array }.compactMap { a -> (Double, Double, Double)? in
      guard a.count >= 3, let s = a[0].number, let t = a[1].number, let v = a[2].number else { return nil }
      return (s, t, v)
    }
    Canvas { ctx, size in
      guard let last = pts.last else { return }
      let maxS = last.0
      let lo = 3.0, hi = 6.0
      func p(_ s: Double, _ l: Double) -> CGPoint {
        CGPoint(x: size.width * s / maxS, y: size.height * (1 - (min(hi, l) - lo) / (hi - lo)))
      }
      for (k, color) in [(1, Theme.amber), (2, Theme.phosphor)] {
        var path = Path()
        for (i, q) in pts.enumerated() {
          let v = k == 1 ? q.1 : q.2
          if i == 0 { path.move(to: p(q.0, v)) } else { path.addLine(to: p(q.0, v)) }
        }
        ctx.stroke(path, with: .color(color), lineWidth: 1.5)
      }
      for m in marks {
        var tick = Path()
        let x = size.width * Double(m) / maxS
        tick.move(to: CGPoint(x: x, y: 0))
        tick.addLine(to: CGPoint(x: x, y: size.height))
        ctx.stroke(tick, with: .color(.white.opacity(m == selected ? 0.8 : 0.2)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
      }
    }
    .frame(height: 90)
    .accessibilityElement()
    .accessibilityLabel("Training loss curve")
    .accessibilityValue(pts.last.map { String(format: "ends at %.2f", $0.2) } ?? "")
  }
}
