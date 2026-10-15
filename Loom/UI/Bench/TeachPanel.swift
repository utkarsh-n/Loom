import LoomEngine
import SwiftUI

/// A real fine-tune on this phone, on lines you give it. It trains Loom 1.4M only: under Qwen it shows why,
/// and the way back.
struct TeachPanel: View {
  @Bindable var model: LoomModel
  @FocusState private var editing: Bool

  var body: some View {
    if model.isQwen {
      FineTuneEngineNotice(model: model, number: "5")
    } else {
      teach
    }
  }

  static let example = """
    the cipher spins and the city hums below
    I write my name in the steam on the window
    cold coffee, warm verse, the night is a slow flow
    every bar a brick and the wall keeps on growing
    the cipher spins and the bass hits the floor
    I came with a pen and I left with the door
    the moon is a mic and the street is a chorus
    we rhyme till the sun comes and rhyme a bit more
    the cipher spins and the words come round again
    a notebook of thunder, a pocket of rain
    """

  private var teach: some View {
    InstrumentPanel(number: "5", title: "Fine-tune on your text · gradient descent", badge: .train,
      nameplate: "Give it a few lines, one per line; every fifth is held back to test on. Each step runs the model forward, measures how wrong it was (the loss), works backward to find each weight's slope (backpropagation), and moves every weight a little downhill (AdamW). When the held-back loss turns upward while the training loss keeps falling, it has started memorising.", tint: Theme.yellow) {
      Formula("w ← w − η · ∂loss/∂w", color: Theme.yellow)
      ZStack(alignment: .topLeading) {
        TextEditor(text: $model.teachText)
          .font(.system(.callout, design: .monospaced))
          .scrollContentBackground(.hidden)
          .focused($editing)
          .frame(minHeight: 130)
          .padding(8)
          .background(Theme.well, in: .rect(cornerRadius: 14))
          .accessibilityIdentifier("teachText")
        if model.teachText.isEmpty {
          Text("Your bars, a poem, anything. One line each.")
            .font(.system(.callout, design: .monospaced))
            .foregroundStyle(.tertiary)
            .padding(14)
            .allowsHitTesting(false)
        }
      }
      HStack {
        Text("\(model.teachSequences.count) lines").font(.system(.caption, design: .monospaced)).foregroundStyle(Theme.muted)
        Spacer()
        Button("Use example bars") { model.teachText = Self.example }
          .font(.caption)
          .buttonStyle(.glass)
      }

      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow {
          ForEach(Self.knobs.prefix(4)) { BenchKnob(model: model, spec: $0) }
        }
      }
      .disabled(model.teaching)
      .opacity(model.teaching ? 0.6 : 1)

      Picker("What trains", selection: $model.teachFull) {
        Text("LoRA (6%)").tag(0.0)
        Text("Every weight").tag(1.0)
      }
      .pickerStyle(.segmented)
      .disabled(model.teaching)

      HStack(spacing: 10) {
        Button {
          editing = false
          model.teaching ? model.stopTeaching() : model.startTeaching()
        } label: {
          Label(model.teaching ? "Stop" : "Train", systemImage: model.teaching ? "stop.fill" : "chart.line.downtrend.xyaxis")
            .font(.subheadline.weight(.semibold))
            .frame(minHeight: 36)
            .padding(.horizontal, 6)
        }
        .buttonStyle(.glassProminent)
        .tint(Theme.yellow.opacity(0.75))
        .disabled(!model.teaching && model.teachSequences.count < 2)
        .accessibilityIdentifier("teach")
        if !model.teaching, !model.teachProgress.isEmpty {
          Button("Keep it", systemImage: "checkmark") { model.keepTaught() }
            .buttonStyle(.glass)
            .font(.subheadline)
        }
        if model.taught != nil || model.fullTuned != nil {
          Button("Put away", systemImage: "trash") { model.discardTaught() }
            .buttonStyle(.glass)
            .font(.subheadline)
        }
      }

      if !model.teachProgress.isEmpty {
        TeachChart(progress: model.teachProgress, before: model.teachBefore?.train)
        TeachReadouts(model: model)
      }
      if model.fullTuned != nil {
        Text("The fully fine-tuned model is in use on every tab. Put it away to go back to the base.")
          .font(.caption).foregroundStyle(Theme.orange)
      }
    }
  }

  static let knobs: [KnobSpec] = [
    KnobSpec(
      id: .teachRate, title: "Learning rate", glyph: "η", tint: Theme.flame, range: 0.0001...0.1, defaultValue: 0.003, logarithmic: true,
      detents: [KnobDetent(value: 0.001, label: "Gentle"), KnobDetent(value: 0.003, label: "Lively"), KnobDetent(value: 0.03, label: "Red line")],
      badge: .train,
      hint: "The step size downhill. Too high and the loss spikes and the model forgets; too low and nothing moves.",
      detail: "", keyPath: \.teachRate, format: { String(format: "%.4f", $0) }),
    KnobSpec(
      id: .teachRank, title: "Rank", glyph: "r", tint: Theme.negative, range: 1...16, defaultValue: 8,
      step: 1, badge: .train,
      hint: "Width of the LoRA matrices: how much the fine-tune can change. Ignored when every weight trains.",
      detail: "", keyPath: \.teachRank, format: { "\(Int($0))" }),
    KnobSpec(
      id: .teachSteps, title: "Steps", glyph: "N", tint: Color(red: 0.35, green: 0.94, blue: 0.69), range: 10...300, defaultValue: 60,
      step: 5, badge: .train,
      hint: "How many steps downhill. Each is a forward pass, a backward pass and a weight update.",
      detail: "", keyPath: \.teachSteps, format: { "\(Int($0))" }),
    KnobSpec(
      id: .teachBatch, title: "Batch", glyph: "B", tint: Theme.violet, range: 1...16, defaultValue: 4,
      step: 1, badge: .train,
      hint: "How many of your lines are averaged per step. Bigger batches make a smoother, slower curve.",
      detail: "", keyPath: \.teachBatch, format: { "\(Int($0))" }),
  ]
}

/// Training loss every step (amber) and held-back loss every fifth (phosphor).
struct TeachChart: View {
  let progress: [TeachProgress]
  let before: Double?

  var body: some View {
    let maxStep = Double(max(progress.last?.step ?? 1, 10))
    let values = progress.map(\.trainLoss) + progress.compactMap(\.heldOutLoss) + (before.map { [$0] } ?? [])
    let hi = max(values.max() ?? 6, 1), lo = 0.0
    VStack(alignment: .leading, spacing: 4) {
      Canvas { ctx, size in
        func p(_ s: Double, _ l: Double) -> CGPoint {
          CGPoint(x: size.width * s / maxStep, y: size.height * (1 - (l - lo) / (hi - lo)))
        }
        var train = Path(), held = Path()
        var startedHeld = false
        for (i, q) in progress.enumerated() {
          if i == 0 { train.move(to: p(Double(q.step), q.trainLoss)) } else { train.addLine(to: p(Double(q.step), q.trainLoss)) }
          if let h = q.heldOutLoss {
            if !startedHeld { held.move(to: p(Double(q.step), h)); startedHeld = true } else { held.addLine(to: p(Double(q.step), h)) }
            ctx.fill(Path(ellipseIn: CGRect(origin: p(Double(q.step), h), size: .zero).insetBy(dx: -2.5, dy: -2.5)), with: .color(Theme.phosphor))
          }
        }
        ctx.stroke(train, with: .color(Theme.amber), lineWidth: 1.5)
        ctx.stroke(held, with: .color(Theme.phosphor), lineWidth: 1.5)
        // The lowest held-back point: where training should have stopped.
        if let best = progress.filter({ $0.heldOutLoss != nil }).min(by: { $0.heldOutLoss! < $1.heldOutLoss! }) {
          var tick = Path()
          let x = p(Double(best.step), 0).x
          tick.move(to: CGPoint(x: x, y: 0))
          tick.addLine(to: CGPoint(x: x, y: size.height))
          ctx.stroke(tick, with: .color(.white.opacity(0.35)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
        }
      }
      .frame(height: 120)
      .background(Theme.well, in: .rect(cornerRadius: 12))
      HStack(spacing: 12) {
        Label("training lines", systemImage: "circle.fill").foregroundStyle(Theme.orange)
        Label("held back", systemImage: "circle.fill").foregroundStyle(Theme.phosphor)
        Spacer()
        Text("nats per token").foregroundStyle(Theme.muted)
      }
      .font(.system(size: 10, design: .monospaced))
      .labelStyle(TightLabel())
    }
    .accessibilityElement()
    .accessibilityLabel("Teach loss curves")
    .accessibilityValue(progress.last.map { String(format: "step %d, training loss %.2f", $0.step, $0.trainLoss) } ?? "")
  }
}

struct TightLabel: LabelStyle {
  func makeBody(configuration: Configuration) -> some View {
    HStack(spacing: 4) {
      configuration.icon.font(.system(size: 6))
      configuration.title
    }
  }
}

struct TeachReadouts: View {
  @Bindable var model: LoomModel

  var body: some View {
    let last = model.teachProgress.last!
    let held = model.teachProgress.compactMap(\.heldOutLoss)
    let bestHeld = held.min()
    let memorising = (held.last ?? 0) > (bestHeld ?? 0) + 0.15 && last.trainLoss < (held.last ?? 0) - 0.5
    VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 16) {
        Readout(title: "Step", value: "\(last.step)")
        Readout(title: "Train", value: String(format: "%.2f", last.trainLoss))
        Readout(title: "Held back", value: held.last.map { String(format: "%.2f", $0) } ?? "–", color: Theme.phosphor)
        Readout(title: "Epochs", value: String(format: "%.1f", last.epochs), color: .white)
      }
      Text("\(last.trainedParameters.formatted()) numbers training · \(model.teachLines.train) lines in, \(model.teachLines.heldOut) held back · \(String(format: "%.1f", last.seconds)) s on this phone")
        .font(.system(.caption, design: .monospaced))
        .foregroundStyle(Theme.muted)
      if memorising {
        Label("Overfitting: it fits your lines better while getting worse on the ones it hasn't seen.", systemImage: "exclamationmark.triangle.fill")
          .font(.footnote)
          .foregroundStyle(Theme.orange)
      }
    }
  }
}
