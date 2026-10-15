import LoomEngine
import SwiftUI

/// The bottom console (Gyre's KnobConsole): pages of glass knobs, the weave controls, and a fold
/// button that melts the knobs away so the machine can fill the screen.
struct KnobConsole: View {
  @Bindable var model: LoomModel
  @State private var page: ConsolePage = .decode
  @Namespace private var glass

  var body: some View {
    GlassEffectContainer(spacing: 18) {
      VStack(spacing: 12) {
        if model.consoleOpen {
          VStack(spacing: 10) {
            Picker("Knobs", selection: $page.animation(.spring(duration: 0.35, bounce: 0.2))) {
              ForEach(ConsolePage.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.tabs)
            .frame(maxWidth: 320)
            .accessibilityIdentifier("consolePages")
            switch page {
            case .decode:
              Grid(horizontalSpacing: 2, verticalSpacing: 10) {
                GridRow { ForEach(KnobSpec.decode(for: model).prefix(4)) { knob($0) } }
                GridRow { ForEach(KnobSpec.decode(for: model).suffix(4)) { knob($0) } }
              }
              .transition(.blurReplace)
            case .model:
              modelPage.transition(.blurReplace)
            case .style:
              stylePage.transition(.blurReplace)
            }
          }
          .transition(.blurReplace.combined(with: .scale(0.9, anchor: .bottom)))
        }
        controls
      }
    }
    .overlay(alignment: .top) {
      if let id = model.activeKnob, let spec = KnobSpec.spec(id, for: model) {
        KnobReadout(spec: spec, value: model[keyPath: spec.keyPath])
          .offset(y: -86)
          .transition(.opacity.combined(with: .scale(scale: 0.92, anchor: .bottom)))
      }
    }
    .animation(.spring(duration: 0.3, bounce: 0.25), value: model.activeKnob)
    .onAppear { if let p = model.consolePage { page = p } }
  }

  private func knob(_ spec: KnobSpec) -> some View {
    GlassKnob(
      spec: spec,
      value: Binding(get: { model[keyPath: spec.keyPath] }, set: { model[keyPath: spec.keyPath] = $0 }),
      haptics: model.haptics,
      onActive: { active in
        if active {
          model.activeKnob = spec.id
        } else if model.activeKnob == spec.id {
          model.activeKnob = nil
        }
      })
  }

  // MARK: Model page

  @ViewBuilder private var modelPage: some View {
    if model.isQwen {
      // Qwen: context, layers used and the cache; precision is fixed at the 4 bits it ships with.
      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow {
          ForEach(KnobSpec.model(for: model)) { knob($0) }
          FixedTile(title: "PRECISION", value: "4 bit", note: "as shipped")
        }
        GridRow {
          QwenHeadBreakers(model: model)
            .gridCellColumns(2)
            .padding(.horizontal, 4)
          FloorSwitches(model: model)
            .gridCellColumns(2)
            .padding(.horizontal, 4)
        }
      }
    } else {
      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow { ForEach(KnobSpec.model.prefix(4)) { knob($0) } }
        GridRow {
          knob(KnobSpec.model[4])
          HeadBreakers(model: model)
            .gridCellColumns(2)
            .padding(.horizontal, 4)
          FloorSwitches(model: model)
            .padding(.horizontal, 4)
        }
      }
    }
  }

  // MARK: Style page

  @ViewBuilder private var stylePage: some View {
    if model.isQwen {
      // The five fine-tunes belong to the 1.4M model; steering works on Qwen's own measured directions.
      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow {
          ForEach(KnobSpec.style(for: model)) { knob($0) }
          DirectionMenu(model: model)
          FixedTile(title: "FINE-TUNES", value: "Loom 1.4M", note: "switch engines")
        }
      }
    } else {
      Grid(horizontalSpacing: 2, verticalSpacing: 10) {
        GridRow { ForEach(KnobSpec.style.prefix(4)) { knob($0) } }
        GridRow {
          knob(KnobSpec.style[4])
          knob(KnobSpec.style[5])
          knob(KnobSpec.style[6])
          DirectionMenu(model: model)
        }
      }
    }
  }

  // MARK: Controls

  private var controls: some View {
    HStack(spacing: 10) {
      Button {
        model.playing ? model.stop() : model.play()
      } label: {
        Label(model.playing ? "Stop" : "Generate", systemImage: model.playing ? "stop.fill" : "play.fill")
          .font(.subheadline.weight(.semibold))
          .lineLimit(1)
          .fixedSize()
          .padding(.horizontal, 4)
          .frame(minHeight: 36)
          .contentTransition(.symbolEffect(.replace))
      }
      .buttonStyle(.glassProminent)
      .tint(Theme.green.opacity(0.7))
      .glassEffectID("weave", in: glass)
      .accessibilityIdentifier("weave")
      .accessibilityHint("Generates tokens one after another until the limit or a line break")

      Button {
        model.step()
      } label: {
        Image(systemName: "forward.frame.fill").frame(width: 36, height: 36)
      }
      .buttonStyle(.glass)
      .buttonBorderShape(.circle)
      .glassEffectID("step", in: glass)
      .disabled(model.playing || !model.canWeave)
      .accessibilityLabel("Generate one token")
      .accessibilityIdentifier("step")

      Button {
        model.restart()
        model.haptics.transient(intensity: 0.5, sharpness: 0.6)
      } label: {
        Image(systemName: "eraser.line.dashed").frame(width: 36, height: 36)
      }
      .buttonStyle(.glass)
      .buttonBorderShape(.circle)
      .glassEffectID("clear", in: glass)
      .accessibilityLabel("Clear the generated text")

      Spacer(minLength: 0)

      if model.consoleOpen {
        Button {
          model.resetKnobs(on: page)
        } label: {
          Image(systemName: "arrow.counterclockwise").frame(width: 36, height: 36)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .glassEffectID("reset", in: glass)
        .accessibilityLabel("Reset \(page.rawValue.lowercased()) knobs")
      }

      Button {
        withAnimation(.spring(duration: 0.45, bounce: 0.2)) { model.consoleOpen.toggle() }
        model.haptics.transient(intensity: 0.4, sharpness: 0.5)
      } label: {
        Group {
          if model.consoleOpen {
            Image(systemName: "chevron.down").frame(width: 36, height: 36)
          } else {
            Label("Knobs", systemImage: "dial.medium.fill")
              .font(.subheadline.weight(.semibold))
              .lineLimit(1)
              .fixedSize()
              .frame(minHeight: 36)
          }
        }
      }
      .buttonStyle(.glass)
      .buttonBorderShape(model.consoleOpen ? .circle : .capsule)
      .glassEffectID("fold", in: glass)
      .accessibilityLabel(model.consoleOpen ? "Fold the knobs away" : "Knobs")
      .accessibilityIdentifier("knobs")
    }
  }
}

/// Sixteen switches, one per attention head: tap to turn a head off.
struct HeadBreakers: View {
  @Bindable var model: LoomModel

  var body: some View {
    VStack(spacing: 4) {
      Grid(horizontalSpacing: 4, verticalSpacing: 4) {
        ForEach(0..<model.config.layers, id: \.self) { floor in
          GridRow {
            Text("\(floor + 1)").font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
            ForEach(0..<model.config.heads, id: \.self) { head in
              let id = HeadID(layer: floor, head: head)
              let off = model.muted.contains(id)
              Button {
                model.toggleMute(id)
              } label: {
                RoundedRectangle(cornerRadius: 3)
                  .fill(off ? Color.white.opacity(0.08) : Theme.heads[head].opacity(0.85))
                  .overlay(RoundedRectangle(cornerRadius: 3).strokeBorder(Theme.heads[head].opacity(off ? 0.6 : 0), lineWidth: 1))
                  .frame(height: 11)
              }
              .buttonStyle(.plain)
              .accessibilityLabel("Layer \(floor + 1) head \(head + 1)")
              .accessibilityValue(off ? "Off" : "On")
              .accessibilityAddTraits(.isToggle)
            }
          }
        }
      }
      Text("HEADS").font(.caption2.weight(.semibold)).tracking(0.6).foregroundStyle(.white.opacity(0.62))
      Text(model.muted.isEmpty ? "All \(model.config.layers * model.config.heads) on" : "\(model.muted.count) off")
        .font(.caption.weight(.medium).monospacedDigit())
        .contentTransition(.numericText())
    }
  }
}

/// Skip switches, one per layer: the vector passes straight down the residual stream. Qwen's 36 sit in
/// a 6 × 6 grid.
struct FloorSwitches: View {
  @Bindable var model: LoomModel

  var body: some View {
    let layers = model.config.layers
    let perRow = layers > 8 ? 6 : 1
    VStack(spacing: 4) {
      Grid(horizontalSpacing: 3, verticalSpacing: 3) {
        ForEach(0..<((layers + perRow - 1) / perRow), id: \.self) { r in
          GridRow {
            ForEach((r * perRow)..<min(layers, (r + 1) * perRow), id: \.self) { floor in
              let off = model.bypass.contains(floor)
              Button {
                model.toggleBypass(floor)
              } label: {
                Text(perRow > 1 ? "\(floor + 1)" : "L\(floor + 1)")
                  .font(.system(size: perRow > 1 ? 9 : 10, weight: .semibold, design: .monospaced))
                  .frame(maxWidth: .infinity, minHeight: perRow > 1 ? 15 : 13)
                  .foregroundStyle(off ? Theme.muted : .black)
                  .background(off ? Color.white.opacity(0.08) : Theme.violet.opacity(0.85), in: .rect(cornerRadius: 3))
              }
              .buttonStyle(.plain)
              .accessibilityLabel("Layer \(floor + 1)")
              .accessibilityValue(off ? "Skipped" : "Running")
              .accessibilityAddTraits(.isToggle)
            }
          }
        }
      }
      Text("LAYERS").font(.caption2.weight(.semibold)).tracking(0.6).foregroundStyle(.white.opacity(0.62))
      if layers > 8 {
        Text(model.bypass.isEmpty ? "All \(layers) run" : "\(model.bypass.count) skipped")
          .font(.caption.weight(.medium).monospacedDigit())
          .contentTransition(.numericText())
      }
    }
  }
}

/// Qwen's heads, one layer at a time: 32 switches, in groups of four that share a key/value head.
struct QwenHeadBreakers: View {
  @Bindable var model: LoomModel
  @State private var layer = 0

  var body: some View {
    let heads = model.config.heads
    let layers = model.config.layers
    let group = max(1, heads / max(model.qwen.kvHeads, 1))
    VStack(spacing: 4) {
      HStack(spacing: 4) {
        Button { layer = (layer + layers - 1) % layers } label: { Image(systemName: "chevron.left").frame(width: 24, height: 22) }
          .accessibilityLabel("Previous layer")
        Text("LAYER \(layer + 1)")
          .font(.system(size: 10, weight: .bold, design: .monospaced))
          .frame(maxWidth: .infinity)
          .contentTransition(.numericText())
        Button { layer = (layer + 1) % layers } label: { Image(systemName: "chevron.right").frame(width: 24, height: 22) }
          .accessibilityLabel("Next layer")
      }
      .buttonStyle(.plain)
      .foregroundStyle(Theme.violet)
      VStack(spacing: 3) {
        ForEach(0..<(heads / 8), id: \.self) { r in
          HStack(spacing: 2) {
            ForEach((r * 8)..<((r + 1) * 8), id: \.self) { head in
              let id = HeadID(layer: layer, head: head)
              let off = model.muted.contains(id)
              Button {
                model.toggleMute(id)
              } label: {
                RoundedRectangle(cornerRadius: 2)
                  .fill(off ? Color.white.opacity(0.08) : Theme.heads[head % Theme.heads.count].opacity(0.85))
                  .overlay(RoundedRectangle(cornerRadius: 2).strokeBorder(Theme.heads[head % Theme.heads.count].opacity(off ? 0.6 : 0), lineWidth: 1))
                  .frame(height: 12)
              }
              .buttonStyle(.plain)
              .padding(.trailing, head % group == group - 1 && head % 8 != 7 ? 4 : 0)
              .accessibilityLabel("Layer \(layer + 1) head \(head + 1)")
              .accessibilityValue(off ? "Off" : "On")
              .accessibilityAddTraits(.isToggle)
            }
          }
        }
      }
      Text("HEADS").font(.caption2.weight(.semibold)).tracking(0.6).foregroundStyle(.white.opacity(0.62))
      Text(model.muted.isEmpty ? "All \((heads * layers).formatted()) on" : "\(model.muted.count) off")
        .font(.caption.weight(.medium).monospacedDigit())
        .contentTransition(.numericText())
    }
    .accessibilityElement(children: .contain)
  }
}

/// A setting that doesn't turn for this engine, shown where its knob would be.
struct FixedTile: View {
  let title: String
  let value: String
  let note: String

  var body: some View {
    VStack(spacing: 4) {
      Text(value)
        .font(.system(size: 13, weight: .semibold, design: .monospaced))
        .minimumScaleFactor(0.6)
        .lineLimit(1)
        .frame(width: 60, height: 60)
        .overlay(Circle().strokeBorder(.white.opacity(0.18), style: StrokeStyle(lineWidth: 1, dash: [3, 3])))
        .frame(width: 84, height: 84)
      Text(title).font(.caption2.weight(.semibold)).tracking(0.6).foregroundStyle(.white.opacity(0.62))
      Text(note).font(.caption.weight(.medium)).foregroundStyle(Theme.muted).lineLimit(1).minimumScaleFactor(0.7)
    }
    .frame(maxWidth: .infinity)
    .accessibilityElement(children: .combine)
  }
}

/// Which measured direction the Steer knob pushes along.
struct DirectionMenu: View {
  @Bindable var model: LoomModel

  var body: some View {
    let dirs = model.steering?.directions ?? []
    let current = dirs.first { $0.key == model.steerKey }
    Menu {
      Section("Languages") {
        ForEach(dirs.filter(\.isLanguage)) { d in
          Button(d.title) { model.steerKey = d.key }
        }
      }
      Section("Moods") {
        ForEach(dirs.filter { !$0.isLanguage }) { d in
          Button(d.title) { model.steerKey = d.key }
        }
      }
    } label: {
      VStack(spacing: 4) {
        Image(systemName: "location.north.line.fill")
          .font(.system(size: 16, weight: .semibold))
          .frame(width: 60, height: 60)
          .glassEffect(.regular.interactive(), in: .circle)
          .frame(width: 84, height: 84)
        Text("TOWARD").font(.caption2.weight(.semibold)).tracking(0.6).foregroundStyle(.white.opacity(0.62))
        Text(current?.title ?? "—").font(.caption.weight(.medium)).lineLimit(1).minimumScaleFactor(0.7)
      }
      .foregroundStyle(.white)
    }
    .frame(maxWidth: .infinity)
    .accessibilityLabel("Steering direction")
    .accessibilityValue(current?.title ?? "none")
  }
}

/// Floating glass card that names what you're turning and what it really does (Gyre).
struct KnobReadout: View {
  let spec: KnobSpec
  let value: Double

  var body: some View {
    VStack(alignment: .leading, spacing: 4) {
      HStack(spacing: 8) {
        if let g = spec.glyph {
          Text(g).font(.system(.headline, design: .serif).italic()).foregroundStyle(spec.tint)
        } else {
          Image(systemName: spec.symbol).foregroundStyle(spec.tint)
        }
        Text(spec.title).font(.headline)
        Badge(provenance: spec.badge)
        Spacer(minLength: 8)
        if !spec.isSwitch, let d = spec.detent(at: value) {
          Text(d.label).font(.caption.weight(.semibold)).foregroundStyle(spec.tint).transition(.opacity)
        }
        Text(spec.format(value))
          .font(.headline.monospacedDigit())
          .contentTransition(.numericText(value: value))
      }
      Text(spec.hint)
        .font(.footnote)
        .foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
    }
    .padding(.horizontal, 16)
    .padding(.vertical, 12)
    .frame(maxWidth: 420)
    .glassEffect(.regular, in: .rect(cornerRadius: 22))
    .padding(.horizontal, 4)
    .allowsHitTesting(false)
    .animation(.smooth(duration: 0.15), value: value)
  }
}
