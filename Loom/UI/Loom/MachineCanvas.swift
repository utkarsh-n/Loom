import LoomEngine
import SwiftUI

/// The transformer at a glance: one column per token position, running down through the embedding,
/// every layer and the output. Each cell shows that position's vector (the residual stream) after
/// the stage; arcs show where the focused position's attention heads drew from in each layer.
/// Qwen's 36 layers get compact rows.
struct MachineCanvas: View {
  @Bindable var model: LoomModel
  let gutter: CGFloat = 78
  @State private var width: CGFloat = 390
  /// Threads spread out to fill the width when there are few, and scroll when there are many.
  private var column: CGFloat { min(86, max(54, (width - gutter - 8) / CGFloat(max(model.weave.threads.count, 1)))) }
  let top: CGFloat = 18

  private var rowCount: Int { model.config.layers + 2 }
  private var compact: Bool { model.config.layers > 8 }
  private var rowHeight: CGFloat { compact ? 24 : 58 }

  var body: some View {
    let threads = model.weave.threads
    HStack(spacing: 0) {
      labels
        .frame(width: gutter)
      ScrollViewReader { proxy in
        ScrollView(.horizontal, showsIndicators: false) {
          HStack(spacing: 0) {
            canvas(threads)
              .frame(width: max(CGFloat(threads.count) * column + 8, 10))
            Color.clear.frame(width: 1).id("end")
          }
        }
        .defaultScrollAnchor(.trailing)
        .onChange(of: threads.count) { _, _ in
          withAnimation(.smooth) { proxy.scrollTo("end", anchor: .trailing) }
        }
      }
    }
    .frame(height: top + CGFloat(rowCount) * rowHeight + 6)
    .onGeometryChange(for: CGFloat.self) { $0.size.width } action: { width = $0 }
    .background(Theme.well, in: .rect(cornerRadius: 20))
    .overlay { RoundedRectangle(cornerRadius: 20).strokeBorder(.white.opacity(0.06)) }
    .accessibilityElement(children: .contain)
  }

  // MARK: Row labels and the logit-lens windows

  private var labels: some View {
    let focus = model.focusThread.map { model.weave.threads[$0] }
    return VStack(alignment: .leading, spacing: 0) {
      Color.clear.frame(height: top)
      ForEach(0..<rowCount, id: \.self) { row in
        Button {
          model.haptics.transient(intensity: 0.35, sharpness: 0.7)
          if row == 0 { model.inspector = .punch }
          else if row == rowCount - 1 { model.inspector = .readout }
          else { model.inspector = .floor(row - 1) }
        } label: {
          if compact {
            compactLabel(row, focus)
              .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
              .padding(.leading, 8)
              .contentShape(.rect)
          } else {
          VStack(alignment: .leading, spacing: 2) {
            Text(rowTitle(row))
              .font(.system(size: 10, weight: .semibold, design: .monospaced))
              .tracking(1)
              .foregroundStyle(lit(row) ? .white : rowColor(row))
            if row >= 1, row <= model.config.layers, let f = focus {
              let ran = f.ran[row - 1]
              if ran, row - 1 >= f.lens.count {
                Text("…").font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
              } else if ran {
                let lens = f.lens[row - 1]
                Text("→\(model.cardText(lens.token))")
                  .font(.system(size: 11, weight: .medium, design: .monospaced))
                  .foregroundStyle(.white.opacity(0.85))
                  .lineLimit(1)
                Text(percent(lens.p)).font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
              } else {
                Text(model.bypass.contains(row - 1) ? "SKIPPED" : "OFF")
                  .font(.system(size: 9, weight: .semibold, design: .monospaced)).foregroundStyle(Theme.orange)
              }
            } else if row == rowCount - 1, let p = model.pending {
              Text("→\(model.cardText(p.chosen))")
                .font(.system(size: 11, weight: .medium, design: .monospaced))
                .foregroundStyle(Theme.orange)
                .lineLimit(1)
              Text(percent(p.final[p.chosen])).font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
            }
          }
          .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
          .padding(.leading, 12)
          .contentShape(.rect)
          }
        }
        .buttonStyle(.plain)
        .frame(height: rowHeight)
        .accessibilityLabel(rowAccessibility(row))
        .accessibilityHint("Opens this part of the model")
        .accessibilityIdentifier("row_\(row)")
      }
    }
  }

  /// One line per row for Qwen: the layer number and its logit-lens guess.
  private func compactLabel(_ row: Int, _ focus: Thread?) -> some View {
    HStack(spacing: 4) {
      Text(row == 0 ? "EMB" : row == rowCount - 1 ? "OUT" : "L\(row)")
        .font(.system(size: 9, weight: .semibold, design: .monospaced))
        .foregroundStyle(lit(row) ? .white : rowColor(row))
        .frame(width: 24, alignment: .leading)
      if row >= 1, row <= model.config.layers, let f = focus, row - 1 < f.ran.count {
        if !f.ran[row - 1] {
          Text(model.bypass.contains(row - 1) ? "SKIP" : "OFF")
            .font(.system(size: 8, weight: .semibold, design: .monospaced)).foregroundStyle(Theme.orange)
        } else if row - 1 < f.lens.count {
          let lens = f.lens[row - 1]
          Text(model.cardText(lens.token))
            .font(.system(size: 10, weight: .medium, design: .monospaced))
            .foregroundStyle(.white.opacity(0.35 + 0.65 * min(1, lens.p * 2)))
            .lineLimit(1)
        } else {
          Text("…").font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted)
        }
      } else if row == rowCount - 1, let p = model.pending {
        Text(model.cardText(p.chosen))
          .font(.system(size: 10, weight: .medium, design: .monospaced))
          .foregroundStyle(Theme.orange)
          .lineLimit(1)
      }
    }
  }

  private func rowTitle(_ row: Int) -> String {
    if row == 0 { return "EMBED" }
    if row == rowCount - 1 { return "OUTPUT" }
    return "LAYER \(row)"
  }

  private func rowAccessibility(_ row: Int) -> String {
    if row == 0 { return "Embedding: tokens and their vectors" }
    if row == rowCount - 1 { return "Output: the next-token prediction" }
    return "Layer \(row): \(model.config.heads) attention heads and a feedforward network"
  }

  private func rowColor(_ row: Int) -> Color {
    row == 0 ? Theme.yellow : row == rowCount - 1 ? Theme.orange : Theme.violet
  }

  /// Row lit by the slowed replay of the newest thread's pass.
  private func lit(_ row: Int) -> Bool {
    model.replaying && Int(model.stage) == row + 1
  }

  // MARK: Drawing

  private func canvas(_ threads: [Thread]) -> some View {
    let focus = model.focusThread
    let stage = model.stage
    let layers = model.config.layers
    let replaying = model.replaying
    let predictStage = Int(model.predictStage), sampleStage = Int(model.sampleStage)
    let rowHeight = self.rowHeight
    let compact = self.compact
    let column = self.column
    let bypass = model.bypass
    let texts = threads.map { model.cardText($0.token) }
    return Canvas { ctx, size in
      guard !threads.isEmpty else { return }
      let n = threads.count
      func x(_ i: Int) -> CGFloat { CGFloat(i) * column + column / 2 + 4 }
      func y(_ row: Int) -> CGFloat { top + CGFloat(row) * rowHeight + rowHeight / 2 }
      // Per-row scale so beads compare across threads.
      var scales = [Float](repeating: 1e-6, count: layers + 1)
      for t in threads { for (r, v) in t.residual.enumerated() { for x in v.prefix(32) { scales[r] = max(scales[r], abs(x)) } } }
      let newest = n - 1

      // Warp threads.
      for i in 0..<n {
        let isFocus = i == focus
        var path = Path()
        path.move(to: CGPoint(x: x(i), y: y(0) - 14))
        let bottomRow = (i == newest && replaying) ? max(0, min(layers + 1, Int(stage) - 1)) : layers + 1
        path.addLine(to: CGPoint(x: x(i), y: y(bottomRow)))
        let color = isFocus ? Theme.violet.opacity(0.9) : Theme.violet.opacity(0.22)
        ctx.stroke(path, with: .color(color), style: StrokeStyle(lineWidth: isFocus ? 1.6 : 1))
      }

      // The focused thread's attention on every layer that ran. With 32 heads, only each layer's
      // strongest few pulls are drawn.
      if let f = focus, f < n {
        let t = threads[f]
        for floor in 0..<layers where floor < t.attention.count && t.ran[floor] {
          if f == newest && replaying && Int(stage) < floor + 2 { continue }
          let yy = y(floor + 1) - (compact ? 5 : 9)
          var pulls: [(h: Int, j: Int, w: Float)] = []
          for (h, weights) in t.attention[floor].enumerated() {
            for (j, w) in weights.enumerated() where j != f && w > 0.04 && j < n { pulls.append((h, j, w)) }
          }
          if compact { pulls = Array(pulls.sorted { $0.w > $1.w }.prefix(10)) }
          for (h, j, w) in pulls {
            var arc = Path()
            let x0 = x(f), x1 = x(j)
            let lift = compact ? min(rowHeight * 0.8, 4 + abs(x0 - x1) * 0.08) : min(30, 8 + abs(x0 - x1) * 0.18) + CGFloat(h) * 2.5
            arc.move(to: CGPoint(x: x0, y: yy))
            arc.addQuadCurve(to: CGPoint(x: x1, y: yy), control: CGPoint(x: (x0 + x1) / 2, y: yy - lift))
            ctx.stroke(arc, with: .color(Theme.heads[h % Theme.heads.count].opacity(Double(0.25 + 0.75 * w))),
              style: StrokeStyle(lineWidth: CGFloat((compact ? 0.5 : 0.6) + (compact ? 2 : 3.2) * w), lineCap: .butt))
          }
        }
      }

      // Beads: the residual at each stage (first 32 of 128 values).
      for i in 0..<n {
        let t = threads[i]
        for row in 0...layers {
          if i == newest && replaying && row + 1 > Int(stage) { continue }
          let skipped = row > 0 && !t.ran[row - 1]
          let rect = compact ? CGRect(x: x(i) - 18, y: y(row) - 2, width: 36, height: 8) : CGRect(x: x(i) - 18, y: y(row) - 3, width: 36, height: 12)
          if skipped {
            ctx.stroke(Path(roundedRect: rect, cornerRadius: 2), with: .color(Theme.amber.opacity(bypass.contains(row - 1) ? 0.6 : 0.25)),
              style: StrokeStyle(lineWidth: 0.8, dash: [2, 2]))
            continue
          }
          let vals = t.residual[row].prefix(32)
          let cw = rect.width / 16, ch = rect.height / 2
          for (k, v) in vals.enumerated() {
            let cell = CGRect(x: rect.minX + CGFloat(k % 16) * cw, y: rect.minY + CGFloat(k / 16) * ch, width: cw - 0.4, height: ch - 0.4)
            ctx.fill(Path(cell), with: .color(Theme.value(v, scale: scales[row])))
          }
          if i == focus {
            ctx.stroke(Path(roundedRect: rect.insetBy(dx: -2, dy: -2), cornerRadius: 3), with: .color(Theme.violet), lineWidth: 1)
          }
        }
        // Card label at the top of each thread.
        let label = Text(texts[i])
          .font(.system(size: 10, weight: i == focus ? .bold : .medium, design: .monospaced))
          .foregroundStyle(i == focus ? Theme.green : .white.opacity(0.8))
        ctx.draw(label, at: CGPoint(x: x(i), y: top - 6), anchor: .center)
      }

      // The readout row: the newest thread's next-card odds as a little bar.
      let ry = y(layers + 1)
      if !(replaying && Int(stage) < predictStage) {
        let bar = CGRect(x: x(newest) - 18, y: ry - 4, width: 36, height: 8)
        ctx.fill(Path(roundedRect: bar, cornerRadius: 2), with: .color(Theme.orange.opacity(0.35)))
        if replaying && Int(stage) >= sampleStage {
          ctx.fill(Path(ellipseIn: CGRect(x: x(newest) - 4, y: ry + 8, width: 8, height: 8)), with: .color(Theme.pink))
        }
      }

      // The slowed replay's glow on the stage being shown.
      if replaying, Int(stage) >= 1, Int(stage) <= layers + 2 {
        let row = Int(stage) - 1
        let glow = CGRect(x: x(newest) - 24, y: y(row) - rowHeight * 0.31, width: 48, height: rowHeight * 0.62)
        ctx.fill(Path(roundedRect: glow, cornerRadius: 8), with: .color(Theme.violet.opacity(0.2)))
      }
    }
    .contentShape(.rect)
    .gesture(
      SpatialTapGesture().onEnded { e in
        let i = Int((e.location.x - 4) / column)
        guard i >= 0, i < threads.count else { return }
        model.selectedThread = (i == threads.count - 1) ? nil : i
        model.haptics.transient(intensity: 0.3, sharpness: 0.8)
      }
    )
    .accessibilityElement()
    .accessibilityLabel("The transformer: \(threads.count) positions through \(layers) layers")
    .accessibilityValue(focus.map { "Focused on \(texts[$0])" } ?? "")
    .accessibilityAdjustableAction { dir in
      let current = focus ?? threads.count - 1
      let next = dir == .increment ? min(threads.count - 1, current + 1) : max(0, current - 1)
      model.selectedThread = next == threads.count - 1 ? nil : next
    }
    .accessibilityIdentifier("machine")
  }
}
