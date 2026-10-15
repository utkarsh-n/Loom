import LoomEngine
import SwiftUI

/// The embedding space as a star map: the catalogue's tokens (all of the 1.4M model's; for Qwen3 the
/// ones commonest in Cipher's lines), each vector projected onto the first three principal axes. Your
/// prompt is a constellation. Neighbours are always measured by cosine in all of the model's dimensions.
struct GalaxyTab: View {
  @Bindable var model: LoomModel
  @State private var yaw = 0.6
  @State private var pitch = -0.35
  @State private var zoom = 1.0
  @State private var dragStart: (Double, Double)?
  @State private var pinchStart: Double?
  @State private var selected: Int?
  @State private var search = ""
  @State private var a = "king"
  @State private var b = "man"
  @State private var c = "woman"
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    ScrollView {
      VStack(spacing: 14) {
        starfield
        if let s = selected { StarCard(model: model, token: s, select: { selected = $0 }) }
        finder
        arithmetic
      }
      .padding(.horizontal, 14)
      .padding(.bottom, 24)
    }
    .scrollDismissesKeyboard(.interactively)
    .onChange(of: model.engine) { selected = nil }
  }

  /// Control tokens, and the small model's ids below 8 (unknown, newline, tags): left out of your
  /// constellation and of the token lists.
  private func offMap(_ t: Int) -> Bool { model.isControl(t) || (!model.isQwen && t < 8) }

  /// Qwen is the engine but isn't on the GPU yet (or couldn't load): a short line instead of an empty map.
  private var qwenWait: (text: String, failed: Bool)? {
    guard model.isQwen, !model.qwen.isReady else { return nil }
    switch model.qwen.status {
    case .failed(let why), .unavailable(let why): return (text: "Qwen3 isn't running. \(why)", failed: true)
    default: return (text: "Loading Qwen3…", failed: false)
    }
  }

  // MARK: The map

  /// Your tokens in order as map rows, and how many distinct ones aren't on the map (so aren't drawn).
  private func constellation(_ cat: Catalogue?) -> (rows: [Int], missing: Int) {
    guard let cat else { return ([], 0) }
    var rows: [Int] = []
    var missing = Set<Int>()
    for t in model.weave.ids where !offMap(t) {
      if let r = cat.row(of: t), r < cat.points.count { rows.append(r) } else { missing.insert(t) }
    }
    return (rows, missing.count)
  }

  private var starfield: some View {
    let cat = model.catalogue
    let mine = constellation(cat)
    return VStack(alignment: .leading, spacing: 8) {
      HStack {
        Text(cat.map { "EMBEDDING SPACE · \($0.count.formatted()) TOKENS" } ?? "EMBEDDING SPACE")
          .font(.system(.subheadline, design: .monospaced).weight(.bold)).tracking(1).foregroundStyle(Theme.yellow)
          .lineLimit(1)
          .minimumScaleFactor(0.75)
        Spacer()
        Badge(provenance: .projection)
      }
      TimelineView(.animation(minimumInterval: 1 / 30, paused: reduceMotion || dragStart != nil)) { tl in
        let spin = reduceMotion ? 0 : tl.date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: 3600) * 0.04
        Canvas { ctx, size in
          guard let cat else { return }
          let cy = cos(yaw + spin), sy = sin(yaw + spin), cp = cos(pitch), sp = sin(pitch)
          let scale = min(size.width, size.height) * 0.42 * zoom
          let mid = CGPoint(x: size.width / 2, y: size.height / 2)
          func project(_ p: SIMD3<Float>) -> (CGPoint, Double) {
            let x = Double(p.x), y = Double(p.y), z = Double(p.z)
            let x1 = x * cy + z * sy, z1 = -x * sy + z * cy
            let y2 = y * cp - z1 * sp, z2 = y * sp + z1 * cp
            let persp = 2.6 / (2.6 + z2)
            return (CGPoint(x: mid.x + x1 * scale * persp, y: mid.y - y2 * scale * persp), persp)
          }
          var paths: [Galaxy.Script: Path] = [:]
          for r in 0..<min(cat.points.count, cat.scripts.count) {
            let (pt, persp) = project(cat.points[r])
            let rad = 0.7 * persp + 0.3
            paths[cat.scripts[r], default: Path()].addRect(CGRect(x: pt.x - rad, y: pt.y - rad, width: 2 * rad, height: 2 * rad))
          }
          for (script, path) in paths {
            ctx.fill(path, with: .color(Theme.script(script).opacity(0.55)))
          }
          // Your tokens as a constellation, in order (those on the map).
          var line = Path()
          for r in mine.rows {
            let (pt, _) = project(cat.points[r])
            if line.isEmpty { line.move(to: pt) } else { line.addLine(to: pt) }
          }
          ctx.stroke(line, with: .color(.white.opacity(0.45)), lineWidth: 0.8)
          for r in Set(mine.rows) {
            let (pt, _) = project(cat.points[r])
            ctx.fill(Path(ellipseIn: CGRect(x: pt.x - 3, y: pt.y - 3, width: 6, height: 6)), with: .color(.white))
          }
          if let s = selected, let r = cat.row(of: s), r < cat.points.count {
            let (pt, _) = project(cat.points[r])
            ctx.stroke(Path(ellipseIn: CGRect(x: pt.x - 8, y: pt.y - 8, width: 16, height: 16)), with: .color(Theme.phosphor), lineWidth: 1.5)
            ctx.draw(Text(model.cardText(s)).font(.system(size: 12, weight: .semibold, design: .monospaced)).foregroundStyle(Theme.phosphor),
              at: CGPoint(x: pt.x, y: pt.y - 16))
          }
        }
      }
      .frame(height: 360)
      .background(Theme.well, in: .rect(cornerRadius: 20))
      .overlay {
        if let wait = qwenWait {
          VStack(spacing: 10) {
            if !wait.failed { ProgressView().tint(Theme.yellow) }
            Text(wait.text)
              .font(.system(.footnote, design: .monospaced))
              .foregroundStyle(Theme.muted)
              .multilineTextAlignment(.center)
          }
          .padding()
        }
      }
      .contentShape(.rect)
      .gesture(
        DragGesture(minimumDistance: 3)
          .onChanged { v in
            if dragStart == nil { dragStart = (yaw, pitch) }
            yaw = dragStart!.0 + v.translation.width / 160
            pitch = min(1.4, max(-1.4, dragStart!.1 + v.translation.height / 160))
          }
          .onEnded { _ in dragStart = nil }
      )
      .simultaneousGesture(
        MagnifyGesture()
          .onChanged { v in
            if pinchStart == nil { pinchStart = zoom }
            zoom = min(6, max(0.6, pinchStart! * v.magnification))
          }
          .onEnded { _ in pinchStart = nil }
      )
      .accessibilityElement()
      .accessibilityLabel("Star map of \((cat?.count ?? 0).formatted()) token vectors")
      .accessibilityValue("\(mine.rows.count) tokens of your prompt lit")
      .accessibilityAdjustableAction { d in yaw += d == .increment ? 0.3 : -0.3 }
      .accessibilityIdentifier("galaxy")
      legend(cat, missing: mine.missing)
    }
  }

  private func legend(_ cat: Catalogue?, missing: Int) -> some View {
    VStack(alignment: .leading, spacing: 6) {
      HStack(spacing: 12) {
        ForEach(Galaxy.Script.allCases, id: \.self) { s in
          HStack(spacing: 4) {
            Circle().fill(Theme.script(s)).frame(width: 7, height: 7)
            Text(s.rawValue.capitalized).font(.caption2)
          }
        }
      }
      .foregroundStyle(.secondary)
      if let cat {
        Text(caption(cat, missing: missing))
          .font(.caption)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
      }
    }
  }

  /// What the map holds, how much of the space its three axes keep, and what that means.
  private func caption(_ cat: Catalogue, missing: Int) -> String {
    let vocab = model.config.vocab
    let kept = String(format: "%.1f", cat.variance.reduce(0, +) * 100)
    var parts: [String] = []
    if model.isQwen {
      parts.append("The map holds \(cat.source), of the \(vocab.formatted()) in the vocabulary, laid out by a PCA of their vectors.")
    } else {
      parts.append("The map holds \(cat.count.formatted()) of the \(vocab.formatted()) tokens (all but the unknown mark, the newline and the control tags), laid out by a PCA of the trained model's vectors. The layout stays put when the knobs change the weights; the neighbours below use the current weights.")
    }
    parts.append("Its 3 axes keep \(kept)% of the variance of \(cat.d.formatted()) dimensions, so neighbours on the map may not be neighbours in the model. Each axis is scaled to its own spread, and the few far outliers sit at the edge. Drag to turn, pinch to zoom. White stars are your tokens, joined in order.")
    if missing == 1 {
      parts.append("1 of your tokens isn't on the map, so it isn't drawn.")
    } else if missing > 1 {
      parts.append("\(missing) of your tokens aren't on the map, so they aren't drawn.")
    }
    return parts.joined(separator: " ")
  }

  // MARK: Find a token

  private var finder: some View {
    InstrumentPanel(number: "1", title: "Find a token", badge: .live, tint: Theme.yellow) {
      TextField(model.catalogue.map { "Search the map's \($0.count.formatted()) tokens" } ?? "Search tokens", text: $search)
        .textInputAutocapitalization(.never)
        .autocorrectionDisabled()
        .padding(10)
        .background(Theme.well, in: .rect(cornerRadius: 12))
      let matches = find(search)
      FlowLayout(spacing: 6) {
        ForEach(matches, id: \.self) { id in
          Button {
            selected = id
            model.haptics.transient(intensity: 0.3, sharpness: 0.8)
          } label: {
            Text(model.cardText(id))
              .font(.system(size: 13, design: .monospaced))
              .padding(.horizontal, 7).padding(.vertical, 4)
              .background(id == selected ? Theme.yellow.opacity(0.3) : Color.white.opacity(0.07), in: .rect(cornerRadius: 6))
          }
          .buttonStyle(.plain)
        }
      }
    }
  }

  /// Tokens on the map whose text starts with the query (ignoring case and surrounding spaces); with no
  /// query, the tokens of your prompt.
  private func find(_ q: String) -> [Int] {
    guard let cat = model.catalogue else { return [] }
    let query = q.trimmingCharacters(in: .whitespaces).lowercased()
    if query.isEmpty { return Array(Set(model.weave.ids.filter { !offMap($0) })).sorted().prefix(16).map { $0 } }
    var found: [Int] = []
    for id in cat.ids where model.text(id).trimmingCharacters(in: .whitespaces).lowercased().hasPrefix(query) {
      found.append(id)
      if found.count == 24 { break }
    }
    return found
  }

  // MARK: Vector arithmetic

  private var arithmetic: some View {
    InstrumentPanel(number: "2", title: "Vector arithmetic", badge: .live, nameplate: arithmeticNote, tint: Theme.yellow) {
      HStack(spacing: 6) {
        field($a)
        Text("−").foregroundStyle(Theme.muted)
        field($b)
        Text("+").foregroundStyle(Theme.muted)
        field($c)
      }
      if let cat = model.catalogue {
        let results = analogy(cat)
        if results.isEmpty {
          Text("Each word must be a single token on the map (try common words).").font(.footnote).foregroundStyle(.secondary)
        } else {
          FlowLayout(spacing: 6) {
            ForEach(results, id: \.0) { id, sim in
              Button { selected = id } label: {
                HStack(spacing: 4) {
                  Text(model.cardText(id)).font(.system(size: 13, weight: .medium, design: .monospaced))
                  Text(String(format: "%.2f", sim)).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
                }
                .padding(.horizontal, 7).padding(.vertical, 4)
                .background(Color.white.opacity(0.07), in: .rect(cornerRadius: 6))
              }
              .buttonStyle(.plain)
            }
          }
        }
      } else if let wait = qwenWait {
        Text(wait.text).font(.footnote).foregroundStyle(.secondary)
      }
    }
  }

  private var arithmeticNote: String {
    let d = model.config.d.formatted()
    let method = "Directions in this space can carry meaning: a − b + c, with each token's vector scaled to length 1, then the nearest"
    if model.isQwen {
      let count = model.catalogue.map { " of the map's \($0.count.formatted())" } ?? ""
      return "\(method) tokens\(count) to the result by cosine similarity over all \(d) numbers. These are the vectors tokens enter the first layer with, before any context."
    }
    return "\(method) tokens to the result by cosine similarity over all \(d) numbers. At this size it's hit and miss, and that's the honest answer."
  }

  private func field(_ s: Binding<String>) -> some View {
    TextField("", text: s)
      .textInputAutocapitalization(.never)
      .autocorrectionDisabled()
      .font(.system(.body, design: .monospaced))
      .multilineTextAlignment(.center)
      .padding(8)
      .background(Theme.well, in: .rect(cornerRadius: 10))
  }

  /// The catalogue row of each word's single token, preferring the form with a leading space (how words
  /// appear mid-line). Nil where the word isn't one token on the map.
  private func rows(for words: [String], in cat: Catalogue) -> [Int?] {
    let wanted = words.map { $0.trimmingCharacters(in: .whitespaces) }
    var spaced = [Int?](repeating: nil, count: wanted.count)
    var bare = spaced
    for (r, id) in cat.ids.enumerated() {
      let text = model.text(id)
      for (k, w) in wanted.enumerated() where !w.isEmpty {
        if text == " " + w { spaced[k] = r } else if text == w { bare[k] = r }
      }
    }
    return wanted.indices.map { spaced[$0] ?? bare[$0] }
  }

  /// a − b + c over the unit rows, then the nearest rows to it by cosine.
  private func analogy(_ cat: Catalogue) -> [(Int, Double)] {
    let found = rows(for: [a, b, c], in: cat)
    guard found.count == 3, let ra = found[0], let rb = found[1], let rc = found[2] else { return [] }
    let da = cat.direction(row: ra), db = cat.direction(row: rb), dc = cat.direction(row: rc)
    var v = [Float](repeating: 0, count: cat.d)
    for j in 0..<cat.d {
      v[j] = da[da.startIndex + j] - db[db.startIndex + j] + dc[dc.startIndex + j]
    }
    return cat.nearest(to: v, count: 8, excluding: [cat.ids[ra], cat.ids[rb], cat.ids[rc]]).map { ($0.token, Double($0.cos)) }
  }
}

/// A selected token: its id, its vector and its nearest neighbours.
struct StarCard: View {
  @Bindable var model: LoomModel
  let token: Int
  let select: (Int) -> Void
  /// Bumped when this token's vector arrives (Qwen's rows outside the map are fetched on demand).
  @State private var arrived = 0

  var body: some View {
    InstrumentPanel(number: "\(token)", title: "\"\(model.cardText(token))\"", badge: .live, tint: Theme.yellow) {
      let _ = arrived
      if let cat = model.catalogue, let dir = model.direction(token), dir.count == cat.d {
        let d = cat.d
        let row = cat.row(of: token)
        HStack(alignment: .top, spacing: 14) {
          TokenChip(token: token, text: model.cardText(token), tint: Theme.yellow)
          VStack(alignment: .leading, spacing: 4) {
            Text(vectorTitle(d: d, length: row.map { cat.norms[$0] })).eyebrow()
            ValueStrip(values: Array(dir), columns: d > 512 ? 80 : 32, cell: d > 512 ? 3 : 6)
          }
        }
        if row == nil {
          Text("Not on the map: it isn't among \(cat.source). Its neighbours are still measured in all \(d.formatted()) dimensions.")
            .font(.footnote)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        Text(model.isQwen ? "NEAREST OF THE MAP'S \(cat.count.formatted()) · COSINE IN ALL \(d.formatted()) DIMENSIONS" : "NEAREST BY COSINE IN ALL \(d.formatted()) DIMENSIONS").eyebrow()
        FlowLayout(spacing: 6) {
          ForEach(cat.nearest(to: Array(dir), count: 10, excluding: [token]).map { ($0.token, Double($0.cos)) }, id: \.0) { id, sim in
            Button { select(id) } label: {
              HStack(spacing: 4) {
                Text(model.cardText(id)).font(.system(size: 13, weight: .medium, design: .monospaced))
                Text(String(format: "%.2f", sim)).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
              }
              .padding(.horizontal, 7).padding(.vertical, 4)
              .background(Color.white.opacity(0.07), in: .rect(cornerRadius: 6))
            }
            .buttonStyle(.plain)
          }
        }
        if !model.isControl(token) {
          Button("Add to the prompt", systemImage: "plus") {
            model.prompt += model.text(token)
            model.tab = .loom
          }
          .buttonStyle(.glass)
          .font(.subheadline)
        }
      } else if model.catalogue != nil {
        HStack(spacing: 8) {
          ProgressView()
          Text("Fetching its vector…").font(.footnote).foregroundStyle(.secondary)
        }
      }
    }
    .task(id: token) { await fetch() }
  }

  /// "ITS VECTOR: 128 NUMBERS · LENGTH 0.83". The strip shows the direction; its colours are relative to
  /// the largest number, so the picture is the same at any length.
  private func vectorTitle(d: Int, length: Float?) -> String {
    let base = "ITS VECTOR: \(d.formatted()) NUMBERS"
    guard let length else { return base }
    return base + String(format: " · LENGTH %.2f", length)
  }

  /// Qwen's rows outside the map come on demand: ask for this one and redraw when it arrives.
  private func fetch() async {
    guard model.isQwen, model.direction(token) == nil else { return }
    model.qwen.prefetch([token])
    for _ in 0..<100 {
      try? await Task.sleep(for: .milliseconds(100))
      if Task.isCancelled { return }
      if model.direction(token) != nil {
        arrived += 1
        return
      }
    }
  }
}
