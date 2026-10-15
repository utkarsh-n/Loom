import Accelerate
import LoomEngine
import SwiftUI

/// The Sphere tab: type a word or phrase, then enter the sphere and ride.
struct SphereTab: View {
  @Bindable var model: LoomModel

  var body: some View {
    ScrollView {
      VStack(spacing: 14) {
        InstrumentPanel(number: "↻", title: "Ride the sphere", badge: .live,
          nameplate: "Stand inside the model's \(model.config.d.formatted())-dimensional space and look along the newest token's vector. Every token sits at its true angle from your gaze, and the prediction is a dot product with that gaze, so the tokens drifting toward the centre are the likely next ones. Each layer turns your gaze; sampling latches a tether onto a token, and you ride it to its place.", tint: Theme.violet) {
          Text(engineLine)
            .font(.system(.caption, design: .monospaced))
            .foregroundStyle(Theme.muted)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("sphereEngine")
          HStack(spacing: 8) {
            TextField("A word or a phrase", text: $model.prompt)
              .textInputAutocapitalization(.sentences)
              .autocorrectionDisabled()
              .submitLabel(.go)
              .onSubmit(enter)
              .padding(12)
              .background(Theme.well, in: .rect(cornerRadius: 14))
              .accessibilityIdentifier("spherePrompt")
          }
          Button(action: enter) {
            Label(enterTitle, systemImage: model.engineReady ? "circle.hexagonpath" : "hourglass")
              .font(.headline)
              .frame(maxWidth: .infinity, minHeight: 44)
          }
          .buttonStyle(.glassProminent)
          .tint(Theme.phosphor.opacity(0.75))
          .disabled(!model.engineReady)
          .accessibilityIdentifier("enterSphere")
          if let why = qwenProblem {
            Text(why)
              .font(.footnote)
              .foregroundStyle(Theme.orange)
              .fixedSize(horizontal: false, vertical: true)
          }
        }
        InstrumentPanel(number: "?", title: "What you're looking at", badge: .projection, tint: Theme.violet) {
          note("Your gaze", "The direction of the newest position's vector after the final norm: the same vector the prediction compares with every token.")
          note("Distance from the centre", distanceNote)
          note("Which tokens", tokensNote)
          note("Direction around the centre", "A projection: the part of each token's vector that is not along your gaze, read on two axes carried smoothly as you turn. Neighbours around the ring may not be neighbours.")
          note("The slow orbit", "A metaphor. The view turns gently around your gaze so the sky reads as a sphere; turning it changes nothing.")
          note("Beams", "On each layer, earlier tokens glow with the attention the newest position pays them, coloured by head (\(model.config.heads.formatted()) per layer). Weights under 5% aren't drawn, and never more than the \(SphereRide.maxBeams) strongest at once.")
          note("The latch", "Sampling (or your tap) picks a token; a tether reaches it and your gaze rides to it. Its embedding is where the next position starts.")
        }
      }
      .padding(.horizontal, 14)
      .padding(.bottom, 24)
    }
    .scrollDismissesKeyboard(.interactively)
  }

  /// Which model the sphere is inside (switching happens on the Pipeline tab).
  private var engineLine: String {
    let c = model.config
    return "Engine: \(model.engine.title) · \(c.layers.formatted()) layers · \(c.d.formatted()) dimensions. Switch engines on the Pipeline tab."
  }

  private var enterTitle: String {
    if model.engineReady { return "Enter the sphere" }
    if qwenProblem != nil { return "Qwen3 isn't running" }
    return model.isQwen ? "Loading Qwen3…" : "Loading…"
  }

  /// Why Qwen can't run, when it is the engine and failed to load.
  private var qwenProblem: String? {
    guard model.isQwen else { return nil }
    switch model.qwen.status {
    case .failed(let why), .unavailable(let why): return why
    default: return nil
    }
  }

  private var distanceNote: String {
    let d = model.config.d.formatted()
    return "1 − cos θ, the angle between a token and your gaze in all \(d) dimensions. It's exact, and linear in the token's score for vectors of equal length. The solid ring is cos 0, at right angles to you: in \(d) dimensions, unrelated vectors mostly sit near it."
  }

  private var tokensNote: String {
    let c = model.config
    guard let cat = model.catalogue else {
      return model.isQwen ? "Qwen3's tokens appear once it has loaded." : "The tokens appear once the model has loaded."
    }
    let stars = cat.ids.reduce(0) { $0 + (offSky($1) ? 0 : 1) }
    if model.isQwen {
      let gb = Double(c.vocab) * Double(c.d) * 4 / 1e9
      return "The sky holds the \(stars.formatted()) tokens commonest in Cipher's lines, plus any token in play: yours, the candidates and each layer's best guess. All \(c.vocab.formatted()) rows of \(c.d.formatted()) 32-bit numbers would take \(String(format: "%.1f", gb)) GB."
    }
    return "The sky holds \(stars.formatted()) of the \(c.vocab.formatted()) tokens: all but the unknown mark, the newline and the control tags."
  }

  /// Tokens the sky leaves out: control tokens, and the small model's ids below 8 (unknown, newline, tags).
  private func offSky(_ t: Int) -> Bool { model.isControl(t) || (!model.isQwen && t < 8) }

  private func enter() {
    guard model.engineReady else { return }
    model.restart()
    model.sphereOpen = true
    model.haptics.transient(intensity: 0.7, sharpness: 0.5)
  }

  private func note(_ title: String, _ body: String) -> some View {
    VStack(alignment: .leading, spacing: 3) {
      Text(title).font(.subheadline.weight(.medium)).foregroundStyle(Theme.phosphor)
      Text(body).font(.callout).foregroundStyle(.white.opacity(0.85)).fixedSize(horizontal: false, vertical: true)
    }
  }
}

// MARK: - The ride

/// The camera's working state. A plain class: it is advanced inside the drawing closure every frame,
/// which SwiftUI state must not be.
@MainActor final class SphereCamera {
  /// The engine and catalogue the state below belongs to; a change starts it afresh.
  var engineKey = ""
  var catalogueKey = ""
  /// Catalogue rows drawn as stars (control tokens left out). Depends only on the catalogue.
  var starRows: [Int] = []
  /// This frame's cos, x and y of every catalogue row (rows × 3), and the axes they're read along (d × 3).
  /// Reused from frame to frame.
  var sky: [Float] = []
  var axes: [Float] = []
  /// What `sky` was computed for: it is reused while the catalogue, gaze and axes stay put.
  var skyKey = ""
  var skyGaze: [Float] = []
  var skyRight: [Float] = []
  var skyUp: [Float] = []
  var right: [Float] = []
  var up: [Float] = []
  var stage: Double = 0
  var lastTime: Double = 0
  var threadKey = ""
  var headings: [[Float]] = []
  var targetToken = -1
  var target: [Float] = []
  var roll: Double = 0
  var look = CGSize.zero
  var lookHeld = false
  var zoom: CGFloat = 1.35
  /// Star positions by catalogue row: a spare buffer and last frame's (for motion streaks), and last
  /// frame's gaze.
  var positions: [CGPoint] = []
  var previous: [CGPoint] = []
  var previousGaze: [Float] = []
  /// Last frame's screen positions of the labelled tokens, for taps.
  var hits: [(token: Int, point: CGPoint)] = []
}

struct SphereRide: View {
  @Bindable var model: LoomModel
  @Environment(\.dismiss) private var dismiss
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @State private var camera = SphereCamera()
  @State private var dragStart: CGSize?
  @State private var pinchStart: CGFloat?
  @State private var fast = false

  /// Attention beams: weights below this aren't drawn, and at most `maxBeams` (the strongest) are.
  static let beamFloor: Float = 0.05
  static let maxBeams = 200

  var body: some View {
    ZStack {
      Theme.background.ignoresSafeArea()
      TimelineView(.animation(minimumInterval: 1 / 60)) { tl in
        Canvas { ctx, size in
          draw(&ctx, size: size, now: tl.date.timeIntervalSinceReferenceDate)
        }
      }
      .ignoresSafeArea()
      .contentShape(.rect)
      .gesture(
        DragGesture(minimumDistance: 4)
          .onChanged { v in
            if dragStart == nil { dragStart = camera.look }
            camera.look = CGSize(width: dragStart!.width + v.translation.width, height: dragStart!.height + v.translation.height)
            camera.lookHeld = true
          }
          .onEnded { _ in
            dragStart = nil
            camera.lookHeld = false
          }
      )
      .simultaneousGesture(
        MagnifyGesture()
          .onChanged { v in
            if pinchStart == nil { pinchStart = camera.zoom }
            camera.zoom = min(3, max(0.6, pinchStart! * v.magnification))
          }
          .onEnded { _ in pinchStart = nil }
      )
      .simultaneousGesture(TapGesture(count: 2).onEnded { camera.look = .zero; camera.zoom = 1.35 })
      .simultaneousGesture(
        SpatialTapGesture().onEnded { e in
          if let hit = camera.hits.min(by: { distance($0.point, e.location) < distance($1.point, e.location) }),
            distance(hit.point, e.location) < 34, model.stage >= model.predictStage || !model.playing
          {
            model.latch(hit.token)
          }
        }
      )
      .accessibilityElement()
      .accessibilityLabel("Inside the sphere")
      .accessibilityValue(accessibilitySummary)
      .accessibilityIdentifier("sphere")
      if let wait = waitText {
        Text(wait)
          .font(.system(.callout, design: .monospaced))
          .foregroundStyle(Theme.phosphor)
          .multilineTextAlignment(.center)
          .padding(.horizontal, 18)
          .padding(.vertical, 12)
          .glassEffect(.regular, in: .rect(cornerRadius: 18))
          .padding(24)
          .accessibilityIdentifier("sphereWaiting")
      }
      VStack(spacing: 0) {
        topBar
        Spacer()
        bottomPanel
      }
    }
    .statusBarHidden()
    .onAppear {
      model.rideSeconds = 3.6
      camera.lastTime = 0
    }
    .onDisappear {
      model.stop()
      model.rideSeconds = nil
    }
  }

  private func distance(_ a: CGPoint, _ b: CGPoint) -> CGFloat { hypot(a.x - b.x, a.y - b.y) }

  /// Tokens the sky leaves out: control tokens, and the small model's ids below 8 (unknown, newline, tags).
  private func offSky(_ t: Int) -> Bool { model.isControl(t) || (!model.isQwen && t < 8) }

  /// Shown instead of an empty sky while the engine can't draw.
  private var waitText: String? {
    if model.isQwen, !model.qwen.isReady {
      switch model.qwen.status {
      case .failed(let why), .unavailable(let why): return "Qwen3 isn't running. \(why)"
      default: return "Loading Qwen3…"
      }
    }
    if model.engineReady, model.catalogue == nil { return "This build has no token catalogue to draw." }
    return nil
  }

  // MARK: HUD

  private var topBar: some View {
    HStack(alignment: .top, spacing: 10) {
      VStack(alignment: .leading, spacing: 4) {
        Text(stageTitle).font(.system(size: 12, weight: .semibold, design: .monospaced)).tracking(1.2).foregroundStyle(Theme.phosphor)
        Text(stageDetail).font(.footnote).foregroundStyle(.white.opacity(0.85)).lineLimit(2)
      }
      .padding(.horizontal, 14)
      .padding(.vertical, 10)
      .frame(maxWidth: .infinity, alignment: .leading)
      .glassEffect(.regular, in: .rect(cornerRadius: 18))
      Button {
        dismiss()
      } label: {
        Image(systemName: "xmark").font(.body.weight(.semibold)).frame(width: 44, height: 44)
      }
      .buttonStyle(.glass)
      .buttonBorderShape(.circle)
      .accessibilityLabel("Leave the sphere")
      .accessibilityIdentifier("leaveSphere")
    }
    .padding(.horizontal, 14)
    .padding(.top, 8)
  }

  private var bottomPanel: some View {
    VStack(spacing: 10) {
      Text(clothText)
        .font(.system(.title3, design: .serif))
        .foregroundStyle(.white)
        .lineLimit(2)
        .truncationMode(.head)
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentTransition(.opacity)
      if let p = model.pending, model.canWeave {
        ScrollView(.horizontal, showsIndicators: false) {
          GlassEffectContainer(spacing: 6) {
            HStack(spacing: 6) {
              Text("LATCH").font(.system(size: 10, weight: .semibold, design: .monospaced)).foregroundStyle(Theme.muted)
              ForEach(Array(p.order.prefix(6)), id: \.self) { t in
                Button {
                  model.latch(t)
                } label: {
                  HStack(spacing: 4) {
                    Text(model.cardText(t)).font(.system(.subheadline, design: .monospaced).weight(.semibold))
                    Text(percent(p.final[t])).font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.muted)
                  }
                  .padding(.horizontal, 6)
                  .frame(minHeight: 32)
                }
                .buttonStyle(.glass(t == model.latchTarget ? .regular.tint(Theme.phosphor.opacity(0.4)) : .regular))
                .accessibilityLabel("Latch onto \(model.cardText(t)), \(percent(p.final[t]))")
              }
            }
          }
        }
      }
      HStack(spacing: 10) {
        Button {
          if model.playing { model.stop() } else { model.rideSeconds = fast ? 1.2 : 3.6; model.ride() }
        } label: {
          Label(model.playing ? "Hold" : "Ride", systemImage: model.playing ? "pause.fill" : "play.fill")
            .font(.subheadline.weight(.semibold))
            .frame(minHeight: 36)
            .padding(.horizontal, 6)
            .contentTransition(.symbolEffect(.replace))
        }
        .buttonStyle(.glassProminent)
        .tint(Theme.phosphor.opacity(0.75))
        .disabled(!model.canWeave && !model.playing)
        .accessibilityIdentifier("ride")
        Button {
          if let t = model.latchTarget { model.latch(t) }
        } label: {
          Image(systemName: "forward.frame.fill").frame(width: 36, height: 36)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .disabled(model.playing || !model.canWeave)
        .accessibilityLabel("Ride one token")
        Button {
          model.restart()
          camera.look = .zero
        } label: {
          Image(systemName: "arrow.counterclockwise").frame(width: 36, height: 36)
        }
        .buttonStyle(.glass)
        .buttonBorderShape(.circle)
        .disabled(!model.engineReady)
        .accessibilityLabel("Back to the start of the phrase")
        Spacer(minLength: 0)
        Picker("Speed", selection: $fast) {
          Text("Cruise").tag(false)
          Text("Fast").tag(true)
        }
        .pickerStyle(.segmented)
        .frame(width: 140)
        .onChange(of: fast) { _, f in model.rideSeconds = f ? 1.2 : 3.6 }
      }
    }
    .padding(14)
    .background(
      LinearGradient(colors: [.clear, Theme.background.opacity(0.9)], startPoint: .top, endPoint: .center)
        .ignoresSafeArea())
  }

  /// The running text. The small model's lines open with a newline token, left off here; `decode` leaves
  /// out control tokens (the small model's tags, Qwen's chat-template markers).
  private var clothText: String {
    let ids = model.weave.ids
    let shown = !model.isQwen && ids.first == Tokenizer.newline ? ids.dropFirst() : ids[...]
    let text = model.decode(shown).replacingOccurrences(of: "\n", with: " ↵ ").trimmingCharacters(in: .whitespaces)
    return text.isEmpty ? model.prompt : text
  }

  private var stageTitle: String {
    let s = Int(model.stage)
    let floors = model.config.layers
    switch s {
    case 0, 1: return "EMBEDDING"
    case 2...(floors + 1): return "LAYER \(s - 1) OF \(floors)"
    case floors + 2: return "PREDICTION"
    case floors + 3: return "LATCH"
    default: return model.playing ? "RIDING" : "AT THE PREDICTION"
    }
  }

  private var stageDetail: String {
    guard let fi = model.focusThread.map({ _ in model.weave.threads.count - 1 }), fi >= 0 else { return "" }
    let t = model.weave.threads[fi]
    let s = Int(model.stage)
    let floors = model.config.layers
    switch s {
    case 0, 1:
      return "You are \"\(model.cardText(t.token))\". Your gaze starts on your own token's vector."
    case 2...(floors + 1):
      let f = s - 2
      guard f < t.lens.count, f < t.ran.count, t.ran[f] else { return "This layer is switched off." }
      let deg = camera.headings.count > f + 1 ? Gaze.angle(camera.headings[f], camera.headings[f + 1]) * 180 / .pi : 0
      return String(format: "Attention pulls, feedforward pushes: gaze turned %.0f°. Best guess \"%@\" (%@).", deg, model.cardText(t.lens[f].token), percent(t.lens[f].p))
    case floors + 2:
      return "The nearest tokens score highest. \(model.pending?.kept ?? 0) are in the running."
    case floors + 3:
      guard let target = model.latchTarget, let p = model.pending else { return "" }
      return "Latching onto \"\(model.cardText(target))\" (\(percent(p.final[target]))). Riding its vector."
    default:
      if let p = model.pending, model.canWeave {
        return "Looking at the prediction. Tap a token, or Ride. Likeliest: \"\(model.cardText(p.order[0]))\" \(percent(p.final[p.order[0]]))."
      }
      return model.weave.finished ? "The line is finished. Start again, or leave." : "Type a phrase on the Sphere tab."
    }
  }

  private var accessibilitySummary: String {
    guard let p = model.pending else { return stageTitle }
    let names = p.order.prefix(3).map { "\(model.cardText($0)) \(percent(p.final[$0]))" }.joined(separator: ", ")
    return "\(stageTitle). Nearest tokens: \(names)."
  }

  // MARK: Drawing

  private func draw(_ ctx: inout GraphicsContext, size: CGSize, now: Double) {
    let threads = model.weave.threads
    guard model.engineReady, let cat = model.catalogue, cat.count > 0, let newest = threads.last else { return }
    let cam = camera
    let c = model.config
    let d = c.d, layers = c.layers
    let gain = model.finalGain
    guard d > 0, layers > 0, cat.d == d, cat.unit.count == cat.count * d, gain.count == d else { return }
    let dt = cam.lastTime == 0 ? 0 : min(0.1, now - cam.lastTime)
    cam.lastTime = now

    // Another engine: start the camera afresh.
    let engineKey = "\(model.engine.rawValue)|\(d)|\(layers)"
    if engineKey != cam.engineKey {
      cam.engineKey = engineKey
      cam.right = []
      cam.up = []
      cam.headings = []
      cam.threadKey = ""
      cam.targetToken = -1
      cam.target = []
      cam.stage = model.predictStage
      cam.catalogueKey = ""
    }
    // Work that depends only on the catalogue: which rows are stars. Rows keep their tokens when the
    // small model's catalogue is rebuilt for new weights, so last frame's positions stay comparable.
    let catalogueKey = "\(cat.count)|\(cat.ids.first ?? -1)|\(cat.ids.last ?? -1)"
    if catalogueKey != cam.catalogueKey {
      cam.catalogueKey = catalogueKey
      cam.starRows = cat.ids.indices.filter { !offSky(cat.ids[$0]) }
      cam.positions = []
      cam.previous = []
      cam.previousGaze = []
    }

    // The newest position's gaze after each stage, in readout space. Only the newest position keeps its
    // whole vector (Qwen trims older ones), so only it is read here.
    let tkey = "\(model.weave.ids.count)|\(newest.token)|\(newest.residual.last?.first ?? 0)"
    if tkey != cam.threadKey {
      guard newest.residual.count >= 2, newest.residual.allSatisfy({ $0.count == d }) else { return }
      let seen = Int(cam.threadKey.split(separator: "|").first ?? "") ?? 0
      cam.threadKey = tkey
      cam.headings = newest.residual.map { Gaze.heading($0[...], gain: gain) }
      if seen < model.weave.ids.count, model.stage < 2 { cam.stage = 0 }
      if cam.right.count != d {
        cam.right = Gaze.fallback(orthogonalTo: [cam.headings[0]])
        cam.up = Gaze.fallback(orthogonalTo: [cam.headings[0], cam.right])
      }
      if model.isQwen { model.qwen.prefetch(Array(model.weave.window) + newest.lens.map { $0.token }) }
    }
    let target = model.latchTarget ?? -1
    if target >= 0, target != cam.targetToken {
      if let dir = model.direction(target), dir.count == d {
        cam.targetToken = target
        cam.target = Gaze.heading(dir, gain: gain)
      } else {
        cam.target = []
        if model.isQwen { model.qwen.prefetch([target]) }
      }
    }

    // Ease the camera's stage toward the model's.
    let ms = model.stage
    let goal = ms >= model.idleStage ? model.predictStage : (ms >= model.sampleStage ? model.idleStage : ms)
    if goal < cam.stage - 0.5 { cam.stage = goal } else {
      let tau = max(0.08, (model.rideSeconds ?? 2.4) / Double(3 * (layers + 4)))
      cam.stage += (goal - cam.stage) * (1 - exp(-dt / tau))
    }
    let gaze0 = gazeAt(cam.stage, layers: layers)
    guard gaze0.count == d, cam.right.count == d, cam.up.count == d else { return }
    Gaze.transport(right: &cam.right, up: &cam.up, gaze: gaze0)

    // Look around: tilt the gaze toward the drag, then let it drift back while you ride.
    if !cam.lookHeld && model.playing {
      let k = CGFloat(1 - exp(-dt / 0.9))
      cam.look.width -= cam.look.width * k
      cam.look.height -= cam.look.height * k
    }
    let R = min(size.width, size.height) * 0.47 * cam.zoom
    let lx = Float(cam.look.width / R), ly = Float(-cam.look.height / R)
    var gaze = gaze0
    var right = cam.right, up = cam.up
    let a = (lx * lx + ly * ly).squareRoot() * 0.9
    if a > 1e-3 {
      let ca = cos(a), sa = sin(a)
      gaze = (0..<d).map { ca * gaze0[$0] + sa * (lx * right[$0] + ly * up[$0]) / (a / 0.9) }
      Gaze.normalize(&gaze)
      Gaze.transport(right: &right, up: &up, gaze: gaze)
    }

    if !reduceMotion { cam.roll += dt * 0.05 }
    let roll = cam.roll
    // The matrix product streams the whole catalogue (84 MB for Qwen's), so it only runs when the view
    // has moved; the slow orbit is applied after it.
    let skyKey = "\(catalogueKey)|\(cat.norms.first ?? 0)|\(cat.norms[cat.count / 2])|\(cat.norms.last ?? 0)"
    if skyKey != cam.skyKey || cam.sky.count != cat.count * 3 || !Self.close(gaze, cam.skyGaze)
      || !Self.close(right, cam.skyRight) || !Self.close(up, cam.skyUp)
    {
      project(cat, gaze: gaze, right: right, up: up)
      cam.skyKey = skyKey
      cam.skyGaze = gaze
      cam.skyRight = right
      cam.skyUp = up
    }
    let sky = cam.sky
    let center = CGPoint(x: size.width / 2, y: size.height * 0.46)
    func spot(_ cv: Float, _ x: Float, _ y: Float) -> CGPoint {
      let r = CGFloat((1 - cv) / 1.3) * R
      let phi = Double(atan2(y, x)) + roll
      return CGPoint(x: center.x + r * CGFloat(cos(phi)), y: center.y + r * CGFloat(sin(phi)))
    }
    // Any token: its catalogue row, else its own direction (three dot products). Qwen's rows outside the
    // catalogue arrive on demand; until one has, the token isn't drawn and is fetched.
    var extra: [Int: CGPoint] = [:]
    var missing: [Int] = []
    func place(_ t: Int) -> CGPoint? {
      if let r = cat.row(of: t) { return spot(sky[r * 3], sky[r * 3 + 1], sky[r * 3 + 2]) }
      if let p = extra[t] { return p }
      guard let v = model.direction(t), v.count == d else {
        missing.append(t)
        return nil
      }
      let (cv, x, y) = Self.coordinates(of: v, gaze: gaze, right: right, up: up)
      let p = spot(cv, x, y)
      extra[t] = p
      return p
    }

    // The sphere's interior: a glow, rings at cos 0.5, 0 and −0.25, and spokes.
    ctx.fill(Path(ellipseIn: CGRect(x: center.x - R * 1.02, y: center.y - R * 1.02, width: R * 2.04, height: R * 2.04)),
      with: .radialGradient(Gradient(colors: [Color(red: 0.05, green: 0.11, blue: 0.09), Theme.background]), center: center, startRadius: 0, endRadius: R * 1.05))
    for (cv, label) in [(0.5, "cos 0.5"), (0.0, "cos 0 · right angles"), (-0.25, "cos −0.25")] {
      let r = CGFloat((1 - cv) / 1.3) * R
      ctx.stroke(Path(ellipseIn: CGRect(x: center.x - r, y: center.y - r, width: 2 * r, height: 2 * r)),
        with: .color(.white.opacity(cv == 0 ? 0.14 : 0.07)), style: StrokeStyle(lineWidth: 0.7, dash: cv == 0 ? [] : [3, 5]))
      ctx.draw(Text(label).font(.system(size: 9, design: .monospaced)).foregroundStyle(Theme.muted.opacity(0.7)),
        at: CGPoint(x: center.x, y: center.y - r - 6))
    }
    for k in 0..<12 {
      let phi = roll + Double(k) * .pi / 6
      var spoke = Path()
      spoke.move(to: CGPoint(x: center.x + 0.15 * R * CGFloat(cos(phi)), y: center.y + 0.15 * R * CGFloat(sin(phi))))
      spoke.addLine(to: CGPoint(x: center.x + 1.0 * R * CGFloat(cos(phi)), y: center.y + 1.0 * R * CGFloat(sin(phi))))
      ctx.stroke(spoke, with: .color(.white.opacity(0.035)), lineWidth: 0.6)
    }

    // Every token in the catalogue, by script. When the gaze turns fast, each star streaks from where it
    // was a frame ago (the stars stay put; you are the one turning).
    let speed = cam.previousGaze.count == d && dt > 0 ? Double(Gaze.angle(cam.previousGaze, gaze)) / dt : 0
    let streak = reduceMotion ? 0 : min(1, max(0, (speed - 0.25) / 2))
    let rows = cat.count
    // Two position buffers take turns, so a frame allocates none.
    var positions = cam.positions
    cam.positions = []
    if positions.count != rows { positions = [CGPoint](repeating: .zero, count: rows) }
    let previous = cam.previous
    let canStreak = streak > 0 && previous.count == rows
    var paths: [Galaxy.Script: Path] = [:]
    var streaks: [Galaxy.Script: Path] = [:]
    for r in cam.starRows {
      let p = spot(sky[r * 3], sky[r * 3 + 1], sky[r * 3 + 2])
      positions[r] = p
      let s: CGFloat = sky[r * 3] > 0.15 ? 1.8 : 1.1
      let script = r < cat.scripts.count ? cat.scripts[r] : .mark
      paths[script, default: Path()].addRect(CGRect(x: p.x - s / 2, y: p.y - s / 2, width: s, height: s))
      if canStreak && r % 2 == 0 {
        let q = previous[r]
        if hypot(q.x - p.x, q.y - p.y) < R * 0.6 {
          streaks[script, default: Path()].move(to: q)
          streaks[script, default: Path()].addLine(to: p)
        }
      }
    }
    for (script, path) in streaks { ctx.stroke(path, with: .color(Theme.script(script).opacity(0.35 * streak)), lineWidth: 1) }
    for (script, path) in paths { ctx.fill(path, with: .color(Theme.script(script).opacity(0.55))) }
    cam.positions = previous
    cam.previous = positions
    cam.previousGaze = gaze

    // The tokens already ridden: your path through the sky, and on a layer, the attention beams.
    let window = Array(model.weave.window)
    let hidden = window.map { offSky($0) }
    var trail = Path()
    for (k, t) in window.enumerated() where !hidden[k] {
      guard let p = place(t) else { continue }
      if trail.isEmpty { trail.move(to: p) } else { trail.addLine(to: p) }
    }
    ctx.stroke(trail, with: .color(.white.opacity(0.22)), style: StrokeStyle(lineWidth: 0.8, dash: [2, 3]))
    // The layer being worked is the model's (the camera's turn lags it slightly).
    let floorShown = Int(model.stage) - 2
    if model.stage < model.idleStage, floorShown >= 0, floorShown < layers, floorShown < newest.attention.count,
      floorShown < newest.ran.count, newest.ran[floorShown]
    {
      let beamStrength = min(1, max(0.25, (cam.stage - Double(floorShown) - 1) * 1.5))
      var beams: [(head: Int, position: Int, weight: Float)] = []
      for (h, row) in newest.attention[floorShown].enumerated() {
        for (j, wgt) in row.enumerated() where wgt > Self.beamFloor && j < window.count && !hidden[j] {
          beams.append((head: h, position: j, weight: wgt))
        }
      }
      // The strongest, drawn last so they sit on top.
      beams.sort { $0.weight < $1.weight }
      for b in beams.suffix(Self.maxBeams) {
        guard let p = place(window[b.position]) else { continue }
        var beam = Path()
        beam.move(to: p)
        beam.addLine(to: center)
        let colour = Theme.heads[b.head % Theme.heads.count]
        ctx.stroke(beam, with: .color(colour.opacity(Double(b.weight) * 0.9 * max(0.2, beamStrength))), lineWidth: CGFloat(0.5 + 4 * b.weight))
      }
    }
    for t in Set(window) where !offSky(t) {
      guard let p = place(t) else { continue }
      ctx.fill(Path(ellipseIn: CGRect(x: p.x - 2.5, y: p.y - 2.5, width: 5, height: 5)), with: .color(.white.opacity(0.9)))
      ctx.draw(Text(model.cardText(t)).font(.system(size: 10, design: .monospaced)).foregroundStyle(.white.opacity(0.75)),
        at: CGPoint(x: p.x, y: p.y + 10))
    }

    // On a layer, its own best guess so far (the logit lens): the readout hasn't happened yet.
    let atReadout = model.stage >= model.predictStage
    if !atReadout, floorShown >= 0, floorShown < newest.lens.count, floorShown < newest.ran.count, newest.ran[floorShown] {
      let guess = newest.lens[floorShown]
      if let p = place(guess.token) {
        ctx.stroke(Path(ellipseIn: CGRect(x: p.x - 12, y: p.y - 12, width: 24, height: 24)), with: .color(Theme.phosphor.opacity(0.8)), lineWidth: 1.2)
        ctx.draw(Text("layer \(floorShown + 1) guesses \(model.cardText(guess.token)) \(percent(guess.p))")
          .font(.system(size: 11, weight: .semibold, design: .monospaced)).foregroundStyle(Theme.phosphor),
          at: CGPoint(x: p.x, y: p.y - 22))
      }
    }

    // The candidates: bright orbs sized by the odds the knobs give them. They appear at the readout,
    // where the scores are actually taken; before that they're faint.
    var hits: [(token: Int, point: CGPoint)] = []
    if let draw = model.pending, model.canWeave {
      let latching = Int(cam.stage) >= layers + 3
      let presence = atReadout ? 1.0 : 0.15
      for t in draw.order.prefix(8) {
        let pr = draw.final[t]
        guard pr > 0, let p = place(t) else { continue }
        let rad = CGFloat(3 + 16 * pr.squareRoot())
        let isTarget = t == target
        ctx.fill(Path(ellipseIn: CGRect(x: p.x - rad * 2.2, y: p.y - rad * 2.2, width: rad * 4.4, height: rad * 4.4)),
          with: .color((isTarget ? Theme.phosphor : Theme.amber).opacity(0.12 * presence)))
        ctx.fill(Path(ellipseIn: CGRect(x: p.x - rad, y: p.y - rad, width: rad * 2, height: rad * 2)),
          with: .color((isTarget && latching ? Theme.phosphor : Theme.amber).opacity(0.85 * presence)))
        if atReadout {
          ctx.draw(Text("\(model.cardText(t)) \(percent(pr))").font(.system(size: 11, weight: .semibold, design: .monospaced)).foregroundStyle(.white),
            at: CGPoint(x: p.x, y: p.y - rad - 9))
          hits.append((token: t, point: p))
        }
      }
      // The tether: from your gaze to the token you're about to become.
      if latching, target >= 0, let p = place(target) {
        var tether = Path()
        tether.move(to: center)
        tether.addLine(to: p)
        ctx.stroke(tether, with: .color(Theme.phosphor.opacity(0.35)), lineWidth: 6)
        ctx.stroke(tether, with: .color(Theme.phosphor), lineWidth: 1.6)
      }
    }
    cam.hits = hits
    if model.isQwen, !missing.isEmpty { model.qwen.prefetch(missing) }

    // Where you are looking.
    let off = a > 1e-3
    ctx.stroke(Path(ellipseIn: CGRect(x: center.x - 11, y: center.y - 11, width: 22, height: 22)), with: .color(Theme.phosphor.opacity(off ? 0.4 : 0.9)), lineWidth: 1.2)
    var cross = Path()
    cross.move(to: CGPoint(x: center.x - 18, y: center.y)); cross.addLine(to: CGPoint(x: center.x - 13, y: center.y))
    cross.move(to: CGPoint(x: center.x + 13, y: center.y)); cross.addLine(to: CGPoint(x: center.x + 18, y: center.y))
    cross.move(to: CGPoint(x: center.x, y: center.y - 18)); cross.addLine(to: CGPoint(x: center.x, y: center.y - 13))
    cross.move(to: CGPoint(x: center.x, y: center.y + 13)); cross.addLine(to: CGPoint(x: center.x, y: center.y + 18))
    ctx.stroke(cross, with: .color(Theme.phosphor.opacity(0.8)), lineWidth: 1.2)
    ctx.draw(Text(off ? "looking around · double-tap to return" : "your gaze · \(model.cardText(newest.token))")
      .font(.system(size: 10, design: .monospaced)).foregroundStyle(Theme.phosphor.opacity(0.85)),
      at: CGPoint(x: center.x, y: center.y + 30))
  }

  /// Cos, x and y of every catalogue row along this frame's gaze and axes, into `camera.sky` (rows × 3):
  /// one matrix product into buffers reused from frame to frame (the same arithmetic as `Gaze.view`).
  private func project(_ cat: Catalogue, gaze: [Float], right: [Float], up: [Float]) {
    let cam = camera
    let d = cat.d, rows = cat.count
    guard gaze.count == d, right.count == d, up.count == d else { return }
    if cam.axes.count != d * 3 { cam.axes = [Float](repeating: 0, count: d * 3) }
    if cam.sky.count != rows * 3 { cam.sky = [Float](repeating: 0, count: rows * 3) }
    cam.axes.withUnsafeMutableBufferPointer { a in
      for j in 0..<d {
        a[j * 3] = gaze[j]
        a[j * 3 + 1] = right[j]
        a[j * 3 + 2] = up[j]
      }
    }
    cat.unit.withUnsafeBufferPointer { u in
      cam.axes.withUnsafeBufferPointer { a in
        cam.sky.withUnsafeMutableBufferPointer { o in
          // (rows × d) · (d × 3), row-major.
          vDSP_mmul(u.baseAddress!, 1, a.baseAddress!, 1, o.baseAddress!, 1, vDSP_Length(rows), 3, vDSP_Length(d))
        }
      }
    }
  }

  /// Two vectors equal to within 1e-6 everywhere (a sky drawn for one is, to the pixel, the other's).
  private static func close(_ a: [Float], _ b: [Float]) -> Bool {
    guard a.count == b.count else { return false }
    for i in a.indices where abs(a[i] - b[i]) > 1e-6 { return false }
    return true
  }

  /// Cos, x and y of one length-1 direction outside the catalogue: three dot products.
  private static func coordinates(of v: ArraySlice<Float>, gaze: [Float], right: [Float], up: [Float]) -> (Float, Float, Float) {
    var c: Float = 0, x: Float = 0, y: Float = 0
    let n = vDSP_Length(v.count)
    guard v.count == gaze.count, v.count == right.count, v.count == up.count else { return (0, 0, 0) }
    v.withUnsafeBufferPointer { p in
      guard let base = p.baseAddress else { return }
      vDSP_dotpr(base, 1, gaze, 1, &c, n)
      vDSP_dotpr(base, 1, right, 1, &x, n)
      vDSP_dotpr(base, 1, up, 1, &y, n)
    }
    return (c, x, y)
  }

  /// Gaze at a continuous stage: 0–1 your own token, 1 to layers + 1 the layers, layers + 1 to layers + 3
  /// the readout, then riding the latch to layers + 4.
  private func gazeAt(_ s: Double, layers: Int) -> [Float] {
    let h = camera.headings
    guard h.count >= 2 else { return h.first ?? [] }
    if s <= 1 { return h[0] }
    if s < Double(layers + 1) {
      let i = max(0, min(Int(s - 1), h.count - 2))
      return Gaze.slerp(h[i], h[i + 1], Float(min(1, max(0, s - 1 - Double(i)))))
    }
    if s <= Double(layers + 3) || camera.target.count != h[0].count { return h[h.count - 1] }
    let t = min(1, s - Double(layers + 3))
    return Gaze.slerp(h[h.count - 1], camera.target, Float(t * t * (3 - 2 * t)))
  }
}
