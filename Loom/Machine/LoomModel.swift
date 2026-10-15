import Foundation
import LoomEngine
import Observation
import OSLog
import SwiftUI

private let log = Logger(subsystem: "com.utkarsh.loom", category: "Model")

enum AppTab: String, CaseIterable, Identifiable {
  case loom, sphere, galaxy, bench
  var id: String { rawValue }
}

enum AppSheet: String, Identifiable {
  case fieldNotes, settings, offlineAssistant
  var id: String { rawValue }
}

/// A part of the machine opened for inspection.
enum Inspector: Identifiable, Hashable {
  case punch
  case floor(Int)
  case readout
  var id: String {
    switch self {
    case .punch: "punch"
    case .floor(let f): "floor\(f)"
    case .readout: "readout"
    }
  }
}

/// Which model runs the pipeline.
enum EngineKind: String, CaseIterable, Identifiable {
  /// Loom's own 1.4M-parameter model, trained on Cipher's lines: everything, including fine-tuning, runs here.
  case tiny
  /// Qwen3-4B-Instruct-2507, 4-bit, on this iPhone's GPU.
  case qwen
  var id: String { rawValue }
  var title: String { self == .tiny ? "Loom 1.4M" : "Qwen3 4B" }
}

/// How fast the loom weaves. The pass itself takes about a millisecond; anything slower is for watching.
enum Pace: Int, CaseIterable {
  case step, slow, fast, instant
  var title: String { ["Step", "Slow", "Fast", "Instant"][rawValue] }
  /// Seconds per card.
  var seconds: Double { [0, 2.4, 0.35, 0][rawValue] }
}


@MainActor @Observable final class LoomModel {
  enum Phase: Equatable { case loading, ready, failed(String) }
  var phase: Phase = .loading
  @ObservationIgnored private(set) var library: Library?
  @ObservationIgnored let haptics = Haptics()
  /// Qwen3-4B, loaded onto the GPU when it's the engine.
  let qwen = QwenRuntime()
  let offlineAssistant = LoomOfflineAssistant()
  private(set) var offlineAssistantOpen = false
  @ObservationIgnored private var offlineSession = UUID()
  @ObservationIgnored private var suspended = false
  @ObservationIgnored private var residencyTask: Task<Void, Never>?

  // MARK: Engine
  var engine: EngineKind = .tiny {
    didSet { if oldValue != engine { engineChanged() } }
  }
  var isQwen: Bool { engine == .qwen }

  // MARK: Navigation
  var tab: AppTab = AppTab(rawValue: UserDefaults.standard.string(forKey: "tab") ?? "") ?? .loom {
    didSet { UserDefaults.standard.set(tab.rawValue, forKey: "tab") }
  }
  var sheet: AppSheet?
  var inspector: Inspector?
  var consoleOpen = false
  /// The first-person sphere is open (full screen).
  var sphereOpen = false
  /// Seconds per card while riding the sphere; overrides the pace knob.
  var rideSeconds: Double?
  /// A card someone tapped to latch onto, used instead of the generator's draw.
  private(set) var forcedNext: Int?
  /// Set by a DEBUG launch argument to open the console on a page.
  var consolePage: ConsolePage?
  var activeKnob: KnobID?

  // MARK: Prompt
  var prompt = UserDefaults.standard.string(forKey: "prompt") ?? "The night was" {
    didSet { UserDefaults.standard.set(prompt, forKey: "prompt") }
  }
  /// A control token loaded after the opening newline, or nil.
  var cartridge: Int? { didSet { if oldValue != cartridge { restart() } } }

  // MARK: Decode knobs (they change the draw, not the pass)
  var temperature = 0.7 { didSet { redraw() } }
  var topK = 0.0 { didSet { redraw() } }
  var topP = 0.9 { didSet { redraw() } }
  var minP = 0.0 { didSet { redraw() } }
  var repetition = 1.1 { didSet { redraw() } }
  var length = 24.0
  var seed = 6.0 { didSet { if oldValue != seed { restartRNG() } } }
  var pace = Double(Pace.slow.rawValue)
  var bias: [Int: Double] = [:] { didSet { redraw() } }

  // MARK: Model knobs (they change the pass)
  var bitsIndex = 0.0 { didSet { if oldValue != bitsIndex { recompose() } } }
  var ageIndex = 3.0 { didSet { if oldValue != ageIndex { recompose() } } }
  var context = 128.0 { didSet { if oldValue != context { rerun() } } }
  var depth = 4.0 { didSet { if oldValue != depth { rerun() } } }
  var kvCache = 1.0
  var muted: Set<HeadID> = [] { didSet { rerun() } }
  var bypass: Set<Int> = [] { didSet { rerun() } }
  var clamps: [Clamp] = [] { didSet { rerun() } }

  // MARK: Style knobs
  var scripture = 0.0 { didSet { if oldValue != scripture { recompose() } } }
  var bard = 0.0 { didSet { if oldValue != bard { recompose() } } }
  var gothic = 0.0 { didSet { if oldValue != gothic { recompose() } } }
  var twain = 0.0 { didSet { if oldValue != twain { recompose() } } }
  var almanac = 0.0 { didSet { if oldValue != almanac { recompose() } } }
  var taughtStrength = 1.0 { didSet { if oldValue != taughtStrength { recompose() } } }
  var taught: Adapter? { didSet { recompose() } }
  var steerKey = "lang:es" { didSet { if oldValue != steerKey { rerun() } } }
  var steerFloor = 1.0 { didSet { if oldValue != steerFloor { rerun() } } }
  var steerStrength = 0.0 { didSet { if oldValue != steerStrength { rerun() } } }

  // MARK: Teach
  var teachText = UserDefaults.standard.string(forKey: "teachText") ?? "" {
    didSet { UserDefaults.standard.set(teachText, forKey: "teachText") }
  }
  var teachRate = 0.003
  var teachRank = 8.0
  var teachSteps = 60.0
  var teachBatch = 4.0
  var teachFull = 0.0
  private(set) var teaching = false
  private(set) var teachProgress: [TeachProgress] = []
  private(set) var teachBefore: (train: Double, heldOut: Double?)?
  private(set) var teachLines = (train: 0, heldOut: 0)
  /// A fully fine-tuned loom, used in place of the base until it is put away.
  var fullTuned: ModelWeights? { didSet { recompose() } }
  @ObservationIgnored private var teachTask: Task<Void, Never>?
  @ObservationIgnored private var teacher: Teacher?

  // MARK: The machine
  private(set) var weights: ModelWeights?
  private(set) var weave = Weave()
  /// The draw waiting for the next card, recomputed live as decode knobs turn.
  private(set) var pending: Draw?
  private(set) var playing = false
  /// Animation progress through the pass of the newest thread.
  /// The stage of the pass the slowed replay is showing for the newest position:
  /// 0 tokenize · 1 embedding · 2…5 layers 1–4 · 6 prediction · 7 sample · 8 idle.
  var stage: Double = 8
  /// The thread being inspected (index into `weave.threads`); nil follows the newest.
  var selectedThread: Int? { didSet { if let s = selectedThread { fillLens(s) } } }
  private(set) var composing = false

  @ObservationIgnored private var rng = SplitMix64(seed: 6)
  @ObservationIgnored private var weaveTask: Task<Void, Never>?
  @ObservationIgnored private var passTask: Task<Void, Never>?
  @ObservationIgnored private var composeTask: Task<Void, Never>?
  @ObservationIgnored private var passGeneration = 0
  @ObservationIgnored private var cache: KVCache?
  /// While true, knob changes don't each start a pass (an engine switch resets several at once).
  @ObservationIgnored private var quiet = false
  /// The small model's catalogue for the current weights (rebuilt when they change).
  @ObservationIgnored private var tinyCatalogue: (key: Int, catalogue: Catalogue)?

  // MARK: Lifecycle

  func load() async {
    guard library == nil else { return }
    do {
      let lib = try await Task.detached(priority: .userInitiated) { try Library.load() }.value
      library = lib
      weights = lib.base
      let saved = UserDefaults.standard.string(forKey: "engine").flatMap(EngineKind.init(rawValue:))
      quiet = true
      engine = qwen.isAvailable ? (saved ?? .qwen) : .tiny
      quiet = false
      applyEngineDefaults()
      phase = .ready
      if ProcessInfo.processInfo.arguments.contains("-resetPrompt") { prompt = "The night was" }
      restart()
      applyLaunchArguments()
      if isQwen { await startQwen() }
    } catch {
      log.error("load failed: \(error.localizedDescription)")
      phase = .failed(error.localizedDescription)
    }
  }

  var tokenizer: Tokenizer? { library?.tokenizer }
  var config: ModelConfig { isQwen ? qwen.config : weights?.config ?? ModelConfig() }
  /// Parameters of the model running the pipeline.
  var parameterCount: Int { isQwen ? qwen.parameterCount : weights?.parameterCount ?? 1_377_408 }
  /// The engine can run a pass now (Qwen may still be loading).
  var engineReady: Bool { isQwen ? qwen.isReady : weights != nil }
  /// The measured directions the Steer knob can push along, for the running engine.
  var steering: Steering? { isQwen ? qwen.steering : library?.steering }

  // MARK: The replay's stages
  // 0 tokenize · 1 embedding · 2 … layers + 1 the layers · layers + 2 prediction · layers + 3 sample · layers + 4 idle.
  var predictStage: Double { Double(config.layers + 2) }
  var sampleStage: Double { Double(config.layers + 3) }
  var idleStage: Double { Double(config.layers + 4) }
  var replaying: Bool { stage < idleStage }
  var bits: Int { Quantizer.choices[Int(bitsIndex)] }
  var paceValue: Pace { Pace(rawValue: Int(pace)) ?? .slow }

  var sampler: SamplerSettings {
    var s = SamplerSettings()
    s.temperature = temperature
    s.topK = Int(topK)
    s.topP = topP
    s.minP = minP
    s.repetition = repetition
    s.bias = bias
    return s
  }

  var passSettings: PassSettings {
    var k = RunKnobs()
    k.muted = muted
    k.bypass = bypass
    k.depth = Int(depth) < config.layers ? Int(depth) : nil
    k.clamp = clamps
    if steerStrength != 0, let steering {
      let floor = min(Int(steerFloor), config.layers - 1)
      let vec = steering.vector(steerKey, floor: floor)
      if !vec.isEmpty { k.steer = [Steer(layer: floor, vector: vec, strength: Float(steerStrength))] }
    }
    return PassSettings(knobs: k, context: Int(context), kvCache: kvCache > 0.5)
  }

  var voiceStrengths: [Voice: Double] {
    [.scripture: scripture, .bard: bard, .gothic: gothic, .twain: twain, .almanac: almanac]
  }

  // MARK: Weaving

  /// The ids a fresh weave starts from: newline, the cartridge, then the prompt's cards.
  func promptIDs() -> [Int] {
    if isQwen { return Array(qwen.encode(prompt).prefix(config.ctx - 1)) }
    guard let tok = tokenizer else { return [Tokenizer.newline] }
    var ids = [Tokenizer.newline]
    if let c = cartridge { ids.append(c) }
    ids += tok.encode(Tokenizer.normalize(prompt).trimmingCharacters(in: .newlines))
    return Array(ids.prefix(config.ctx - 1))
  }

  /// Clears the cloth and prefills the prompt.
  func restart() {
    stop()
    var w = Weave()
    w.ids = promptIDs()
    w.promptEnd = w.ids.count
    weave = w
    selectedThread = nil
    restartRNG()
    rerun()
  }

  private func restartRNG() {
    rng = SplitMix64(seed: UInt64(seed))
    redraw()
  }

  /// Weave until the length is reached or a line ends.
  func play() {
    guard phase == .ready else { return }
    if weave.finished || weave.ids != Array(promptIDs() + weave.woven) { restart() }
    if paceValue == .step {
      step()
      return
    }
    playing = true
    haptics.transient(intensity: 0.55, sharpness: 0.5)
    weaveTask?.cancel()
    weaveTask = Task { [weak self] in
      guard let self else { return }
      while self.playing, !Task.isCancelled, self.canWeave {
        await self.weaveOne(animated: self.paceValue != .instant)
        if self.paceValue == .instant { await Task.yield() }
      }
      self.playing = false
    }
  }

  /// Rides the sphere: weaves continuously at `rideSeconds`, whatever the pace knob says.
  func ride() {
    guard phase == .ready else { return }
    if weave.finished || weave.ids != Array(promptIDs() + weave.woven) { restart() }
    playing = true
    weaveTask?.cancel()
    weaveTask = Task { [weak self] in
      guard let self else { return }
      while self.playing, !Task.isCancelled, self.canWeave {
        await self.weaveOne(animated: true)
      }
      self.playing = false
    }
  }

  /// The card the next latch will land on.
  var latchTarget: Int? { forcedNext ?? pending?.chosen }

  /// Latch onto a card by hand: the next woven card is this one, whatever the generator drew.
  func latch(_ token: Int) {
    forcedNext = token
    haptics.transient(intensity: 0.8, sharpness: 0.6)
    if !playing, canWeave {
      weaveTask?.cancel()
      weaveTask = Task { [weak self] in await self?.weaveOne(animated: true) }
    }
  }

  func stop() {
    playing = false
    weaveTask?.cancel()
    weaveTask = nil
    if stage < idleStage { stage = idleStage }
  }

  func step() {
    guard !playing, canWeave else { return }
    weaveTask?.cancel()
    weaveTask = Task { [weak self] in await self?.weaveOne(animated: true) }
  }

  var canWeave: Bool {
    !weave.finished && weave.woven.count < Int(length) && weave.ids.count < config.ctx && pending != nil
  }

  /// One card: show the drop, punch the chosen card, feed it back and run its pass.
  private func weaveOne(animated: Bool) async {
    guard pending != nil, engineReady else { return }
    let seconds = rideSeconds ?? paceValue.seconds
    if animated && seconds > 0 {
      // The pass for the newest thread has already run; the animation replays it in order, slowed.
      let layers = config.layers
      let tick = seconds / Double(layers + 4)
      for s in stride(from: 1.0, through: sampleStage, by: 1.0) {
        withAnimation(.easeInOut(duration: tick * 8 / 9)) { stage = s }
        let layer = Int(s) - 2
        if layer >= 0 && layer < layers && (layers <= 8 || layer % 6 == 5) { haptics.transient(intensity: 0.18, sharpness: 0.8) }
        try? await Task.sleep(for: .seconds(tick))
        if Task.isCancelled { return }
      }
    }
    // Commit the draw: the generator moves on, the card is punched. A hand-picked latch wins.
    guard let drawn = pending else { return }
    let draw = forcedNext.map { drawn.choosing($0) } ?? drawn
    forcedNext = nil
    _ = rng.nextUnit()
    let surprise = min(1, draw.surprise / 12)
    haptics.transient(intensity: Float(0.35 + 0.6 * surprise), sharpness: Float(0.3 + 0.5 * surprise))
    var next = weave
    next.ids.append(draw.chosen)
    next.stitches.append(Stitch(position: next.ids.count - 1, draw: draw))
    if ends(draw.chosen) || next.woven.count >= Int(length) || next.ids.count >= config.ctx {
      next.finished = ends(draw.chosen) || next.ids.count >= config.ctx
    }
    if isQwen {
      await runQwen(next, animated: animated && seconds > 0)
      return
    }
    guard let w = weights else { return }
    let settings = passSettings
    let useCache = settings.kvCache && cache != nil && next.ids.count <= settings.context && next.windowStart == 0
    passGeneration += 1
    let generation = passGeneration
    let cacheIn = cache
    let ids = next.ids
    let clock = Date()
    let result = await Task.detached(priority: .userInitiated) { () -> (PassTrace, KVCache, Int) in
      if useCache, var kv = cacheIn {
        let t = Forward.run(w, ids: ids, start: ids.count - 1, cache: &kv, knobs: settings.knobs)
        return (t, kv, 0)
      }
      let start = max(0, ids.count - settings.context)
      let (t, kv) = Forward.prefill(w, ids: Array(ids[start...]), knobs: settings.knobs)
      return (t, kv, start)
    }.value
    guard generation == passGeneration else { return }
    let (trace, kv, start) = result
    cache = kv
    if useCache {
      next.threads += Loomwork.threads(from: trace, ids: ids, weights: w)
    } else {
      next.windowStart = start
      next.threads = Loomwork.threads(from: trace, ids: Array(ids[start...]), weights: w)
    }
    let V = w.config.vocab
    next.lastLogits = Array(trace.logits[((trace.rows - 1) * V)...])
    next.macsLastStep = trace.macs
    next.macsTotal += trace.macs
    next.secondsLastStep = Date().timeIntervalSince(clock)
    stage = animated && seconds > 0 ? 0 : idleStage
    weave = next
    redraw()
    if animated && seconds > 0 {
      withAnimation(.easeOut(duration: 0.25)) { stage = 1 }
    }
    if weave.finished || !canWeave { playing = false }
  }

  /// Re-runs the whole window with the current pass knobs (any knob that changes the pass).
  func rerun() {
    guard !quiet else { return }
    if isQwen { rerunQwen(); return }
    guard let w = weights, !weave.ids.isEmpty else { return }
    passTask?.cancel()
    passGeneration += 1
    let generation = passGeneration
    let settings = passSettings
    let ids = weave.ids
    passTask = Task { [weak self] in
      let start = max(0, ids.count - settings.context)
      let result = await Task.detached(priority: .userInitiated) { () -> (PassTrace, KVCache, [Thread]) in
        let window = Array(ids[start...])
        let (t, kv) = Forward.prefill(w, ids: window, knobs: settings.knobs)
        return (t, kv, Loomwork.threads(from: t, ids: window, weights: w))
      }.value
      guard let self, generation == self.passGeneration, !Task.isCancelled else { return }
      let (trace, kv, threads) = result
      self.cache = kv
      var next = self.weave
      next.windowStart = start
      next.threads = threads
      let V = w.config.vocab
      next.lastLogits = Array(trace.logits[((trace.rows - 1) * V)...])
      next.macsLastStep = trace.macs
      next.macsTotal += trace.macs
      self.weave = next
      self.redraw()
    }
  }

  /// Recomputes the pending draw from the newest logits (decode knobs only; the pass is unchanged).
  func redraw() {
    guard !weave.lastLogits.isEmpty else { pending = nil; return }
    var peek = rng
    pending = Sampler.draw(logits: weave.lastLogits[...], context: weave.window.map { $0 }, settings: sampler, rng: &peek)
  }

  /// Builds the weights the pass uses: the chosen age, rounded to the precision knob, plus the adapters.
  func recompose() {
    guard let lib = library else { return }
    let age = fullTuned ?? lib.ages[min(Int(ageIndex), lib.ages.count - 1)].weights
    var mix: [(Adapter, Float)] = []
    for (i, voice) in Voice.allCases.enumerated() {
      let s = voiceStrengths[voice] ?? 0
      if s != 0, i < lib.adapters.count { mix.append((lib.adapters[i], Float(s))) }
    }
    if let t = taught, taughtStrength != 0 { mix.append((t, Float(taughtStrength))) }
    let bits = self.bits
    let adapters = mix
    composeTask?.cancel()
    composing = true
    composeTask = Task { [weak self] in
      try? await Task.sleep(for: .milliseconds(40))
      if Task.isCancelled { return }
      let w = await Task.detached(priority: .userInitiated) { Workshop.compose(base: age, bits: bits, adapters: adapters) }.value
      guard let self, !Task.isCancelled else { return }
      self.weights = w
      self.composing = false
      if !self.isQwen { self.rerun() }
    }
  }

  // MARK: Teach (a real fine-tune on this phone)

  var teachSequences: [[Int]] {
    guard let tok = tokenizer else { return [] }
    return teachText.split(whereSeparator: \.isNewline)
      .map { $0.trimmingCharacters(in: .whitespaces) }
      .filter { !$0.isEmpty }
      .map { [Tokenizer.newline] + tok.encode(Tokenizer.normalize($0)) + [Tokenizer.newline] }
  }

  func startTeaching() {
    guard let lib = library, !teaching else { return }
    let lines = teachSequences
    guard lines.count >= 2 else { return }
    var s = TeachSettings()
    s.learningRate = teachRate
    s.rank = Int(teachRank)
    s.steps = Int(teachSteps)
    s.batch = Int(teachBatch)
    s.full = teachFull > 0.5
    let base = lib.ages.last!.weights
    let t = Teacher(base: base, lines: lines, settings: s, seed: UInt64(seed))
    teacher = t
    teachLines = (t.trainLineCount, t.heldOutLineCount)
    teachProgress = []
    teaching = true
    haptics.transient(intensity: 0.7, sharpness: 0.4)
    teachTask = Task { [weak self] in
      let before = await Task.detached(priority: .userInitiated) { () -> (Double, Double?) in
        let train = Backprop.loss(base, sequences: lines)
        return (train, nil)
      }.value
      self?.teachBefore = (before.0, before.1)
      for _ in 0..<s.steps {
        if Task.isCancelled { break }
        let p = await Task.detached(priority: .userInitiated) { t.step() }.value
        guard let self else { return }
        self.teachProgress.append(p)
        if p.step % 10 == 0 { self.haptics.transient(intensity: 0.2, sharpness: 0.9) }
      }
      guard let self else { return }
      self.teaching = false
      self.haptics.transient(intensity: 0.8, sharpness: 0.5)
    }
  }

  func stopTeaching() {
    teachTask?.cancel()
    teaching = false
  }

  /// Puts what Teach learned into the loom: the adapter as a sixth fader, or the fully tuned weights.
  func keepTaught() {
    guard let t = teacher else { return }
    if t.settings.full {
      fullTuned = t.tuned
    } else {
      taughtStrength = 1
      taught = t.adapter
    }
    haptics.transient(intensity: 0.8, sharpness: 0.4)
  }

  func discardTaught() {
    taught = nil
    fullTuned = nil
  }

  // MARK: Side-by-side weaves (Bench)

  /// Weaves `count` cards from the current prompt with the given weights and knobs, off the main thread.
  func preview(weights w: ModelWeights, knobs: RunKnobs = RunKnobs(), count: Int = 28) async -> String {
    let ids = promptIDs()
    let s = sampler
    let seed = UInt64(self.seed)
    let tok = tokenizer
    return await Task.detached(priority: .utility) { () -> String in
      var seq = ids
      var rng = SplitMix64(seed: seed)
      var (t, cache) = Forward.prefill(w, ids: seq, knobs: knobs)
      let V = w.config.vocab
      for _ in 0..<count {
        let d = Sampler.draw(logits: t.logits[((t.rows - 1) * V)...], context: seq, settings: s, rng: &rng)
        seq.append(d.chosen)
        if d.chosen == Tokenizer.newline || seq.count >= w.config.ctx { break }
        t = Forward.run(w, ids: seq, start: seq.count - 1, cache: &cache, knobs: knobs)
      }
      return tok?.decode(Array(seq[ids.count...])).trimmingCharacters(in: .newlines) ?? ""
    }.value
  }

  /// Generates `count` tokens from the current prompt with Qwen and the given knobs (its own keys and
  /// values, so the pipeline's pass is untouched).
  func previewQwen(knobs: RunKnobs = RunKnobs(), count: Int = 28) async -> String {
    guard !offlineAssistantOpen, !suspended, qwen.isReady else { return "" }
    let ids = promptIDs()
    guard let out = try? await qwen.worker.generate(
      ids: ids, count: count, knobs: knobs, settings: sampler, seed: UInt64(seed), stops: qwen.stopIDs)
    else { return "" }
    return qwen.decode(out).trimmingCharacters(in: .whitespacesAndNewlines)
  }

  // MARK: Interventions

  func toggleMute(_ head: HeadID) {
    if muted.contains(head) { muted.remove(head) } else { muted.insert(head) }
    haptics.transient(intensity: 0.6, sharpness: 0.6)
  }

  func toggleBypass(_ floor: Int) {
    if bypass.contains(floor) { bypass.remove(floor) } else { bypass.insert(floor) }
    haptics.transient(intensity: 0.6, sharpness: 0.6)
  }

  func setClamp(floor: Int, neuron: Int, value: Double?) {
    var c = clamps.filter { !($0.layer == floor && $0.neuron == neuron) }
    if let value { c.append(Clamp(layer: floor, neuron: neuron, value: Float(value))) }
    clamps = c
  }

  func clamp(floor: Int, neuron: Int) -> Double? {
    clamps.first { $0.layer == floor && $0.neuron == neuron }.map { Double($0.value) }
  }

  func toggleBan(_ token: Int) {
    if bias[token] != nil { bias[token] = nil } else { bias[token] = -.infinity }
    haptics.transient(intensity: 0.7, sharpness: 0.5)
  }

  func boost(_ token: Int) {
    bias[token] = (bias[token] ?? 0) == 2 ? nil : 2
    haptics.transient(intensity: 0.5, sharpness: 0.7)
  }

  /// Puts every knob that changes the pass or the style back, leaving the cloth.
  func resetKnobs(on page: ConsolePage) {
    withAnimation(.smooth(duration: 0.5)) {
      for spec in KnobSpec.page(page, for: self) { self[keyPath: spec.keyPath] = spec.defaultValue }
      if page == .model { muted = []; bypass = []; clamps = [] }
      if page == .decode { bias = [:] }
    }
    haptics.transient(intensity: 0.6, sharpness: 0.5)
  }

  /// Unplugs every adapter, steer, cartridge and surgery: the plain base loom.
  func baseOnly() {
    quiet = true
    scripture = 0; bard = 0; gothic = 0; twain = 0; almanac = 0; steerStrength = 0
    muted = []; bypass = []; clamps = []; bitsIndex = 0; ageIndex = 3
    depth = Double(config.layers); context = Double(config.ctx)
    quiet = false
    rerun()
    cartridge = nil
    haptics.transient(intensity: 0.7, sharpness: 0.4)
  }

  // MARK: Reading the machine

  /// The thread the panels describe: the selected one, else the newest.
  var focusThread: Int? {
    guard !weave.threads.isEmpty else { return nil }
    if let s = selectedThread, s < weave.threads.count { return s }
    return weave.threads.count - 1
  }

  func text(_ token: Int) -> String { isQwen ? qwen.text(token) : tokenizer?.text(token) ?? "?" }

  /// A token as shown on a card: newline as ↵, and a leading space as ▁ (SentencePiece's mark for
  /// "starts a word"), so you can see which cards begin words and which continue them.
  func cardText(_ token: Int) -> String {
    if isQwen { return qwen.card(token) }
    let t = text(token)
    if t == "\n" { return "↵" }
    if t == " " { return "␣" }
    return t.hasPrefix(" ") ? "▁" + t.dropFirst() : t
  }

  func startsWord(_ token: Int) -> Bool { text(token).hasPrefix(" ") }

  /// A control token: the small model's language tags, or Qwen's chat-template markers.
  func isControl(_ token: Int) -> Bool { isQwen ? qwen.isSpecial(token) : tokenizer?.isControl(token) ?? false }

  /// Whether this token ends generation: a line break for the small model; for Qwen the end of its reply
  /// (or, raw, a line break).
  func ends(_ token: Int) -> Bool { isQwen ? qwen.ends(token) : token == Tokenizer.newline }

  /// Running text of a run of tokens, control tokens left out.
  func decode(_ ids: some Sequence<Int>) -> String {
    if isQwen { return qwen.decode(ids) }
    guard let tok = tokenizer else { return "" }
    return tok.decode(ids.filter { !tok.isControl($0) })
  }

  /// The tokens the embedding map and the sphere draw, with their directions.
  var catalogue: Catalogue? {
    if isQwen { return qwen.catalogue }
    guard let w = weights, let g = library?.galaxy else { return nil }
    let key = w.wte.count &+ Int(w.wte[w.config.d * 500].bitPattern) &+ Int(w.wte[w.config.d * 3000 + 7].bitPattern)
    if let c = tinyCatalogue, c.key == key { return c.catalogue }
    let c = Catalogue.tiny(weights: w, galaxy: g)
    tinyCatalogue = (key, c)
    return c
  }

  /// A token's embedding direction (length 1), for the sphere; Qwen's may still be on its way.
  func direction(_ token: Int) -> ArraySlice<Float>? {
    if isQwen { return qwen.direction(token) }
    guard let w = weights, token >= 0, token < w.config.vocab else { return nil }
    let d = w.config.d
    var v = Array(w.wte[(token * d)..<((token + 1) * d)])
    Gaze.normalize(&v)
    return v[...]
  }

  /// The final norm's gains: the gaze is the newest vector through them.
  var finalGain: [Float] { isQwen ? qwen.finalGain : weights?.gf ?? [] }

  // MARK: Launch arguments (DEBUG only), for screenshots and UI tests

  private func applyLaunchArguments() {
    // Qwen's self-check runs in every build, so the Release build that ships is the one checked.
    if ProcessInfo.processInfo.arguments.contains("-verifyQwen") {
      Task {
        while !self.qwen.isReady { try? await Task.sleep(for: .milliseconds(200)) }
        await QwenCheck.run(self)
      }
    }
    #if DEBUG
      let args = ProcessInfo.processInfo.arguments
      func value(_ flag: String) -> String? {
        guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
        return args[i + 1]
      }
      if let t = value("-tab"), let tab = AppTab(rawValue: t) { self.tab = tab }
      if let e = value("-engine"), let kind = EngineKind(rawValue: e), kind != engine, kind == .tiny || qwen.isAvailable { engine = kind }
      if let c = value("-chat") { qwen.chat = c == "1" }
      if let p = value("-prompt") { prompt = p; restart() }
      if args.contains("-instant") { pace = Double(Pace.instant.rawValue) }
      if args.contains("-weave") { Task { try? await Task.sleep(for: .milliseconds(300)); self.play() } }
      if args.contains("-knobs") { consoleOpen = true }
      // -knob scripture=1.2 sets any knob by its id.
      for (i, a) in args.enumerated() where a == "-knob" && i + 1 < args.count {
        let parts = args[i + 1].split(separator: "=")
        if parts.count == 2, let id = KnobID(rawValue: String(parts[0])), let v = Double(parts[1]),
          let spec = KnobSpec.spec(id, for: self)
        {
          self[keyPath: spec.keyPath] = v
        }
      }
      if let page = value("-page") { consolePage = ConsolePage(rawValue: page) }
      if let f = value("-inspect") {
        Task {
          try? await Task.sleep(for: .milliseconds(isQwen ? 9000 : 1500))
          switch f {
          case "punch": self.inspector = .punch
          case "readout": self.inspector = .readout
          default: self.inspector = .floor(Int(f) ?? 0)
          }
        }
      }
      if let s = value("-sheet") { sheet = AppSheet(rawValue: s) }
      if args.contains("-sphere") {
        Task {
          try? await Task.sleep(for: .milliseconds(500))
          self.sphereOpen = true
          if args.contains("-ride") {
            try? await Task.sleep(for: .milliseconds(700))
            self.ride()
          }
        }
      }
      if args.contains("-teach") {
        teachText = TeachPanel.example
        tab = .bench
        Task {
          try? await Task.sleep(for: .milliseconds(600))
          self.startTeaching()
        }
      }
    #endif
  }

  // MARK: Qwen

  /// One Qwen pass for the cloth `next` (its newest token just chosen): only the new position when the
  /// kept keys and values still fit, else the whole window.
  private func runQwen(_ next: Weave, animated: Bool) async {
    guard !offlineAssistantOpen, !suspended, !Task.isCancelled else { return }
    var next = next
    let settings = passSettings
    passGeneration += 1
    let generation = passGeneration
    let start = max(0, next.ids.count - settings.context)
    let window = Array(next.ids[start...])
    let reuse = settings.kvCache && start == next.windowStart
    let pass: QwenPass
    do {
      pass = try await qwen.worker.run(ids: window, knobs: settings.knobs, reuse: reuse)
    } catch {
      log.error("Qwen pass failed: \(error.localizedDescription)")
      playing = false
      return
    }
    guard generation == passGeneration else { return }
    if pass.start > 0 {
      // Older positions keep a few numbers of each vector; the newest keeps everything.
      if let last = next.threads.indices.last { next.threads[last].trim(keeping: QwenWorker.keptDims) }
      next.threads += pass.threads
    } else {
      next.windowStart = start
      next.threads = pass.threads
    }
    next.lastLogits = pass.lastLogits
    next.macsLastStep = pass.macs
    next.macsTotal += pass.macs
    next.secondsLastStep = pass.seconds
    stage = animated ? 0 : idleStage
    weave = next
    redraw()
    if animated { withAnimation(.easeOut(duration: 0.25)) { stage = 1 } }
    if weave.finished || !canWeave { playing = false }
  }

  /// Re-runs the window with Qwen once a turning knob settles.
  private func rerunQwen() {
    guard !offlineAssistantOpen, !suspended, qwen.isReady, !weave.ids.isEmpty else { return }
    passTask?.cancel()
    passGeneration += 1
    let generation = passGeneration
    let settings = passSettings
    let ids = weave.ids
    passTask = Task { [weak self] in
      try? await Task.sleep(for: .milliseconds(120))
      guard let self, !Task.isCancelled else { return }
      let start = max(0, ids.count - settings.context)
      guard let pass = try? await self.qwen.worker.run(ids: Array(ids[start...]), knobs: settings.knobs, reuse: false),
        generation == self.passGeneration
      else { return }
      var next = self.weave
      next.windowStart = start
      next.threads = pass.threads
      next.lastLogits = pass.lastLogits
      next.macsLastStep = pass.macs
      next.macsTotal += pass.macs
      next.secondsLastStep = pass.seconds
      self.weave = next
      self.redraw()
    }
  }

  /// Loads Qwen if it isn't, then runs the prompt.
  func startQwen() async {
    guard !offlineAssistantOpen, !suspended, isQwen, qwen.isAvailable else { return }
    if !qwen.isReady {
      await qwen.load()
      guard isQwen, qwen.isReady else { return }
    }
    restart()
  }

  /// The logit lens of an earlier position, read back from the kept pass (Qwen keeps only the newest's).
  func fillLens(_ index: Int) {
    guard isQwen, index < weave.threads.count, weave.threads[index].lens.isEmpty else { return }
    let ids = Array(weave.window)
    let count = weave.ids.count
    Task {
      guard let lens = try? await qwen.worker.lens(position: index, ids: ids), weave.ids.count == count,
        index < weave.threads.count
      else { return }
      weave.threads[index].lens = lens
    }
  }

  /// Knobs that name layers, heads, neurons or tokens belong to one model: put them back for the new one.
  private func applyEngineDefaults() {
    quiet = true
    muted = []; bypass = []; clamps = []; bias = [:]
    depth = Double(config.layers)
    context = Double(config.ctx)
    steerFloor = isQwen ? 15 : 1
    steerStrength = 0
    stage = idleStage
    quiet = false
  }

  private func engineChanged() {
    guard !quiet else { return }
    UserDefaults.standard.set(engine.rawValue, forKey: "engine")
    stop()
    inspector = nil
    selectedThread = nil
    applyEngineDefaults()
    if cartridge != nil { quiet = true; cartridge = nil; quiet = false }
    weave = Weave()
    pending = nil
    haptics.transient(intensity: 0.7, sharpness: 0.4)
    if isQwen {
      Task { await startQwen() }
    } else {
      restart()
    }
  }

  /// The app went to the background: give Qwen's 2.3 GB back.
  func suspend() {
    suspended = true
    stop()
    passTask?.cancel()
    passGeneration += 1
    let previous = residencyTask
    residencyTask = Task {
      await previous?.value
      await offlineAssistant.unload(reason: "Interrupted when Loom left the foreground")
      await qwen.unload()
    }
  }

  /// Back in front: reload Qwen and rebuild the pass for the cloth on screen.
  func resume() {
    suspended = false
    Task {
      await residencyTask?.value
      guard !suspended else { return }
      if offlineAssistantOpen {
        await offlineAssistant.prepare(context: offlineContext())
        return
      }
      guard isQwen, qwen.isAvailable, !qwen.isReady else { return }
      await qwen.load()
      guard isQwen, qwen.isReady else { return }
      if weave.ids.isEmpty { restart() } else { rerun() }
    }
  }

  /// Explanations and tensor inspection use different checkpoints. Drain the specialist before
  /// loading the shared ordinary text model so only one set of GPU weights is resident.
  func prepareOfflineAssistant() async {
    let identity = UUID()
    offlineSession = identity
    offlineAssistantOpen = true
    stop()
    passTask?.cancel()
    passGeneration += 1
    await residencyTask?.value
    await qwen.unload()
    guard offlineSession == identity, offlineAssistantOpen, !suspended else { return }
    await offlineAssistant.prepare(context: offlineContext())
  }

  func offlineAssistantDidClose() {
    guard offlineAssistantOpen else { return }
    let identity = offlineSession
    Task {
      await offlineAssistant.unload(reason: "Explanation model memory released")
      await residencyTask?.value
      guard offlineSession == identity else { return }
      offlineAssistantOpen = false
      if !suspended { resume() }
    }
  }

  func receivedMemoryWarning() {
    stop()
    passTask?.cancel()
    passGeneration += 1
    let previous = residencyTask
    residencyTask = Task {
      await previous?.value
      await offlineAssistant.unload(reason: "Interrupted by iOS memory pressure; partial output retained")
      await qwen.unload()
    }
  }

  private func offlineContext() -> String {
    let facts: [String: Any] = [
      "instrument_model": isQwen ? QwenRuntime.name : EngineKind.tiny.title,
      "parameter_count": parameterCount,
      "layers": config.layers,
      "attention_heads_per_layer": config.heads,
      "hidden_width": config.d,
      "context_capacity_tokens": config.ctx,
      "layers_used": Int(depth),
      "muted_head_count": muted.count,
      "skipped_layers_zero_based": bypass.sorted(),
      "neuron_clamp_count": clamps.count,
      "steering_direction": steerKey,
      "steering_strength": steerStrength,
      "temperature": temperature,
      "top_p": topP,
      "prompt": String(prompt.prefix(1200)),
      "generated_text": String(decode(weave.woven).prefix(1200)),
      "measurement_boundary": "The instrument model supplies actual tensors. The explanation model does not inspect its own tensors and cannot infer missing measured values.",
      "control_definitions": [
        "muted_heads": "Selected attention heads contribute zero to their layer output.",
        "skipped_layers": "The residual vector passes through a skipped layer unchanged.",
        "layers_used": "Layers after the selected depth are skipped.",
        "steering": "A measured direction vector multiplied by strength is added to the residual after the chosen layer."
      ]
    ]
    guard let data = try? JSONSerialization.data(withJSONObject: facts, options: [.sortedKeys]),
      let text = String(data: data, encoding: .utf8) else { return "{}" }
    return text
  }
}
