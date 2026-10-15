import CoreHaptics
import Foundation
import QuartzCore

/// Core Haptics, from Gyre: transients for knob ticks and detents, and a knock for every woven card
/// that grows with how surprising the card was. A user switch and a strength live in Settings › Feel.
final class Haptics {
  private var engine: CHHapticEngine?
  private var hum: CHHapticAdvancedPatternPlayer?
  private var humRunning = false
  private var lastSend: CFTimeInterval = 0
  private var lastTransient: CFTimeInterval = 0
  let supported = CHHapticEngine.capabilitiesForHardware().supportsHaptics
  var enabled = UserDefaults.standard.object(forKey: "haptics") as? Bool ?? true {
    didSet {
      UserDefaults.standard.set(enabled, forKey: "haptics")
      if !enabled { stopHum() }
    }
  }
  /// Scales every pulse (Orbitals' strength setting).
  var strength = UserDefaults.standard.object(forKey: "hapticStrength") as? Float ?? 0.85 {
    didSet { UserDefaults.standard.set(strength, forKey: "hapticStrength") }
  }

  func start() {
    guard supported else { return }
    if engine == nil {
      guard let e = try? CHHapticEngine() else { return }
      e.playsHapticsOnly = true
      e.isAutoShutdownEnabled = false
      e.resetHandler = { [weak self] in
        Task { @MainActor in self?.recover() }
      }
      e.stoppedHandler = { [weak self] _ in
        Task { @MainActor in
          self?.hum = nil
          self?.humRunning = false
        }
      }
      engine = e
    }
    try? engine?.start()
  }

  func stop() {
    stopHum()
    engine?.stop()
  }

  private func recover() {
    hum = nil
    humRunning = false
    try? engine?.start()
  }

  private func stopHum() {
    if humRunning { try? hum?.stop(atTime: CHHapticTimeImmediate) }
    humRunning = false
  }

  /// Called every frame; parameters go out at most 40 times a second.
  func updateHum(intensity: Float, sharpness: Float) {
    guard enabled, supported, let engine else { return }
    let now = CACurrentMediaTime()
    if intensity < 0.02 {
      stopHum()
      return
    }
    if hum == nil {
      let event = CHHapticEvent(
        eventType: .hapticContinuous,
        parameters: [
          CHHapticEventParameter(parameterID: .hapticIntensity, value: 1),
          CHHapticEventParameter(parameterID: .hapticSharpness, value: 0.5),
        ], relativeTime: 0, duration: 30)
      guard let pattern = try? CHHapticPattern(events: [event], parameters: []),
        let player = try? engine.makeAdvancedPlayer(with: pattern)
      else { return }
      player.loopEnabled = true
      hum = player
    }
    if !humRunning {
      do {
        try hum?.start(atTime: CHHapticTimeImmediate)
        humRunning = true
      } catch {
        hum = nil
        return
      }
    }
    guard now - lastSend > 1.0 / 40 else { return }
    lastSend = now
    try? hum?.sendParameters(
      [
        CHHapticDynamicParameter(parameterID: .hapticIntensityControl, value: clamped(intensity, 0, 1), relativeTime: 0),
        CHHapticDynamicParameter(parameterID: .hapticSharpnessControl, value: clamped(sharpness, 0, 1) - 0.5, relativeTime: 0),
      ], atTime: CHHapticTimeImmediate)
  }

  /// A single tap. `minGap` throttles bursts (string pulls, field crackle).
  func transient(intensity: Float, sharpness: Float, minGap: CFTimeInterval = 0) {
    guard enabled, supported, let engine else { return }
    let now = CACurrentMediaTime()
    guard now - lastTransient >= max(minGap, 0.028) else { return }
    lastTransient = now
    let event = CHHapticEvent(
      eventType: .hapticTransient,
      parameters: [
        CHHapticEventParameter(parameterID: .hapticIntensity, value: clamped(intensity * strength, 0, 1)),
        CHHapticEventParameter(parameterID: .hapticSharpness, value: clamped(sharpness, 0, 1)),
      ], relativeTime: 0)
    if let pattern = try? CHHapticPattern(events: [event], parameters: []),
      let player = try? engine.makePlayer(with: pattern)
    {
      try? player.start(atTime: CHHapticTimeImmediate)
    }
  }
}

func clamped<T: Comparable>(_ v: T, _ lo: T, _ hi: T) -> T { min(max(v, lo), hi) }
