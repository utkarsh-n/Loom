import Foundation
import LoomEngine
import OSLog

private let log = Logger(subsystem: "com.utkarsh.loom", category: "QwenCheck")

/// `-verifyQwen` (in any build, so it can check the Release build that ships): runs the instrumented pass on the two reference prompts and compares it with MLX LM's
/// own forward and with the float32 trace computed on the Mac (`qwen-reference.json`, made by
/// Tools/qwen/precompute.py), then times decoding. Writes Documents/qwen-check.json.
@MainActor enum QwenCheck {
  static func run(_ model: LoomModel) async {
    var report: [String: Any] = ["started": ISO8601DateFormatter().string(from: Date())]
    defer { write(report) }
    guard let dir = QwenRuntime.directory,
      let data = try? Data(contentsOf: dir.appending(path: "qwen-reference.json")),
      let ref = try? JSONDecoder().decode(JSONValue.self, from: data)
    else {
      report["error"] = "no qwen-reference.json in the bundle"
      return
    }
    report["loadSeconds"] = model.qwen.loadSeconds
    report["footprintAfterLoadMB"] = footprintMB()
    let worker = model.qwen.worker
    for key in ["raw", "chat"] {
      guard let rec = ref["records"]?[key] else { continue }
      let ids = (rec["ids"]?.array ?? []).map { Int($0.number ?? 0) }
      var out: [String: Any] = ["tokens": ids.count]
      do {
        let pass = try await worker.run(ids: ids, knobs: RunKnobs(), reuse: false)
        let plain = try await worker.plainLogits(ids)
        let logits = pass.lastLogits
        let newest = pass.threads[pass.threads.count - 1]
        out["prefillSeconds"] = pass.seconds
        out["instrumentedVsPlainMaxAbs"] = zip(logits, plain).map { abs($0 - $1) }.max() ?? -1
        let mine = top(logits, 10)
        let theirs = (rec["top10"]?.array ?? []).compactMap { $0.array?.first?.number.map(Int.init) }
        out["top10"] = mine.map { [model.qwen.text($0), $0] as [Any] }
        out["top1Matches"] = mine.first == theirs.first
        out["top10Overlap"] = Set(mine).intersection(theirs).count
        var probe: [[Any]] = []
        for p in rec["logits_probe"]?.array ?? [] {
          guard let a = p.array, a.count == 2, let id = a[0].number.map(Int.init), let v = a[1].number else { continue }
          probe.append([id, v, Double(logits[id])])
        }
        out["logitsProbe ref,ours"] = probe
        out["logitsProbeMaxAbs"] = probe.map { abs(($0[1] as! Double) - ($0[2] as! Double)) }.max() ?? -1
        var att: [String: Double] = [:]
        for (name, row) in rec["attention"]?.object ?? [:] {
          let parts = name.dropFirst().split(separator: "H")
          guard parts.count == 2, let l = Int(parts[0]), let h = Int(parts[1]), l < newest.attention.count,
            h < newest.attention[l].count
          else { continue }
          let ours = newest.attention[l][h]
          att[name] = Double(zip(ours, row.floats).map { abs($0 - $1) }.max() ?? -1)
        }
        out["attentionMaxAbs"] = att
        let norms = newest.residual.map { Double(Gaze.norm($0)) }
        let refNorms = rec["resid_norm"]?.doubles ?? []
        out["residualNormMaxRelative"] = zip(norms, refNorms).map { abs($0 - $1) / max($1, 1e-6) }.max() ?? -1
        let refLens = (rec["lens_top1"]?.array ?? []).compactMap { $0.array?.first?.number.map(Int.init) }
        out["lensTop1Agreement"] = "\(zip(newest.lens.map(\.token), refLens).filter { $0 == $1 }.count) of \(refLens.count)"
        var mlp: [[Double]] = []
        for m in rec["mlp17_top"]?.array ?? [] {
          guard let a = m.array, a.count == 2, let n = a[0].number.map(Int.init), let v = a[1].number,
            newest.neurons.count > 17, n < newest.neurons[17].count
          else { continue }
          mlp.append([Double(n), v, Double(newest.neurons[17][n])])
        }
        out["mlp17 neuron,ref,ours"] = mlp
      } catch {
        out["error"] = error.localizedDescription
      }
      report[key] = out
    }

    // Decoding speed with the kept keys and values, the way the pipeline runs.
    do {
      var ids = model.qwen.encode("Write one line about the sea.")
      var seconds: [Double] = []
      var pass = try await worker.run(ids: ids, knobs: RunKnobs(), reuse: false)
      report["promptPrefillSeconds"] = pass.seconds
      for _ in 0..<16 {
        let next = top(pass.lastLogits, 1)[0]
        if QwenRuntime.stops.contains(next) { break }
        ids.append(next)
        pass = try await worker.run(ids: ids, knobs: RunKnobs(), reuse: true)
        seconds.append(pass.seconds)
      }
      report["greedy"] = model.qwen.decode(ids, keepSpecial: true)
      report["decodeSecondsPerToken"] = seconds.isEmpty ? -1 : seconds.reduce(0, +) / Double(seconds.count)
      report["footprintAfterDecodeMB"] = footprintMB()
    } catch {
      report["decodeError"] = error.localizedDescription
    }
    model.restart()
  }

  private static func top(_ logits: [Float], _ n: Int) -> [Int] {
    Array(logits.indices.sorted { logits[$0] > logits[$1] }.prefix(n))
  }

  private static func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
      $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
  }

  private static func write(_ report: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]) else { return }
    let url = URL.documentsDirectory.appending(path: "qwen-check.json")
    try? data.write(to: url)
    log.info("qwen-check: \(String(decoding: data, as: UTF8.self), privacy: .public)")
  }
}
