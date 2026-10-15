// A Mac harness for LoomEngine, compiled together with the engine sources by Tools/cli/run.sh.
//   run.sh weave "<prompt>" [seed] [temperature]
//   run.sh teach <lines.txt> [steps] [rank] [learning rate] [batch] [full]
//   run.sh speed
import Foundation

let args = CommandLine.arguments
let data = ProcessInfo.processInfo.environment["LOOM_DATA"] ?? NSHomeDirectory() + "/Developer/loom-data"
let run = ProcessInfo.processInfo.environment["LOOM_RUN"] ?? "v3"
let tok = try Tokenizer(url: URL(filePath: data + "/tokenizer.json"))
let base = try ModelWeights(file: WeightFile(url: URL(filePath: "\(data)/runs/\(run)/loom-base.loom")))

func weave(_ w: ModelWeights, _ prompt: String, seed: UInt64, temperature: Double, topP: Double = 0.9) -> String {
  var ids = [Tokenizer.newline] + tok.encode(prompt)
  var rng = SplitMix64(seed: seed)
  var s = SamplerSettings()
  s.temperature = temperature
  s.topP = topP
  var (trace, cache) = Forward.prefill(w, ids: ids)
  for _ in 0..<32 {
    let V = w.config.vocab
    let last = trace.logits[((trace.rows - 1) * V)...]
    let d = Sampler.draw(logits: last, context: ids, settings: s, rng: &rng)
    ids.append(d.chosen)
    if d.chosen == Tokenizer.newline || ids.count >= w.config.ctx { break }
    trace = Forward.run(w, ids: ids, start: ids.count - 1, cache: &cache)
  }
  return tok.decode(Array(ids.dropFirst()))
}

switch args.count > 1 ? args[1] : "speed" {
case "weave":
  print(weave(base, args[2], seed: UInt64(args.count > 3 ? args[3] : "1")!, temperature: Double(args.count > 4 ? args[4] : "0.7")!))
case "seeds":
  for seed in 1...Int(args[3])! { print(seed, weave(base, args[2], seed: UInt64(seed), temperature: 0.7).replacingOccurrences(of: "\n", with: " ")) }
case "teach":
  let text = try String(contentsOfFile: args[2], encoding: .utf8)
  let lines = text.split(whereSeparator: \.isNewline).map { [Tokenizer.newline] + tok.encode(Tokenizer.normalize(String($0))) + [Tokenizer.newline] }
  var s = TeachSettings()
  s.steps = Int(args.count > 3 ? args[3] : "60")!
  s.rank = Int(args.count > 4 ? args[4] : "8")!
  s.learningRate = Double(args.count > 5 ? args[5] : "0.01")!
  s.batch = Int(args.count > 6 ? args[6] : "4")!
  s.full = args.count > 7 && args[7] == "full"
  let t = Teacher(base: base, lines: lines, settings: s)
  print("train lines \(t.trainLineCount), held out \(t.heldOutLineCount), trained parameters \(t.trainedParameters)")
  print("before: train \(Backprop.loss(base, sequences: lines))")
  for _ in 0..<s.steps {
    let p = t.step()
    print(String(format: "%3d  train %.3f  held %@  epochs %.1f  %.2fs", p.step, p.trainLoss, p.heldOutLoss.map { String(format: "%.3f", $0) } ?? "  –  ", p.epochs, p.seconds))
  }
  for seed in 1...3 { print("→", weave(t.tuned, "the cipher", seed: UInt64(seed), temperature: 0.8)) }
default:
  let ids = [Tokenizer.newline] + tok.encode("The night was dark and the sea was calm and the stars came out one by one over the water")
  let clock = ContinuousClock()
  let e = clock.measure { for _ in 0..<50 { _ = Forward.prefill(base, ids: ids) } }
  print("prefill \(ids.count) tokens: \(e / 50)")
  let g = clock.measure { _ = Backprop.gradients(base, sequences: Array(repeating: ids, count: 8)) }
  print("gradients, 8 × \(ids.count) tokens: \(g)")
}
