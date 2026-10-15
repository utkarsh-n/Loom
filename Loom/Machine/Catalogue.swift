import Accelerate
import Foundation
import LoomEngine

/// The tokens the embedding map and the sphere draw: each one's place on the 3-D map, its script and its
/// embedding direction (length 1). The 1.4M model's catalogue is its whole vocabulary; Qwen's is the
/// 8,192 tokens commonest in Cipher's corpus, because all 151,936 rows at 2,560 numbers each would take
/// 1.5 GB.
nonisolated struct Catalogue: Sendable {
  /// Token id of each row.
  let ids: [Int]
  let points: [SIMD3<Float>]
  let scripts: [Galaxy.Script]
  /// Share of the variance the three map axes carry.
  let variance: [Double]
  let d: Int
  /// rows × d, each row length 1.
  let unit: [Float]
  /// Each row's length before it was scaled to 1.
  let norms: [Float]
  /// Where the rows came from, for the map's caption.
  let source: String
  private let rowOf: [Int: Int]

  var count: Int { ids.count }
  func row(of token: Int) -> Int? { rowOf[token] }
  func direction(row r: Int) -> ArraySlice<Float> { unit[(r * d)..<((r + 1) * d)] }

  init(ids: [Int], points: [SIMD3<Float>], scripts: [Galaxy.Script], variance: [Double], d: Int, embeddings: [Float], source: String) {
    self.ids = ids
    self.points = points
    self.scripts = scripts
    self.variance = variance
    self.d = d
    (unit, norms) = Gaze.unitCatalogue(embeddings, rows: ids.count, d: d)
    self.source = source
    rowOf = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
  }

  /// The small model's whole vocabulary after its 8 control tokens, from the current weights.
  static func tiny(weights w: ModelWeights, galaxy: Galaxy) -> Catalogue {
    let d = w.config.d, V = w.config.vocab
    let ids = Array(8..<V)
    return Catalogue(
      ids: ids, points: ids.map { $0 < galaxy.points.count ? galaxy.points[$0] : .zero },
      scripts: ids.map { $0 < galaxy.scripts.count ? galaxy.scripts[$0] : .mark }, variance: galaxy.variance, d: d,
      embeddings: Array(w.wte[(8 * d)...]), source: "all \((V - 8).formatted()) tokens after the 8 control tokens")
  }

  /// Qwen's catalogue: ids and map from `qwen-catalogue.json` (Tools/qwen/precompute.py), rows from the model.
  nonisolated struct QwenFile: Sendable {
    let ids: [Int]
    let points: [SIMD3<Float>]
    let scripts: [Galaxy.Script]
    let variance: [Double]

    init(url: URL) throws {
      let json = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: url))
      ids = (json["ids"]?.array ?? []).map { Int($0.number ?? 0) }
      let flat = json["points"]?.floats ?? []
      points = stride(from: 0, to: flat.count - 2, by: 3).map { SIMD3(flat[$0], flat[$0 + 1], flat[$0 + 2]) }
      scripts = (json["script"]?.array ?? []).map { Galaxy.Script(rawValue: $0.string ?? "") ?? .mark }
      variance = json["variance"]?.doubles ?? []
    }
  }

  /// The rows whose direction is closest to `v` (cosine), best first.
  func nearest(to v: [Float], count n: Int, excluding: Set<Int> = []) -> [(token: Int, cos: Float)] {
    guard v.count == d, !ids.isEmpty else { return [] }
    var q = v
    Gaze.normalize(&q)
    var sims = [Float](repeating: 0, count: ids.count)
    unit.withUnsafeBufferPointer { u in
      q.withUnsafeBufferPointer { x in
        sims.withUnsafeMutableBufferPointer { y in
          cblas_sgemv(CblasRowMajor, CblasNoTrans, Int32(ids.count), Int32(d), 1, u.baseAddress!, Int32(d), x.baseAddress!, 1, 0, y.baseAddress!, 1)
        }
      }
    }
    var best: [(Int, Float)] = []
    for r in sims.indices where !excluding.contains(ids[r]) {
      let s = sims[r]
      if best.count < n {
        best.append((ids[r], s))
        best.sort { $0.1 > $1.1 }
      } else if s > best[n - 1].1 {
        best[n - 1] = (ids[r], s)
        best.sort { $0.1 > $1.1 }
      }
    }
    return best.map { (token: $0.0, cos: $0.1) }
  }
}
