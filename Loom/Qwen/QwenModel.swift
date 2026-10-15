import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN

/* The Qwen3 module layout, adapted from ml-explore/mlx-swift-lm Qwen3.swift at tag 3.32.3 (via the
user's AttentionMachine app). Loom never calls these modules' own forward passes: QwenWorker walks the
layers itself so it can read every intermediate and apply Loom's knobs.

MIT License

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

nonisolated final class QwenAttention: Module {
  let config: QwenConfig
  let scale: Float
  @ModuleInfo(key: "q_proj") var wq: Linear
  @ModuleInfo(key: "k_proj") var wk: Linear
  @ModuleInfo(key: "v_proj") var wv: Linear
  @ModuleInfo(key: "o_proj") var wo: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
  let rope: RoPELayer

  init(_ c: QwenConfig) {
    config = c
    scale = pow(Float(c.headDim), -0.5)
    _wq.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: false)
    _wk.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
    _wv.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
    _wo.wrappedValue = Linear(c.attentionHeads * c.headDim, c.hiddenSize, bias: false)
    _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
    _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
    rope = initializeRope(
      dims: c.headDim, base: c.ropeTheta, traditional: false, scalingConfig: c.ropeScaling,
      maxPositionEmbeddings: c.maxPositionEmbeddings)
  }
}

nonisolated final class QwenMLP: Module {
  @ModuleInfo(key: "gate_proj") var gate: Linear
  @ModuleInfo(key: "down_proj") var down: Linear
  @ModuleInfo(key: "up_proj") var up: Linear

  init(dimensions: Int, hidden: Int) {
    _gate.wrappedValue = Linear(dimensions, hidden, bias: false)
    _down.wrappedValue = Linear(hidden, dimensions, bias: false)
    _up.wrappedValue = Linear(dimensions, hidden, bias: false)
  }
}

nonisolated final class QwenBlock: Module {
  @ModuleInfo(key: "self_attn") var attention: QwenAttention
  let mlp: QwenMLP
  @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionNorm: RMSNorm

  init(_ c: QwenConfig) {
    _attention.wrappedValue = QwenAttention(c)
    mlp = QwenMLP(dimensions: c.hiddenSize, hidden: c.intermediateSize)
    _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    _postAttentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
  }
}

nonisolated final class QwenInner: Module {
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  let layers: [QwenBlock]
  let norm: RMSNorm

  init(_ c: QwenConfig) {
    _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
    layers = (0..<c.hiddenLayers).map { _ in QwenBlock(c) }
    norm = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
  }
}

/// The whole model. It conforms to LLMModel only so MLX LM's loader can quantize and fill it; the
/// plain forward below is used once, to check the instrumented pass against it. Qwen3-4B ties its
/// readout to the embedding table, so there is no separate output matrix.
nonisolated final class QwenModel: Module, LLMModel, KVCacheDimensionProvider, @unchecked Sendable {
  let vocabularySize: Int
  let kvHeads: [Int]
  let model: QwenInner
  let config: QwenConfig
  var loraLayers: [Module] { model.layers }

  init(_ c: QwenConfig) {
    config = c
    vocabularySize = c.vocabularySize
    kvHeads = Array(repeating: c.kvHeads, count: c.hiddenLayers)
    model = QwenInner(c)
  }

  func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
    var h = model.embedTokens(inputs)
    let mask = createAttentionMask(h: h, cache: cache?.first)
    for (i, layer) in model.layers.enumerated() {
      let a = layer.attention
      let (B, L) = (h.dim(0), h.dim(1))
      let x = layer.inputNorm(h)
      var q = a.qNorm(a.wq(x).reshaped(B, L, config.attentionHeads, -1)).transposed(0, 2, 1, 3)
      var k = a.kNorm(a.wk(x).reshaped(B, L, config.kvHeads, -1)).transposed(0, 2, 1, 3)
      let v = a.wv(x).reshaped(B, L, config.kvHeads, -1).transposed(0, 2, 1, 3)
      let offset = cache?[i].ropeOffset
      q = applyRotaryPosition(a.rope, to: q, offset: offset)
      k = applyRotaryPosition(a.rope, to: k, offset: offset)
      let out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache?[i], scale: a.scale, mask: mask)
        .transposed(0, 2, 1, 3).reshaped(B, L, -1)
      h = h + a.wo(out)
      let n2 = layer.postAttentionNorm(h)
      h = h + layer.mlp.down(silu(layer.mlp.gate(n2)) * layer.mlp.up(n2))
    }
    return model.embedTokens.asLinear(model.norm(h))
  }

  func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
    weights.filter { !$0.key.hasPrefix("lm_head.") }
  }
}

nonisolated struct QwenConfig: Codable, Sendable {
  var hiddenSize: Int
  var hiddenLayers: Int
  var intermediateSize: Int
  var attentionHeads: Int
  var rmsNormEps: Float
  var vocabularySize: Int
  var kvHeads: Int
  var ropeTheta: Float = 1_000_000
  var headDim: Int
  var ropeScaling: [String: StringOrNumber]? = nil
  var tieWordEmbeddings = false
  var maxPositionEmbeddings: Int = 32768

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "hidden_size"
    case hiddenLayers = "num_hidden_layers"
    case intermediateSize = "intermediate_size"
    case attentionHeads = "num_attention_heads"
    case rmsNormEps = "rms_norm_eps"
    case vocabularySize = "vocab_size"
    case kvHeads = "num_key_value_heads"
    case ropeTheta = "rope_theta"
    case headDim = "head_dim"
    case ropeScaling = "rope_scaling"
    case tieWordEmbeddings = "tie_word_embeddings"
    case maxPositionEmbeddings = "max_position_embeddings"
  }

  init(from decoder: any Swift.Decoder) throws {
    let c = try decoder.container(keyedBy: CodingKeys.self)
    hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
    hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
    intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
    attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
    rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
    vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
    kvHeads = try c.decode(Int.self, forKey: .kvHeads)
    ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
    headDim = try c.decode(Int.self, forKey: .headDim)
    ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
    tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    maxPositionEmbeddings = try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
  }
}
