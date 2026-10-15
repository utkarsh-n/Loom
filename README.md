# Loom

Real language models you can see inside, for iPhone on iOS 27. Type a message and watch a transformer predict the next token, with every number on screen coming from that computation. The interface is a neon "math engine": the pipeline drawn the way the best LLM explainer diagrams draw it, with one colour per stage, and every one of those numbers is live.

There are two engines, switched at the top of the Pipeline tab (and in Settings):

- **Qwen3 4B** (the default on a phone): Qwen3-4B-Instruct-2507 by Alibaba's Qwen team, 4,022,468,096 parameters at 4 bits (2.3 GB, shipped inside the app), running on the iPhone's GPU with MLX. Loom walks its 36 layers itself so it can read every attention weight, neuron and logit-lens guess, and turn the same knobs: heads off, layers skipped, early exit, neuron clamps, steering along directions measured on Cipher's lines.
- **Loom 1.4M**: Loom's own 1,377,408-parameter transformer, trained only on what Cipher already carries. It keeps everything Qwen can't do on a phone: on-device fine-tuning, the five voice fine-tunes, training checkpoints, precision from 16 bits to 1.

SwiftUI and Liquid Glass draw the interface, SwiftUI Canvas draws the diagrams and the sphere, Accelerate (BLAS) runs Loom 1.4M and its backpropagation, MLX (`mlx-swift` 0.32.3, `mlx-swift-lm` 3.32.3) runs Qwen, `swift-transformers` 1.3.0 reads Qwen's tokenizer, and Core Haptics gives the knobs their detents. Nothing is downloaded at run time: there is no network call and no permission prompt.

## Run

```bash
Tools/build.sh sim       # Debug build for the iPhone 17 Pro simulator
Tools/build.sh device    # Release build for a connected iPhone
Tools/build.sh test      # LoomEngine's parity and training tests, on the Mac
```

`xcodebuild` hangs when it loads a project from this iCloud-synced folder, so the script mirrors the project to `$TMPDIR/LoomBuild` and builds there (Gyre and Cipher do the same). After adding source files, the script's `xcodegen generate` picks them up.

Device builds carry Qwen: the script clones (APFS, instantly) the weights and tokenizer from `~/Developer/loom-data/qwen/Qwen3-4B-Instruct-2507-4bit/` and the Mac-side extras from `~/Developer/loom-data/qwen/extras/` into the mirror's `Loom/Resources/Qwen/`; nothing of it enters the iCloud folder or git. The extras come from `Tools/qwen/precompute.py` (see `Tools/qwen/README.md`). Simulator builds leave Qwen out: MLX needs the phone's GPU, so the simulator runs Loom 1.4M only. To install after a device build (the app is 2.3 GB, so this takes a minute or two):

```bash
xcrun devicectl device install app --device <device-id> "$TMPDIR/LoomBuild/dd/Build/Products/Release-iphoneos/Loom.app"
```

## The pipeline

| Stage | Colour | Loom 1.4M | Qwen3 4B |
| --- | --- | --- | --- |
| 1 Input | cyan | Your message plus every generated token; optional language tag | The same, wrapped in Qwen's chat template (or raw) |
| 2 Tokenize | green | Character-level BPE, 4,096 tokens | Byte-level BPE, 151,936 tokens (151,387 merges over UTF-8 bytes) |
| 3 Embedding | yellow | A row of a 4,096 × 128 table: `x = E[id]` | A row of a 151,936 × 2,560 table |
| 4 Transformer | violet | 4 layers × 4 heads: attention (RoPE), SwiGLU 384, residual adds with RMSNorm | 36 layers × 32 heads sharing 8 key/value heads, QK-norm, RoPE base 5,000,000, SwiGLU 9,728 |
| 5 Predict next token | orange | Dot product with all 4,096 rows, then `p = softmax(z / τ)` | With all 151,936 rows (tied embeddings) |
| 6 Sample | pink | Inverse-CDF draw from a seeded SplitMix64 | The same sampler |
| 7 Repeat | blue | KV cache; until a line break or the limit | KV cache; until ⟨im_end⟩ (or a line break, raw) or the limit; context capped at 128 |

Below the pipeline, **The big picture** traces the running model's real numbers. Loom 1.4M: Data (217,798 lines) → Optimization (12,000 steps of gradient descent) → Weights (1,377,408) → Prediction (4,096 probabilities) → Generation. Qwen3 4B: about 36 trillion tokens in 119 languages and dialects (the Qwen3 technical report) → pre-training, instruction tuning and reinforcement learning by the Qwen team → 4,022,468,096 weights at 4 bits → 151,936 probabilities → Generation.

## Four tabs: Pipeline, Sphere, Embeddings, Tune

- **Pipeline.** The seven stages above, each a live card that lights up in turn while a token is generated. Below them, the big picture.
  - Tap in for detail:
    - **Inside the tokenizer:** the byte-pair merge replay, and why some languages cost more tokens.
    - **Inside layer N:**
      - multi-head attention: arcs, matrix, a switch to turn each head off;
      - the feedforward network: 384 neurons, each with the tokens it raises and lowers and the text that fires it, plus a clamp;
      - the residual connection and norm.
    - **Prediction and sampling:** logit histogram, the probability chain, ban and boost buttons, the sample.
  - The Knobs console (Gyre's) has three pages:
    - **Decode:** temperature τ, top-k, top-p, min-p, repetition ρ, max tokens, seed, speed.
    - **Model:** precision 16→1 bit, context, layers used (logit lens), checkpoint, KV cache, 16 head switches, layer skip.
    - **Style:** five fine-tunes, plus a steering strength, layer and direction.
- **Sphere.** Type a word or phrase and enter a first-person sphere. You stand inside the model's 128-dimensional readout space, looking along your thread.
  - **Distance from the centre** is 1 − cos θ, the token's exact angle from your gaze in all 128 dimensions. That is linear in its score for vectors of equal length, so the tokens drifting inward are the likely next ones. Most of the 4,096 sit on the cos 0 ring, at right angles to you.
  - **The ride:** each layer turns your gaze (with the layer's attention beams and its logit-lens guess). At the prediction the candidate orbs light up. Sampling, or your tap on a candidate, latches a tether onto a token, and your gaze rides it to the centre, where the next position starts.
  - **What is approximate:** direction around the centre is a projection on two axes carried smoothly as you turn. The slow orbit is decoration. Stars streak when your gaze turns fast.
  - **Controls:** drag to look around, pinch to zoom, double-tap to recentre.
- **Embeddings.** All 4,096 token vectors projected on their first three principal axes, coloured by script, with your prompt as a constellation. Tap a token to see its 128 numbers and its nearest neighbours by cosine over all 128 dimensions. Vector arithmetic (a − b + c) lives here too.
- **Tune.**
  - **Fine-tunes:** five LoRA fine-tunes (Scripture, Bard, Gothic, Twain, Almanac), each trained on text the base model never saw, with losses on unseen lines and a same-seed side-by-side generation.
  - **Language tags:** control tokens ⟨en⟩ ⟨es⟩ ⟨fr⟩ ⟨hi⟩ ⟨gu⟩ ⟨card⟩.
  - **Steering:** difference-of-means directions for each language and Cipher mood family.
  - **Checkpoints:** real checkpoints at steps 40, 250, 1,500 and 12,000, with the training curve.
  - **Fine-tune on your text:** gradient descent on the phone (LoRA rank 1–16 or every weight) with live training and held-back curves, then keep the result as a sixth knob.

Top-left **Dials** opens the console from any tab; top-right are **How it works** (the pipeline, the model, "not guessing: optimization", what is real) and **Settings** (Feel, a reset to the base model, privacy, credits).

## Qwen on the phone

**Runtime** (`Loom/Qwen/`). `QwenModel.swift` is the Qwen3 module layout from `mlx-swift-lm` (MIT), filled by its quantized-weight loader. `QwenEngine.swift` (`QwenWorker`, an actor) never calls the model's fused forward: it walks every layer by hand, keeps its own keys and values, computes attention in float32 so all 32 heads' weights can be read back, and applies the pass knobs (muted heads, skipped layers, early exit, neuron clamps, steering). Each pass returns threads in the same shape as Loom 1.4M's, so every view draws both. The newest position keeps everything (37 vectors of 2,560 numbers, 36 × 9,728 neuron values, the logit lens); older positions keep 33 numbers of each vector and their attention, and their logit lens is read back from the GPU when you select them. `QwenRuntime.swift` loads the weights on demand, reads the tokenizer, decodes the byte-level vocabulary for display (‹hex› for tokens that hold part of a character), and unloads Qwen when the app goes to the background. `LoomModel` switches engines and runs either.

**Extras** (`Tools/qwen/precompute.py`, on the Mac with MLX). Steering directions after each of the 36 layers (200 Cipher lines per group, difference of means, float32), the token catalogue for the sphere and the embedding map (the 8,192 tokens commonest in Cipher's lines plus the 3 chat tokens, with a PCA of their embedding rows: the three axes keep 2.7% of the variance of 2,560 dimensions), and a float32 reference trace for checking the phone's pass.

**Memory.** Qwen's weights take 2.3 GB of GPU memory. Loom caps the context at 128 tokens and reads attention back as float32 for every position, so a full context's attention is about 75 MB on the way through.

## How it works

**Corpus** (`Tools/train/prepare.py`). This reads `Cipher/Cipher/Resources/culture-lines.tsv` and `culture-works.tsv` without changing them. The result is 217,798 cleaned, de-duplicated lines (16.2 M characters) in English, Spanish, French, Hindi and Gujarati, plus 22,254 fact cards written from culture-works rows. Lines are grouped by work.

**Tokenizer** (`Tools/train/bpe.py`, `LoomEngine/Sources/LoomEngine/Tokenizer.swift`). A character-level BPE with 4,096 tokens, trained on the corpus. Merges never cross pre-token pieces (newline; an optional leading space plus a run of letters, numbers or punctuation). Ids 2–7 are control tokens, and half the training lines begin with their language tag. Swift splits text exactly as Python does; this is tested on all five scripts.

**Model** (`Tools/train/model.py`, `train.py`, MLX on the Mac). It has the Llama family's shape: RMSNorm, RoPE (base 10,000), SwiGLU (384 hidden), no biases, and a readout tied to the embedding. The size is 4 layers, 4 heads, width 128, context 128 and 1,377,408 parameters. It trained for 12,000 steps of 64 × 128 tokens with AdamW (lr 2e-3 cosine), about 8 minutes on an M1 Max, and reached a held-out loss of 3.70 nats per token. The 32,023 rows the voice adapters teach are held out of it. `Tools/train/pipeline.sh v3` runs everything, and `Tools/train/install.sh v3` copies the results into `Loom/Resources/Model` (15 MB).

**Voices** (`Tools/train/adapters.py`). LoRA rank 8 (α 16) on all seven matrices of every block: 81,920 parameters, 6% of the model. Each adapter trains on 90% of its texts with early stopping on the other 10%:

| Voice | Taught from | Held-out loss, base → adapter |
| --- | --- | --- |
| Scripture | Book of Mormon, The Woman's Bible | 3.99 → 3.52 |
| Bard | Eight Shakespeare plays | 4.58 → 4.39 |
| Gothic | Dracula, Frankenstein, Dorian Gray, Jekyll and Hyde, Usher, The Raven | 3.71 → 3.67 |
| Twain | Tom Sawyer, Huckleberry Finn, A Connecticut Yankee | 3.96 → 3.86 |
| Almanac | 22,254 fact cards | 7.58 → 3.49 |

The first attempt used Cipher's five mood families and failed. Mood is too subtle for a model this size to learn from text it has already read. Distinct voices on unseen text work, and the subtle ones (Gothic, Twain) show it in their numbers.

**Extras** (`Tools/train/extras.py`). These are measured with the float64 reference:
- steering directions per layer (each language minus the other four, each mood family minus the corpus);
- for each neuron, its three strongest moments in 2,500 corpus lines;
- the PCA of the embedding table.

**LoomEngine** (`LoomEngine/`, a Swift package using only Foundation and Accelerate). It contains:
- `WeightFile`: reads `.loom` files (JSON header plus f32/f16 tensors).
- `Model`: the forward pass with every intermediate kept, the KV cache, and the pass knobs (muted heads, bypass, depth, steering, neuron clamps).
- `Sampler`: bias, then repetition, then temperature, then top-k, top-p and min-p; inverse-CDF draws; SplitMix64.
- `Workshop`: per-column quantisation and LoRA mixing (QLoRA order).
- `Backprop`: a hand-written reverse pass.
- `Teacher`: AdamW on a LoRA adapter or on every weight.
- `Gaze`: the sphere's geometry: exact cosines with the gaze, the projected azimuth, slerp, and parallel-transported reference axes.

It compiles headless: `Tools/cli/run.sh weave|teach|seeds|speed` builds a Mac harness from the engine sources.

**App** (`Loom/`):
- `Machine/LoomModel.swift` owns the knobs and the weave loop. A prompt is prefilled (all threads at once); each card is one decode step on the KV cache. At slow pace the newest thread's pass is replayed floor by floor, labelled with its slowdown factor.
- `Machine/Library.swift` loads the shipped files.
- The views are in `UI/`.

## What is calculated, and what is drawn

- **LIVE:** every number on the Pipeline tab, the inspection panels, neighbours and arithmetic, steering, surgery and decode knobs. All computed by the real pass on the phone as you turn.
- **TRAINED HERE:** the fine-tune curves, from real training on the phone.
- **PRE-TRAINED:** trained on the Mac and shipped: the base, the age checkpoints, the voice adapters, steering directions and neuron moments.
- **PROJECTION:** the star map shows 3 of 128 dimensions (22% of the variance), each axis scaled to its own spread.
- **ILLUSTRATION:** the pace of the replay and the sphere's slow orbit. A token really takes about a millisecond.
- **Limits:**
  - Attention arcs are mixing weights, not explanations.
  - Neuron moments are examples, not definitions.
  - The model writes dreamy, half-right sentences and invents its facts.

## Simulator and test hooks (DEBUG only)

| Argument | Effect |
| --- | --- |
| `-resetPrompt` | Starts from "The night was" |
| `-tab loom\|sphere\|galaxy\|bench` | Opens a tab (Pipeline, Sphere, Embeddings, Tune) |
| `-prompt "<text>"` | Sets the prompt |
| `-instant` | Pace Instant |
| `-weave` | Starts generating at launch |
| `-knobs` / `-page Decode\|Model\|Style` | Opens the console, on a page |
| `-knob <id>=<value>` | Sets any knob, e.g. `-knob scripture=1.2`, `-knob precision=4` |
| `-inspect punch\|readout\|<floor>` | Opens an inspection panel |
| `-sheet fieldNotes\|settings` | Opens a sheet |
| `-teach` | Fills the fine-tune text with the example bars and starts training |
| `-sphere` / `-ride` | Opens the sphere, and starts riding |
| `-engine tiny\|qwen` | Picks the engine |
| `-chat 0\|1` | Qwen's chat template off or on |
| `-verifyQwen` | Once Qwen loads, checks its pass against MLX's own forward and the Mac's float32 trace, times decoding, and writes `Documents/qwen-check.json` |

## Privacy

Loom makes no network calls and asks for no permissions; Qwen's weights ship inside the app. The prompt and the Teach lines stay in the app's preferences on the phone.

## Credits and licences

Qwen3-4B-Instruct-2507 © Alibaba Cloud's Qwen team, Apache-2.0 (the 4-bit MLX conversion is `mlx-community/Qwen3-4B-Instruct-2507-4bit`). MLX and mlx-swift-lm © Apple / ml-explore, MIT. swift-transformers © Hugging Face, Apache-2.0.

## Verification

See `VERIFICATION.md`. Research behind the design is in `Docs/research/llm-visualization-research.md`.


## Public snapshot

See [PUBLICATION.md](PUBLICATION.md) for publication exclusions and setup requirements.
