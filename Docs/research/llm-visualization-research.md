# LLM-as-a-machine: visualization and tunables research brief

Research date: 3 October 2026. For: a native iOS "LLM toy", a real tiny GPT (about 1–3M parameters, trained on about 19 MB of public-domain EN/ES/FR/HI/GU text) drawn as a physical machine with dials.

**TL;DR**
- No native Apple-platform app found does this "true to form"; the strong explainers are all web or video (§1.11).
- Best evidence on what teaches: Polo Club's CHI 2026 study. Live input plus live knobs beat blogs and videos, and one consistent visual element lowered mental load. The MLP, the only stage with no interactive control, was understood worst. Video produced "illusory understanding" ([paper](https://poloclub.github.io/papers/26-chi-transformer-explainer.pdf)).
- At 1–3M parameters, every inference-time knob in §2 can be real and live on an iPhone, and most training-time knobs can be real on-device mini-trainings (seconds to minutes). Only MoE, vocabulary size, positional scheme and full RLHF need shipped pre-trained variants or a labelled analogue.
- Prefer interventions (clamp, ablate, inject, steer) to passive pictures. As Anthropic's work shows, the honest way to say "this part does X" is to break it and watch the output change.

---

## 1. Survey of the best-known LLM explainers

**1.1 Brendan Bycroft, LLM Visualization** ([bbycroft.net/llm](https://bbycroft.net/llm), [source](https://github.com/bbycroft/llm-viz))
- **What it shows:** a 3D walkthrough of one forward pass of "nano-gpt" (about 85k parameters, real minGPT weights) sorting six letters, with scale comparisons against GPT-2 and GPT-3.
- **Metaphor:** a landscape of translucent slabs, one per matrix. Green blocks are computed activations, blue blocks are learned parameters ([GIGAZINE](https://gigazine.net/gsc_news/en/20231204-llm-visualization/)).
- **Interaction:** text on the left, 3D model on the right. Spacebar or "Continue" advances and the camera flies to the highlighted region. Hovering a block shows its dimensions and formula.
- **True to form:** every operation is present (layer norm, biases, per-cell dot products), on real weights.
- **Falls short:** one token, one fixed task, no language, no sampling knobs. [HN](https://news.ycombinator.com/item?id=38505211) readers couldn't tell whether it was accurate and wanted help with each part's purpose.
- **Borrow:** the material split between parameters and activations (fixed brass machinery vs glowing fluid), and the "zoom out to GPT-3 scale" moment.

**1.2 Polo Club, Transformer Explainer** ([live](https://poloclub.github.io/transformer-explainer/), [arXiv 2408.04619](https://arxiv.org/abs/2408.04619), CHI 2026)
- **What it shows:** live GPT-2 small (124M) running in the browser through ONNX. You type your own text and set temperature, top-k and top-p.
- **Metaphor:** a Sankey-style flow. Each token's embedding is a 1-D heatmap strip whose band width is proportional to its dimensionality. The strip branches into Q, K and V, expands in the MLP, and the attention heads appear as a looping stack of cards.
- **Interaction:** hover any matrix cell to see its value, and open "formula cards" on demand.
- **Study results (n=90):** it beat blog and video on quiz accuracy, and 26 of 30 participants named interactivity as the most helpful feature.
- **Criticism:** in a Sept 2026 [HN thread](https://news.ycombinator.com/item?id=49792342), commenters pointed out GPT-2's absolute positions instead of modern RoPE, 2.2 GB of RAM use, and poor behaviour on mobile.
- **Borrow:** one recurring visual token, the vector strip, that is never replaced by a new chart type. Also give every stage a knob, the MLP included.

**1.3 3Blue1Brown, Deep Learning chapters 5–7** ([GPT](https://www.3blue1brown.com/lessons/gpt), [attention](https://www.3blue1brown.com/lessons/attention/), [MLP](https://www.3blue1brown.com/lessons/mlp/))
- **Central metaphors:**
  - Embeddings are directions: "woman − man" is a gender direction, and a dot product measures alignment.
  - Attention is words passing information: in "a fluffy blue creature", the adjectives' values update the noun. Queries ask, keys answer, values carry the edit.
  - MLPs store facts: a neuron acts as an AND gate on "Michael" + "Jordan" that adds the "basketball" direction.
  - Superposition: about 10,000 nearly-orthogonal vectors fit in 100 dimensions (89–91° apart).
- **Interaction:** none; it is a linear narrative.
- **True to form:** real GPT-3 dimensions and honest parameter counts.
- **Falls short:** the directions are idealised. Real heads and neurons are rarely this clean, which the videos hedge with "might".
- **Borrow:** a compass-needle readout projecting a vector onto a few labelled directions (language, plural, gender), computed for real from your model by difference of means.

**1.4 Jay Alammar, The Illustrated Transformer / Illustrated GPT-2** ([Transformer](https://jalammar.github.io/illustrated-transformer/), [GPT-2](https://jalammar.github.io/illustrated-gpt2/))
- **What it shows:** static, colour-consistent diagrams. Q/K/V appear as cell strips, masked self-attention is explained, and GPT-2 is a stack of decoder blocks processing one token at a time.
- **Interaction:** none.
- **True to form:** shapes and order are right.
- **Where it falls short:** the numbers are schematic and the 2017 post centres on the encoder-decoder architecture.
- **Borrow:** Alammar's filing-cabinet analogy for Q/K/V. The query is a sticky note, the keys are folder labels, the values are folder contents. It maps directly onto a physical card catalogue.

**1.5 Financial Times, "Generative AI exists because of the transformer"** ([ig.ft.com/generative-ai](https://ig.ft.com/generative-ai/), Sept 2023, FT Visual Storytelling Team and Madhumita Murgia, vetted by Vaswani, Uszkoreit, Gomez and Petrov per [JournalismAI](https://journalismai.com/2023/09/12/generative-ai-exists-because-of-the-transformer-ft/))
- **What it shows:** a scroll-driven newspaper explainer. Words become tokens, then number vectors, then a cloud of related words. Self-attention is shown as links between words in a sentence that resolve context. It closes on failure modes such as hallucination.
- **Caveat:** the page wouldn't fetch for this brief, so these details come from secondary coverage.
- **True to form:** strong on narrative.
- **Where it falls short:** no live model, and the numbers are illustrative.
- **Borrow:** one example sentence followed through every chapter, and an honest ending on where it fails.

**1.6 Anthropic interpretability visuals**
- **Residual stream:** [A Mathematical Framework](https://transformer-circuits.pub/2021/framework/index.html) recast the residual stream as a shared communication bus that every head reads from and writes into.
- **Features and Golden Gate Claude:** [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) used sparse autoencoder (SAE) features with dashboards. [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude) clamped one feature to 10× its maximum activation, and the model began identifying as the bridge.
- **Attribution graphs:** "[Tracing the thoughts](https://www.anthropic.com/research/tracing-thoughts-language-model)" (2025; [methods](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)) draws node-link diagrams of feature groups. In one, Dallas → Texas → Austin, and swapping in California yields Sacramento. Another shows a poem pre-planning "rabbit"; another shows shared multilingual concepts.
- **Later work:** [persona vectors](https://arxiv.org/abs/2507.21509) (Aug 2025), [concept injection / introspection](https://www.anthropic.com/research/introspection) (Oct 2025), and the [Assistant Axis with activation capping](https://www.anthropic.com/research/assistant-axis) (Jan 2026).
- **True to form:** the claims are causal, because they rest on interventions.
- **Where it falls short:** the graphs are partial explanations from a "replacement model", and they are dense.
- **Borrow:** every visual of an internal state should double as an intervention point (clamp, suppress, inject), with the effect shown on the output.

**1.7 Logit lens / tuned lens** ([nostalgebraist 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens), [Belrose et al. 2023](https://arxiv.org/abs/2303.08112))
- **What it shows:** each layer's residual is decoded through the final norm and the unembedding. The result is a grid of layers × positions showing the top-guess token, which reveals a guess forming and sharpening mid-network.
- **True to form:** the logit lens is free and exact.
- **Where it falls short:** early layers suffer "basis drift". The tuned lens fixes this with a small affine probe per layer.
- **Borrow:** for a 4–6 layer toy this is ideal. Give every floor of the machine an inspection window showing its current best guess.

**1.8 Karpathy: nanoGPT, "Let's build GPT", microgpt, nanochat** ([nanoGPT](https://github.com/karpathy/nanoGPT), [video](https://www.youtube.com/watch?v=kCc8FmEb1nY), [microgpt, Feb 2026](http://karpathy.github.io/2026/02/12/microgpt/), [nanochat, Oct 2025](https://simonwillison.net/2025/Oct/13/nanochat/))
- **What they show:** the explainer is the code. nanoGPT is a 10.65M-parameter character-level model on 1.1 MB of Shakespeare, set against a bigram baseline. microgpt is about 200 lines of dependency-free Python with 4,192 parameters, trained on names. nanochat adds SFT and RL.
- **True to form:** perfectly. **Falls short:** barely visual.
- **Borrow:** a training-time scrubber over real shipped checkpoints (step 0, 100, 1k, final), so samples visibly go from noise to language. Offer the bigram baseline as an "attention removed" comparison machine.

**1.9 Neuronpedia** ([neuronpedia.org](https://www.neuronpedia.org/))
- **What it shows:** an open interpretability platform. Its feature dashboards list top-activating examples, logit effects and auto-generated labels.
- **Tools:** a steering playground (strength, temperature, seed), the [circuit-tracer](https://www.neuronpedia.org/blog/circuit-tracer) attribution-graph frontend, and [Gemma Scope 2 plus the Assistant Axis demo](https://www.neuronpedia.org/blog/assistant-axis) (64M+ latents).
- **True to form:** real models and real features.
- **Where it falls short:** it is built for researchers and is dense.
- **Borrow:** a two-sided feature card: what lights it up, and what it pushes the output toward.

**1.10 BertViz** ([GitHub](https://github.com/jessevig/bertviz), [paper](https://arxiv.org/abs/1906.05714))
- **What it shows:** real attention weights at three scales:
  - head view: lines between tokens, coloured per head
  - model view: a thumbnail grid of every layer × head
  - neuron view: element-wise q·k products
- **Where it falls short:** it invites the reading that attention is explanation, which [Jain & Wallace](https://arxiv.org/abs/1902.10186) dispute. It also ignores value norms.
- **Borrow:** the model-view thumbnail wall, built as a bank of tiny oscilloscope screens with one per head. Tap one to open its head view.

**1.11 Apple platforms**
- No native app was found that visualizes a running transformer's internals.
- **Closest:** the iPad "[Neural Network](https://apps.apple.com/us/app/neural-network/id1529389288)" app (4.7★) has Transformer and language-model tutorials with small real networks. On-device chat apps are black boxes.
- **Apple's own model:** the [Foundation Models framework](https://developer.apple.com/videos/play/wwdc2025/301/) exposes only greedy or random top-k / top-p (probability threshold) sampling with a seed, plus temperature and a token cap. That is the sampler chain and nothing internal.
- **Feasible stack:** [MLX Swift](https://github.com/ml-explore/mlx-swift) runs on iOS, including autograd, optimizers and a training example. A hand-written Metal or Accelerate forward pass is also trivial at this scale.

**1.12 Notable 2025–26 newcomers**
- **[AnimatedLLM](https://animatedllm.github.io/)** ([arXiv 2601.04213](https://arxiv.org/abs/2601.04213), Charles University): steps through precomputed traces of OLMo 3, Aya, Llama 3.2, Qwen 3 and GPT-2, with a multilingual UI and keyboard stepping. The traces are real but not live.
- **[Anatomy of an LLM](https://www.royvanrijn.com/anatomy-of-an-llm/):** 15 chapters with KV-cache and quantization widgets. It uses small illustrative vectors and says so.
- **[llm-internal](https://github.com/ctbot000/llm-internal):** trains BPE and a roughly 700-parameter bigram live in the page. Its stated rule is to show "the real algorithm at a scale you can watch, not a picture", which is the toy's thesis exactly.
- **Also relevant:** [AttentionViz](https://arxiv.org/abs/2305.03210) (a joint Q/K embedding map) and Welch Labs' [MLA / KV-cache video](https://www.youtube.com/watch?v=0VLAoVGf_74) (Mar 2025).

---

## 2. Catalogue of real LLM tunables

### Reference model and feasibility

**Reference model (assumption):** d_model 160, 6 layers, 4–8k BPE vocabulary, context 256. That comes to about 2.5M parameters, with roughly 25% of them in the tied embedding/unembedding table.

**Compute (back-of-envelope estimates; benchmark them):**
- **Inference:** about 2N ≈ 5–6 MFLOPs per token over roughly 12 MB of fp32 weights, so a decode step is well under 1 ms on an A17/A18. Prefilling 256 tokens is a few milliseconds. Rendering, not math, is the bottleneck, so every inference-time knob can re-run the full forward pass on each slider tick. A full activation trace is about 40 KB per token, so scrubbing history is cheap.
- **Training:** about 6N ≈ 15–18 MFLOPs per token. A 4k-token step is about 75 GFLOPs, roughly 0.1–0.3 s on the iPhone GPU through MLX Swift, so 100–300 fine-tuning steps take tens of seconds before thermal throttling.

**Data:** 19 MB of mixed-script UTF-8 (Devanagari and Gujarati use 3 bytes per character) is probably only 5–10M tokens. Multi-epoch pretraining is likely, which makes overfitting real and demonstrable.

**Phase:** **[I]** inference-time · **[T]** training-time · **[A]** architecture (fixed when the model is trained).

**Feasibility tags:**
- **LIVE:** exact and real-time on device.
- **TRAIN:** real on-device training, taking seconds to minutes.
- **PREBAKED:** real, but trained offline. Ship the variants with the app.
- **ANALOGUE:** a labelled stand-in, not the real technique.

### A. Sampling and decoding

| Tunable | What it actually does | Real on toy? | Machine metaphors |
|---|---|---|---|
| **Temperature** [I] | p = softmax(z/T). T→0 approaches argmax; T=1 is the model's own distribution; T>1 flattens it. It doesn't add "creativity", it only reshapes the distribution. | LIVE | Furnace thermostat over sand-pile histograms (heat slumps the peaks); boiler pressure gauge; heat lamp over a mountain range |
| **Top-k** [I] | Keep the k highest logits, then renormalise. | LIVE | Turnstile admitting k; rack with k slots; comb with k teeth |
| **Top-p (nucleus)** [I] | Keep the smallest high-probability set whose mass is ≥ p ([Holtzman 2019](https://arxiv.org/abs/1904.09751)). The set shrinks when the model is confident and grows when it isn't. | LIVE | Measuring jug filled largest-first to the p line; self-adjusting funnel; sieve whose mesh tracks the flow |
| **Min-p** [I] | Keep tokens with p ≥ p_min × p_max ([Nguyen 2024](https://arxiv.org/abs/2407.01082); see the [critique](https://arxiv.org/abs/2506.13681)). | LIVE | Float valve tied to the tallest column; "fraction-of-the-leader" gate |
| **Repetition / frequency / presence penalty** [I] | CTRL repetition penalty: logits of tokens already seen are divided by θ≈1.2 ([Keskar 2019](https://arxiv.org/abs/1909.05858)). OpenAI style: z_j −= c_j·α_freq + 1[c_j>0]·α_pres ([docs](https://developers.openai.com/api/docs/guides/advanced-usage)). | LIVE | Wear counter or brake that stamps each used token; printer's "used type" tray; ratchet damper |
| **Logit bias** [I] | z_j += b_j per token. −100 effectively bans the token ([OpenAI](https://help.openai.com/en/articles/5247780-using-logit-bias-to-alter-token-probability-with-the-openai-api)). | LIVE | Magnets or shims under individual columns; ban stamp; counterweights on scale pans |
| **Sampler-chain order** [I] | Order matters. llama.cpp defaults to penalties → truncation → temperature last, and the order is reorderable ([guide](https://smcleod.net/2025/04/llm-sampling-parameters-guide/)). | LIVE | Reorderable pipe segments; plumbing manifold |
| **Greedy** [I] | Always take the argmax. Deterministic, and prone to loops. | LIVE | Lever that locks the dice; "tallest column wins" latch |
| **Beam search** [I] | Keep the B best partial sequences by cumulative log-probability, expanding and pruning each step. Tends toward bland, repetitive text on open-ended prompts (Holtzman). | LIVE (B≤8 is cheap) | Parallel conveyor tracks with a culling gate; branching pipework with pressure gauges |
| **Seed / determinism** [I] | Sets the pseudo-random generator state for sampling. Same seed + knobs + input gives the same output on the same build. | LIVE | Numbered dice; punched random tape; music-box cylinder |
| **Max tokens** [I] | Hard cap on loop iterations. | LIVE | Fuel tank; ticket roll; hourglass |
| **Stop sequences** [I] | A string match on the decoded output halts the loop. The match can straddle token boundaries. | LIVE | Trip-wire or limit switch; end-stop |

### B. Context and memory

| Tunable | What it actually does | Real on toy? | Machine metaphors |
|---|---|---|---|
| **Context window** [A/I] | The number of positions attention can see. Running shorter than the trained length (truncation or a sliding window) is LIVE. Going longer fails: learned absolute positions don't exist beyond the trained length, and RoPE degrades beyond it. | LIVE (shorter), PREBAKED (longer) | Length of the workbench or conveyor; reading slot over a player-piano roll |
| **KV cache** [I] | Stores K and V for every past position in each layer, so each decode step computes only the new token's q, k, v. The output is identical; compute per step drops from re-running the full prefix to a single position. Memory = 2·L·n·d (about 2 MB fp32 here). | LIVE. Toggle it to show identical text and a FLOP counter. | Card-catalogue drawers filling one card per token; shelf of stamped plates; pigeonholes |
| **System prompt** [I/T] | In chat models, a prefix that SFT taught the model to obey. In a base model it's just conditioning text. Honest toy version: pretrain with language or style control tokens (CTRL "control codes"), so a cartridge inserts a real prefix the model was trained to respect. | PREBAKED (control tokens), LIVE (prefix) | Punched card loaded before the roll; cartridge; stencil |

### C. Internal surgery

| Tunable | What it actually does | Real on toy? | Machine metaphors |
|---|---|---|---|
| **Attention head ablation** [I] | Zero (or mean-replace) a head's output before W_O, then measure the change in loss and output. Many heads are removable ([Michel 2019](https://arxiv.org/abs/1905.10650)). | LIVE | Fuse or breaker per head; mute switch on a mixing desk; lifting a shuttle off the loom |
| **Layer skip / early exit (logit lens)** [I] | Decode h_ℓ through the final norm and W_U. Alternatively, bypass block ℓ so the residual passes straight through. Production versions train for it ([CALM](https://arxiv.org/abs/2207.07061), [LayerSkip](https://arxiv.org/abs/2404.16710)). | LIVE. A tuned lens is PREBAKED and cheap to train. | Periscope or inspection window per floor; bypass valve; express elevator |
| **Activation steering** [I] | h_ℓ += α·v, where v = mean(h given prompt set A) − mean(h given prompt set B) ([ActAdd](https://arxiv.org/abs/2308.10248), [CAA](https://aclanthology.org/2024.acl-long.828/)). SAE-feature clamping is the Golden Gate variant. With a multilingual corpus, "Spanish − English" makes a real language dial. | LIVE (difference-of-means takes milliseconds); SAE features PREBAKED | Dye injector into the main pipe; magnet deflecting a compass needle; weighted pendulum |
| **Quantization (8/4/2-bit, per-channel)** [I] | w ≈ s·clamp(round(w/s)), with scale s per tensor or per output channel. 8-bit gives 256 levels, 4-bit 16, 2-bit 4. Ternary [BitNet b1.58](https://arxiv.org/abs/2402.17764) needs training for it. Expect steep collapse at ≤4-bit on a tiny model; 2-bit [breaks even mid-size models](https://arxiv.org/abs/2410.14766). | LIVE. Fake-quantize in milliseconds and show held-out perplexity per language. | Staircase replacing a ramp; ruler with fewer ticks; gear with fewer teeth; coarse sieve snapping pebbles to a grid |
| **Pruning** [I/T] | Unstructured: zero the smallest \|w\| (or \|w\|·‖x‖, [Wanda](https://arxiv.org/abs/2306.11695)). Structured: drop whole neurons or heads. | LIVE (one-shot); TRAIN (recovery fine-tune) | Pruning shears on a vine; taping over Jacquard-card holes; pulling pins from a pinboard |

### D. Adaptation and training

| Tunable | What it actually does | Real on toy? | Machine metaphors |
|---|---|---|---|
| **LoRA (rank, alpha)** [T] | W' = W + (α/r)·B·A, with A∈ℝ^{r×d}, B∈ℝ^{d×r}, and B initialised to 0. Only A and B train ([Hu 2021](https://arxiv.org/abs/2106.09685)). Rank sets expressivity; α/r is the gain. | TRAIN (rank 1–8 on a few KB of user text in under a minute, estimated) | Transparent overlay stencils on the punch cards; clip-on lens or filter; plug-in cartridge with a gain knob |
| **Mixing adapters** [I] | W + Σ λ_i·ΔW_i ([LoraHub](https://arxiv.org/abs/2307.13269)). | LIVE once adapters exist | Mixing-desk faders, one per cartridge |
| **Full fine-tune vs adapters** [T] | Full fine-tuning updates all of θ and risks catastrophic forgetting, which here would be visible as other languages degrading. Adapters leave the base model untouched. | TRAIN (full fine-tuning of 2.5M parameters is feasible) | Re-cutting the gearbox vs bolting on an attachment |
| **Learning rate** [T] | Step size of the AdamW update. Too high gives loss spikes or NaN; too low barely moves. | TRAIN | Boiler throttle with a red-lined pressure gauge; feed rate on a lathe |
| **Epochs / overfitting** [T] | Number of passes over the data. Many passes over a small set: training loss falls, held-out loss rises, and the model starts regurgitating the training text. Probably the single most vivid real demo. | TRAIN | Feeding the same roll through again and again; a mould that takes the exact shape; worn grooves |
| **Batch size** [T] | Examples averaged per gradient step. Larger means a smoother, less noisy gradient estimate and fewer steps per epoch. | TRAIN | Scoop or hopper size; committee averaging votes |
| **Warmup** [T] | Learning rate ramps linearly from 0 over the first N steps, which stabilises the early Adam statistics. | TRAIN | Choke or engine warm-up; soft-start motor |
| **Weight decay** [T] | Decoupled shrinkage θ −= η·λ·θ ([AdamW](https://arxiv.org/abs/1711.05101)). Linked to generalisation and [grokking](https://arxiv.org/abs/2201.02177). | TRAIN | Rust or friction eroding unused gears; return springs pulling dials to zero |
| **Dropout** [T] | During training, zero each activation with probability p and scale the rest by 1/(1−p). Off at inference ([Srivastava 2014](https://jmlr.org/papers/v15/srivastava14a.html)). | TRAIN. Leaving it on at inference (MC dropout) is real but non-standard, so label it. | Flickering bulbs; shutters blinking closed at random |

### E. Preference and alignment

**Real techniques:**
- **RLHF:** fit a reward model r(x,y) to human pairwise choices, then optimise the policy with PPO to maximise r − β·KL(π‖π_ref).
- **DPO:** skips the reward model, using the loss −log σ(β[(log π − log π_ref)(y_w) − (log π − log π_ref)(y_l)]) ([Rafailov 2023](https://arxiv.org/abs/2305.18290)).
- **Training-time only.** Full PPO on device is overkill and unstable, so it would be an **ANALOGUE**.

**Toy-honest options, all real techniques:**
1. **"Preference press":** the user picks between two continuations. The pairs accumulate, and DPO trains a LoRA adapter on device. The β knob is the tether to the reference model. **TRAIN**
2. **Best-of-N with a tiny reward model:** a linear head on the final residual, trained on the user's picks, generates N continuations and keeps the top one. **LIVE + TRAIN**
3. **Reward-guided decoding:** add β·r(prefix+token) to the logits ([FUDGE](https://arxiv.org/abs/2104.05218)), or follow a classifier's gradient on the activations ([PPLM](https://arxiv.org/abs/1912.02164)). **LIVE**

**Metaphors:** a tasting panel or judge's scale; a Watt flyball governor; a KL tether spring back to the original machine.

### F. Architecture-time (fixed per trained model)

| Tunable | What it actually does | Real on toy? | Machine metaphors |
|---|---|---|---|
| **Distillation** [T] | Train a student on the teacher's softened distribution: loss = T²·KL(softmax(z_t/T) ‖ softmax(z_s/T)) ([Hinton 2015](https://arxiv.org/abs/1503.02531)). | PREBAKED (ship a 3M teacher and a 300k student); on-device distillation is TRAIN but slow | Master and apprentice; casting a smaller copy from a mould; tracing paper |
| **Mixture-of-experts routing** [A] | Router g = softmax(W_r·h) chooses the top-k expert MLPs; output = Σ g_i·E_i(h); an auxiliary loss balances load across experts ([Switch](https://arxiv.org/abs/2101.03961)). | PREBAKED only if trained as MoE (e.g. 4 tiny experts, top-1). Routing and expert ablation are then LIVE; otherwise ANALOGUE. | Railway switchyard; sorting office; pneumatic-tube router |
| **RoPE / positional encoding** [A] | Learned absolute embeddings add a position vector. RoPE instead rotates each 2-D pair of q and k by pos·θ_i, with θ_i = 10000^(−2i/d), so q·k depends only on relative offset ([Su 2021](https://arxiv.org/abs/2104.09864)). Choose RoPE: HN criticised GPT-2's absolute positions as dated. | PREBAKED; the rotation angles themselves can be shown LIVE | Clockwork: a gear train of hands turning at geometric speed ratios (an orrery); combination-lock dials |
| **Tokenizer vocab (BPE merges)** [A] | Start from bytes and repeatedly merge the most frequent adjacent pair ([Sennrich 2016](https://arxiv.org/abs/1508.07909)); vocabulary = 256 + number of merges. A bigger vocabulary means shorter sequences but a larger V·d embedding table. Indic scripts fragment if English dominates the merges ([Regional TinyStories](https://arxiv.org/abs/2504.07989)). | LIVE for replaying or truncating the merge list (tokenizer only); PREBAKED for running the model at another vocabulary size | Printer's type case where each merge is a cast ligature; die-cutter; bead stringing |

**Unifying metaphor:** if the app is "Loom", the Jacquard loom fits unusually well. Punched cards are fixed weights. Warp threads are per-position residual streams. Shuttles are attention heads carrying colour between threads. Each weft pass is a layer, and the cloth is the generated text. Historically, the Jacquard loom is also a precursor of programmable computing.

---

## 3. What "true to form" should mean: honesty rules

1. **Every displayed number is read from the live tensor of this pass.** No decorative random digits. When downsampled, say so on screen: "showing 16 of 160 dims (top-|value|)".
2. **Every metaphor carries a nameplate with the real term and formula.** For example: "Shuttle 3.2 = attention head, softmax(QKᵀ/√d)·V".
3. **Every knob is badged LIVE / TRAIN / PREBAKED / ANALOGUE.** ANALOGUE controls are visually distinct, for example a different material, and are kept rare.
4. **Computation order is faithful.**
   - Prefill processes all prompt positions in parallel; decode does one position at a time, reading the KV cache.
   - Attention weights are produced by one matrix multiply, not by beams travelling over time.
   - The sampler chain runs in its displayed order.
   - Residual updates are added to the stream, never shown replacing it.
   - Layer norm and biases are present, not silently skipped.
5. **Projections are disclosed.** A 2-D map is labelled "PCA, x% variance". Don't use t-SNE or UMAP to imply distances.
6. **Colour scales are fixed per tensor type and zero-centred diverging, with a legend.** Any per-frame renormalisation is visible.
7. **Determinism is shown.** Display the seed. "Replay" reproduces byte-identical output, and any knob change shows a before/after diff.
8. **Attention arcs are "mixing weights", not "what the model looked at".** Feature labels are marked as guesses alongside their top-activating evidence. Attention isn't explanation ([Jain & Wallace](https://arxiv.org/abs/1902.10186)).
9. **Fail visibly.** Show the gibberish at 2-bit, the overfit regurgitation, and how an Indic word splits into many byte-pieces.
10. **A permanent scale card.** For example: "This machine: 2.5M params, 19 MB text. GPT-3: 175B (about 70,000×)."
11. **Training curves come only from real runs**, either on device or from shipped logs of the actual pretraining. Never fake a curve.

---

## 4. Interaction patterns and pitfalls

### Patterns that work

- **Live input + live knobs, with immediate feedback.** The top-rated feature in the Transformer Explainer study. Reshape the distribution while the user drags, with ghost bars of the previous state.
- **Step / play / scrub.** Space or N steps one token; play runs the autoregressive loop (as in AnimatedLLM and Bycroft). Store every token's trace to scrub back through generation, and scrub training time through the shipped checkpoints.
- **Tap a token to see its attention.** BertViz-style arcs for the selected head, plus a thumbnail wall of all heads with pinning.
- **A zoom ladder with one consistent visual token:** machine → floor (layer) → part (head or MLP) → vector strip → single cell with a formula card. This follows Bret Victor's "[ladder of abstraction](http://worrydream.com/LadderOfAbstraction/)" and Polo Club's progressive disclosure.
- **Interventions are first-class.** Clamp a feature lamp, ablate a head, inject a steering vector, then diff the output. Give the MLP its own knob (clampable neuron lamps, a fact probe); without one it was the worst-understood stage.
- **Small challenges mid-flow,** e.g. "make it answer in Gujarati without typing Gujarati". Study participants asked for these, and they counter illusory understanding.
- **Physical feel:** detented dials with Core Haptics ticks, a dice-roll haptic per sampling step, a counter that ticks when the KV cache is reused.

### Pitfalls

- **Information overload.** Blog and video participants said "Too much information at once". Don't add a new chart type per subtopic; keep one visual language.
- **Fake animation that misrepresents computation:** words marching through layers one at a time during prefill; "thinking" spinners; the MLP drawn as a lookup database; layers shown running in parallel, when depth is sequential and positions are parallel.
- **Over-clean semantics.** 3Blue1Brown-style labelled directions are idealised; show the messy real projections beside them.
- **Attention-as-explanation,** and normalised heatmaps that hide value norms.
- **Precomputed traces passed off as live.** AnimatedLLM makes this trade-off openly; here everything can be live, so make that a selling point.
- **Mobile performance.** Transformer Explainer hit 2.2 GB of RAM and crashes. A native Metal renderer at this model size avoids that, but limit live activation readouts to what is on screen.
- **Calling temperature "creativity", or hiding tokenization.** Show tokens, not words, and per-language tokens-per-word, since Hindi and Gujarati will fragment.
