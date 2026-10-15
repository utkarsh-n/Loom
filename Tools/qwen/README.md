# Qwen precompute

Mac-side data that lets the phone run Qwen3-4B-Instruct-2507 (4-bit MLX) as Loom's engine.
`precompute.py` reads the model at `~/Developer/loom-data/qwen/Qwen3-4B-Instruct-2507-4bit/`
(falling back to `~/Developer/cipher-llm/models/`) and the corpus `~/Developer/loom-data/lines.tsv`, and
writes into `~/Developer/loom-data/qwen/extras/`:

| file | what it is |
|---|---|
| `qwen-catalogue.json` | The tokens the star map and first-person sphere use: the 8,192 most frequent token ids in the corpus (tokenized with no special tokens) plus 151643/151644/151645, sorted. `points` holds the first 3 PCA components of their dequantized float32 embedding rows, each axis divided by its 98th-percentile absolute value and clipped to ±1.3 (Galaxy's rule in `Library.swift`, so the app's own pass leaves it unchanged). Also `variance`, `script`, `frequency`, `source`, plus `tokens` (decoded text), `mean`, `axes` and `scale` so anything else can be projected onto the same map. |
| `qwen-reference.json` | A float32 reference trace for checking the Swift forward pass number for number, for `"The night was"` raw and in the chat template. Per record: `ids`, `top10` and `logits_probe` (last-position logits), `attention` (last-query softmax rows for layers 0/17/35 × heads 0/7/31), `resid_norm` (embedding + 36 layer outputs), `mlp17_top` (top 8 by abs value of the layer-17 SwiGLU activation), `lens_top1` (final norm + tied readout per layer). It is computed by an explicit numpy forward over weights dequantized to float32; `checks` records how far that and the manual MLX walk are from `model(x)`. |
| `qwen-steering.loom` | Loom tensor file (`loomio.write_loom`, float32): `lang:en/es/fr/hi/gu` (that language's mean residual minus the mean of the other four, literature rows only) and `mood:dread/tender/wonder/rousing/wry` (mean minus 400 random lines), each `[36, 2560]`, one row per layer output. 200 lines per group, raw text, first 48 tokens, first position excluded. Meta carries `residual_norm` per layer. |

Rerun (no downloads; uses the cipher-llm venv, mlx 0.32 / mlx-lm 0.31):

```sh
cd ~/Developer/loom-data
~/Developer/cipher-llm/venv/bin/python "$HOME/Documents/AI Stuff/Loom/Tools/qwen/precompute.py"            # all three
~/Developer/cipher-llm/venv/bin/python "$HOME/Documents/AI Stuff/Loom/Tools/qwen/precompute.py" reference  # or one stage
```

Stages run in the order catalogue, reference, steering, and each writes its file as soon as it finishes.
The steering pass checkpoints to `extras/.steering-partial.npz` and picks up from it if interrupted.
All activations are float32: the model's bf16 scales, biases and norm weights are widened (exactly) first.
