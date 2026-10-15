#!/usr/bin/env python3
"""Write the parity fixtures the Swift tests compare against (LoomEngine/Tests/LoomEngineTests/Fixtures).

  fixtures.py <model .loom>

Copies the model and tokenizer beside the fixtures, writes a small random adapter, and records the
float64 reference's answers for: tokenisation, the forward pass (plain and with every knob), the logit
lens, quantisation and the sampler chain.
"""
import json, os, shutil, sys

import numpy as np

from bpe import Tokenizer, pieces
from loomio import DATA, write_loom
from ref import MATRICES, Weights, forward, lens, quantize, sampling_distribution

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "LoomEngine", "Tests", "LoomEngineTests", "Fixtures")

TEXTS = [
    "The night was dark and the sea was calm.",
    "Il était une fois, dans un pays lointain…",
    "El caballero se quitó el sombrero — ¿y tú?",
    "आज आबिद होता तो क्या इसी तरह ईद आती",
    "અમારો નિવાસ ઉજળી વસ્તીથી દૂર",
    "Forrest Gump (1994 film, Robert Zemeckis)",
    "  double  spaces, tabs\tand 'quotes' don't break it 12,000x!",
    "emoji 🙂 and rare 漢字 become unknown cards",
    "line one\nline two",
]


def r(a, n=6):
    return [round(float(x), n) for x in np.asarray(a).ravel()]


def mlx_gradients(path, tok):
    """Loss and gradients from MLX autodiff on two fixed sequences, for the Swift backprop test."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from loomio import read_loom
    from model import GPT, Config

    meta, t = read_loom(path)
    m = GPT(Config(**meta["config"]))
    m.wte, m.gf = mx.array(t["wte"]), mx.array(t["gf"])
    for i, b in enumerate(m.blocks):
        for n in ("g1", "g2") + MATRICES:
            setattr(b, n, mx.array(t[f"b{i}.{n}"]))
    seqs = [[1] + tok.encode("The night was dark and the sea was calm and still")[:20],
            [1] + tok.encode("El caballero se quitó el sombrero y miró la luna")[:20]]
    n = min(len(x) for x in seqs)
    seqs = [x[:n] for x in seqs]
    X = mx.array([x[:-1] for x in seqs]); Y = mx.array([x[1:] for x in seqs])
    loss_fn = lambda model: nn.losses.cross_entropy(model(X), Y, reduction="mean")
    loss, grads = nn.value_and_grad(m, loss_fn)(m)
    g = {k: np.array(v) for k, v in tree_flatten(grads)}
    out = {"sequences": seqs, "loss": float(loss.item()), "slices": {}}
    for name, key in [("b0.wq", "blocks.0.wq"), ("b1.wg", "blocks.1.wg"), ("b2.wv", "blocks.2.wv"),
                      ("b3.wd", "blocks.3.wd"), ("b3.wo", "blocks.3.wo"), ("b0.wk", "blocks.0.wk"), ("b1.wu", "blocks.1.wu")]:
        out["slices"][name] = r(g[key][:6, :10], 9)
    out["slices"]["b2.g1"] = r(g["blocks.2.g1"], 9)
    out["slices"]["b0.g2"] = r(g["blocks.0.g2"], 9)
    out["slices"]["gf"] = r(g["gf"], 9)
    rows = sorted(set(seqs[0][:4]))
    out["wte_rows"] = rows
    out["slices"]["wte"] = r(g["wte"][rows], 9)
    return out


def main():
    model = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    shutil.copy(model, os.path.join(OUT, "model.loom"))
    shutil.copy(os.path.join(DATA, "tokenizer.json"), os.path.join(OUT, "tokenizer.json"))
    W = Weights(model)
    tok = Tokenizer.load()

    fx = {"tokenizer": [{"text": t, "pieces": pieces(t), "ids": tok.encode(t)} for t in TEXTS]}

    rng = np.random.default_rng(5)
    # A random rank-4 adapter so mixing is tested with non-zero B.
    lora = {}
    dims = {"wq": (W.d, W.d), "wk": (W.d, W.d), "wv": (W.d, W.d), "wo": (W.d, W.d),
            "wg": (W.d, W.ff), "wu": (W.d, W.ff), "wd": (W.ff, W.d)}
    for l in range(W.layers):
        for m in MATRICES:
            n_in, n_out = dims[m]
            lora[f"b{l}.{m}_a"] = rng.normal(0, 1 / np.sqrt(n_in), (n_in, 4)).astype(np.float32)
            lora[f"b{l}.{m}_b"] = rng.normal(0, 0.02, (4, n_out)).astype(np.float32)
    write_loom(os.path.join(OUT, "adapter.loom"), {"name": "test", "rank": 4, "alpha": 8.0}, lora)
    lora64 = {k: v.astype(np.float64) for k, v in lora.items()}
    steer_vec = rng.normal(0, 1, W.d)

    ids = [1] + tok.encode("The night was dark and the sea")
    cases = {
        "plain": ({}, None),
        "muted": ({"muted": {(0, 1), (2, 3)}}, None),
        "bypass": ({"bypass": {1}}, None),
        "depth2": ({"bypass": {2, 3}}, None),
        "steer": ({"steer": [(1, steer_vec, 2.5)]}, None),
        "clamp": ({"clamp": [(2, 17, 6.0)]}, None),
        "bits4": ({"bits": 4}, None),
        "bits2": ({"bits": 2}, None),
        "bits1": ({"bits": 1}, None),
        "adapter": ({}, [(lora64, 1.5, 8.0, 4)]),
        "adapter_bits4": ({"bits": 4}, [(lora64, -0.7, 8.0, 4)]),
    }
    fx["forward"] = {"ids": ids, "steer_vector": r(steer_vec, 9), "cases": {}}
    for name, (knobs, adapters) in cases.items():
        f = forward(W, ids, knobs, adapters)
        entry = {"logits_last": r(f["logits"][-1]), "final_last": r(f["final"][-1])}
        if name == "plain":
            entry["x0_last"] = r(f["x0"][-1])
            entry["att0"] = r(f["layers"][0]["att"])
            entry["q1_first_rows"] = r(f["layers"][1]["q"][:, :3, :])
            entry["act2_last"] = r(f["layers"][2]["act"][-1])
            entry["out3_all"] = r(f["layers"][3]["out"])
            entry["logits_all_argmax"] = [int(np.argmax(x)) for x in f["logits"]]
            entry["lens"] = [r(lens(W, L["out"][-1])[:64]) for L in f["layers"]]
        fx["forward"]["cases"][name] = entry

    m = W["b0.wg"]
    fx["quantize"] = {f"b{b}": r(quantize(m, b)[:8, :8]) for b in (8, 4, 3, 2, 1)}

    f = forward(W, ids)
    logits = f["logits"][-1]
    settings = [
        {"temperature": 1.0},
        {"temperature": 0.7, "top_k": 20},
        {"temperature": 1.3, "top_p": 0.9},
        {"temperature": 0.9, "min_p": 0.05},
        {"temperature": 1.0, "repetition": 1.3},
        {"temperature": 0.8, "top_k": 50, "top_p": 0.8, "min_p": 0.02, "repetition": 1.2, "bias": {"1": -1e9, "9": 2.0}},
        {"temperature": 0},
    ]
    fx["sampler"] = {"logits": r(logits, 7), "context": ids, "cases": []}
    for s in settings:
        args = dict(s)
        bias = {int(k): v for k, v in args.pop("bias", {}).items()}
        q, heat = sampling_distribution(logits, ids, bias=bias, **args)
        order = np.argsort(-q, kind="stable")
        fx["sampler"]["cases"].append({
            "settings": s, "kept": int((q > 0).sum()),
            "top": [[int(i), round(float(q[i]), 9)] for i in order[:12]],
            "heat_top": [[int(i), round(float(heat[i]), 9)] for i in np.argsort(-heat, kind="stable")[:12]],
        })
    fx["grads"] = mlx_gradients(model, tok)
    json.dump(fx, open(os.path.join(OUT, "fixtures.json"), "w"), ensure_ascii=False)
    print("wrote", OUT, os.path.getsize(os.path.join(OUT, "fixtures.json")) // 1024, "KB")


if __name__ == "__main__":
    main()
