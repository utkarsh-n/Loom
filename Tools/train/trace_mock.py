#!/usr/bin/env python3
"""Real numbers for the design mockups: one prompt traced through the float64 reference.

  trace_mock.py <run name> "<prompt>"     prints JSON
"""
import json, sys

import numpy as np

from bpe import Tokenizer, pieces
from loomio import DATA
from ref import Weights, forward, lens, sampling_distribution, softmax


def script_of(tok):
    for ch in tok:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: return "hi"
        if 0x0A80 <= o <= 0x0AFF: return "gu"
        if ch.isalpha(): return "lat"
        if ch.isdigit(): return "num"
    return "punct"


def main():
    run, prompt = sys.argv[1], sys.argv[2]
    W = Weights(f"{DATA}/runs/{run}/loom-base.loom")
    tok = Tokenizer.load()
    ids = [1] + tok.encode(prompt)
    out = {"prompt": prompt, "tokens": [[i, tok.vocab[i]] for i in ids], "pieces": pieces(prompt)}
    f = forward(W, ids)
    p, _ = sampling_distribution(f["logits"][-1], ids, temperature=1.0)
    order = np.argsort(-p)[:10]
    out["next"] = [[tok.vocab[i], round(float(p[i]), 4), round(float(f["logits"][-1][i]), 2)] for i in order]
    for t in (0.5, 1.6):
        q, _ = sampling_distribution(f["logits"][-1], ids, temperature=t)
        out[f"next_t{t}"] = [[tok.vocab[i], round(float(q[i]), 4)] for i in order]
    q, _ = sampling_distribution(f["logits"][-1], ids, temperature=1.0, top_p=0.9)
    out["top_p_0.9_kept"] = int((q > 0).sum())
    out["attention_last"] = [[[round(float(x), 3) for x in L["att"][h][-1]] for h in range(W.heads)] for L in f["layers"]]
    out["lens"] = []
    for l, L in enumerate(f["layers"]):
        row = []
        for pos in range(len(ids)):
            lp = softmax(lens(W, L["out"][pos]))
            j = int(np.argmax(lp))
            row.append([tok.vocab[j], round(float(lp[j]), 3)])
        out["lens"].append(row)
    act = f["layers"][1]["act"][-1]
    top = np.argsort(-np.abs(act))[:6]
    neurons = []
    for n in top:
        push = W["b1.wd"][n] @ W["wte"].T
        neurons.append({"n": int(n), "act": round(float(act[n]), 3),
                        "pushes": [tok.vocab[i] for i in np.argsort(-push)[:5]],
                        "pulls": [tok.vocab[i] for i in np.argsort(push)[:3]]})
    out["neurons_floor2"] = neurons
    out["act_floor2_last"] = [round(float(x), 2) for x in act]
    E = W["wte"]
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    for word in [" night", " love", " king"]:
        i = tok.index.get(word)
        if i is None: continue
        sims = En @ En[i]
        out[f"neighbours{word}"] = [[tok.vocab[j], round(float(sims[j]), 3)] for j in np.argsort(-sims)[1:9]]
    X = E - E.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / (S ** 2).sum()
    out["pca_var3"] = round(float(var[:3].sum()), 4)
    P = X @ Vt[:3].T
    P /= np.abs(P).max()
    rng = np.random.default_rng(3)
    pick = sorted(set(rng.choice(np.arange(8, W.vocab), 420, replace=False).tolist()) | set(ids[1:]))
    out["stars"] = [[round(float(P[i, 0]), 3), round(float(P[i, 1]), 3), round(float(P[i, 2]), 3), script_of(tok.vocab[i]), tok.vocab[i]] for i in pick]
    out["prompt_stars"] = [[round(float(P[i, 0]), 3), round(float(P[i, 1]), 3), round(float(P[i, 2]), 3), tok.vocab[i]] for i in ids[1:]]
    # a seeded weave at τ 0.8, top-p 0.95
    rng = np.random.default_rng(11)
    seq, cloth = list(ids), []
    for _ in range(24):
        g = forward(W, seq[-W.ctx:])
        q, _ = sampling_distribution(g["logits"][-1], seq, temperature=0.8, top_p=0.95)
        u = rng.random()
        j = int(np.searchsorted(np.cumsum(q), u))
        cloth.append([tok.vocab[j], round(float(q[j]), 3)])
        seq.append(j)
        if j == 1: break
    out["cloth"] = cloth
    # the residual strip of the last thread at x0 and after each floor (first 32 dims)
    out["strip"] = [[round(float(v), 3) for v in f["x0"][-1][:32]]] + [[round(float(v), 3) for v in L["out"][-1][:32]] for L in f["layers"]]
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
