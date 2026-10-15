"""Shared I/O for Loom's training tools: the token stream and the .loom tensor file.

.loom file (read by Loom/Engine/WeightFile.swift):
  8 bytes   magic "LOOMv1\\0\\0"
  4 bytes   little-endian uint32 header length H
  H bytes   UTF-8 JSON: {"meta": {...}, "tensors": [{"name", "shape", "offset"}]}
  padding   to a 16-byte boundary; offsets count from here, each tensor 16-byte aligned
  data      little-endian, row-major; each tensor's "dtype" is "f32" (default) or "f16"
"""
import json, os, struct, zlib

import numpy as np

DATA = os.environ.get("LOOM_DATA", os.path.expanduser("~/Developer/loom-data"))
FAMILIES = ["dread", "tender", "wonder", "rousing", "wry"]
LANGS = ["en", "es", "fr", "hi", "gu"]
MAGIC = b"LOOMv1\0\0"

# Five voices, each taught by its adapter from text the base model never reads (train.py holds these
# rows out). Strong, surface-level voices on purpose: a 1.4M-parameter model can learn "And it came to
# pass" from 2,000 lines, but not a mood from them.
VOICES = ["scripture", "bard", "gothic", "twain", "almanac"]
SETS = {
    "scripture": ["Book of Mormon", "The Woman's Bible"],
    "bard": ["Hamlet", "Macbeth", "Romeo and Juliet", "King Lear", "A Midsummer Night's Dream", "Julius Caesar",
             "The Tempest", "The Merchant of Venice"],
    "gothic": ["Dracula", "Frankenstein; or, The Modern Prometheus", "The Picture of Dorian Gray",
               "Strange Case of Dr Jekyll and Mr Hyde", "The Fall of the House of Usher", "The Raven"],
    "twain": ["The Adventures of Tom Sawyer", "Adventures of Huckleberry Finn", "A Connecticut Yankee in King Arthur's Court",
              "Tom Sawyer Abroad", "Tom Sawyer, Detective"],
    "almanac": [],      # every fact card written from Cipher's culture index
}


def line_works():
    """Slice, work title and language of every row of lines.tsv, in stream order."""
    slices, works, langs = [], [], []
    with open(os.path.join(DATA, "lines.tsv"), encoding="utf-8") as f:
        for row in f:
            c = row.split("\t")
            slices.append(c[0]); works.append(c[2]); langs.append(c[1])
    return slices, works, langs


def voice_rows(voice):
    slices, works, langs = line_works()
    if voice == "almanac":
        return np.array([s == "card" for s in slices])
    return np.array([s == "lit" and w in SETS[voice] for s, w in zip(slices, works)])


def adapter_rows():
    """Every row an adapter teaches: never shown to the base."""
    out = voice_rows(VOICES[0])
    for v in VOICES[1:]:
        out |= voice_rows(v)
    return out


def write_loom(path, meta, tensors, half=False):
    dt = "<f2" if half else "<f4"
    table, offset = [], 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=dt)
        table.append({"name": name, "shape": list(arr.shape), "offset": offset, "dtype": "f16" if half else "f32"})
        offset += (arr.nbytes + 15) // 16 * 16
    header = json.dumps({"meta": meta, "tensors": table}, ensure_ascii=False).encode()
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<I", len(header)) + header)
        f.write(b"\0" * ((-f.tell()) % 16))
        for name, arr in tensors.items():
            arr = np.ascontiguousarray(arr, dtype=dt)
            f.write(arr.tobytes())
            f.write(b"\0" * ((-arr.nbytes) % 16))


def read_loom(path):
    raw = open(path, "rb").read()
    assert raw[:8] == MAGIC
    (n,) = struct.unpack("<I", raw[8:12])
    head = json.loads(raw[12:12 + n])
    base = 12 + n + ((-(12 + n)) % 16)
    out = {}
    for t in head["tensors"]:
        count = int(np.prod(t["shape"])) if t["shape"] else 1
        dt = "<f2" if t.get("dtype") == "f16" else "<f4"
        out[t["name"]] = np.frombuffer(raw, dtype=dt, count=count, offset=base + t["offset"]).reshape(t["shape"]).astype("<f4")
    return head["meta"], out


def is_validation(text):
    return zlib.crc32(text.encode()) % 50 == 0


def build_stream(tok):
    """Encode lines.tsv once. Every line starts with token 1 ("\\n")."""
    cache = os.path.join(DATA, "stream.npz")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(os.path.join(DATA, "tokenizer.json")):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    ids, starts, fam, lang, val, card = [], [], [], [], [], []
    with open(os.path.join(DATA, "lines.tsv"), encoding="utf-8") as f:
        for row in f:
            s, l, w, fams, text = row.rstrip("\n").split("\t")
            starts.append(len(ids))
            ids.append(1)
            if zlib.crc32(("tag:" + text).encode()) % 2 == 0:          # half the lines carry their cartridge tag
                ids.append(tok.index["⟨card⟩"] if s == "card" else tok.index["⟨%s⟩" % l])
            ids.extend(tok.encode(text))
            fam.append(sum(1 << FAMILIES.index(x) for x in fams.split("|") if x))
            lang.append(LANGS.index(l) if l in LANGS else 0)
            val.append(is_validation(text))
            card.append(s == "card")
    starts.append(len(ids))
    out = {
        "ids": np.array(ids, dtype=np.uint16),
        "starts": np.array(starts, dtype=np.int64),
        "family": np.array(fam, dtype=np.uint8),
        "lang": np.array(lang, dtype=np.uint8),
        "val": np.array(val, dtype=bool),
        "card": np.array(card, dtype=bool),
    }
    np.savez(cache, **out)
    return out
