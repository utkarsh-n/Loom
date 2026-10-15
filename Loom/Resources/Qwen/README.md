Qwen3-4B-Instruct-2507 (4-bit MLX, Apache-2.0) ships in this folder of the app, but it isn't kept here:
`Tools/build.sh device` clones the weights and tokenizer from
`~/Developer/loom-data/qwen/Qwen3-4B-Instruct-2507-4bit/` and the extras from
`~/Developer/loom-data/qwen/extras/` (made by `Tools/qwen/precompute.py`) into the build mirror.
