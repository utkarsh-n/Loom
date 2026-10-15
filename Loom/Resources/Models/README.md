The shared offline explanation layer uses pinned Qwen3.5 2B (default) or optional 4B.
Device build staging provisions verified `qwen-2b` and `qwen-4b` folders from
`~/Developer/offline-intelligence-models`; do not put the weight files in the source tree.

The separate `Resources/Qwen` checkpoint belongs to Loom's instrumented Qwen3 pipeline.
It supplies real attention, residuals, steering and interventions. Explanation models
do not replace or imitate those tensors. Loom drains that specialist before loading
the shared explanation model and drains explanations before resuming the specialist.
