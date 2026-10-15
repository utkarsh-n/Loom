# Shared offline explanations

Loom's **Ask offline** sheet uses `OfflineIntelligence` and `OfflineIntelligenceMLX`
from the sibling `OfflineIntelligenceKit` packages. The persisted explanation model
defaults to Qwen3.5 2B; Qwen3.5 4B is an explicit option. Both resolve verified local
folders through the shared `LocalModelFiles` catalog. Missing selected weights are
reported rather than silently substituted.

The instrumented pipeline continues to use its existing Qwen3 4B checkpoint and
custom layer walk. Its real attention, residuals, interventions and same-checkpoint
baseline/tuned comparisons cannot be represented by ordinary Qwen3.5 text generation.
Explanations receive a bounded JSON snapshot of the current instrument; they are
generated prose, not additional measured tensors.

Opening the explanation sheet stops pipeline work and drains its Qwen3 load/worker
before allowing the shared runtime to load. Closing the sheet drains explanations
before resuming the specialist. The specialist also owns the shared external model
lease, preventing simultaneous GPU model residency even if a host lifecycle path
is called unexpectedly. Background and memory-pressure paths cancel and unload;
partial explanation output stays visible and an interrupted answer is not resumed
automatically. Foreground re-entry re-enables an explicit new request.

Shared checkpoints are staged as `Models/qwen-2b` and `Models/qwen-4b`, with immutable
revision metadata. The original specialist remains under `Qwen`. Weight files stay
outside the iCloud source tree.

Validation for this integration: changed Swift sources passed frontend syntax
parsing. Full physical-target compilation, signing and phone behavior are handled
by the parent integration run. No simulator tools were used during the active hold.
