# Public source snapshot

This repository contains a reviewed source snapshot. Original Git history, credentials, personal records, local signing identities, device diagnostics, build outputs, downloaded model weights, firmware, and unreviewed media are excluded. Existing upstream license and attribution files apply to their respective components. Publication does not grant a new license to third-party content.

Configure your own Apple signing team to run native apps on a physical device. Downloaded datasets, binary runtimes, fonts, demo media, and models may need to be obtained separately using the documented upstream sources or fetch/build tools. This publication review does not certify that this snapshot builds or runs.

## Project setup and omitted assets

Clone [OfflineIntelligenceKit](https://github.com/utkarsh-n/OfflineIntelligenceKit) beside this repository; `project.yml` references its core package and `MLX` subpackage. Qwen/shared model weights under `Loom/Resources/Qwen/` and `Loom/Resources/Models/` are omitted; the placeholder setup README files are retained.

The original tiny-model tensor binaries (`Loom/Resources/Model/*.loom`) and `LoomEngine/Tests/LoomEngineTests/Fixtures/model.loom` are also omitted. The retained tokenizer/visualization JSON does not replace these tensors. `Tools/train/` retains the training/export pipeline; it expects the separately prepared public-domain Cipher corpus. Set `CIPHER_RES` to that resource directory, or clone the published Cipher repository beside this one as `Cipher-public`. `Tools/qwen/README.md` documents the additional precomputed Qwen metadata. Supply/rebuild those assets and review the pinned upstream licenses before regenerating the Xcode project. Full-model tests require the missing tensor fixture.
