#!/bin/zsh
# xcodebuild hangs loading projects from this iCloud-synced folder, so mirror the project
# to a local scratch directory and build there (same workaround as Gyre and Cipher).
#   Tools/build.sh sim       → Debug build for the iPhone 17 Pro simulator
#   Tools/build.sh device    → Release build for a connected iPhone
#   Tools/build.sh test      → the engine's parity and training tests on the Mac
# Device builds carry Qwen3-4B (2.3 GB): its files are cloned into the mirror (APFS clones, instant),
# never into the iCloud folder. Simulator builds leave them out; MLX can't run there.
set -e
SRC="$(cd "$(dirname "$0")/.." && pwd)"
if [[ "${1:-sim}" == device ]]; then
  exec python3 "$SRC/../OfflineIntelligenceKit/Tools/build_hosts.py" --app Loom
fi
if [[ "${1:-sim}" == sim ]]; then
  print "Simulator hold is active. See OfflineIntelligenceKit/Documentation/SIMULATOR_HOLD.md; use device or test until explicitly released."
  exit 2
fi
WORK="${LOOM_BUILD_DIR:-${TMPDIR:-/tmp}/LoomBuild}"
mkdir -p "$WORK"
rsync -a --delete --exclude 'Loom.xcodeproj/xcuserdata' --exclude '.build' --exclude '* 2.*' \
  --exclude 'Loom/Resources/Qwen/*' "$SRC/" "$WORK/src/"
cp "$SRC/Loom/Resources/Qwen/README.md" "$WORK/src/Loom/Resources/Qwen/README.md"
QWEN="${LOOM_QWEN:-$HOME/Developer/loom-data/qwen}"
QDIR="$WORK/src/Loom/Resources/Qwen"
qwen_files() {
  for f in config.json tokenizer.json tokenizer_config.json model.safetensors; do echo "$QWEN/Qwen3-4B-Instruct-2507-4bit/$f"; done
  for f in qwen-catalogue.json qwen-steering.loom qwen-reference.json; do echo "$QWEN/extras/$f"; done
}
if [[ "${1:-sim}" == device ]]; then
  for f in $(qwen_files); do
    [[ -f "$f" ]] || { echo "missing $f (see Tools/qwen/README.md)"; exit 1; }
    t="$QDIR/${f:t}"
    [[ -f "$t" && $(stat -f %z "$f") == $(stat -f %z "$t") && ! "$f" -nt "$t" ]] || cp -c -p "$f" "$t"
  done
else
  for f in $(qwen_files); do rm -f "$QDIR/${f:t}"; done
fi
cd "$WORK/src"
case "${1:-sim}" in
  test)
    cd LoomEngine && swift test -c release --scratch-path "$WORK/engine" ;;
  sim)
    xcodegen generate --quiet
    xcodebuild -project Loom.xcodeproj -scheme Loom -configuration Debug \
      -destination 'platform=iOS Simulator,name=iPhone 17 Pro' -derivedDataPath "$WORK/dd" \
      -skipPackagePluginValidation -skipMacroValidation build ;;
  device)
    xcodegen generate --quiet
    xcodebuild -project Loom.xcodeproj -scheme Loom -configuration Release \
      -destination 'generic/platform=iOS' -allowProvisioningUpdates -derivedDataPath "$WORK/dd" \
      -skipPackagePluginValidation -skipMacroValidation build ;;
esac
