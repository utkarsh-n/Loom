#!/bin/zsh
# Builds the LoomEngine harness (engine sources + main.swift) with -O and runs it.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
BIN="${TMPDIR:-/tmp}/loom-cli"
swiftc -O -Xcc -DACCELERATE_NEW_LAPACK -o "$BIN" "$HERE"/../../LoomEngine/Sources/LoomEngine/*.swift "$HERE/main.swift"
"$BIN" "$@"
