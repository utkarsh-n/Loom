#!/bin/zsh
# Copies a finished run's files into the app bundle's Model folder.
#   install.sh [run name]
set -e
RUN=${1:-v3}
D=${LOOM_DATA:-$HOME/Developer/loom-data}
DST="$(cd "$(dirname "$0")/../.." && pwd)/Loom/Resources/Model"
mkdir -p "$DST"
find "$DST" -name "*.loom" -delete; find "$DST" -name "*.json" -delete
cp "$D/tokenizer.json" "$DST/"
cp "$D/runs/$RUN/loom-base.loom" "$D/runs/$RUN"/loom-age-{40,250,1500}.loom "$DST/"
for v in scripture bard gothic twain almanac; do cp "$D/runs/$RUN/adapter-$v.loom" "$DST/"; done
cp "$D/runs/$RUN/steering.loom" "$D/runs/$RUN/neurons.json" "$D/runs/$RUN/galaxy.json" "$DST/"
du -sh "$DST"; ls "$DST"
