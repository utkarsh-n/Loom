#!/bin/zsh
# The whole Mac-side build of Loom's model files, in order. Logs go to $LOOM_DATA/<run>-*.log.
#   pipeline.sh <run name> [base steps]
set -e
RUN=${1:-v2}
STEPS=${2:-12000}
D=${LOOM_DATA:-$HOME/Developer/loom-data}
cd "$(dirname "$0")"
python3 train.py --name $RUN --steps $STEPS --batch 64 --ages 40,250,1500 > $D/$RUN-train.log 2>&1
python3 adapters.py $RUN --steps 400 --lr 2e-3 > $D/$RUN-adapters.log 2>&1
python3 extras.py $RUN > $D/$RUN-extras.log 2>&1
python3 fixtures.py $D/runs/$RUN/loom-age-1500.loom > $D/$RUN-fixtures.log 2>&1
echo done > $D/$RUN-done.txt
