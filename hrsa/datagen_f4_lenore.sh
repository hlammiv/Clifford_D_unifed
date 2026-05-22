#!/bin/bash
# f=4 dataset generation, LENORE (low-θ half).
# Serial, OMP=32, 90-min timeout per cell.

set -u
cd "$(dirname "$0")"

OUT=/home/hlamm/Desktop/efficent_gates/unified/canddump_f4_lenore
mkdir -p "$OUT"
LOG=$OUT/dispatch.log
: > "$LOG"

OMP=32
TIMEOUT=18000  # 5 hours
MAX_SOLNS=30   # was 100; bumped down 2026-05-10 since K_3=1 + max-solns 100 was 5-15x slower than v4

THETAS="0.5 0.7854 1.0053 1.5708"
EPSILONS="0.001 0.0005 0.0001"

for theta in $THETAS; do
    for eps in $EPSILONS; do
        tag="t${theta}_e${eps}"
        out_dump="$OUT/${tag}.txt"
        out_json="$OUT/${tag}.json"
        if [ -s "$out_dump" ] && grep -q "^CANDDUMP" "$out_dump"; then
            echo "[skip] $tag already has data" >> "$LOG"
            continue
        fi
        start=$(date +%s)
        echo ">>> $(date -Iseconds)  starting theta=$theta eps=$eps  OMP=$OMP" >> "$LOG"
        timeout "$TIMEOUT" env OMP_NUM_THREADS="$OMP" \
            ./HRSA_tester "$theta" "$eps" 4 --max-solns "$MAX_SOLNS" --k3 1 --json "$out_json" \
            > "$out_dump" 2>&1
        rc=$?
        end=$(date +%s)
        ncands=$(grep -c "^CANDDUMP" "$out_dump" 2>/dev/null || echo 0)
        method=$(grep "^Method:" "$out_dump" | tail -1 | awk '{print $NF}')
        nd=$(grep "^Total D gates" "$out_dump" | tail -1 | awk '{print $NF}')
        echo "    rc=$rc wall=$((end-start))s ncands=$ncands method=$method N_D=$nd" >> "$LOG"
    done
done
echo "DATAGEN_F4_LENORE_DONE $(date -Iseconds)" >> "$LOG"
