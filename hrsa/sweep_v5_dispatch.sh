#!/bin/bash
# v5 expansion sweep: 16 new cells at existing θ × tighter ε,
# plus 24 new cells at 8 interleaved θ × {1e-3, 5e-4, 1e-4}.
#
# Phase 1: cells expected to land at f<=3, run 6 in parallel @ 4 cores each.
# Phase 2: cells expected to land at f=4, run serial @ 32 cores each.

set -u
cd "$(dirname "$0")"

OUT_DIR=/home/hlamm/Desktop/efficent_gates/unified/sweep_v5_2026-05-10
mkdir -p "$OUT_DIR"
DISP_LOG=$OUT_DIR/dispatch.log
: > "$DISP_LOG"

BIN=./HRSA_tester
MAX_F=4
MAX_SOLNS=10

PARALLEL_F3=6
OMP_F3=4
TIMEOUT_F3=5400     # 90 min (bumped from 30 min after first batch all timed out)
OMP_F4=32
TIMEOUT_F4=5400     # 90 min

# 8 existing θ:    0.5  0.7854  1.0053  1.5708  2.0106  2.14  2.5133  2.879
# 8 new (interleaved) θ:  0.25  0.6427  0.8954  1.2880  1.7907  2.0753  2.3267  2.6962

# --------- Phase 1: f<=3 likely cells (parallel, 4-core) ---------
# 8 new θ × {1e-3, 5e-4} = 16 cells.
read -r -d '' CELLS_F3 <<'EOF'
0.25    0.001
0.25    0.0005
0.6427  0.001
0.6427  0.0005
0.8954  0.001
0.8954  0.0005
1.2880  0.001
1.2880  0.0005
1.7907  0.001
1.7907  0.0005
2.0753  0.001
2.0753  0.0005
2.3267  0.001
2.3267  0.0005
2.6962  0.001
2.6962  0.0005
EOF

# --------- Phase 2: f=4 likely cells (serial, 32-core) ---------
# 8 existing θ × {3e-4, 1e-4}  +  8 new θ × {1e-4} = 24 cells.
read -r -d '' CELLS_F4 <<'EOF'
0.5     0.0003
0.5     0.0001
0.7854  0.0003
0.7854  0.0001
1.0053  0.0003
1.0053  0.0001
1.5708  0.0003
1.5708  0.0001
2.0106  0.0003
2.0106  0.0001
2.14    0.0003
2.14    0.0001
2.5133  0.0003
2.5133  0.0001
2.879   0.0003
2.879   0.0001
0.25    0.0001
0.6427  0.0001
0.8954  0.0001
1.2880  0.0001
1.7907  0.0001
2.0753  0.0001
2.3267  0.0001
2.6962  0.0001
EOF

# Per-cell runner
run_cell() {
    local theta=$1 eps=$2 cores=$3 timeout_s=$4 label=$5
    local tag="${label}_t${theta}_e${eps}"
    local out_json="$OUT_DIR/${tag}.json"
    local out_log="$OUT_DIR/${tag}.log"
    local start=$(date +%s)
    timeout "$timeout_s" env OMP_NUM_THREADS="$cores" \
        "$BIN" "$theta" "$eps" "$MAX_F" --max-solns "$MAX_SOLNS" --json "$out_json" \
        > "$out_log" 2>&1
    local rc=$?
    local end=$(date +%s)
    printf '[%s] theta=%s eps=%s cores=%d rc=%d wall=%ds\n' \
        "$label" "$theta" "$eps" "$cores" "$rc" "$((end-start))" >> "$DISP_LOG"
}
export -f run_cell
export OUT_DIR DISP_LOG BIN MAX_F MAX_SOLNS

echo "=== Phase 1: f<=3 cells, $PARALLEL_F3 parallel @ $OMP_F3 cores ===" | tee -a "$DISP_LOG"
echo "$CELLS_F3" | awk 'NF==2{print $1, $2}' | \
    xargs -n2 -P "$PARALLEL_F3" bash -c \
    'run_cell "$1" "$2" '"$OMP_F3"' '"$TIMEOUT_F3"' f3' _

echo "=== Phase 2: f=4 cells, serial @ $OMP_F4 cores ===" | tee -a "$DISP_LOG"
while read -r theta eps _; do
    [ -z "${theta:-}" ] && continue
    run_cell "$theta" "$eps" "$OMP_F4" "$TIMEOUT_F4" f4
done < <(echo "$CELLS_F4" | awk 'NF==2{print}')

echo "=== DONE ===" | tee -a "$DISP_LOG"
