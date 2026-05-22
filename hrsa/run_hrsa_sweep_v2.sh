#!/bin/bash
# v2: sequential per machine, --max-solns 10, OMP cap.
#
# Args:
#   $1  = parity ("even" → cells 0,2,4,..; "odd" → cells 1,3,5,..)
#   $2  = output dir (relative to hrsa cwd)
#   $3  = OMP_NUM_THREADS cap

set -uo pipefail
cd /home/hlamm/Desktop/efficent_gates/unified/hrsa

PARITY=${1:-all}
OUT=${2:-sweep_v2}
OMP=${3:-8}
mkdir -p $OUT
export OMP_NUM_THREADS=$OMP

THETAS=(0.5 0.7854 1.0053 1.5708 2.0106 2.14 2.5133 2.879)
EPSILONS=(0.5 0.3 0.1 0.05 0.01 0.005 0.001)

echo "theta,epsilon,method,N_D,f,achieved_frob,validate_frob,wall_s,all_checks_pass" > $OUT/sweep.csv.partial

idx=0
for th in "${THETAS[@]}"; do
  for eps in "${EPSILONS[@]}"; do
    if [ "$PARITY" = "even" ] && (( idx % 2 != 0 )); then idx=$((idx+1)); continue; fi
    if [ "$PARITY" = "odd"  ] && (( idx % 2 == 0 )); then idx=$((idx+1)); continue; fi
    idx=$((idx+1))

    log=$OUT/cell_${th}_${eps}.log
    json=$OUT/h_${th}_${eps}.json
    vjson=$OUT/v_${th}_${eps}.json
    echo ">>> [$idx] theta=$th eps=$eps OMP=$OMP starting at $(date +%H:%M:%S)" >&2
    t0=$(date +%s)
    timeout 1800 ./HRSA_tester $th $eps 6 \
        --max-direct 3 --use-bidir 5 --max-solns 10 --json $json \
        > $log 2>&1
    rc=$?
    t1=$(date +%s)
    wall=$((t1-t0))

    if [ $rc -eq 124 ]; then
      echo "$th,$eps,TIMEOUT,,,,,${wall},false" >> $OUT/sweep.csv.partial
      echo "    TIMEOUT wall=${wall}s" >&2
      continue
    fi
    if [ $rc -ne 0 ] || [ ! -s $json ]; then
      echo "$th,$eps,RUN_FAIL,,,,,${wall},false" >> $OUT/sweep.csv.partial
      echo "    FAIL rc=$rc" >&2
      continue
    fi

    /home/hlamm/Desktop/efficent_gates/unified/v_validate.py $json > $vjson 2>>$log
    python3 - <<PYEOF
import json
with open("$json") as f: r = json.load(f)
with open("$vjson") as f: v = json.load(f)
ach = r.get("achieved", {}); dec = r.get("decomposition", {}); un = r.get("unitary", {})
method = ach.get("method","?")
N_D    = dec.get("N_D","")
f_lvl  = un.get("f","")
frob   = ach.get("achieved_frob","")
ok     = bool(v.get("all_checks_pass", False)) and bool(v.get("is_unitary", False))
with open("$OUT/sweep.csv.partial","a") as o:
    o.write(f"$th,$eps,{method},{N_D},{f_lvl},{frob},{v.get('frobenius','')},{$wall},{str(ok).lower()}\n")
print(f"    DONE method={method} N_D={N_D} wall=${wall}s ok={ok}", flush=True)
PYEOF
  done
done

{ head -1 $OUT/sweep.csv.partial; tail -n +2 $OUT/sweep.csv.partial | sort -t, -k1,1g -k2,2g; } > $OUT/sweep.csv
rm -f $OUT/sweep.csv.partial
echo "DONE. CSV: $OUT/sweep.csv"
wc -l $OUT/sweep.csv
