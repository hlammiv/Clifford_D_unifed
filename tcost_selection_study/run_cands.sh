#!/bin/bash
# Usage: run_cands.sh EPS MAXSOLNS MAXF "THETAS" [TIMEOUT_S]
# Runs HRSA_tester (HRSA_bestD, --no-direct) per angle sequentially; emits CANDDUMP/CANDTCOST.
EPS=$1; NS=$2; MF=$3; THS=$4; TO=${5:-5400}
HR=/home/hlamm/Desktop/efficent_gates/unified/hrsa/HRSA_tester
OUT=/home/hlamm/Desktop/efficent_gates/unified/tcost_selection_study/raw
for TH in $THS; do
  /usr/bin/time -v env OMP_NUM_THREADS=${OMP:-8} timeout -s INT $TO $HR $TH $EPS $MF --no-direct --max-solns $NS \
     > $OUT/hrsa_t${TH}_e${EPS}.log 2> $OUT/hrsa_t${TH}_e${EPS}.err
  echo "done $TH $EPS rc=$? $(date)"
done
