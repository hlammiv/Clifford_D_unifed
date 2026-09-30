#!/bin/bash
# Usage: run_hrsa_cands.sh EPS MAXSOLNS MAXF  -- runs HRSA_bestD (CANDDUMP) for 5 angles sequentially
EPS=$1; NS=$2; MF=$3
HR=/home/hlamm/Desktop/efficent_gates/unified/hrsa/HRSA_tester
OUT=/home/hlamm/Desktop/efficent_gates/unified/r_count_study/raw
for TH in 0.3 0.5 1.0 1.7 2.5; do
  OMP_NUM_THREADS=${OMP:-8} timeout 1800 $HR $TH $EPS $MF --no-direct --max-solns $NS \
     > $OUT/hrsa_t${TH}_e${EPS}.log 2> $OUT/hrsa_t${TH}_e${EPS}.err
  echo "done $TH $EPS rc=$?"
done
