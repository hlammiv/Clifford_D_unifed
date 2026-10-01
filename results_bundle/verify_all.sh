#!/usr/bin/env bash
# verify_all.sh — re-run the checks behind results_bundle/*.md.
#   ./verify_all.sh          quick checks (~2–5 min)
#   ./verify_all.sh --full   also the slow ones (reducer tests, 2-ancilla controls, factory model)
# Run from anywhere; paths are relative to unified/.
set -u
U="$(cd "$(dirname "$0")/.." && pwd)"
FULL=0; [ "${1:-}" = "--full" ] && FULL=1
pass=0; fail=0
run() {  # run <label> <dir> <cmd...>
  local label=$1 dir=$2; shift 2
  printf '%-58s ' "$label"
  if out=$(cd "$U/$dir" && "$@" 2>&1); then echo "OK"; pass=$((pass+1));
  else echo "FAIL"; echo "$out" | tail -5 | sed 's/^/    /'; fail=$((fail+1)); fi
}
run "tables regenerate from data"                       results_bundle python3 compute_tables.py
run "R = 7 T (and level-4 word), exact action"          r_from_d       python3 verify_R7T.py
run "level-4 = 7 T (C2X ; T ; C2X†), exact action"     level4         python3 l4_7T_verify.py
run "level-4 = 8 T from GRVY block (history)"           level4         python3 l4_from_8T.py
run "prior level-4 construction counts 224 T"           level4/novelty python3 yeh_vdw_cor4_tcount.py
run "2-ancilla ≤6-T exclusion certificate"              two_ancilla    python3 certify.py
run "Clifford+T emitter (f=4, 2 matrices) T = formula"  compiler       python3 cd_to_ct.py --f 4 --n 2
run "Galois shadows are unitary (demo)"                 nick_test      python3 galois_shadows_demo.py --rows 0
if [ $FULL = 1 ]; then
  run "canonical reducer regression (residual-R fix)"   hrsa           python3 test_canonical_reducer.py
  run "3-qutrit Clifford-equivalence controls"          two_ancilla    python3 test_equiv.py
  run "R optimal with 1 ancilla (merge-free, ≤6 none)"  r_from_d       python3 raw_lb_1anc.py
  run "factory conversions (1/2, 1/4, 2.67x)"           factory_model  python3 conversions.py
  run "factory cost model"                              factory_model  python3 factory_cost.py
fi
echo "passed $pass, failed $fail"
[ $fail = 0 ]
