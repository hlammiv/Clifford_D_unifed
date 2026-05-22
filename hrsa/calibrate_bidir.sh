#!/bin/bash
# Calibrate bidir Frob at K_f=K_b ∈ {4, 5} across 8 thetas.
# Outputs table to stdout; per-run logs to /tmp/bidir_<theta>_<K>.log.

cd /home/hlamm/Desktop/efficent_gates/unified/hrsa

THETAS=(0.5 0.7854 1.0053 1.5708 2.0106 2.14 2.5133 2.879)
KS=(4 5)

printf "theta,K,frob,wall_seconds\n" > /tmp/bidir_cal.csv

for K in "${KS[@]}"; do
  for th in "${THETAS[@]}"; do
    log=/tmp/bidir_${th}_${K}.log
    echo ">>> theta=$th K=$K starting at $(date +%H:%M:%S)" >&2
    /usr/bin/time -f "wall=%es" -o ${log}.time \
      timeout 3000 ./bidir_bfs $th $K $K > $log 2>&1
    rc=$?
    frob=$(grep "Approximate-match best" $log 2>/dev/null | awk -F': ' '{print $2}' | awk '{print $1}')
    wall=$(grep wall= ${log}.time 2>/dev/null | head -1 | sed 's/wall=//;s/s$//')
    if [ -z "$frob" ]; then frob="ERR"; fi
    if [ -z "$wall" ]; then wall="?"; fi
    printf "theta=%-7s K=%d frob=%-22s wall=%ss rc=%s\n" "$th" "$K" "$frob" "$wall" "$rc"
    printf "%s,%d,%s,%s\n" "$th" "$K" "$frob" "$wall" >> /tmp/bidir_cal.csv
  done
done
echo "DONE. CSV at /tmp/bidir_cal.csv"
