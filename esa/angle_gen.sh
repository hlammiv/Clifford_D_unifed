rm run_file
maxf=4
for power in 0.1; do
for j in $(seq 1 1 40);
do
angle=`awk -v seed="$RANDOM" 'BEGIN { srand(seed); printf("%.10f\n", (rand()-0.5) * 3.1415926535897) }'`
#echo $angle, $power
	echo "./ESA_tester $angle $power $maxf >> data/out_a${angle}_e${power}_f${maxf}" >> run_file
done
done
chmod +x run_file
