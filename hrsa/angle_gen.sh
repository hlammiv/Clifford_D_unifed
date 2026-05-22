rm run_file
maxf=12
for power in 0.00001; do
#for power in 1 0.1 0.01 0.001; do
for j in $(seq 1 1 100);
do
angle=`awk -v seed="$RANDOM" 'BEGIN { srand(seed); printf("%.10f\n", (rand()-0.5) * 3.1415926535897) }'`
#echo $angle, $power
	#echo "./HRSA_tester $angle $power $maxf 0.8 --max-solns 10 >> data/out_a${angle}_e${power}_f${maxf}" >> run_file
	echo "./HRSA_tester $angle $power $maxf 0.8 --no-direct >> data/out_a${angle}_e${power}_f${maxf}" >> run_file
done
done
chmod +x run_file
