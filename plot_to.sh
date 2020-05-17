logs=$1
for log in $logs/*; do 
	if [ -d $log ]; then 
		blog=$(basename $log)
		mkdir results/$blog
		. plot.sh $log results/$blog
	fi
done
