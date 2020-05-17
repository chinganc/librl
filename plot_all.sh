logs=$1
for log in $logs/*; do 
	if [ -d $log ]; then 
		. plot.sh $log $log
	fi
done
