log_folder=$1
plot_folder=$2
python utils/plot.py -v MeanSumOfRewards -d $log_folder -o $plot_folder &
python utils/plot.py -v BestSumOfRewards -d $log_folder -o $plot_folder &
python utils/plot.py -v std -d $log_folder -o $plot_folder &
python utils/plot.py -v MeanRolloutLens -d $log_folder -o $plot_folder &


python utils/plot.py -v "MeanNrmseBefore" -d $log_folder -o $plot_folder &
python utils/plot.py -v "MeanNrmseAfter"  -d $log_folder -o $plot_folder &
python utils/plot.py -v "NrmseBefore" -d $log_folder -o $plot_folder &
python utils/plot.py -v "NrmseAfter"  -d $log_folder -o $plot_folder &

