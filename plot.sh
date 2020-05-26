log_folder=$1
plot_folder=$2
python utils/plot.py -v MeanSumOfRewards -d $log_folder -o $plot_folder &
python utils/plot.py -v BestSumOfRewards -d $log_folder -o $plot_folder &
python utils/plot.py -v std -d $log_folder -o $plot_folder &


python utils/plot.py -v "ExplainVarianceBefore(AE)" -d $log_folder -o $plot_folder &
python utils/plot.py -v "ExplainVarianceAfter(AE)"  -d $log_folder -o $plot_folder &
python utils/plot.py -v "MeanExplainVarianceAfter(AE)"  -d $log_folder -o $plot_folder &
python utils/plot.py -v "MeanExplainVarianceBefore(AE)"  -d $log_folder -o $plot_folder &
