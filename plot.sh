log_folder='log_mamba'
plot_folder='results'

python utils/plot.py -v MeanSumOfRewards -d $log_folder -o $plot_folder &
python utils/plot.py -v BestSumOfRewards -d $log_folder -o $plot_folder &
python utils/plot.py -v std -d $log_folder -o $plot_folder &
