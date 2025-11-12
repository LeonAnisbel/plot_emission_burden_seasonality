import os
import global_vars
import plot_seasonality
import read_files

try:
    os.makedirs(global_vars.plot_dir)
except OSError:
    pass

dict_seasonality = read_files.read_vars_per_months()
print('Read all data and calculate monthly mean')
print('Calculate weighted average per region')
plot_seasonality.seasonality_region_to_pickle_file(dict_seasonality)
print('Done')
print('Read pickle files and create scatter plot')
plot_seasonality.plot_seasonality_region()
print('Done')
