import os
import plot_maps, read_files, global_vars
import plot_seasonality

thirty_yrs = True  # change to False for 10 yrs mean plot (2009-2019)


try:
    os.makedirs(global_vars.plot_dir)
except OSError:
    pass

# Plot map of burden and emission over the central Atlantic
plot_maps.plot_atlantic()

# Plot global map of burden and emission
plot_maps.plot_global_average_maps(thirty_yrs)
print('Plot complete emission and Burden')
print('start with Arctic analysis')

seasons = global_vars.seasons_months
years_list = [[1990, 2004], [2005, 2019], [1990, 2019]]
years_names = ['1990-2004', '2005-2019', '1990-2019']

datasets_dict = {}
for idx, yrs in enumerate(years_list):
    datasets_dict[years_names[idx]] = {}
    # read data per year
    omf_tot, wind, emi_ss, sst, emi_moa, burden_moa, wdep_moa, ice = read_files.read_vars_per_seasons(seasons['summer'],
                                                                                                      seasons['winter'],
                                                                                                      yrs)
    # save data in dictionary
    for i, s in zip([0,1], seasons.keys()):
        datasets_dict[years_names[idx]][s] = {'wind': wind[i],
                                    'omf': omf_tot[i],
                                    'burden_moa': burden_moa[i],
                                    'wdep_moa': wdep_moa[i],
                                    'ice': ice[i],
                                    'sst': sst[i],
                                    'emi_moa': emi_moa[i],
                                    'emi_SS': emi_ss[i],}

# store dictionary in a pkl file
plot_seasonality.create_pkl_files(datasets_dict, global_vars.pkl_file_title+'_15_years_aver')

