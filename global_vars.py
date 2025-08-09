main = '/work/bb1005/b381361/'
experiments = ['ac3_arctic', 'echam_base']
main_dir = main + 'echam_postproc/emi_burden_sfc_box_plots/'
project_dir_glb = main_dir + 'mean_burden_and_emission/global_emission_burden_'
lat_arctic_lim = 63

plot_dir = main_dir + f'plot_emi_burden_maps/plots_{lat_arctic_lim}/'
main_new = '/work/bb1178/b324073/'
model_output = [main_new + experiments[0] + '/',
                main_new + experiments[1] + '/']
late_season = False
if late_season:
    seasons_months = {'summer': [7, 8, 9],
                    'winter': [1, 2, 3]}
    pkl_file_title = 'late_season_data'
else:
    seasons_months = {'summer': [4, 5, 6, 7, 8, 9],
                    'winter': [10, 11, 12, 1, 2, 3]}
    pkl_file_title = 'whole_season_data'
emi_lim = [0.032, 0.15, 2.1]

titles = {'MOA': [['Surface emission flux', r'$\bf{(a)}$'], ['Total burden', r'$\bf{(b)}$']],
          'SS': [['Surface emission flux', r'$\bf{(a)}$'], ['Total burden', r'$\bf{(b)}$']],
          'wind': [['Surface emission flux', r'$\bf{(a)}$'], ['Total burden', r'$\bf{(b)}$'],
                   ['10 m wind speed', r'$\bf{(c)}$'], ['Large scale precipitation', r'$\bf{(d)}$']]}
unit = {'MOA': ['ng m$^{-2}$ s$^{-1}$', 'mg m$^{-2}$'],
        'SS': ['µg m$^{-2}$ s$^{-1}$', 'mg m$^{-2}$'],
        'wind': ['µg m$^{-2}$ s$^{-1}$', 'mg m$^{-2}$', 'm/s', 'mg m$^{-2}$ s$^{-1}$']}
vmax = {'MOA': [[0, 3.5], [0, 0.6]],
        'SS': [[0, 0.5], [0, 50]],
        'wind': [[0, 0.5], [0, 50], [0, 15], [0, 50]]}  # 0.1
colorbar = {'MOA': ['Spectral_r', 'Spectral_r'],
            'SS': ['Spectral_r', 'Spectral_r'],
            'wind': ['Spectral_r', 'Spectral_r', 'jet', 'rainbow']}
vmax_arctic = {'MOA': {'emi': emi_lim,
                        'wdep': [0.02, 0.1, 1.2],
                       'burden': [0.001, 0.01, 0.1]},
               'wind': [0.4, 0.15, 9, 20]}
unit_arctic = {'MOA': {'emi': ['ng m$^{-2}$ s$^{-1}}$'],
                       'wdep': ['ng m$^{-2}$ s$^{-1}}$'],
                       'burden': ['mg m$^{-2}$']},
               'wind': ['   ', 'µg m$^{-2}$ s$^{-1}$',
                        'm/s', '$^{o}$C']}
factor_kg_to_ng = 1e12
factor_kg_to_mg = 1e6

files_id_30yr = '_mean_whole_grid_glb_annual_total_1990_2019'
files_id_10yr = '_mean_whole_grid_glb_annual_total_2009_2019'
