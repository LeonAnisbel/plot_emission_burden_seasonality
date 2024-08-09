main = '/work/bb1005/b381361/'
experiments = ['ac3_arctic','echam_base']

project_dir = main + 'echam_postproc/emi_burden_sfc_box_plots/burden/'
plot_dir = project_dir + 'plot_emi_burden_maps/plots/'

model_output = [main + 'my_experiments/' + experiments[0] + '/',
                main + 'my_experiments/' + experiments[1] + '/']

titles = {'MOA': [['Surface emission flux', r'$\bf{(a)}$'], ['Total burden', r'$\bf{(b)}$']],
          'SS': [['Surface emission flux', r'$\bf{(a)}$'], ['Total burden', r'$\bf{(b)}$']],
          'wind': [['Surface emission flux', r'$\bf{(a)}$'], ['Total burden', r'$\bf{(b)}$'],
                   ['10 m wind speed', r'$\bf{(c)}$'], ['Large scale precipitation', r'$\bf{(d)}$']]}
unit = {'MOA': ['${ng m^{-2} s^{-1}}$', '${mg m^{-2}}$'],
        'SS': ['${{\mu}g m^{-2} s^{-1}}$', '${mg m^{-2}}$'],
        'wind': ['${{\mu}g m^{-2} s^{-1}}$', '${mg m^{-2}}$', '${m/s}$', '${mg m^{-2} s^{-1}}$']}
vmax = {'MOA': [[0,5], [0,0.8]],
        'SS': [[0,0.5], [0, 50]],
        'wind': [[0, 0.5], [0, 50], [0, 15], [0, 50]]}  #0.1
colorbar = {'MOA': ['Spectral_r', 'Spectral_r'],
            'SS': ['Spectral_r', 'Spectral_r'],
            'wind': ['Spectral_r', 'Spectral_r', 'jet', 'rainbow']}
vmax_arctic = {'MOA': [0.04, 0.2, 1.7],
               'wind': [0.4, 0.15, 9, 20]}
unit_arctic = {'MOA': ['${ng m^{-2} s^{-1}}$'],
               'wind': ['   ', '${{\mu}g m^{-2} s^{-1}}$',
                        'm/s', '$^{o}C$']}
unit_factor = 1e12
files_id = '_mean_whole_grid_glb_annual_total.nc'
