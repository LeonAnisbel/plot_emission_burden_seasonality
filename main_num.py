import read_files
import global_vars, plot_maps
import matplotlib.pyplot as plt
import xarray as xr

num_summer, num_winter, dens_summer, dens_winter = [], [], [], []
for idx, file_dir in enumerate(global_vars.experiments):
    num_sum, num_win, dens_sum, dens_win = (
        read_files.read_emi_ice_files(
            'tracer',
            ['NUM_AS', 'NUM_CS'],
            'vphysc',
            'rhoam1',
            idx,
            isice=False))
    num_summer.append(num_sum)
    num_winter.append(num_win)
    dens_summer.append(dens_sum)
    dens_winter.append(dens_win)

var_names = ['NUM_AS', 'NUM_CS']
units = ['10$^{7}$ m$^{-3}$', '10$^{5}$ m$^{-3}$']
factor = [1e7, 1e5]
for idx,var_na in enumerate(var_names):
    fig = plot_maps.create_fig(10, 7)
    num_diff = ((num_summer[0][var_na] * dens_summer[0]['rhoam1']).compute() -
                (num_summer[1][var_na] * dens_summer[1]['rhoam1']).compute()).to_dataset(name='diff')
    print(num_diff.max().values, num_diff.min().values)
    plot_maps.each_fig(fig, num_diff['diff'].isel(lev=46)/factor[idx], ['', f'{var_na} MOA_exp - BASE'], units[idx], [-0.5, 0.5], 'coolwarm')
    plt.savefig(f'{global_vars.plot_dir}/Diff_MOA_exp_BASE_{var_na}.png', dpi=300, bbox_inches="tight")
