import pickle

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import global_vars
import utils
from utils import get_var_reg, regions, get_conds
import pandas as pd
import seaborn as sns

def format_func(value, tick_number):
    N = int(value + 1)
    return N


font = 12


def plot_monthly_series_pannel(axes, fig, C_emi, C_atmos, std_conc, std_omf, title, limits, pos, left_axis=False):
    print('starting plots')
    t_ax = C_atmos[0].time
    ax0 = axes[0]
    ax1 = ax0.twinx()
    ax2 = ax0.twinx()
    ax2.spines.right.set_position(("axes", 1.2))

    ax3 = axes[1]
    ax4 = ax3.twinx()
    ax5 = ax3.twinx()
    ax5.spines.right.set_position(("axes", 1.2))

    axes = [ax0, ax0, ax1, ax2, ax3, ax4, ax5]
    variables = [C_emi[0], C_emi[1], C_emi[2], C_emi[3],
                 C_atmos[0], C_atmos[1], C_atmos[2]]
    labels = ['PCHO$_{aer}$', 'DCAA$_{aer}$', 'PL$_{aer}$',
              'SS', 'SIC', 'SST', 'Wind 10m']
    legend = []
    colors = ['b', 'g', 'darkred', 'm',
              'lightblue', 'gray', 'k']
    for i in range(len(variables)):
        p, = axes[i].plot(t_ax,
                          variables[i].values,
                          label=labels[i],
                          linewidth=1.5,
                          color=colors[i])
        legend.append(p)

        if labels[i] == 'PCHO$_{aer}$' or labels[i] == 'DCAA$_{aer}$':
            ylabel = 'PCHO$_{aer}$ and DCAA$_{aer}$'
        else:
            ylabel = labels[i]

    for ax in enumerate(axes):
        ax.set_ylabel(ylabel,
                      fontsize=font)
        ax.yaxis.set_tick_params(labelsize=font)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.xaxis.set_tick_params(labelsize=font)
        ax.set_xlabel("Months",
                           fontsize=font)

    fig.legend(loc='upper left',
               handles=legend[4:],
               ncol=3,
               fontsize=font,
               bbox_to_anchor=(0.1, 1)
               )

    fig.legend(loc='upper right',
               handles=legend[:3],
               ncol=2,
               fontsize=font,
               bbox_to_anchor=(0.94, 1))


def plot_seasons_reg(ax, C_conc, c, mark, ylabels, reg_gray_line=False, botton_label=True):
    t_ax = C_conc[0].index
    print(C_conc[0])
    p2 = sns.lineplot(data=C_conc[0],
                     palette=c,
                     linewidth=1.5,
                     linestyle='dashed',
                     ax = ax)
    for i, region in enumerate(C_conc[0].columns):
        ax.errorbar(
            t_ax,
            C_conc[0][region],
            yerr=C_conc[1][region],
            fmt='none',
            ecolor=c[i],
            linestyle='dashed',
            linewidth=1.5
        )

    ax.set_ylabel(ylabels,
                  fontsize=font)
    ax.get_legend().remove()

    # ax.set_ylim(0, ylims[0])
    ax.yaxis.set_tick_params(labelsize=font)

    def format_func(value, tick_number):
        N = int(value + 1)
        return N

    if botton_label:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax.xaxis.set_tick_params(labelsize=font)
        ax.set_xlabel("Months",
                      fontsize=font)

    return p2

def get_mean(dict, var_type, var_na ):
    da = dict[var_type]['seasonality'][var_na]
    # if var_type == 'emi':
    #     da = da.sum(dim=['lat', 'lon'],
    #                  skipna=True)
    # else:
    da = da.mean(dim=['lat', 'lon'],
                 skipna=True)
    return da.compute()


def plot_all_seasonality(dict_seasonality):
    fig, axs = plt.subplots(1, 2,
                            figsize=(12, 6))  # 15,8
    ax = axs.flatten()

    emi_pol_mean = get_mean(dict_seasonality,'emi', 'emi_POL')
    emi_pro_mean = get_mean(dict_seasonality,'emi', 'emi_PRO')
    emi_lip_mean = get_mean(dict_seasonality,'emi', 'emi_LIP')
    emi_ss_mean = get_mean(dict_seasonality,'emi', 'emi_SS')
    print('finished computing means emi')

    seaice_mean = get_mean(dict_seasonality, 'echam', 'seaice')
    sst_mean = get_mean(dict_seasonality, 'echam', 'tsw') - 273.16
    wind = get_mean(dict_seasonality, 'vphysc', 'velo10m')

    print('finished computing means')
    plot_monthly_series_pannel(ax, fig,
                               [emi_pol_mean,
                                emi_pro_mean,
                                emi_lip_mean,
                                emi_ss_mean],
                               [seaice_mean,
                                sst_mean,
                                wind],
                               [],
                               [],
                               'title',
                               [],
                               0.27,
                               left_axis=False)
    fig.tight_layout()
    plt.savefig(f'Multiannual_monthly_emission_trends_poles.png',
                dpi=300,
                bbox_inches="tight")


def get_mean_reg(data_ds, gboxarea, var_type):
    mod_dir = global_vars.model_output[0]
    exp = global_vars.experiments[0]

    lon = ((data_ds.lon + 180) % 360) - 180

    reg_data = regions()
    for idx, reg_na in enumerate(list(reg_data.keys())):
        conditions = get_conds(data_ds.lat,
                               lon)
        reg_sel_vals = get_var_reg(data_ds,
                                   conditions[idx])
        reg_sel_vals_gba = get_var_reg(gboxarea,
                                       conditions[idx])

        _, weights = utils.get_weights_pole(mod_dir, exp, '201001', reg_sel_vals_gba)
        if var_type == 'emi':
            reg_sel_vals_gbx = reg_sel_vals * reg_sel_vals_gba # Tg/month/m2 to Tg/month
        else:
            reg_sel_vals_gbx = reg_sel_vals
        reg_sel_vals_mean = utils.get_lalo_mean_pole(reg_sel_vals_gbx, weights, whole_arctic=True)

        print('finished computing reg. weights', reg_sel_vals_mean.time, '\n')
        reg_data[reg_na]['values'] = reg_sel_vals_mean.groupby("time.month").mean(dim="time", skipna=True)
        reg_data[reg_na]['std'] = reg_sel_vals_mean.groupby("time.month").std(dim="time", skipna=True)

    df_reg_data = pd.DataFrame(reg_data)

    return df_reg_data

def create_pkl_files(data, var_na):
    with open(f"./pd_files_{global_vars.lat_arctic_lim}/{var_na}_seasonality.pkl", "wb") as File:
        pickle.dump(data, File)

def read_pkl_files(var_na):
    with open(f"./pd_files_{global_vars.lat_arctic_lim}/{var_na}_seasonality.pkl", "rb") as File:
        var = pickle.load(File)
    return var

def seasonality_region_to_pickle_file(dict_seasonality):
    gboxarea = dict_seasonality['emi_gbx']['seasonality']['gboxarea']

    emi_means_reg = {key: get_mean_reg(value, gboxarea, 'emi')
                     for key, value in dict_seasonality['emi']['seasonality'].items()}
    emi_ss = emi_means_reg['emi_SS']
    emi_lip = emi_means_reg['emi_LIP']
    emi_pro = emi_means_reg['emi_PRO']
    emi_pol = emi_means_reg['emi_POL']
    emi_pol_pro = emi_pro + emi_pol

    echam_means_reg = {key: get_mean_reg(value, gboxarea, 'echam')
                       for key, value in dict_seasonality['echam']['seasonality'].items()}
    seaice_m = echam_means_reg['seaice']

    open_ocean_fraction = 1 - dict_seasonality['echam']['seasonality']['seaice']
    open_ocean_fraction_m =  get_mean_reg(open_ocean_fraction, gboxarea, 'open_ocean_frac')

    sst_m = echam_means_reg['tsw']
    wind_mean = get_mean_reg(dict_seasonality['vphysc']['seasonality']['velo10m'], gboxarea, 'vphysc')

    print('Save variable in pickle files')
    create_pkl_files(emi_lip, 'emi_LIP')
    create_pkl_files(emi_pol, 'emi_POL')
    create_pkl_files(emi_pol_pro, 'emi_POL_PRO')
    create_pkl_files(emi_pro, 'emi_PRO')
    create_pkl_files(emi_ss, 'emi_SS')
    create_pkl_files(seaice_m, 'seaice')
    create_pkl_files(sst_m, 'sst')
    create_pkl_files(wind_mean, 'veloc10m')
    create_pkl_files(open_ocean_fraction_m, 'open_ocean_frac')


def data_df_new(data_df):
    reg_data = regions()
    reg_data_std = regions()
    for idx, reg_na in enumerate(list(reg_data.keys())):
        reg_data[reg_na] = data_df[reg_na]['values']
        reg_data_std[reg_na] = data_df[reg_na]['std']

    df_reg_data = pd.DataFrame(reg_data)
    df_reg_data_std = pd.DataFrame(reg_data_std)
    return [df_reg_data, df_reg_data_std]

def plot_seasonality_region():

    var_ids = ['10m wind speed \n (m s${^{-1}}$)',
               'SIC (%)',
               'SST (C$^{o}$)',
               'SS emission\n (Tg month$^{-1}$)',
               'PCHO$_{aer}$+DCAA$_{aer}$ emission\n (Tg month$^{-1}$)',
               'PL$_{aer}$ emission\n (Tg month$^{-1}$)',
               ]

    emi_lip = read_pkl_files('emi_LIP')
    emi_pol_pro = read_pkl_files('emi_POL_PRO')
    emi_ss = read_pkl_files('emi_SS')
    seaice = read_pkl_files('seaice')
    sst = read_pkl_files('sst')
    wind = read_pkl_files('veloc10m')

    # adapt dataframe containing data and std to plot with sns


    variables = [data_df_new(wind),
                 data_df_new(seaice)*100,
                 data_df_new(sst),
                 data_df_new(emi_ss),
                 data_df_new(emi_pol_pro),
                 data_df_new(emi_lip),
                 ]

    var_values = [i[0] for i in variables]
    var_std = [i[1] for i in variables]

    plot_seanonality_reg_species([var_values, var_std],
                                 var_ids)


def plot_seanonality_reg_species(variables, ylabels):
    fig, axs = plt.subplots(2, 3,
                            figsize=(12, 6))  # 15,8
    ax = axs.flatten()

    color_reg = ['k', 'r', 'm', 'pink',
                 'lightgreen', 'darkblue', 'orange',
                 'brown', 'lightblue', 'y', 'gray']
    mark = ['__' for i in range(len(color_reg)-1)]
    mark.insert(0, '_')

    xaxislabel = False
    for i in range(len(variables[0])):
        leg_list = []
        if ylabels[i] == 'SST (C$^{o}$)':
            ff = 273.16
        else:
            ff = 0

        C_var = variables[0][i] - ff
        if i > 2:
            xaxislabel = True

        p2 = plot_seasons_reg(ax[i],
                              [C_var, variables[1][i]],
                              color_reg,
                              mark,
                              ylabels[i],
                              botton_label=xaxislabel)
        leg_list.append(p2)

    titles = [r'$\bf{(a)}$', r'$\bf{(b)}$',
              r'$\bf{(c)}$', r'$\bf{(d)}$',
              r'$\bf{(e)}$', r'$\bf{(f)}$']
    for i, axs in enumerate(ax):
        axs.set_title(titles[i],
                      loc='right',
                      fontsize=font)
        axs.xaxis.set_tick_params(labelsize=font)
        axs.grid(linestyle='--',
                 linewidth=0.4)

        def format_func(value, tick_number):
            N = int(value + 1)
            return N

        axs.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    box = ax[-3].get_position()
    ax[-3].set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

    handles, labels = ax[-1].get_legend_handles_labels()

    fig.legend(handles=handles,
               labels= labels,
               ncol=3,
               bbox_to_anchor=(0.5, 0.),
               loc='upper center',
               fontsize=font)
    fig.tight_layout()

    plt.savefig(global_vars.plot_dir + 'Multiannual_monthly_seasonality_subregions_emission_sic_sst.png',
                dpi=300,
                bbox_inches="tight")
    plt.close()
