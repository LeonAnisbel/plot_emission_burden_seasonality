import pickle
from matplotlib.ticker import FormatStrFormatter

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

def yearly_seasonality_arctic_and_reg(reg_data,variable):#, variable, mv
    """ This function creates the four-panel plot of biomolecules and OMF seasonality for the Arctic and Arctic subregions
    :returns None"""
    fig, axs = plt.subplots(4, 3, figsize=(14, 12))  # 15,8
    fig.suptitle(f'Arctic and Arctic subregions seasonality for {variable} \n',
                 fontsize=font+2,
                 weight='bold')
    ax = axs.flatten()
    fig.subplots_adjust(right=0.75)

    cmap = plt.get_cmap('turbo')  # choose any cmap
    colors = [tuple(c) for c in cmap(np.linspace(0.05, 0.95, 30))]  # avoid extreme ends
    for idx, na in enumerate(reg_data.keys()):
        print(na)
        ax[idx].set_title(na,
                          loc='center',
                          fontsize=font,
                          weight='bold')
        if variable == 'SST ($^{o}$C)':
            ff = 273.16
        else:
            ff = 0
        C_conc = reg_data[na]- ff

        year_list = C_conc.time.dt.year.values
        month_list = C_conc.time.dt.month.values
        C_conc_pd = pd.DataFrame({'values':C_conc, 'years': year_list, 'months': month_list})
        C_conc_pd['years'] = C_conc_pd['years'].astype(str)

        p1 = sns.lineplot(data=C_conc_pd,
                     x = 'months',
                     y = 'values',
                     hue = 'years',
                    palette = colors,
                    linestyle='dashed',
                    ax=ax[idx])
        ax[idx].set_ylabel(f"Values", fontsize=font)
        # ax[idx].set_ylim(0, mv)
        ax[idx].set_xlim(1, 12)

        ax[idx].yaxis.set_tick_params(labelsize=font)
        ax[idx].xaxis.set_tick_params(labelsize=font)

        ax[idx].set_xlabel("Months", fontsize=font)
        ax[idx].get_legend().remove()

    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
    handles, labels = ax[0].get_legend_handles_labels()
    ax[-1].axis('off')

    fig.legend(handles=handles,
               labels=labels,
               ncol=8,
               bbox_to_anchor=(0.5, 0.),
               loc='upper center',
               fontsize=font)
    fig.tight_layout()
    plt.savefig(f'{global_vars.plot_dir}Yearly_monthly_poles_subregions_{variable}.png',
                dpi=300,
                bbox_inches="tight")
    plt.close()


def yearly_seasonality_arctic_and_reg_heatmap(reg_data, variable, norm_label=False):
    """ This function creates the four-panel plot of biomolecules and OMF seasonality for the Arctic and Arctic subregions
    :returns None"""
    fig, axs = plt.subplots(3, 4,
                            figsize=(20, 18))  # 15,8
    if norm_label:
        norm = ' normalized'
    else:
        norm = ''
    fig.suptitle(f'Arctic and Arctic subregions seasonality for {variable}{norm} values \n',
                 fontsize=font+2,
                 weight='bold')
    ax = axs.flatten()
    fig.subplots_adjust(right=0.75)
    yr = 'months_30_yr'

    for idx, na in enumerate(reg_data.keys()):
        print(na)
        ax[idx].set_title(na,
                          loc='center',
                          fontsize=font,
                          weight='bold')
        if variable == 'SST ($^{o}$C)':
            ff = 273.16
        else:
            ff = 0
        C_conc = reg_data[na]- ff
        C_min = C_conc.min().values
        C_max = C_conc.max().values
        if norm_label:
            C_conc_norm = (C_conc.values - C_min) / (C_max - C_min)
        else:
            C_conc_norm = C_conc.values
        year_list = C_conc.time.dt.year.values
        month_list = C_conc.time.dt.month.values
        C_conc_pd = pd.DataFrame({'values':C_conc_norm, 'years': year_list, 'months': month_list})
        C_conc_pd['years'] = C_conc_pd['years'].astype(str)
        C_conc_pd_sel = C_conc_pd[C_conc_pd['months']>3]
        C_conc_pd_sel2 = C_conc_pd_sel[C_conc_pd_sel['months']<12]

        C_conc_pd_piv = C_conc_pd_sel2.pivot(index='years',
                                        columns='months',
                                        values='values')
        p1 = sns.heatmap(C_conc_pd_piv,
                    annot=True,
                    cmap = 'Reds',
                    fmt='.1f',
                    ax = ax[idx],
                    linewidths = .5,)
        ax[idx].set(xlabel='Months', ylabel='', )
        ax[idx].set_xlabel('Months', fontsize=font)
        p1.figure.axes[-1].tick_params(labelsize=font)
        ax[idx].tick_params(labelsize=font)
        p1.invert_yaxis()

        # ax[idx].axis.tick_top()

    ax[-1].axis('off')

    fig.tight_layout()
    plt.savefig(f'{global_vars.plot_dir}Yearly_monthly_poles_subregions_{variable}_heatmaps.png',
                dpi=300,
                bbox_inches="tight")
    plt.close()



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


def plot_seasons_reg(ax, C_conc, c, mark, ylabels, font, reg_gray_line=False, botton_label=True):
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
            # reg_sel_vals_mean = reg_sel_vals_gbx.sum(dim=['lat', 'lon'], skipna=True)
        else:
            reg_sel_vals_gbx = reg_sel_vals
        reg_sel_vals_mean = utils.get_lalo_mean_pole(reg_sel_vals_gbx, weights, whole_arctic=True)

        print('finished computing reg. weights', reg_sel_vals_mean.time, '\n')
        reg_data[reg_na]['data'] = reg_sel_vals_mean
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
    reg_all_data = regions()
    reg_data_std = regions()
    for idx, reg_na in enumerate(list(reg_data.keys())):
        reg_data[reg_na] = data_df[reg_na]['values']
        reg_data_std[reg_na] = data_df[reg_na]['std']
        reg_all_data[reg_na] = data_df[reg_na]['data']

    df_reg_data = pd.DataFrame(reg_data)
    df_reg_data_std = pd.DataFrame(reg_data_std)
    return [df_reg_data, df_reg_data_std, reg_all_data]

def plot_seasonality_region():

    var_ids = ['10m wind speed \n (m s${^{-1}}$)',
               'Open ocean fraction (%)',
               'SST ($^{o}$C)',
               'SS emission\n (10$^{-3}$ Tg month$^{-1}$)',
               'PCHO$_{aer}$+DCAA$_{aer}$ emission\n (10$^{-6}$ Tg month$^{-1}$)',
               'PL$_{aer}$ emission \n (10$^{-5}$ Tg month$^{-1}$)',
               ]

    emi_lip = read_pkl_files('emi_LIP') * 1e5
    emi_pol_pro = read_pkl_files('emi_POL_PRO') * 1e6
    emi_ss = read_pkl_files('emi_SS') * 1e3
    seaice = read_pkl_files('open_ocean_frac')
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
    var_all_data = [i[2] for i in variables]

    var_ids_heatmap = ['10m wind speed \n (m s${^{-1}}$)',
                       'Open ocean fraction (%)',
                       'SST ($^{o}$C)',
                        'SS emission',
                        'PCHO$_{aer}$+DCAA$_{aer}$ emission',
                        'PL$_{aer}$ emission',
                        ]
    norm_labels = [False, False, False, True, True, True]

    for idx,var_na in enumerate(var_ids):
        yearly_seasonality_arctic_and_reg(var_all_data[idx],var_na)
        yearly_seasonality_arctic_and_reg_heatmap(var_all_data[idx],var_ids_heatmap[idx], norm_label=norm_labels[idx])

    plot_seanonality_reg_species([var_values, var_std],
                                 var_ids)


def plot_seanonality_reg_species(variables, ylabels):
    if global_vars.lat_arctic_lim == 63: # then do thesis plot
        fig, axs = plt.subplots(3, 2,
                                figsize=(10, 12))  # 15,8
        id_ax = 3
        font = 14
    else:
        fig, axs = plt.subplots(2, 3,
                                figsize=(12, 6))  # 15,8
        id_ax = 2
        font = 12

    ax = axs.flatten()

    color_reg = ['k', 'r', 'm', 'pink',
                 'lightgreen', 'darkblue', 'orange',
                 'brown', 'lightblue', 'y', 'gray']
    mark = ['__' for i in range(len(color_reg)-1)]
    mark.insert(0, '_')

    xaxislabel = False
    for i in range(len(variables[0])):
        leg_list = []
        if ylabels[i] == 'SST ($^{o}$C)':
            ff = 273.16
        else:
            ff = 0

        C_var = variables[0][i] - ff
        if i > id_ax:
            xaxislabel = True

        p2 = plot_seasons_reg(ax[i],
                              [C_var, variables[1][i]],
                              color_reg,
                              mark,
                              ylabels[i],
                              font,
                              botton_label=xaxislabel)
        leg_list.append(p2)
        # ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

        # if global_vars.lat_arctic_lim == 63 and i > 2:
        #     ax[i].set_yscale("log")

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
