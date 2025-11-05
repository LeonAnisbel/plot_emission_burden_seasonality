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
from matplotlib.ticker import AutoMinorLocator


def format_func(value, tick_number):
    """
    This function will create the months labels considering that January is month 0, it returns 1
    :returns value+1
    """
    N = int(value + 1)
    return N


font = 12
def yearly_seasonality_arctic_and_reg(reg_data,variable):#, variable, mv
    """
    This function creates the multi-panel plot of yearly seasonality for the Arctic and Arctic subregions
    :var reg_data: dictionary-like data with all Arctic subregions
    :var variable: variable name to plot
    :returns None
    """
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
    """
    This function creates the multi-panel plot of yearly seasonality for the Arctic and Arctic subregions as heatmaps
    :var reg_data: dictionary-like data with all Arctic subregions
    :var variable: variable name to plot
    :param norm_label: boolean flag to normalize the values or not
    :returns None
    """
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



def plot_monthly_series_pannel(axes, fig, C_emi, C_atmos):
    """
    This function plots the monthly seasonality for Arctic and Arctic subregions for emissions fluxes of marine species
    and emission drivers
    :param axes: axes object
    :param fig: figure object
    :var C_emi: list of emission flux for various species
    :var C_atmos: list of emission drivers
    :return: None
    """
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


def plot_seasons_reg(ax, C_conc, c, mark, ylabels, font, reg_names, botton_label=True):
    """
    This function plots the monthly seasonality and customize axis and labels
    :param ax: axes object
    :var C_conc: dataArray as multiannual monthly mean values to plot
    :param c: color
    :param mark: line style
    :param ylabels: yaxis label
    :param font: label and ticks font size
    :param reg_names: name of Arctic subregions
    :param botton_label: boolean flag to indicate whether it is a bottom panel to add x-label
    :return: None
    """
    t_ax = C_conc[0].index
    # if len(mark) > 1 :
    p2 = sns.lineplot(data=C_conc[0],
                      # dashes=mark,
                      palette=c,
                      linewidth=1.5,
                      ax = ax)
    for ln, seq in zip(ax.get_lines(), mark):
        if not seq:
            ln.set_linestyle('solid')
            ln.set_dashes(())
        else:
            ln.set_linestyle('solid')
            ln.set_dashes(seq)


    for i, region in enumerate(reg_names):
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
    """
    Computes non-weighted spatial mean
    """
    da = dict[var_type]['seasonality'][var_na]
    # if var_type == 'emi':
    #     da = da.sum(dim=['lat', 'lon'],
    #                  skipna=True)
    # else:
    da = da.mean(dim=['lat', 'lon'],
                 skipna=True)
    return da.compute()


def plot_all_seasonality(dict_seasonality):
    """
    Creates multipanel plot of monthly seasonality for Arctic and Arctic subregions
    :param dict_seasonality: dictionary of emission drivers and emission fluxes
    :return: None
    """
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
                                wind])
    fig.tight_layout()
    plt.savefig(f'Multiannual_monthly_emission_trends_poles.png',
                dpi=300,
                bbox_inches="tight")


def get_mean_reg(data_ds, gboxarea, var_type):
    """
     Computes weighted mean of data_ds considering the grid area from ECHAM-HAM. In the case of emissions the total
     values are computed
    :var data_ds: dataset-like
    :var gboxarea: dataset of gridbox area
    :param var_type: variable type to discern emissions fluxes from emission drivers
    :return: df_reg_data as a dataframe with the whole data, monthly values and multiannual std
    """
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
            reg_sel_vals_mean = reg_sel_vals_gbx.sum(dim=['lat', 'lon'], skipna=True)
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
    """
    Saves data into a pickle file
    :var data:
    :param var_na: file name
    :return: None
    """
    with open(f"./pd_files_{global_vars.lat_arctic_lim}/{var_na}_seasonality.pkl", "wb") as File:
        pickle.dump(data, File)

def read_pkl_files(var_na):
    """
    Reads data from a pickle file
    :param var_na: file name
    :return: data
    """
    with open(f"./pd_files_{global_vars.lat_arctic_lim}/{var_na}_seasonality.pkl", "rb") as File:
        var = pickle.load(File)
    return var

def seasonality_region_to_pickle_file(dict_seasonality):
    """
    Save dataframe data into a pickle file
    :param dict_seasonality: dataframe
    :return:
    """
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

    print('Save variable into pickle files')
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
    """
    Creates three dataframe to split information from the original dataframe
    :var data_df: data frame data
    :return: list of dataframes as monthly values, std and whole data
    """
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
    """
    Reads pickle files of marine aerosol emissions and emission drivers to plot seasonality
    :return: None
    """
    # Define y-axis labels
    var_ids = ['10m wind speed \n (m s${^{-1}}$)',
               'Open ocean fraction (%)',
               'SST ($^{o}$C)',
               'SS emission\n (Tg month$^{-1}$)',
               'PCHO$_{aer}$+DCAA$_{aer}$ emission\n (Tg month$^{-1}$)',
               'PL$_{aer}$ emission \n (Tg month$^{-1}$)',
               ]

    #Read data
    emi_lip = read_pkl_files('emi_LIP') #* 1e5
    emi_pol_pro = read_pkl_files('emi_POL_PRO')# * 1e6
    emi_ss = read_pkl_files('emi_SS') #* 1e3
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

    # Plot yearly monthly data
    for idx,var_na in enumerate(var_ids):
        yearly_seasonality_arctic_and_reg(var_all_data[idx],var_na)
        yearly_seasonality_arctic_and_reg_heatmap(var_all_data[idx],var_ids_heatmap[idx], norm_label=norm_labels[idx])

    # Plot multiannual monthly seasonality as 9-panel Figure
    plot_seanonality_reg_species_acp_plot([var_values, var_std],
                                 var_ids)
    # Plot multiannual monthly seasonality Figure of six-panel plot
    # plot_seanonality_reg_species([var_values, var_std],
    #                              var_ids)



def plot_seanonality_reg_species(variables, ylabels):
    """ Creates six-panel figure with multiannual monthly seasonal values for all Arctic subregions
    :var variables: list of dataframes with monthly values and std
    :param ylabels: y-axis labels names
    :return: None
    """
    if global_vars.lat_arctic_lim == 63: # then do thesis plot
        fig, axs = plt.subplots(3, 2,
                                figsize=(10, 12))  # 15,8
        id_ax = 3
        font = 14

    else:
        fig, axs = plt.subplots(2, 3,
                                figsize=(12, 6))  # 15,8
        id_ax = 7
        font = 12
    ax = axs.flatten()


    _, color_reg, linestyle = utils.line_style_regions()

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
                              linestyle,
                              ylabels[i],
                              font,
                              C_var.columns,
                              botton_label=xaxislabel)
        leg_list.append(p2)
        ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

        if global_vars.lat_arctic_lim == 66 and i > 2:
            ax[i].set_yscale("log")

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


def plot_seanonality_reg_species_acp_plot(variables, ylabels):
    """ Creates nine-panel figure with multiannual monthly seasonal values for all Arctic subregions. An Additional
    panel was added for regions with high total emission fluxes (Arctic, Greenland & Norwegian Sea and Barents Sea)
    :var variables: list of dataframes with monthly values and std
    :param ylabels: y-axis labels names
    :return: None
    """
    id_ax = 7
    font = 14
    height_ratios = [1., 0.7, 0.7]

    fig, axes = plt.subplots(
        nrows=3, ncols=3,
        figsize=(14, 10),
        gridspec_kw={
            'height_ratios': height_ratios,

        },
        sharex=True,
    )
    ax = axes[0, :]  # top row, 3 axes
    ax2 = axes[1, :]  # third row, 3 axes
    ax3 = axes[2, :]  # fourth row, 3 axes

    _, color_reg, linestyle = utils.line_style_regions()

    # plot emission drivers
    xaxislabel = False
    for i in range(len(variables[0][:3])):
        leg_list = []
        if ylabels[i] == 'SST ($^{o}$C)':
            ff = 273.16
        else:
            ff = 0

        C_var = variables[0][i] - ff

        p2 = plot_seasons_reg(ax[i],
                              [C_var, variables[1][i]],
                              color_reg,
                              linestyle,
                              ylabels[i],
                              font,
                              C_var.columns,
                              botton_label=xaxislabel)
        leg_list.append(p2)
        ax[i].grid(linestyle='--',
                 linewidth=0.4)
        ax[i].xaxis.set_tick_params(labelsize=font)
        ax[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))

        ax[i].xaxis.set_minor_locator(AutoMinorLocator(2))  # 4 minor intervals → 3 minor ticks


    # Define parameters for high-total emission fluxes Arctic, Greenland & Norwegian Sea and Barents Sea as well as for
    # the remaining regions
    factors = [1.e1, 1.e4, 1e2]
    factors_lab = ['10$^{-1}$', '10$^{-4}$', '10$^{-2}$']
    reg_style_dict, _, _ = utils.line_style_regions()
    color_reg_sel1, color_reg_sel2 = [], []
    line_style1, line_style2 = [], []
    region_sel1, region_sel2   = [], []
    for reg in list(reg_style_dict.keys()):
        if reg == 'Arctic' or reg == 'Barents Sea' or reg == 'Greenland & Norwegian Sea':
            region_sel1.append(reg)
            color_reg_sel1.append(reg_style_dict[reg]['color'])
            line_style1.append(reg_style_dict[reg]['linestyle'])
        else:
            region_sel2.append(reg)
            color_reg_sel2.append(reg_style_dict[reg]['color'])
            line_style2.append(reg_style_dict[reg]['linestyle'])

    # plot total emission for Arctic, Greenland & Norwegian Sea and Barents Sea
    for i in range(len(variables[0][3:])):
        C_var = variables[0][i+3]*factors[i]
        C_sel = C_var[region_sel1]
        C_sel_std = variables[1][i+3][region_sel1]*factors[i]
        p2 = plot_seasons_reg(ax2[i],
                              [C_sel, C_sel_std],
                              color_reg_sel1,
                              line_style1,
                              ylabels[i+3],
                              font,
                              region_sel1,
                              botton_label=xaxislabel)
        ax2[i].grid(linestyle='--',
                 linewidth=0.4)
        ax2[i].set_ylabel(' \n ')
        ax2[i].xaxis.set_minor_locator(AutoMinorLocator(2))  # 4 minor intervals → 3 minor ticks


    # plot total emission for other Arctic subregions except for the Arctic, Greenland & Norwegian Sea and Barents Sea
    for i in range(len(variables[0][3:])):
        xaxislabel = True
        C_var = variables[0][i+3]
        C_sel = C_var[region_sel2]*factors[i]
        C_sel_std = variables[1][i+3][region_sel2]*factors[i]
        p2 = plot_seasons_reg(ax3[i],
                              [C_sel, C_sel_std],
                              color_reg_sel2,
                              line_style2,
                              ylabels[i+3],
                              font,
                              region_sel2,
                              botton_label=xaxislabel)
        ax3[i].grid(linestyle='--',
                 linewidth=0.4)
        ax3[i].xaxis.set_tick_params(labelsize=font)
        ax3[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax3[i].set_ylabel('\n ')
        ax3[i].xaxis.set_minor_locator(AutoMinorLocator(2))  # 4 minor intervals → 3 minor ticks

    titles = [r'$\bf{(a)}$', r'$\bf{(b)}$',
              r'$\bf{(c)}$', r'$\bf{(d)}$',
              r'$\bf{(e)}$', r'$\bf{(f)}$']

    ax12 = list(ax)+ list(ax2)
    for i, axs in enumerate(ax12):
        axs.set_title(titles[i],
                      loc='right',
                      fontsize=font)

    for a, b, c in zip(ax, ax2, ax3):
        a.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        b.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        c.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    # Collocate ylabels where I want
    fig.text(0.025, 0.17,
             f'SS emission ({factors_lab[0]}'+' Tg month$^{-1}$)',
             fontsize=font,
             rotation=90)
    fig.text(0.36, 0.11,
             'PCHO$_{aer}$+DCAA$_{aer}$ emission ('+f'{factors_lab[1]}'+' Tg month$^{-1}$)',
             fontsize=font,
             rotation=90)
    fig.text(0.68, 0.16,
             'PL$_{aer}$ emission ('+f'{factors_lab[2]}'+' Tg month$^{-1}$)',
             fontsize=font,
             rotation=90)

    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(handles=handles,
               labels= labels,
               ncol=3,
               bbox_to_anchor=(0.5, 0.),
               loc='upper center',
               fontsize=font)
    fig.tight_layout()

    plt.savefig(global_vars.plot_dir + 'Multiannual_monthly_seasonality_subregions_emission_sic_sst_acp_plot.png',
                dpi=300,  bbox_inches='tight')
    plt.close()