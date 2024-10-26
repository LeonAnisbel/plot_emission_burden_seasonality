import numpy as np
import matplotlib.pyplot as plt

import global_vars
from utils import get_var_reg, regions, get_conds

fac = global_vars.factor_kg_to_ng  # factor to convert kg to ng


def format_func(value, tick_number):
    N = int(value + 1)
    return N


font = 14


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
              'SS$_{aer}$', 'SIC', 'SST', 'Wind 10m']
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
        axes[i].set_ylabel(ylabel,
                           fontsize=font)
        axes[i].yaxis.set_tick_params(labelsize=font)
        axes[i].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        axes[i].xaxis.set_tick_params(labelsize=font)
        axes[i].set_xlabel("Months",
                           fontsize=font)

    # ax1.set_ylabel("Emission flux PL$_{aer}$",
    #                color='darkred',
    #                fontsize=font)
    # ax2.set_ylabel("Emission flux SS$_{aer}$",
    #                color='m',
    #                fontsize=font)
    # ax3.spines['right'].set_color('darkred')
    # ax3.tick_params(axis='y',
    #                 colors='darkred')
    #
    # ax4.spines['right'].set_color('m')
    # ax4.tick_params(axis='y',
    #                 colors='m')

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


def plot_seasons_reg(ax, C_conc, C_omf, na, c, lw, ylabels, ylims, reg_gray_line=False):
    t_ax = C_conc.time
    ax, ax2 = ax[0], ax[1]
    if na != 'Arctic':
        line_sty = '--'
    else:
        line_sty = '-'

    #   Plot gray lines only
    if reg_gray_line:
        p2 = ax.plot(t_ax,
                     C_conc,
                     color=c,
                     label='Arctic',
                     linewidth=lw)
        p3 = ax2.plot(t_ax,
                      C_omf,
                      color=c,
                      label='Arctic',
                      linewidth=lw)

    else:
        p2, = ax.plot(t_ax, C_conc,
                      linewidth=lw,
                      label=na,
                      color=c,
                      linestyle=line_sty)  # linestyle = li_style,
        p3, = ax2.plot(t_ax, C_omf,
                       linewidth=lw,
                       label=na,
                       color=c,
                       linestyle=line_sty)  # linestyle = li_style,

    ax.set_ylabel(ylabels[0],
                  fontsize=font)
    # ax.set_ylim(0, ylims[0])
    ax.yaxis.set_tick_params(labelsize=font)

    ax2.set_ylabel(ylabels[1],
                   fontsize=font)
    # ax2.set_ylim(0, ylims[1])
    ax2.yaxis.set_tick_params(labelsize=font)

    def format_func(value, tick_number):
        N = int(value + 1)
        return N

    ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    ax.xaxis.set_tick_params(labelsize=font)

    ax.set_xlabel("Months",
                  fontsize=font)
    ax2.set_xlabel("Months",
                   fontsize=font)

    if na == 'Arctic' and reg_gray_line:
        print('here')
        ax.legend(loc='upper left',
                  fontsize=14)
        ax2.legend(loc='upper left',
                   fontsize=14)

    if reg_gray_line:
        pass
    else:
        return p2, p3


def get_mean(da):
    da = da.mean(dim=['lat', 'lon'],
                 skipna=True)
    return da.compute()


def plot_all_seasonality(dict_seasonality):
    fig, axs = plt.subplots(1, 2,
                            figsize=(12, 6))  # 15,8
    ax = axs.flatten()

    emi_pol_mean = get_mean(dict_seasonality['emi']['seasonality']['emi_POL']) * fac
    emi_pro_mean = get_mean(dict_seasonality['emi']['seasonality']['emi_PRO']) * fac
    emi_lip_mean = get_mean(dict_seasonality['emi']['seasonality']['emi_LIP']) * fac
    emi_ss_mean = get_mean(dict_seasonality['emi']['seasonality']['emi_SS']) * fac
    print('finished computing means emi')

    seaice_mean = get_mean(dict_seasonality['echam']['seasonality']['seaice'])
    sst_mean = get_mean(dict_seasonality['echam']['seasonality']['tsw']) - 273.16
    wind = get_mean(dict_seasonality['vphysc']['seasonality']['velo10m'])

    print('finished computing means')
    print(seaice_mean, sst_mean)
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


def get_mean_reg(data_ds):
    reg_data = regions()
    for idx, reg_na in enumerate(list(reg_data.keys())):
        conditions = get_conds(data_ds.lat,
                               data_ds.lon)
        reg_sel_vals = get_var_reg(data_ds,
                                   conditions[idx])
        reg_data[reg_na] = reg_sel_vals.mean(
            dim=['lat', 'lon'],
            skipna=True)
    print(reg_data)
    return reg_data


def plot_seasonality_region(dict_seasonality):
    emi_means = [get_mean_reg(dict_seasonality['emi']['seasonality']['emi_POL'] * fac),
                 get_mean_reg(dict_seasonality['emi']['seasonality']['emi_PRO'] * fac),
                 get_mean_reg(dict_seasonality['emi']['seasonality']['emi_LIP'] * fac),
                 get_mean_reg(dict_seasonality['emi']['seasonality']['emi_SS'] * fac)]
    print('finished computing means emi')

    var_ids = ['PCHO', 'DCAA', 'PL', 'SS']

    seaice_mean = get_mean_reg(dict_seasonality['echam']['seasonality']['seaice'])
    sst_mean = get_mean_reg(dict_seasonality['echam']['seasonality']['tsw'] - 273.16)
    wind_mean = get_mean_reg(dict_seasonality['vphysc']['seasonality']['velo10m'])

    atmo_vars = [seaice_mean,
                 sst_mean,
                 wind_mean,
                 wind_mean]
    atmos_var_title = ['SIC',
                       'SSt',
                       'Wind 10m',
                       'Wind 10m']

    print('finished computing means')

    lims = []
    for i in range(len(emi_means)):
        plot_seanonality_reg_species(atmo_vars[i],
                                     atmos_var_title[i],
                                     emi_means[i],
                                     var_ids[i],
                                     lims)


def plot_seanonality_reg_species(C_ice_all, title, C_emi_all, var_id, lims, ):
    fig, axs = plt.subplots(1, 2,
                            figsize=(12, 6))  # 15,8
    ax = axs.flatten()

    leg_list = []
    color_reg = ['k', 'r', 'm', 'pink',
                 'lightgreen', 'darkblue', 'orange',
                 'brown', 'lightblue', 'y', 'gray']
    for idx, na in enumerate(C_emi_all.keys()):
        print(na)
        C_emi = C_emi_all[na]
        C_ice = C_ice_all[na]
        if na == 'Arctic':
            lw = 1.5
        else:
            lw = 1.5

        ylabels = [f'Emission flux of {var_id}',
                   title]

        p2, p3 = plot_seasons_reg(axs,
                                  C_emi,
                                  C_ice,
                                  na,
                                  color_reg[idx],
                                  lw,
                                  ylabels,
                                  lims)
        leg_list.append(p2)

    titles = [r'$\bf{(a)}$',
              r'$\bf{(b)}$']
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

    box = ax[0].get_position()
    ax[0].set_position([box.x0, box.y0 + box.height * 0.1,
                        box.width, box.height * 0.9])
    fig.legend(handles=leg_list,
               ncol=3,
               bbox_to_anchor=(0.5, 0.),
               loc='upper center',
               fontsize=font)
    fig.tight_layout()

    plt.savefig(f'Multiannual monthly trends poles and subregions_color_reg_ice_{var_id}.png',
                dpi=300,
                bbox_inches="tight")
    plt.close()
