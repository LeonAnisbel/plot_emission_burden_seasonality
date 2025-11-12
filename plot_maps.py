import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter
import global_vars
import matplotlib.path as mpath
from matplotlib import ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.colors as mplcolors

import plot_seasonality
import read_files
import utils


def create_fig(h, w):
    """
    Creates figure given the height and width parameters
    :param h: height of the figure
    :param w: width of the figure
    :return: figure instance
    """
    return plt.figure(constrained_layout=True, figsize=(h, w))


def customize_axis(ax, titles, polar_proj, plot_diff=False, labels=False, color_land=True):
    """
    Customize map
    :param ax: ax object
    :param titles: list with upper title labels
    :param polar_proj: boolean to discern whether to customize a polar projection plot
    :param plot_diff: boolean to discern whether to plot the 15-year difference
    :param labels: boolean to discern whether to add labels to the figure
    :param color_land: boolean to discern whether to color the land or not
    :return: None
    """
    if plot_diff or not polar_proj:
        font = '16'
    else:
        font = '20'
    if polar_proj:
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land',
                                                    '10m', edgecolor='black',
                                                    facecolor='oldlace'))
        ax.coastlines()
        if not plot_diff:
            gl = ax.gridlines(draw_labels=True, )
            gl.ylocator = mticker.FixedLocator([65, 75, 85])
            gl.yformatter = LATITUDE_FORMATTER
            gl.xlabel_style = {'size': 10}
            gl.ylabel_style = {'size': 10}
    else:
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          linewidth=0.5,
                          color='gray',
                          alpha=0.5,
                          linestyle='--')
        gl.top_labels = False
        gl.right_labels = labels
        gl.bottom_labels = labels
        gl.left_labels = False

        ax.coastlines()
        if color_land:
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',
                                                               'land', '110m',
                                                               edgecolor='black',
                                                               facecolor='oldlace'))  # lightgray

    ax.set_title(titles[0],
                 loc='right',
                 fontsize=font)
    ax.set_title(titles[1],
                 loc='left',
                 fontsize=font)


def add_ice_colorbar(fig, ic, ff, plot_ice=True):
    """
    Adds ice color bar to figure
    :param fig:
    :param ic: plot instance
    :param ff: fontsize of colorbar labels
    :param plot_ice: boolean to plot ice colorbar or not
    :return:
    """
    if plot_ice:
        cbar_ax = fig.add_axes([0.43, -0.05, 0.15, 0.02])
        ic_bar = fig.colorbar(ic, extendfrac='auto',
                              cax=cbar_ax,
                              orientation='horizontal') # 10
        ic_bar.set_label('Sea ice concentration (%)', fontsize=str(ff))
        ic_bar.ax.tick_params(labelsize=ff)
    else:
        handles, _ = ic.legend_elements()
        plt.legend(handles,
                   ["sic 10%"],
                   # loc='lower right',
                   bbox_to_anchor=(0.75, 6))


def each_fig(subfig, moa, titles, unit, vm, colorb, polar_proj=False, atlantic=False, labels=False, color_land=True):
    """
    Plots moa data and customize axis
    :param subfig:
    :var moa: dataArray as a temporal mean with lat and lon dimensions
    :param titles: figure titles
    :param unit: unit to display at colorbar
    :param vm: max value for colorbar
    :param colorb: color bar style
    :param polar_proj: boolean to discern whether to create a map with a polar projection
    :param atlantic: boolean to discern whether to create a map for the Atlantic region
    :param labels: boolean to discern whether to add labels to the figure
    :param color_land: boolean to discern whether to color the land or not
    :return: None
    """
    if polar_proj:
        axes = subfig.subplots(nrows=1, ncols=1, sharex=True,
                               subplot_kw={'projection': ccrs.NorthPolarStereo()})
        axes.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

    elif atlantic:
        axes = subfig.subplots(nrows=1, ncols=1, sharex=True,
                               subplot_kw={'projection': ccrs.Robinson()})
        axes.set_extent([-65, 35, -10, 40], ccrs.PlateCarree())

    else:
        axes = subfig.subplots(nrows=1, ncols=1, sharex=True,
                               subplot_kw={'projection': ccrs.Robinson()})

    orig_cmap = plt.get_cmap(colorb)
    colors = orig_cmap(np.linspace(0.1, 1, 14))
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)

    im = axes.pcolormesh(moa.lon, moa.lat, moa,
                         cmap=cmap, transform=ccrs.PlateCarree(),
                         vmin=vm[0], vmax=vm[1])
    # im = axes.pcolormesh(moa.lon, moa.lat, moa,
    #                      norm=mplcolors.LogNorm(vmin=float(moa.min().values), vmax=float(moa.max().values)),
    #                      cmap=colorb, transform=ccrs.PlateCarree(),)

    cbar = subfig.colorbar(im, orientation="horizontal", extend='max')  # ,cax = cbar_ax
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label=unit, fontsize=12)
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    customize_axis(axes, titles, polar_proj=polar_proj, labels=labels, color_land=color_land)


def plot_emi_burden_maps(moa_emi, moa_burden, var, polar_proj=False, thirty_yrs=False):
    """
    Creates two-panel plot of global emission flux and burden of total PMOA
    :var moa_emi: temporally averaged PMOA emission flux
    :var moa_burden: temporally averaged PMOA burden flux
    :param var: variable id
    :param polar_proj: boolean to discern whether to create a map with a polar projection
    :param thirty_yrs: boolean to discern whether it is a 10-year or 30-year mean (important to define figure name)
    :return: None
    """
    fig = create_fig(10, 7)

    (subfig1, subfig2) = fig.subfigures(nrows=1, ncols=2)
    subfigs = [subfig1, subfig2]

    title = f'global_emiss_burden_{var}.png'
    moa = [moa_emi, moa_burden]
    names = global_vars.titles[var]
    units = global_vars.unit[var]
    vm = global_vars.vmax[var]
    colorb = global_vars.colorbar[var]
    if polar_proj:
        title = f'arctic_emiss_burden_{var}.png'
        vm = [[0, 1.2], [0, 0.005]]

    color_land = [True, False]
    for idx, subf in enumerate(subfigs):
        each_fig(subf,
                 moa[idx],
                 names[idx],
                 units[idx],
                 vm[idx],
                 colorb[idx],
                 polar_proj=polar_proj,
                 color_land=color_land[idx])

    if thirty_yrs: yr_name = '30yr_'
    else: yr_name = '10yr_'

    plt.savefig(f'{global_vars.plot_dir}/{yr_name}{title}',
                dpi=300,
                bbox_inches="tight")
    #     fig.tight_layout()


def plot_emi_burden_maps_vert_profile(moa_emi, moa_burden, moa_conc, var, polar_proj=False, thirty_yrs=False):
    """
    Creates a figure of global emission flux and burden of total PMOA as well as vertical cross-section
    :var moa_emi: temporally averaged PMOA emission flux
    :var moa_burden: temporally averaged PMOA burden flux
    :param var: variable id
    :param polar_proj: boolean to discern whether to create a map with a polar projection
    :param thirty_yrs: boolean to discern whether it is a 10-year or 30-year mean (important to define figure name)
    :return: None
    """
    fig = create_fig(7, 7)

    subfigs = fig.subfigures(nrows=2, ncols=1, height_ratios=[2, 1])
    (subfig1, subfig2) = subfigs[0].subfigures(nrows=1, ncols=2)

    subfigs0 = [subfig1, subfig2,]

    title = f'global_emiss_burden_{var}.png'
    moa = [moa_emi, moa_burden]
    names = global_vars.titles[var]
    units = global_vars.unit[var]
    vm = global_vars.vmax[var]
    colorb = global_vars.colorbar[var]
    if polar_proj:
        title = f'arctic_emiss_burden_{var}.png'
        vm = [[0, 1.2], [0, 0.005]]

    for idx, subf in enumerate(subfigs0):
        each_fig(subf, moa[idx], names[idx], units[idx], vm[idx], colorb[idx], polar_proj=polar_proj)

    if thirty_yrs:
        yr_name = '30yr_'
    else:
        yr_name = '10yr_'

    orig_cmap = plt.get_cmap(colorb[0])
    colors = orig_cmap(np.linspace(0.1, 1, 14))
    cmap = mcolors.LinearSegmentedColormap.from_list("mycmap", colors)
    ax = subfigs[-1].subplots(nrows=1, ncols=1)

    moa_conc_lat_cross_section = moa_conc.mean('lon', skipna=True)
    im = ax.pcolormesh(moa_conc_lat_cross_section.lat.values,
                       moa_conc_lat_cross_section.lev.values/100,
                       moa_conc_lat_cross_section,
                       cmap=cmap)
    ax.set_xlabel('Latitude ($^{o}$)', fontsize=12)
    ax.set_ylabel('Pressure (hPa)', fontsize=12)
    ax.set_ylim([35, None])
    cbar = subfigs[-1].colorbar(im, orientation="horizontal", extend='max')  # ,cax = cbar_ax
    cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label(label='kg m$^{-3}$', fontsize=12)
    plt.gca().invert_yaxis()

    plt.show()
    plt.savefig(f'{global_vars.plot_dir}/{yr_name}vert_profile_{title}', dpi=300, bbox_inches="tight")
    plt.close()


def plot_wind_prep_emi_burden_global(var, var_id, fig_na, polar_proj=False):
    """
    Creates figure of meteorological variables influencing emissions
    :var var: dataArray with temporally averaged values
    :param var: variable id
    :param fig_na: figure name
    :param polar_proj: boolean to discern whether to create a map with a polar projection
    :return: None
    """
    fig = create_fig(10, 7)

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2, ncols=2)
    subfigs = [subfig1, subfig2, subfig3, subfig4]

    names = global_vars.titles[var_id]
    units = global_vars.unit[var_id]
    vm = global_vars.vmax[var_id]
    colorb = global_vars.colorbar[var_id]

    for idx, subf in enumerate(subfigs):
        each_fig(subf, var[idx], names[idx], units[idx], vm[idx], colorb[idx], polar_proj=polar_proj)
    plt.savefig( f'{global_vars.plot_dir}/{fig_na}{var_id}.png', dpi=300, bbox_inches="tight")


def each_fig_season(subfig, moa, ice, titles, unit, vma, colorb, sh, name,
                    lower=False, polar_proj=True, plot_ice=True, plot_diff=False):
    """
    Creates the subplot or subplots for each subfigure
    :param subfig: subfigure instance
    :var moa: list of dataArray with temporally averaged values (lat, lon dimensions)
    :var ice: list of dataArray with temporally averaged values (lat, lon dimensions)
    :param titles: list of titles
    :param unit: units for colorbar
    :param vma: max values for colorbar
    :param colorb: colorbar style
    :param sh: shrink factor for colorbar
    :param name:
    :param lower: boolean to discern whether this is a bottom panel
    :param polar_proj: boolean to discern whether to create a map with a polar projection
    :param plot_ice: boolean to plot ice colorbar or not
    :param plot_diff: boolean to discern whether to plot the 15-year difference
    :return: return plot instance if plot_ice or None if not
    """
    if plot_diff: col = 2
    else: col = 1

    axes = subfig.subplots(nrows=1, ncols=col, sharex=True,
                           subplot_kw={'projection': ccrs.NorthPolarStereo()})
    cmap = plt.get_cmap(colorb, 15)  # 11 discrete colors

    if plot_diff:
        axes.flatten()
            #     for i,ax in enumerate(axes):
        for i in range(len(moa)):
            axes[i].set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
            im = axes[i].pcolormesh(moa[i].lon, moa[i].lat, moa[i],
                                 cmap=cmap, vmin=-vma, vmax=vma,transform=ccrs.PlateCarree(), alpha=0.8)
            customize_axis(axes[i], titles[i], polar_proj=polar_proj, plot_diff=plot_diff)
            # if titles[0][0] == 'DCAA$_{aer}$ \n':
            #     subfig.suptitle('\n'+ name + '\n \n \n \n',
            #                      fontsize='16',
            #                      weight='bold')
            if plot_ice:
                ic = axes[i].contourf(ice[i].lon,
                                      ice[i].lat,
                                      ice[i],
                                      np.arange(10, 110, 30),  # levels=5,
                                      cmap='Greys_r',
                                      transform=ccrs.PlateCarree())
            axes[i].set_position([0, 0, 1, 1], which='both')

        ext = 'both'
        cbar_ax = subfig.add_axes([0.28, 0.17, 0.45, 0.05]) #(left, bottom, width, height)
        cbar = subfig.colorbar(im,
                               orientation="horizontal",
                               shrink=sh,
                               cax=cbar_ax,
                               ax=axes,
                               extend=ext,
                               pad=0.000001,)
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label(label=unit, fontsize='16')  # , weight='bold'
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    else:
        axes.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())
        im = axes.pcolormesh(moa.lon, moa.lat, moa,
                             cmap=cmap, transform=ccrs.PlateCarree(),
                             vmin=0, vmax=vma, alpha=0.8)
        customize_axis(axes, titles, polar_proj=polar_proj)
        ext = 'max'

        if lower:
            # cbar = subfig.colorbar(im, orientation="horizontal", extend='max', shrink=sh)  # ,cax = cbar_ax
            ticks_bar = [round(i, 2) for i in np.linspace(0, vma, 4)]
            cbar = subfig.colorbar(im,
                                   orientation="horizontal",
                                   ticks=ticks_bar,
                                   shrink=sh,
                                   extend=ext)
            cbar.ax.tick_params(labelsize=18)
            cbar.ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))
            cbar.set_label(label=unit, fontsize='18') #, weight='bold'

        if plot_ice:
            ic = axes.contourf(ice.lon,
                               ice.lat,
                               ice,
                               np.arange(10, 110, 30),  # levels=5,
                               cmap='Greys_r',
                               transform=ccrs.PlateCarree())


    #     ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    if plot_ice:
        return ic
    else:
        return None

        # ic = axes.contour(ice.lon, ice.lat,
        #                  ice, levels=[10],
        #                  linestyles=('solid',),
        #                   colors='green',
        #                   linewidths=1.,
        #                  transform=ccrs.PlateCarree())





def plot_emi_season(moa_emi_summer, moa_emi_winter, ice_summer, ice_winter, var_id, title, var, plot_ice=True):
    """
    Creates figure of seasonal emission maps
    :var moa_emi_summer: list of PMOA species emission for summer
    :var moa_emi_winter: list of PMOA species emission for winter
    :var ice_summer: ice concentration for summer
    :var ice_winter: ice concentration for summer
    :param var_id: ID of variable type
    :param title: subplot title
    :param var: var indicating whether this is an emission or emission driver
    :param plot_ice: boolean to plot ice colorbar or not
    :return: None
    """
    fig = create_fig(12, 8)  # layout='constrained',constrained_layout=True

    (subfig1, subfig2, subfig3), (subfig4, subfig5, subfig6) = fig.subfigures(nrows=2,
                                                                              ncols=3,
                                                                              hspace=0.07,
                                                                              height_ratios=[1, 1.25])
    subfigs_winter = [subfig1, subfig2, subfig3]
    subfigs_summer = [subfig4, subfig5, subfig6]

    moa_summer = [moa_emi_summer[f'{var_id}_POL'], moa_emi_summer[f'{var_id}_PRO'], moa_emi_summer[f'{var_id}_LIP']]
    moa_winter = [moa_emi_winter[f'{var_id}_POL'], moa_emi_winter[f'{var_id}_PRO'], moa_emi_winter[f'{var_id}_LIP']]

    print('summer', moa_summer[0].max().values, moa_summer[1].max().values, moa_summer[2].max().values)
    print('winter', moa_winter[0].max().values, moa_winter[1].max().values, moa_winter[2].max().values)
    names_winter = [[r'PCHO$_{aer}$', r'$\bf{(a)}$'], [r'DCAA$_{aer}$', r'$\bf{(b)}$'], [r'PL$_{aer}$', r'$\bf{(c)}$']]
    names_summer = [['PCHO$_{aer}$', r'$\bf{(d)}$'], ['DCAA$_{aer}$', r'$\bf{(e)}$'], ['PL$_{aer}$', r'$\bf{(f)}$']]
    units = global_vars.unit_arctic[var][var_id][0]
    vm = global_vars.vmax_arctic[var][var_id]
    for idx, subf in enumerate(subfigs_summer):
        ic = each_fig_season(subf,
                             moa_summer[idx],
                             ice_summer['seaice'],
                             names_summer[idx],
                             units,
                             vm[idx],
                             'jet',
                             0.7,
                             None,
                             lower=True,
                             plot_ice=plot_ice)

    for idx, subf in enumerate(subfigs_winter):
        ic = each_fig_season(subf,
                             moa_winter[idx],
                             ice_winter['seaice'],
                             names_winter[idx],
                             units,
                             vm[idx],
                             'jet',
                             0.7,
                             None,
                             plot_ice=plot_ice)

    add_ice_colorbar(fig, ic, 18, plot_ice=plot_ice)

    # labels = np.arange(20, 100, 15)
    # ic_bar.set_ticklabels(labels)

    plt.savefig(f'{global_vars.plot_dir}/{title}_pole_emiss.png', dpi=300, bbox_inches="tight")
    #     fig.tight_layout()


def plot_15yr_difference(moa_emi, moa_burden, moa_wdep, seaice, title, plot_ice=False):
    """
    Creates figure of difference of 15yr emission maps (2005-2019 - 1990-2004) for all PMOA species.
    :var moa_emi: list of emission datasets for winter and summer.
    :var moa_burden: list of burden datasets for winter and summer.
    :var moa_wdep: list of wet deposition datasets for winter and summer.
    :var seaice: sea ice dataArray for winter and summer.
    :param title: title of the subplots
    :param plot_ice: boolean to plot ice colorbar or not
    :return: None
    """
    fig = plt.figure(figsize=(14, 11))

    (subfig1, subfig2, subfig3), (subfig4, subfig5, subfig6), (subfig7, subfig8, subfig9)  = fig.subfigures(nrows=3,
                                                                                                          ncols=3)
    subfigs_omf = [subfig1, subfig2, subfig3]
    subfigs_emi = [subfig4, subfig5, subfig6]
    subfigs_burden = [subfig7, subfig8, subfig9]


    moa_emi = [[moa_emi[0]['emi_POL'], moa_emi[1]['emi_POL']],
               [moa_emi[0]['emi_PRO'], moa_emi[1]['emi_PRO']],
               [moa_emi[0]['emi_LIP'], moa_emi[1]['emi_LIP']],
               ]
    moa_burden = [[moa_burden[0]['burden_POL'], moa_burden[1]['burden_POL']],
               [moa_burden[0]['burden_PRO'], moa_burden[1]['burden_PRO']],
               [moa_burden[0]['burden_LIP'], moa_burden[1]['burden_LIP']],
               ]
    moa_wdep = [[moa_wdep[0]['OMF_POL'], moa_wdep[1]['OMF_POL']],
               [moa_wdep[0]['OMF_PRO'], moa_wdep[1]['OMF_PRO']],
               [moa_wdep[0]['OMF_LIP'], moa_wdep[1]['OMF_LIP']],
               ]
    names = [[['PCHO$_{aer}$ \n', 'Winter'], [r'$\bf{OMF}$'+' \n', 'Summer']], #r'$\bf{(a)}$'
                [['DCAA$_{aer}$ \n', 'Winter'], [r'$\bf{OMF}$'+' \n', 'Summer']],
                [['PL$_{aer}$ \n', 'Winter'], [r'$\bf{OMF}$'+' \n', 'Summer']]]
    units_emi = global_vars.unit_arctic['MOA']['emi'][0]
    units_burden = global_vars.unit_arctic['MOA']['burden'][0]
    vm_emi = [0.003, 0.01, 0.6]
    vm_burden = [0.0001, 0.0005, 0.01]
    vm_omf = [0.0003, 0.001, 0.03]

    for idx, subf in enumerate(subfigs_omf):
        ic = each_fig_season(subf,
                             moa_wdep[idx],
                             seaice,
                             names[idx],
                             ' ',
                             vm_omf[idx],
                             'bwr',
                             0.7,
                        'OMF',
                        lower=True,
                        plot_ice=True,
                        plot_diff=True,)
    add_ice_colorbar(fig, ic,18,  plot_ice=True)

    names = [[['PCHO$_{aer}$ \n', 'Winter'], [r'$\bf{Emission flux}$'+' \n', 'Summer']], #r'$\bf{(a)}$'
                [['DCAA$_{aer}$ \n', 'Winter'], [r'$\bf{Emission flux}$'+' \n', 'Summer']],
                [['PL$_{aer}$ \n', 'Winter'], [r'$\bf{Emission flux}$'+' \n', 'Summer']]]
    for idx, subf in enumerate(subfigs_emi):
        print(idx)
        each_fig_season(subf,
                        moa_emi[idx],
                        moa_emi[idx],
                        names[idx],
                        units_emi,
                       vm_emi[idx],
                     'bwr',
                     0.7,
                        'Emission flux',
                     lower=True,
                     plot_ice=plot_ice,
                    plot_diff=True,)

    names = [[['PCHO$_{aer}$ \n', 'Winter'], [r'$\bf{Burden}$'+' \n', 'Summer']], #r'$\bf{(a)}$'
                [['DCAA$_{aer}$ \n', 'Winter'], [r'$\bf{Burden}$'+' \n', 'Summer']],
                [['PL$_{aer}$ \n', 'Winter'], [r'$\bf{Burden}$'+' \n', 'Summer']]]
    for idx, subf in enumerate(subfigs_burden):
        each_fig_season(subf,
                             moa_burden[idx],
                             moa_burden[idx],
                             names[idx],
                             units_burden,
                             vm_burden[idx],
                             'bwr',
                             0.7,
                        'Burden',
                        lower=True,
                        plot_ice=plot_ice,
                        plot_diff=True,)


    # 3) global spacing tweak
    fig.subplots_adjust(
        left=0.05, right=0.95,
        bottom=0.05, top=0.95,
        wspace=0.10, hspace=0.12
    )
    plt.savefig(f'{global_vars.plot_dir}/{title}_pole_emiss.png', dpi=300, bbox_inches="tight")



def plot_omf_emi_wind_sst_season(variables, ice, title, var_id):
    """
    Creates multi-panel figure of
    :var variables: list of temporally variable dataArray
    :var ice: dataset of temporally sea ice concentration
    :param title: title of the subplots
    :param var_id: ID of variable type
    :return: None
    """
    fig = create_fig(10, 10)  # layout='constrained'

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2, ncols=2,
                                                            wspace=0.07, )
    subfigs = [subfig1, subfig2, subfig3, subfig4]

    names = [['OMF', r'$\bf{(a)}$'],
             ['SS emission flux', r'$\bf{(b)}$'],
             ['10m wind speed', r'$\bf{(c)}$'],
             ['SST', r'$\bf{(d)}$']]

    units = global_vars.unit_arctic[var_id]
    vm = global_vars.vmax_arctic[var_id]
    colorb = global_vars.colorbar[var_id]
    for idx, subf in enumerate(subfigs):
        ic = each_fig_season(subf,
                             variables[idx],
                             ice['seaice'],
                             names[idx],
                             units[idx],
                             vm[idx],
                             colorb[idx],
                             0.5,
                             None,
                             lower=True,
                             polar_proj=True)

    add_ice_colorbar(fig, ic, 18)

    plt.savefig(f'{global_vars.plot_dir}/omf_emiss_wind_pole.png', dpi=300, bbox_inches="tight")  #



def plot_global_average_maps(thirty_yrs):
    """
    Creates figures of temporally averaged emission flux and burden of total PMOA
    :param thirty_yrs: boolean to discern whether it is a 10-year or 30-year mean (important to define figure name)
    :return: None
    """
    ### Plotting wind speed and SS

    # wind, precipitation, emi_ss, burden_ss = read_files.read_wind_prec_emi_ss(thirty_yrs=thirty_yrs)
    # plot_maps.plot_wind_prep_emi_burden_global(read_files.read_wind_prec_emi_ss(), 'wind',
    #                                            'global_wind_prec_emiss_burden_')

    # Plotting SS emission and burden
    # emi_tot, burden_tot = read_files.read_ss_emi_burden(thirty_yrs=thirty_yrs)
    # plot_maps.plot_emi_burden_global(emi_tot, burden_tot, 'SS')

    # # Plotting MOA emission and burden
    ### To uncomment
    emi_tot, burden_tot, conc_tot = read_files.read_tot_moa_emi_burden(thirty_yrs=thirty_yrs)
    print('Read in emission')
    # utils.calculate_mean_values_oceans(emi_tot)

    print('\n Read in burden')
    # utils.calculate_mean_values_oceans(burden_tot)

    plot_emi_burden_maps_vert_profile(emi_tot,
                                                burden_tot,
                                                conc_tot,
                                                'MOA',
                                                thirty_yrs=thirty_yrs)

    plot_emi_burden_maps(emi_tot.where(emi_tot.lat > 50, drop=True),
                                   burden_tot.where(burden_tot.lat > 50, drop=True),
                                   'MOA',
                                   polar_proj=True,
                                   thirty_yrs=thirty_yrs)

    plot_emi_burden_maps(emi_tot,
                                   burden_tot,
                                   'MOA',
                                   thirty_yrs=thirty_yrs)


def plot_atlantic():
    """
    Creates figures of temporally averaged total emission flux and burden for the central atlantic and each biomolecule
    :return: None
    """
    data_dir = global_vars.project_dir_glb+'1990_2004_and_2005_2019/'

    f_id_bur = '_mean_global_whole_grid_total_'
    f_id_emi = '_mean_whole_grid_glb_annual_total_'
    years = ['1990_2004', '2005_2019', '1990_2019']
    fac = 1e-3
    variables = ['LIP', 'POL', 'PRO', 'PMOA', 'SS']
    var_names = ['PL', 'PCHO', 'DCAA', 'PMOA', 'SS']
    units = ['g m$^{-2}$ yr$^{-1}}$', 'mg m$^{-2}$']
    limits = [[[0, 0.2], [0, 0.5]],
              [[0, 0.004], [0, 0.03]],
              [[0, 0.02], [0, 0.06]],
              [[0, 0.2], [0, 0.6]],
              [[0, 15], [0, 30]],
    ]
    for yr in years:
        for id,var in enumerate(variables):
            emi = xr.open_dataset(f'{data_dir}emi/emi_{var}_emi{f_id_emi}{yr}.nc')[f'emi_{var}']*fac
            burden = xr.open_dataset(f'{data_dir}burden/burden_{var}_burden{f_id_bur}{yr}.nc')[f'burden_{var}']

            fig = create_fig(10, 7)
            (subfig1, subfig2) = fig.subfigures(nrows=1, ncols=2)
            subfigs = [subfig1, subfig2]

            title = f'global_emiss_burden_{var}'
            moa = [emi.isel(time=0), burden.isel(time=0)]
            names = [[f'{var_names[id]} total emission flux', r' '],
                     [f'{var_names[id]} total burden', r' ']]
            vm = limits[id]
            colorb = global_vars.colorbar['MOA']
            color_land = [True, False]
            for idx, subf in enumerate(subfigs):
                each_fig(subf,
                         moa[idx],
                         names[idx],
                         units[idx],
                         vm[idx],
                         colorb[idx],
                         atlantic=True,
                         labels=True,
                         color_land=color_land[idx])

            plt.savefig(f'plots_atlantic/{yr}_{title}_atlantic.png',
                        dpi=300,
                        bbox_inches="tight")
            plt.close()



def calculate_diff(season, var, fac):
    """
    Computes the 15-year difference
    :param season:
    :param var:
    :param fac:
    :return:
    """
    years_names = ['1990-2004', '2005-2019', '1990-2019']
    den = data_dict[years_names[0]][season][var]* fac
    diff = (data_dict[years_names[1]][season][var] * fac -
            den )
    return diff


if __name__ == '__main__':

        # plot_maps.plot_omf_emi_wind_sst_season([omf_tot[0],
        #                                         emi_ss[0]['emi_SS']*fac/1e3,
        #                                         wind[0]['velo10m'],
        #                                         sst[0]['tsw'] - 273],
        #                                        ice[0],
        #                                        'OMF_ss_emi_wind_sst_late',
        #                                        'wind')
        # print('start plot')
        # fac_mg = global_vars.factor_kg_to_mg  # factor to convert kg to mg
        # plot_maps.plot_emi_season(burden_moa[0] * fac_mg,
        #                           burden_moa[1] * fac_mg,
        #                           ice[0],
        #                           ice[1],
        #                           'burden',
        #                           'Surface burden late',
        #                           'MOA')
        # exit()
        # plot_maps.plot_emi_season(wdep_moa[0] * fac,
        #                           wdep_moa[1] * fac,
        #                           ice[0],
        #                           ice[1],
        #                           'wdep',
        #                           'Surface wdep flux',
        #                           'MOA',
        #                           plot_ice=False)


    # Uncomment this section to create seasonal Arctic plots of marine species
    file_name = global_vars.pkl_file_title
    data_dict = plot_seasonality.read_pkl_files(file_name)
    fac = global_vars.factor_kg_to_ng

    data_dict_yr = data_dict['1990-2019']
    emi_moa = [data_dict_yr['summer']['emi_moa'], data_dict_yr['winter']['emi_moa'] ]
    emi_ss = [data_dict_yr['summer']['emi_SS'], data_dict_yr['winter']['emi_SS']]
    ice = [data_dict_yr['summer']['ice'], data_dict_yr['winter']['ice'] ]

    print('Done')
    print('Calculate Arctic average values')

    print()
    utils.get_mean_max_moa(emi_moa[0] * fac, 'summer')
    utils.get_mean_max_moa(emi_moa[1] * fac, 'winter')

    utils.get_mean_max_SS_SIC(emi_ss[0] * fac , 'emi_SS', 'summer')
    utils.get_mean_max_SS_SIC(emi_ss[1] * fac , 'emi_SS', 'winter')


    plot_emi_season(emi_moa[0] * fac,
                      emi_moa[1] * fac,
                      ice[0]  ,
                      ice[1] ,
                      'emi',
                      'Surface_emission_flux_'+file_name,
                      'MOA')

    # Plot the difference between the periods 1990-2004 and 2005-2019 for emission, wet deposition and burden
    file_name = global_vars.pkl_file_title + '_15_years_aver'
    data_dict = plot_seasonality.read_pkl_files(file_name)

    ice_diff_winter = data_dict['1990-2019']['winter']['ice']['seaice']
    ice_diff_summer = data_dict['1990-2019']['summer']['ice']['seaice']

    emi_moa_diff_winter = calculate_diff('winter','emi_moa', fac)
    emi_moa_diff_summer = calculate_diff('summer','emi_moa', fac)

    omf_moa_diff_winter = calculate_diff('winter', 'omf', 1)
    omf_moa_diff_summer = calculate_diff('summer', 'omf', 1)

    fac_bur = global_vars.factor_kg_to_mg
    burden_moa_diff_winter = calculate_diff('winter', 'burden_moa', fac_bur)
    burden_moa_diff_summer = calculate_diff('summer','burden_moa', fac_bur)

    wdep_moa_diff_winter = calculate_diff('winter', 'wdep_moa', fac)
    wdep_moa_diff_summer = calculate_diff('summer','wdep_moa', fac)
        # [ice_diff_winter, ice_diff_summer],

    plot_15yr_difference([emi_moa_diff_winter, emi_moa_diff_summer],
                         [burden_moa_diff_winter, burden_moa_diff_summer],
                         [omf_moa_diff_winter, omf_moa_diff_summer],
                         [ice_diff_winter, ice_diff_summer],
                         # [wdep_moa_diff_winter, wdep_moa_diff_summer],
                         'Diff_map_emi_burden',)