import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy
import matplotlib.colors as mcolors
import numpy as np
from matplotlib import ticker, cm
from matplotlib.ticker import FuncFormatter
import global_vars
import matplotlib.path as mpath
from matplotlib import ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import matplotlib.colors as mplcolors


def create_fig(h, w):
    return plt.figure(constrained_layout=True, figsize=(h, w))


def customize_axis(ax, titles, polar_proj):
    if polar_proj:
        font = '20'
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)
        ax.set_boundary(circle, transform=ax.transAxes)

        gl = ax.gridlines(draw_labels=True, )
        ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land',
                                                    '10m', edgecolor='black',
                                                    facecolor='oldlace'))
        ax.coastlines()
        gl.ylocator = mticker.FixedLocator([65, 75, 85])
        gl.yformatter = LATITUDE_FORMATTER
    else:
        font = '16'
        gl = ax.gridlines(crs=ccrs.PlateCarree(),
                          draw_labels=True,
                          linewidth=0.5,
                          color='gray',
                          alpha=0.5,
                          linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = False
        gl.left_labels = False

        ax.coastlines()
        if titles[0] != 'Total burden' and titles[0] != 'Surface emission flux':
            ax.add_feature(cartopy.feature.NaturalEarthFeature('physical',
                                                               'land', '110m',
                                                               edgecolor='black',
                                                               facecolor='oldlace'))  # lightgray

    ax.set_title(titles[0], loc='right', fontsize=font)
    ax.set_title(titles[1], loc='left', fontsize=font)


def add_ice_colorbar(fig, ic):
    cbar_ax = fig.add_axes([0.37, -0.05, 0.25, 0.03])
    ic_bar = fig.colorbar(ic, extendfrac='auto', shrink=0.01,
                          cax=cbar_ax, orientation='horizontal', )
    ic_bar.set_label('Sea ice concentration (%)', fontsize='18')
    ic_bar.ax.tick_params(labelsize=18)


def each_fig(subfig, moa, titles, unit, vm, colorb, polar_proj=False):
    if polar_proj:
        axes = subfig.subplots(nrows=1, ncols=1, sharex=True,
                               subplot_kw={'projection': ccrs.NorthPolarStereo()})
        axes.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

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
    cbar.set_label(label=unit, fontsize=12, weight='bold')

    customize_axis(axes, titles, polar_proj=polar_proj)


def plot_emi_burden_maps(moa_emi, moa_burden, var, polar_proj=False):
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

    for idx, subf in enumerate(subfigs):
        each_fig(subf, moa[idx], names[idx], units[idx], vm[idx], colorb[idx], polar_proj=polar_proj)
    plt.savefig(global_vars.plot_dir + title, dpi=300, bbox_inches="tight")
    #     fig.tight_layout()


def plot_wind_prep_emi_burden_global(var, var_id, fig_na, polar_proj=False):
    fig = create_fig(10, 7)

    (subfig1, subfig2), (subfig3, subfig4) = fig.subfigures(nrows=2, ncols=2)
    subfigs = [subfig1, subfig2, subfig3, subfig4]

    names = global_vars.titles[var_id]
    units = global_vars.unit[var_id]
    vm = global_vars.vmax[var_id]
    colorb = global_vars.colorbar[var_id]

    for idx, subf in enumerate(subfigs):
        each_fig(subf, var[idx], names[idx], units[idx], vm[idx], colorb[idx], polar_proj=polar_proj)
    plt.savefig(global_vars.plot_dir + f'{fig_na}{var_id}.png', dpi=300, bbox_inches="tight")


def each_fig_season(subfig, moa, ice, titles, unit, vma, colorb, sh, lower=False, polar_proj=True):
    axes = subfig.subplots(nrows=1, ncols=1, sharex=True,
                           subplot_kw={'projection': ccrs.NorthPolarStereo()})
    axes.set_extent([-180, 180, 50, 90], ccrs.PlateCarree())

    #     for i,ax in enumerate(axes):
    cmap = plt.get_cmap(colorb, 15)  # 11 discrete colors
    im = axes.pcolormesh(moa.lon, moa.lat, moa,
                         cmap=cmap, transform=ccrs.PlateCarree(),
                         vmin=0, vmax=vma, alpha=0.8)

    if lower:
        cbar = subfig.colorbar(im, orientation="horizontal", extend='max', shrink=sh)  # ,cax = cbar_ax
        cbar.ax.tick_params(labelsize=18)
        cbar.set_label(label=unit, weight='bold', fontsize='18')
    #     ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #                   linewidth=0.5, color='gray', alpha=0.5, linestyle='--')

    ic = axes.contourf(ice.lon,
                       ice.lat,
                       ice,
                       np.arange(10, 110, 30),  # levels=5,
                       cmap='Greys_r',
                       transform=ccrs.PlateCarree())

    customize_axis(axes, titles, polar_proj=polar_proj)
    return ic


def plot_emi_season(moa_emi_summer, moa_emi_winter, ice_summer, ice_winter, var_id, title, var):
    fig = create_fig(14, 10)  # layout='constrained',constrained_layout=True

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
                             lower=True)

    for idx, subf in enumerate(subfigs_winter):
        ic = each_fig_season(subf,
                             moa_winter[idx],
                             ice_winter['seaice'],
                             names_winter[idx],
                             units,
                             vm[idx],
                             'jet',
                             0.7)

    add_ice_colorbar(fig, ic)

    # labels = np.arange(20, 100, 15)
    # ic_bar.set_ticklabels(labels)

    plt.savefig(global_vars.plot_dir + title + '_pole_emiss.png', dpi=300, bbox_inches="tight")
    #     fig.tight_layout()


def plot_omf_emi_wind_sst_season(variables, ice, title, var_id):
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
                             lower=True,
                             polar_proj=True)

    add_ice_colorbar(fig, ic)

    plt.savefig(global_vars.plot_dir + 'omf_emiss_wind_pole.png', dpi=300, bbox_inches="tight")  #
