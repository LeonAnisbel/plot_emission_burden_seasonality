import global_vars
import xarray as xr
import glob

def read_tot_moa_emi_burden():
    data_dir = global_vars.project_dir
    f_id = global_vars.files_id

    emi_li = xr.open_dataset(data_dir + 'emi_LIP_emi' + f_id)['emi_LIP']
    emi_po = xr.open_dataset(data_dir + 'emi_POL_emi' + f_id)['emi_POL']
    emi_pr = xr.open_dataset(data_dir + 'emi_PRO_emi' + f_id)['emi_PRO']

    burden_li = xr.open_dataset(data_dir + 'burden_LIP_burden' + f_id)['burden_LIP']
    burden_po = xr.open_dataset(data_dir + 'burden_POL_burden' + f_id)['burden_POL']
    burden_pr = xr.open_dataset(data_dir + 'burden_PRO_burden' + f_id)['burden_PRO']

    emi_tot = (emi_li + emi_po + emi_pr) * 1e3  # factor to convert ug to ng
    burden_tot = burden_li + burden_pr + burden_po
    return emi_tot.isel(time=0), burden_tot.isel(time=0)


def read_wind_prec_emi_ss():
    data_dir = global_vars.project_dir
    f_id = global_vars.files_id

    wind = xr.open_dataset(data_dir + 'wind10_echam' + f_id)['wind10']
    precip = xr.open_dataset(data_dir + 'aprl_echam' + f_id)['aprl'] * 1e6  # factor to convert kg to mg
    emi_ss = xr.open_dataset(data_dir + 'emi_SS_emi' + f_id)['emi_SS']
    burden_ss = xr.open_dataset(data_dir + 'burden_SS_burden' + f_id)['burden_SS']

    return emi_ss.isel(time=0), burden_ss.isel(time=0), wind.isel(time=0), precip.isel(time=0),


def sel_time(C, month):
    C_ti = []
    for m in month:
        C_ti.append(C.where(C.time.dt.month == m,
                            drop=True))
    C_ti_season = xr.concat(C_ti,
                            dim='time')
    return C_ti_season


def sel_var_season(ds, season, months, isice=False):

    ds_season = sel_time(ds, months)
    # tot_emi = (emi_ds_season['emi_POL'] +
    #            emi_ds_season['emi_PRO'] +
    #            emi_ds_season['emi_LIP'])
    season_mean = ds_season.mean(dim='time',
                                 skipna=True)

    if isice:
        season_mean = season_mean.compute() * 100
        season_mean = season_mean.where(season_mean['seaice'] > 20,
                                        drop=True)

    return season_mean


def read_emi_ice_files(sum_mo, win_mo, file_type1, emi_var, file_type2, atm_var, exp_idx, isice=False):
    mod_dir = global_vars.model_output[exp_idx]
    exp = global_vars.experiments[exp_idx]
    print(mod_dir + exp + f'*01_{file_type1}.nc')
    files1 = glob.glob(mod_dir + exp + f'*01_{file_type1}.nc')
    files2 = glob.glob(mod_dir + exp + f'*01_{file_type2}.nc')

    
    files_filter_1 = []
    files_filter_2 = []

    for ff1,ff2 in zip(files1, files2):
        if int(ff1[58:62]) > 2008:
            
            files_filter_1.append(ff1)
        if int(ff2[58:62]) > 2008:
            files_filter_2.append(ff2)
    files_filter_2.sort()    
    files_filter_1.sort()
    
    

    emi_ds = xr.open_mfdataset(files_filter_1,
                               concat_dim='time',
                               combine='nested',
                               preprocess=lambda ds:
                               ds[emi_var])
    print(emi_ds)
    ice_ds = xr.open_mfdataset(files_filter_2,
                               concat_dim='time',
                               combine='nested',
                               preprocess=lambda ds:
                               ds[[atm_var]])

    emi_ds_summer = sel_var_season(emi_ds, 'summer', sum_mo)
    emi_ds_winter = sel_var_season(emi_ds, 'winter', win_mo)
    ice_ds_summer = sel_var_season(ice_ds, 'summer', sum_mo, isice=isice)
    ice_ds_winter = sel_var_season(ice_ds, 'winter', win_mo, isice=isice)

    return emi_ds_summer, emi_ds_winter, ice_ds_summer, ice_ds_winter


def read_vars_per_seasons(sum_month, win_month):
    omf_summer, omf_winter, wind_summer, wind_winter = (
        read_emi_ice_files(sum_month, win_month,
            'ham',
            ['OMF_POL', 'OMF_PRO', 'OMF_LIP'],
            'echam',
            'wind10',
            0,
            isice=False))

    omf_tot_summer = (omf_summer['OMF_LIP'])
    omf_tot_winter = (omf_winter['OMF_LIP'])

    ss_summer, ss_winter, sst_summer, sst_winter = (
        read_emi_ice_files(sum_month, win_month,
            'emi',
            ['emi_SS'],
            'echam',
            'tsw',
            0,
            isice=False))

    emi_summer, emi_winter, ice_summer, ice_winter = (
        read_emi_ice_files(sum_month, win_month,
            'emi',
            ['emi_POL', 'emi_PRO', 'emi_LIP'],
            'echam',
            'seaice',
            0,
            isice=True))

    return ([omf_tot_summer, omf_tot_winter],
            [wind_summer, wind_winter],
            [ss_summer, ss_winter],
            [sst_summer, sst_winter],
            [emi_summer, emi_winter],
            [ice_summer, ice_winter])
