import numpy as np
import xarray as xr
import global_vars
import read_files


def find_region(var, cond, dicc, sub_na):
    variab = []
    for v in var:
        if len(cond) <= 1:
            v = v.where((cond[0][0] > cond[0][1]) &
                        (cond[0][0] < cond[0][2])
                        , drop=True)
        elif len(cond) > 1:
            v = v.where(((cond[0][0] > cond[0][1]) &
                         (cond[0][0] < cond[0][2]) &
                         (cond[1][0] > cond[1][1]) &
                         (cond[1][0] < cond[1][2]))
                        , drop=True)
        variab.append(v)

    print(sub_na, 'Max value', variab[0].max().values)  # dicc[sub_na], std)

    print(' ')

    return dicc


def calculate_mean_values_oceans(data):
    lat = data.lat
    lon = data.lon


    conditions = [[[lat, 63, 90]],
                  [[lat, -90, -60]],
                  [[lat, -60, 0], [lon, 130, 290]],
                  [[lat, 0, 63], [lon, 130, 290]],
                  [[lat, -60, 0], [lon, 290, 360]],
                  [[lat, 0, 63], [lon, 300, 360]],
                  # [[lat, -23, 23], [lon, 130, 290]],
                  #  [[lat, -23, 23], [lon, 290, 360]],
                  [[lat, -60, 23], [lon, 20, 120]], ]

    reg_data = {'N. Pole': [],
                'S. Pole': [],
                'S. Pacific': [],
                'N. Pacific': [],
                'S. Atlantic': [],
                'N. Atlantic': [],
                # 'Sub. Pacific': [],
                # 'Sub. Atlantic': [],
                'Indian Ocean': []}

    for i, na in enumerate(reg_data.keys()):
        variables = [data]
        reg_data[na] = find_region(variables,
                                   conditions[i],
                                   reg_data,
                                   na)


def get_conds(lat, lon):
    conditions = [[[lat, 63, 90]],
                  [[lat, 66, 82], [lon, 20, 60]],
                  [[lat, 66, 82], [lon, 60, 100]],
                  [[lat, 66, 82], [lon, 100, 140]],
                  [[lat, 66, 82], [lon, 140, 180]],
                  [[lat, 66, 82], [lon, -180 % 360, -160 % 360]],
                  [[lat, 66, 82], [lon, -160 % 360, -120 % 360]],
                  [[lat, 66, 82], [lon, -120 % 360, -70 % 360]],
                  [[lat, 66, 82], [lon, -70 % 360, -50 % 360]],
                  [[[lat, 66, 82], [lon, -30 % 360, -0.001 % 360]], [[lat, 66, 82], [lon, 0, 20]]],
                  [[lat, 82, 90]], ]

    return conditions


def regions():
    reg_data = {'Arctic': {},
                'Barents Sea': {},
                'Kara Sea': {},
                'Laptev Sea': {},
                'East-Siberian Sea': {},
                'Chukchi Sea': {},
                'Beaufort Sea': {},
                'Canadian Archipelago': {},
                'Baffin Bay': {},
                'Greenland & Norwegian Sea': {},
                'Central Arctic': {},
                }
    return reg_data


def get_var_reg(v, cond):
    if len(cond) <= 1:
        v = v.where((cond[0][0] > cond[0][1]) &
                    (cond[0][0] < cond[0][2])
                    , drop=True)
    elif len(cond) > 1:
        v = v.where(((cond[0][0] > cond[0][1]) &
                     (cond[0][0] < cond[0][2]) &
                     (cond[1][0] > cond[1][1]) &
                     (cond[1][0] < cond[1][2]))
                    , drop=True)
    return v


def get_lalo_mean_pole(ds, w):
    ds_pole = ds.where(ds.lat > 63, drop=True)
    ds_weighted = ds_pole.weighted(w)
    ds_weighted_mean = ds_weighted.mean(dim='time', skipna=True)
    return ds_weighted_mean


def get_mean_max_moa(emi_moa, season):
    mod_dir = global_vars.model_output[0]
    exp = global_vars.experiments[0]
    gboxarea, weights = read_files.get_weights_pole(mod_dir + exp + f'*201001.01_emi.nc')

    print('mean vals', season)
    list_vars = [emi_moa['emi_POL'], emi_moa['emi_PRO'], emi_moa['emi_LIP']]
    for var in list_vars:
        ds_w_mean = get_lalo_mean_pole(var, weights)
        print('\n', ds_w_mean.values)

    print('max vals', season)
    print('POL', emi_moa['emi_POL'].where(emi_moa.lat > 63, drop=True).max(skipna=True).values)
    print('PRO', emi_moa['emi_PRO'].where(emi_moa.lat > 63, drop=True).max(skipna=True).values)
    print('LIP', emi_moa['emi_LIP'].where(emi_moa.lat > 63, drop=True).max(skipna=True).values)
    print('  ')


def get_mean_max_SS_SIC(var, name, season):
    print('mean vals', season)
    if name != 'seaice':
        var = var[name]

    mod_dir = global_vars.model_output[0]
    exp = global_vars.experiments[0]
    gboxarea, weights = read_files.get_weights_pole(mod_dir + exp + f'*201001.01_emi.nc')
    ds_w_mean = get_lalo_mean_pole(var, weights)
    print(name, ds_w_mean.values)
    print('max vals', season)
    print(name, var.where(var.lat > 63, drop=True).max(skipna=True).values)
    print('  ')
