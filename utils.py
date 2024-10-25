import numpy as np
import xarray as xr
import global_vars

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

    print('No weight')
    # dicc[sub_na] = variab[0].mean(dim=['lat', 'lon'], skipna=True).values
    # std = variab[0].std(dim=['lat', 'lon'], skipna=True).values
    # print(sub_na, dicc[sub_na], std)

    print(' ')
    # print('weight')
    to_yr = 31557600*1.E-12*1.E-9  # from s to yr and ng to kg and from kg to Tg
    dicc[sub_na] = (variab[0]*variab[1]*to_yr).mean(dim=['lat', 'lon'], skipna=True).values
    std = (variab[0]*variab[1]*to_yr).std(dim=['lat', 'lon'], skipna=True).values
    print(sub_na,variab[0].max().values)#dicc[sub_na], std)

    print(' ')
    print(' ')

    return dicc


def calculate_mean_values_oceans(data):
    lat = data.lat
    lon = data.lon
    # use normalized aera values as weights
    area = xr.open_dataset(global_vars.project_dir + 'grid_box_area.nc')['gboxarea'].mean(dim='time')
    # weights_area = (area - area.min()) / (area.max()-area.min())

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
        variables = [data, area]
        reg_data[na] = find_region(variables,
                                   conditions[i],
                                   reg_data,
                                   na)



def get_conds(lat,lon):
    conditions = [[[lat, 63, 90]],
                  [[lat, 66, 82], [lon, 20, 60]],
                  [[lat, 66, 82], [lon, 60, 100]],
                  [[lat, 66, 82], [lon, 100, 140]],
                  [[lat, 66, 82], [lon, 140, 180]],
                  [[lat, 66, 82], [lon, -180, -160]],
                  [[lat, 66, 82], [lon, -160, -120]],
                  [[lat, 66, 82], [lon, -120, -70]],
                  [[lat, 66, 82], [lon, -70, -50]],
                  [[lat, 66, 82], [lon, -30, 20]],
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




