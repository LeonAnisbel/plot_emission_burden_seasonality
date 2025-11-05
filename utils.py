import global_vars
import read_files


def find_region(var, cond, dicc, sub_na):
    """
    This function is used to select and filter certain regions according to the conditions variable (cond).
    :var var: list of variable datasets
    :var cond: list containing the regions lat and lon limits
    :var dicc: dict containing the regions lat and lon limits
    :return: variable dataset with the data filtered to meet each condition
    """
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
    """
    This function defines a Oceanic subregions
    :return :None
    """
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
    """
    Define the subregions limits
    :param lat: list of latitude values
    :param lon: list of longitudes values
    :return: list with region limits, latitude and longitude arrays from the model grid
    """
    conditions = [[[lat, global_vars.lat_arctic_lim, 90]],
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
    """
    This function defines a dictionary with Arctic subregions
    :return :dictionary with regions names as keys
    """
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


def line_style_regions():
    """
    This function defines the colors and line styles per region to use in seasonality plots
    :return: dictionary with regions names as keys containing colors and line styles and lists with colors and
    line styles
    """
    color_reg = ['k', 'r', 'm', 'pink',
                 'lightgreen', 'darkblue', 'orange',
                 'brown', 'lightblue', 'y', 'gray']
    line_styles = [
        [1.0, 0.0],  # solid
        [5.0, 2.0],  # long dash
        [2.0, 2.0],  # dotted
        [4.0, 2.0, 1.0, 2.0],  # dash-dot
        [3.0, 1.0, 1.0, 1.0],  # dash-dot-dot
        [8.0, 2.0, 2.0, 2.0],  # long dash + dots
        [1.0, 1.0],  # dense dots
        [6.0, 1.0, 1.0, 1.0],  # long-short pattern
        [2.0, 3.0, 1.0, 3.0],  # spaced dots
        [5.0, 1.0, 2.0, 1.0],  # alternating dashes
        [7.0, 3.0, 2.0, 3.0],  # long dash–dot pattern
    ]
    reg_style_dict = regions()
    for i, reg in enumerate(list(reg_style_dict.keys())):
        reg_style_dict[reg]['color'] = color_reg[i]
        reg_style_dict[reg]['linestyle'] = line_styles[i]
    return reg_style_dict, color_reg, line_styles


def get_var_reg(v, cond):
    """
    This function is used to select and filter certain regions according to the conditions variable (cond).
    :var v: variable dataset
    :var cond: dictionary containing the regions lat and lon limits
    :return: variable dataset with the data filtered to meet each condition
    """
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


def get_lalo_mean_pole(ds, w, whole_arctic=False):
    """
    Computes weighted mean of ds considering the grid area "w"
    :var ds: dataArray-like object
    :var w: weights based on grid box area from ECHAM-HAM model
    :param whole_arctic: boolean to determine whether the average is for the whole Arctic (True) or Arctic subregions
    (False)
    :return: dataArray as averaged weighted mean
    """
    if whole_arctic: ds_pole = ds.where(ds.lat > global_vars.lat_arctic_lim, drop=True)
    else: ds_pole = ds
    ds_weighted = ds_pole.weighted(w)
    ds_weighted_mean = ds_weighted.mean(dim=['lat', 'lon'], skipna=True)
    return ds_weighted_mean



def get_weights_pole(mod_dir, exp, m, gboxarea_region, whole_arctic=False):
    """
    Reads grid box area from ECHAM-HAM output or uses the gboxarea_region to create the weights used for weighted mean
    computation later on
    :param mod_dir: data directory
    :param exp: experiment ID
    :param m: month
    :var gboxarea_region: dataArray for predefined Arctic subregion
    :param whole_arctic: boolean to determine whether the average is for the whole Arctic (True) or Arctic subregions
    (False)
    :return: dataArray of gridboxarea and dataArray of weights
    """
    files = mod_dir + exp + f'*{m}.01_emi.nc'
    gboxarea = read_files.read_nc_file(files, 'gboxarea')
    if whole_arctic:
        gboxarea_pole = gboxarea.where(gboxarea.lat > global_vars.lat_arctic_lim
, drop=True)
    else:
        gboxarea_pole = gboxarea_region
    weights = gboxarea_pole/ gboxarea_pole.sum(dim=('lat', 'lon'))

    return gboxarea_pole, weights


def get_mean_max_moa(emi_moa, season):
    """Computes max and mean of PMOA emissions"""
    mod_dir = global_vars.model_output[0]
    exp = global_vars.experiments[0]
    gboxarea, weights = get_weights_pole(mod_dir, exp, '201001', None, whole_arctic=True)

    print('mean vals', season)
    list_vars = [emi_moa['emi_POL'], emi_moa['emi_PRO'], emi_moa['emi_LIP']]
    var_names = ['emi_POL', 'emi_PRO', 'emi_LIP']
    for i, var in enumerate(list_vars):
        ds_w_mean = get_lalo_mean_pole(var, weights, whole_arctic=True)
        print('\n', var_names[i], ds_w_mean.values[0])

    print('max vals', season)
    arctic_lim = global_vars.lat_arctic_lim
    print('POL', emi_moa['emi_POL'].where(emi_moa.lat > arctic_lim, drop=True).max(skipna=True).values)
    print('PRO', emi_moa['emi_PRO'].where(emi_moa.lat > arctic_lim, drop=True).max(skipna=True).values)
    print('LIP', emi_moa['emi_LIP'].where(emi_moa.lat > arctic_lim, drop=True).max(skipna=True).values)
    print('  ')


def get_mean_max_SS_SIC(var, name, season):
    """Computes max and mean of Sea salt and sea ice concentration emissions"""
    print('mean vals', season)
    if name != 'seaice':
        var = var[name]

    mod_dir = global_vars.model_output[0]
    exp = global_vars.experiments[0]
    gboxarea, weights = get_weights_pole(mod_dir, exp, '201001', None, whole_arctic=True)
    ds_w_mean = get_lalo_mean_pole(var, weights, whole_arctic=True)
    print(name, ds_w_mean.values[0])
    print('max vals', season)
    arctic_lim = global_vars.lat_arctic_lim
    print(name, var.where(var.lat > arctic_lim, drop=True).max(skipna=True).values)
    print('  ')
