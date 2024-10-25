import plot_maps, read_files, global_vars, utils
import plot_seasonality

thirty_yrs=False # change to False for 10 yrs mean plot (2009-2019)

### Plotting wind speed and

#wind, precipitation, emi_ss, burden_ss = read_files.read_wind_prec_emi_ss(thirty_yrs=thirty_yrs)
#plot_maps.plot_wind_prep_emi_burden_global(read_files.read_wind_prec_emi_ss(), 'wind',
#                                            'global_wind_prec_emiss_burden_')

# Plotting SS emission and burden
#emi_tot, burden_tot = read_files.read_ss_emi_burden()
#plot_maps.plot_emi_burden_global(emi_tot, burden_tot, 'SS')

# # Plotting MOA emission and burden
### To uncomment
#emi_tot, burden_tot = read_files.read_tot_moa_emi_burden(thirty_yrs=thirty_yrs)
print('EMISSION')
#utils.calculate_mean_values_oceans(emi_tot)
#print('  ')

print('BURDEN')
#utils.calculate_mean_values_oceans(burden_tot)


#if thirty_yrs:
 #   plot_maps.plot_emi_burden_maps(emi_tot.where(emi_tot.lat>50, drop=True), 
  #          burden_tot.where(burden_tot.lat>50, drop=True), 'MOA', polar_proj=thirty_yrs)
#else:
 #   plot_maps.plot_emi_burden_maps(emi_tot, burden_tot, 'MOA')


dict_seasonality = read_files.read_vars_per_months()
plot_seasonality.plot_all_seasonality(dict_seasonality)
plot_seasonality.plot_seasonality_region(dict_seasonality)
exit()


sum_month, win_month = [7, 8, 9], [1, 2, 3]
omf_tot, wind, emi_ss, sst, emi_moa, burden_moa, ice = read_files.read_vars_per_seasons(sum_month, win_month)
#print(sum_month, win_month)
fac = global_vars.factor_kg_to_ng  # factor to convert kg to ng

plot_maps.plot_omf_emi_wind_sst_season([omf_tot[0],
                                        emi_ss[0]['emi_SS']*fac/1e3,
                                        wind[0]['velo10m'],
                                        sst[0]['tsw'] - 273],
                                       ice[0],
                                       'OMF_ss_emi_wind_sst_late',
                                       'wind')
print('start plot')
plot_maps.plot_emi_season(burden_moa[0] * 1e6,
                          burden_moa[1] * 1e6,
                          ice[0],
                          ice[1],
                          'burden',
                          'Surface burden late',
                          'MOA')
print('finished plot')
exit()
plot_maps.plot_emi_season(emi_moa[0] * fac,
                          emi_moa[1] * fac,
                          ice[0],
                          ice[1],
                          'emi',
                          'Surface emission flux late',
                          'MOA')

def get_mean_max_moa(emi_moa, season):
    print('mean vals', season)
    print('POL',emi_moa['emi_POL'].where(emi_moa.lat>63,drop=True).mean(skipna=True).values)
    print('PRO',emi_moa['emi_PRO'].where(emi_moa.lat>63,drop=True).mean(skipna=True).values)
    print('LIP',emi_moa['emi_LIP'].where(emi_moa.lat>63,drop=True).mean(skipna=True).values)
    print('max vals',season)
    print('POL',emi_moa['emi_POL'].where(emi_moa.lat>63,drop=True).max(skipna=True).values)
    print('PRO',emi_moa['emi_PRO'].where(emi_moa.lat>63,drop=True).max(skipna=True).values)
    print('LIP',emi_moa['emi_LIP'].where(emi_moa.lat>63,drop=True).max(skipna=True).values)
    print('  ')

get_mean_max_moa(emi_moa[0]*fac, 'summer')
get_mean_max_moa(emi_moa[1]*fac, 'winter')

def get_mean_max(var, name, season):
    print('mean vals', season)
    if name != 'seaice':
        var = var[name]
    
    print(name,var.where(var.lat>63,drop=True).mean(skipna=True).values)
    print('max vals',season)
    print(name,var.where(var.lat>63,drop=True).max(skipna=True).values)
    print('  ')

get_mean_max(emi_ss[0]*fac/1e3,'emi_SS', 'summer')
get_mean_max(emi_ss[1]*fac/1e3,'emi_SS', 'winter')

get_mean_max(ice[0],'seaice', 'summer')
get_mean_max(ice[1],'seaice', 'winter')



