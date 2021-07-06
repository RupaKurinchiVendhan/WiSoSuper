import netCDF4
import pandas as pd

file = 'S:/Downloads/ua_cfDay_CCSM4_1pctCO2_r2i1p1_00250101-00251231.nc'
nc = netCDF4.Dataset(file, mode='r')
nc.set_auto_mask(False)

nc.variables.keys()

# lat = nc.variables['lat'][:]
# lon = nc.variables['lon'][:]
ua = nc.variables['ua'][:]
print(ua.max())
# time_var = nc.variables['time']
# dtime = netCDF4.num2date(time_var[:],time_var.units)
# precip = nc.variables['precip'][:]

# a pandas.Series designed for time series of a 2D lat,lon grid
# precip_ts = pd.Series(precip, index=dtime) 

# precip_ts.to_csv('precip.csv',index=True, header=True)