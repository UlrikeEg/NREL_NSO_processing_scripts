import numpy as np
from numpy import cos,sin
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.fftpack import *
import scipy as sp
import scipy.signal
import scipy.signal as signal
from scipy.optimize import curve_fit
import sys
import time
import glob
import netCDF4 as nc
import xarray as xr
import pickle
from numba import jit
import scipy.io

sys.path.append("../NSO_data_processing")

from Functions_masts import *
from Functions_general import *
from Functions_loads import *


# select dates
years   =  np.arange(2022,2024)   #[2022] #
months  =  [11,12, 1, 2]   # np.arange(1,13)   # [3] # 
days    =  np.arange(1,32)   #[5] # 

years   =  [2023] #np.arange(2023,2024)   #
months  =  [2]   # np.arange(1,13)   # [3] # 
days    =  np.arange(21,32)   #[5] # 


# Define
resolution_loads = '20Hz'   # one out of: '20Hz', '1min', '10sec'
resolution_winds = '20Hz'   # one out of: '20Hz', '10min'






#%% Read loads



loads_path = 'Y:\Wind-data/Restricted/Projects/NSO/Loads/Data/FastData_FT_formatted/'

loads = pd.DataFrame()


for year in years:

    for month in months:
        
        if year == '2022' and month<11: # no loads data before Nov 2022
            continue
    
        for day in days:

            l = read_raw_loads_at_date(year, month, day, resolution_loads, loads_path)  # .resample("0.1S").mean())
            if len(l)>0:
                loads = pd.concat([loads, l[['Anemometer', 'LabVIEW_Timestamp']]])

# Time conversion: before Feb 23, the GPS did not work and LabView time had a nominal offset of 7 hours, after that GPS provided UTC correctly.           
#loads.index = loads.LabVIEW_Timestamp +pd.to_timedelta(7, unit="H") # LabView time stamp is 7 hours behind UTC, which is Pacific Daylight Time (Summer time)()
loads.index = loads.LabVIEW_Timestamp.where(
    loads.LabVIEW_Timestamp > pd.to_datetime('2023-02-23 19:01:00'), loads.LabVIEW_Timestamp +pd.to_timedelta(7, unit="H"))


loads.index.name = 'UTC'
loads.index = loads.index.round('0.01S')
loads['time'] = (loads.index - loads.index[0]) / np.timedelta64(1,'s') 
loads = loads.drop_duplicates().sort_index().loc['2022-11-18 00:00:00':]

filter_window = '60S'
loads['Anemometer'] = loads['Anemometer'].where( np.abs(loads['Anemometer'] - loads['Anemometer'].rolling(filter_window, center=True, min_periods=1).median() ) 
 
                                                <= (5* loads['Anemometer'].rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)            




#loads2 = pd.read_pickle('Sync_Loads_all.pkl')
# loads3 = pd.concat([loads2, loads])
# loads.to_pickle('Sync_Loads_all.pkl')  




#%% Read winds
inflow_files = []

for year in years:
    year = str(year)

    for month in months:       
        month = f'{month:02d}'
    
        for day in days:
            day = f'{day:02d}'
              
            print (year, month, day) 
            inflow_files = inflow_files + sorted(glob.glob('../NSO_data_processing/data/' +'Inflow_Mast_'+resolution_winds+'_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #

### Read data
inflow = pd.DataFrame()    
for datafile in inflow_files:
    i = pd.read_pickle(datafile)[['WS_15m', 'wspd_7m']] #  .resample("0.1S").mean())
    inflow = pd.concat( [inflow, i]) 
#inflow = inflow['2022-11-18 00:00:00':'2023-02-22 00:00:00']

    
    
#inflow2 = pd.read_pickle('Sync_winds_all.pkl')  
# inflow3 = pd.concat( [inflow2, inflow]) 
# inflow.to_pickle('Sync_winds_all.pkl')




      
        


# inflow = inflow.loc[inflow.index > loads.index[0] ]  
# inflow = inflow.loc[inflow.index < loads.index[-1]]  

      
#%% Combine loads and wind

# fig = plt.figure(figsize=(16,9))
# plt.plot(inflow.wspd_7m)
# plt.plot(inflow.WS_15m,"x", color='red', label='15m wind')
# plt.plot(loads.Anemometer,"-", color='black', label='15m loads')




# plt.figure()
# plt.plot(inflow.WS_15m, inflow.wspd_7m, ".")
# plt.plot(inflow.WS_15m, inflow.WS_15m, ".")




final_lags = pd.DataFrame(columns=['lag', 'corr'])


window = 1*60*60   # window for flux calc, in s
for time, period in loads.groupby( (window/10.) * (loads.time/(window/10.)).round(-1)): # loop over segments in window
    print (period.index[0])   # max time = 6393600
    
    
    inflow_mod = inflow.loc[inflow.index > period.index[0]-pd.to_timedelta(1, unit="H") ]
    inflow_mod = inflow_mod.loc[inflow_mod.index < period.index[-1]+pd.to_timedelta(1, unit="H")]  
    
    if len(inflow_mod) == 0:
        continue
        
    corrs = np.empty(0)
    lags = np.arange(-50,100,1)
    for lag in lags:
        loads_mod = period.copy()
        loads_mod.index = loads_mod.index + pd.to_timedelta(lag, unit="S")
        
        merge =  loads_mod.merge(inflow_mod, left_index=True, right_index=True, how='inner')[['WS_15m', 'Anemometer', 'wspd_7m']]#.dropna()
    
        corrs = np.append(corrs, merge.WS_15m.corr(merge.Anemometer))
        
    if np.isnan(corrs).all():
        continue
              
    index_max = np.nanargmax(corrs)
    final_lag = lags[index_max]
    final_corr = corrs[index_max]
    
    
    # ### Check synchronozation
    # fig = plt.figure(figsize=(16,9))
    # plt.suptitle(loads.index[0].date())
    # ax = plt.subplot(1, 2, 1)
    # plt.plot(inflow_mod.index, inflow_mod.WS_15m,"x", color='red', label='15m wind')
    # plt.plot(period.index, period.Anemometer,".", color='black', label='15m loads', ms=2)
    # plt.plot(period.index +pd.to_timedelta(final_lag, unit="S"), period.Anemometer,".", color='grey', label='15m loads corrected', ms=2)
    # #plt.plot(loads.MS_Excel_Timestamp+pd.to_timedelta(7, unit="H"), loads.Anemometer,"x", color='blue', label='15m loads')
    # plt.ylabel('Wind speed (m/s)')
    # plt.legend()
    # plt.grid()
    # plt.ylim(0,inflow.WS_15m.max())
    # plt.xlim(period.index.mean(), period.index.mean()+pd.to_timedelta(10*60, unit="S"))
    # plt.xlabel("Time (UTC)")
    # ax = plt.subplot(1, 2, 2)
    # plt.plot(lags,corrs,".")
    # plt.plot(final_lag, final_corr,"x", color='red', label = 'lag = {}s'.format(final_lag))
    # plt.xlabel('lag (s)')
    # plt.ylabel('Correlation coefficient')
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    
    
    # create new line with time stamp
    final_lags = pd.concat([final_lags, pd.DataFrame({    
          },
        index = [ period.index[int(len(period)/2)]] )  ]) 
    
    # 7m height for inflow characteristics
    final_lags['lag'].iloc[-1] = final_lag
    final_lags['corr'].iloc[-1] = final_corr
    
    print(final_lag)
  
      
# #final_lags3 = pd.concat( [final_lags2, final_lags]) 

# final_lags2 = pd.read_pickle('Time_lags_for_sync_2022-11-19_to_2023-02-21.pkl')  
final_lags = final_lags[final_lags.lag!=-20]


final_lags.lag = final_lags.lag.where(final_lags['corr']>0.95)


fig = plt.figure()
plt.plot(loads.Anemometer.resample("H").mean(), label = '15m wind', color="lightblue")
plt.plot(final_lags.lag, ".", label = 'lag (s)')       
plt.plot(final_lags['corr']*100, ".", label = 'R$^2$*100', color='grey') 
# plt.plot(loads3.Anemometer.resample("60S").mean(), label = '15m loads')
# plt.plot(inflow3.WS_15m.resample("60S").mean(), label = '15m inflow')
plt.legend()
plt.grid()




### fit linear function to lags

periods = [
    final_lags[:'2022-11-28 00:00:00'],                           # [2.35771007e-06 6.16246944e+01]
    final_lags['2022-11-28 00:00:00':'2023-01-05 00:00:00'],      # [3.12347939e-06 7.95930566e+01]
    final_lags['2023-01-05 00:00:00':'2023-01-13 16:30:00'],      # [ 6.45083640e-06 -4.59459968e+01]
    final_lags['2023-01-13 16:30:00':'2023-02-05 00:00:00'],      # [ 3.17934698e-06 -1.27003379e+01]
    final_lags['2023-02-05 00:00:00':'2023-02-20 00:52:00'],       # [ 3.77645412e-06 -1.73116474e+01]
    final_lags['2023-02-20 00:52:00':'2023-02-23 19:01:00']       # [6.91208681e-06 -1.06307282e+02]
    ]


def func(x, a, b):
    return a * x + b

for period in periods:

    period = period[['lag']].dropna()
    period['time'] = (period.index - pd.to_datetime('2022-11-18 00:00:00')) / np.timedelta64(1,'s')   
    
    popt, pcov = curve_fit(func, period.time, period.lag)
    period['lag_fitted'] = func(period.time, *popt)
    plt.plot(period.lag_fitted)
    
    print (popt)
    



        
  
save=0
if save ==1:

    
    fig.savefig('Quicklookds_loads_processing/Synchronization_{}_to_{}.png'.format(final_lags.index[0].date(), final_lags.index[-1].date() ))
    final_lags3.to_pickle('Time_lags_for_sync_{}_to_{}.pkl'.format(final_lags3.index[0].date(), final_lags3.index[-1].date() ))
    
    
### also correlate WS_15m mit Sonics!
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        