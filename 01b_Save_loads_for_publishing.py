import numpy as np
from numpy import cos,sin
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from scipy.fftpack import *
import scipy as sp
import scipy.signal
import scipy.signal as signal
from scipy.optimize import curve_fit
import sys
import time
import glob
import netCDF4 as nc
import pyarrow.parquet as pq
import xarray
import pickle
from numba import jit
import scipy.io
import matplotlib.colors as colors
from windrose import WindroseAxes
import matplotlib.cm as cm
import os
import shutil

sys.path.append("../NSO_data_processing")

from Functions_masts import *
from Functions_general import *
from Functions_loads import *




# select dates
years   =  np.arange(2022,2024)   
months  =  [11,12,1,2,3,4,5,6]  
days    =  np.arange(1,32)  

years   =  [2023] #
months  =  [2] # 
days    =  [22] # np.arange(19,22)   #


# path to files
mast_path = 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v0/'   
loads_path = 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Loads_v1/' 
save_path = 'Y:\Wind-data/Restricted/Projects/NSO/Data_publish/NSO/loads/' # '../published_data/NSO/loads/' # 


# Define
resolution_loads = '20Hz'   # one out of: '20Hz', '1min'
units = unit_dict


save = 1
plot = 1


#%% Loop over every day


for year in years:

    for month in months:
    
        for day in days:
            
            print (year, month, day) 
            
            day_after = pd.to_datetime(str(year) + f'{month:02d}' + f'{day:02d}').date()+ pd.to_timedelta(1,"D")
            
            # Read loads 
            loads  =  pd.concat( [ read_processed_loads_at_dates([year],[month], [day], 
                                                  resolution_loads, loads_path=loads_path),
                                   read_processed_loads_at_dates([day_after.year],[day_after.month], [day_after.day], 
                                                  resolution_loads, loads_path=loads_path)
                                  ])
            loads = loads.drop_duplicates().sort_index()
            if len(loads)==0:
                continue    
            
            
            ## Make data from   00:00 UTC to 24:00UTC  
            start = pd.to_datetime(str(loads.index[0].date())) +   pd.to_timedelta(1,"D")
            end = start +   pd.to_timedelta(1,"D")    
            
            loads = loads[start:end]   


            ## Prepare data
            
            # drop columns
            #loads = loads[loads.columns.drop(list(loads.filter(regex='_corr')))]
            loads.drop(['LabVIEW_Timestamp'], axis=1, inplace=True)
            loads.drop(loads.filter(regex="Accel_._orig").columns, axis=1, inplace=True)
               
            
            # rename parameters 
            loads.columns = loads.columns.str.replace(r"_orig", "_raw")

            loads.index.name='timestamp_UTC'
 
            # round to 3 digits
            loads = loads.round(3)      
            
            

            ## Make 1 min means
            loads_1min = loads.resample('60S').mean(numeric_only=True)
         
            # add min, max and std values
            columns_no_stats = [col for col in loads.columns if ('raw' in col)] + ['projected_sun_angle']   
            loads_1min_std = loads.drop(columns_no_stats, axis=1).add_suffix('_std').resample('60S').std(numeric_only=True)            
            loads_1min_min = loads.drop(columns_no_stats, axis=1).add_suffix('_min').resample('60S').min(numeric_only=True) 
            loads_1min_max = loads.drop(columns_no_stats, axis=1).add_suffix('_max').resample('60S').max(numeric_only=True) 
            
            loads_1min = pd.concat([loads_1min, 
                                    loads_1min_std, 
                                    loads_1min_min, 
                                    loads_1min_max], axis=1)   




            #%% Overview 
            
        
            if plot == 1:
                
                # Read winds
                resolution_winds = '1min'   # one out of: '20Hz', '1min'
                inflow  =  read_winds_at_dates([loads.index[0].year],[loads.index[0].month], [loads.index[0].day], 
                                                 path = mast_path,   flux_path = mast_path ,
                                                 res= resolution_winds, read_fluxes = False, read_masts=False    )
                inflow = inflow.drop_duplicates().sort_index()              
                inflow = inflow[start:end]    
                
                # # Combine loads and wind
                # all = pd.merge(loads, inflow, left_index=True, right_index=True, how="outer")
                
                
                mpl.rcParams['lines.markersize'] = 1
               
             
                ## Look at time series
                fig = plt.figure(figsize=(17,9))   
                #plt.suptitle("Example time period at NSO, {} to {}".format(start.date(),end.date()))
                
                
                ax1 = plt.subplot(4, 2, 1)
                ax1.set_ylabel('Wind speed 3m (m s$^{-1}$)')    
            
                ax6 = plt.subplot(4, 2, 3, sharex=ax1)
                ax6.set_ylabel('Wind direction 3m ($^\circ$)')
                plt.yticks([0, 90, 180, 270, 360], ['N', 'E', 'S', "w", "N"]) 
            
                ax9 = plt.subplot(4, 2, 4, sharex=ax1)
                ax9.set_ylabel('TI 3m')
                
                ax10 = plt.subplot(4, 2, 2, sharex=ax1)
                ax10.set_ylabel('TKE 3m (m$^{2}$ s$^{-2}$)')
                
                ax7 = plt.subplot(4, 2, 7, sharex=ax1)
                ax7.set_ylabel('Bending moment SO (kNm)')
                
                ax8 = plt.subplot(4, 2, 8, sharex=ax1)
                ax8.set_ylabel('Torque moment (kNm)')    
            
                ax4 = plt.subplot(4, 2, 6, sharex=ax1)
                ax4.set_ylabel('Displacement (mm)')    
            
                ax3 = plt.subplot(4, 2, 5, sharex=ax1)
                ax3.set_ylabel('Tilt ($^\circ$)')
            
                ax1.plot(inflow.wspd_3m,".", label = "inflow")
                ax6.plot(inflow.wdir_3m,".")
                ax9.plot(inflow.TI_3m,".")    
                ax10.plot(inflow.TKE_3m,".")  
            
                ax1.legend(fontsize=7, markerscale= 4, loc='center left', bbox_to_anchor=(1, 0.5))
                for column in loads[[col for col in loads.columns if ('SO_Bending' in col)  & ('_m' not in col) & ('_s' not in col)]]:
                    ax7.plot(loads[column],".", label = loads[column].name[:2]) 
                ax7.legend(fontsize=7, markerscale= 4, loc='center left', bbox_to_anchor=(1, 0.5))
                for column in loads[[col for col in loads.columns if ('Torque' in col)  & ('_m' not in col) & ('_s' not in col)]]:
                    ax8.plot(loads[column],".", label = loads[column].name[:2]) 
                ax8.legend(fontsize=7, markerscale= 4, loc='center left', bbox_to_anchor=(1, 0.5))
                for column in loads[[col for col in loads.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('_raw' not in col)& ('_level' not in col)]]:
                    ax4.plot(loads[column],".", label = loads[column].name.replace('_Disp_', ' ')) 
                ax4.legend(fontsize=7, markerscale= 4, loc='center left', bbox_to_anchor=(1, 0.5))
                for column in loads[[col for col in loads.columns if ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col) & ('SO' in col)]]:
                    ax3.plot(loads[column],".", label = loads[column].name[:2]) 
                ax3.legend(fontsize=7, markerscale= 4, loc='center left', bbox_to_anchor=(1, 0.5))
                
                for ax in [ax1, ax3, ax4, ax6, ax7, ax8, ax9, ax10]:
                    ax.grid(True)
                    ax.set_xlim(start, end)
            
                ax8.set_xlabel('Date/ Time (UTC)')
                ax7.set_xlabel('Date/ Time (UTC)')
            
                fig.autofmt_xdate() 
                plt.tight_layout() 
    
  
    


            #%% Save data files as Parquet

            
            if save == 1:
                    
                print ("Saving data ...")  
                  
                ## Parquet files 
                # Define metadata and units
                loads_metadata = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'author':'Ulrike Egerer',
                            'units': units}
                
                # Create folder stucture and save data
                complete_path = create_file_structure(file_path = save_path, resolution = '20Hz', year=loads.index[0].year, month=loads.index[0].month, day=loads.index[0].day)
                loads.to_parquet(complete_path +
                                  '/Loads_20Hz_{}_{:0>2}h_to_{}_{:0>2}h.parquet'
                                  .format(loads.index[0].date(),loads.index[0].hour, 
                                          loads.index[-1].date(),loads.index[-1].hour), 
                                 # metadata=loads_metadata
                                  )
                
                complete_path = create_file_structure(file_path = save_path, resolution = '1min', year=loads.index[0].year, month=loads.index[0].month, day=loads.index[0].day)
                loads_1min.to_parquet(complete_path +
                                  '/Loads_1min_{}_{:0>2}h_to_{}_{:0>2}h.parquet'
                                  .format(loads.index[0].date(),loads.index[0].hour, 
                                          loads.index[-1].date(),loads.index[-1].hour), 
                                 # metadata=loads_metadata
                                  ) 
                
                
                if plot == 1:
                    
                   plt.savefig('Y:\Wind-data/Restricted/Projects/NSO/Daily_quicklooks' + 
                                     '/Loads_{}_{:0>2}h_to_{}_{:0>2}h.png'
                                     .format(loads.index[0].date(),loads.index[0].hour, 
                                             loads.index[-1].date(),loads.index[-1].hour), dpi=200) 
                   plt.close()
   
                

                  

            print ("ok")            
            
            
            
            
            
            
            
            
            
            
                #  ## Pickle files (for size comparison)
                # loads.to_pickle(save_path + 'Loads_20Hz_{}_{}h_to_{}_{}h.pkl'.format(loads.index[0].date(),loads.index[0].hour, loads.index[-1].date(),loads.index[-1].hour))
                # loads_1min.to_pickle(save_path + 'Loads_1min_{}_{}h_to_{}_{}h.pkl'.format(loads.index[0].date(),loads.index[0].hour, loads.index[-1].date(),loads.index[-1].hour))  


                #   ## Netcdf files
                # dfs = [loads, loads_1min]   # 
                # names = ["Loads_20Hz", "Loads_1min"] #
                    
                    
                # for df, name in zip(dfs, names):
                
                #     try: xr.close()  
                #     except: pass

                #     xr = xarray.Dataset.from_dataframe(df)
                #     xr.attrs = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), 'author':'Ulrike Egerer'}
                    
                #     for col in loads.columns:
                #         u = col.replace('_max', '').replace('_min', '').replace('_std', '').replace('_orig', '')
                #         xr[col].attrs={'units':units[u]}

                #     #xr.info()
                #     xr.to_netcdf(save_path +name+'_{}_{}h_to_{}_{}h.nc'.format(df.index[0].date(),df.index[0].hour, df.index[-1].date(),df.index[-1].hour))
                #     xr.close()  
            
            
                # read netcdf: 
                # test = xarray.open_dataset(save_path + 'Loads_20Hz_{}_{}h_to_{}_{}h.nc'.
                #                            format(loads.index[0].date(),loads.index[0].hour, loads.index[-1].date(),loads.index[-1].hour))
                # test.to_dataframe()
                    







   
    
   
    
  
    
   
    
            
            