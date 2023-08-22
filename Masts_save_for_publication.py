import numpy as np
from numpy import cos,sin
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
from scipy.fftpack import *
import scipy as sp
import scipy.signal
import scipy.signal as signal
from scipy.optimize import curve_fit
import sys
import os
import time
import itertools
import glob
import netCDF4 as nc
import xarray
import pickle
import pyarrow.parquet as pq
import pyarrow
import os
import datetime
import matplotlib as mpl
from netCDF4 import Dataset
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = 1
from windrose import WindroseAxes
import matplotlib.cm as cm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
pd.options.mode.chained_assignment = None  # default='warn'



from Functions_masts import *
from Functions_general import *



#%% Define

path = 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v0/'    
flux_path = 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v0/'

publication_path = 'Y:\Wind-data/Restricted/Projects/NSO/Data_publish/NSO/met_masts/'  # '../published_data/NSO/met_masts/' #

years   =  np.arange(2021,2024)   # 
months  =  np.arange(1,13)   # 
days    =  np.arange(1,32)   # 

years   =  [2023] #
months  =  [1,2,3,4] # 
days    =  np.arange(1,32)   #[28] # 




save  = 1
plot  = 1







#%% Read data            

print ("Processing ...")

for year in years:
    year = str(year)

    for month in months:
        
        if year == '2021' and month<10: # no data before October 2021
            continue

        
        month = f'{month:02d}'
    
        for day in days:      
            
            day = f'{day:02d}'         

            print (year, month, day) 

            
            ## Wind data
            inflow_files = sorted(glob.glob(path +'Inflow_Mast_20Hz_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
            mast_files = sorted(glob.glob(path +'Wake_masts_20Hz_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #        

            inflow = pd.DataFrame()    
            for datafile in inflow_files:
                inflow = pd.concat( [inflow, pd.read_pickle(datafile)]) 
            inflow = inflow[~inflow.index.duplicated(keep='first')]
            inflow = inflow.sort_index()  
            
            if len(inflow)==0:
                continue

            masts = pd.DataFrame()
            for datafile in mast_files:
                masts = pd.concat( [masts, pd.read_pickle(datafile)] ) 
            masts = masts[~masts.index.duplicated(keep='first')]
            masts = masts.sort_index()  
            

            ## Flux data
            inflow_files = sorted(glob.glob(flux_path +'Inflow_Mast_fluxes_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
            mast_files = sorted(glob.glob(flux_path +'Wake_Masts_fluxes_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #        

            inflow_fluxes = pd.DataFrame()    
            for datafile in inflow_files:
                inflow_fluxes = pd.concat( [inflow_fluxes, pd.read_pickle(datafile)]) 
            inflow_fluxes = inflow_fluxes[~inflow_fluxes.index.duplicated(keep='first')]
            inflow_fluxes = inflow_fluxes.sort_index()   

            masts_fluxes = pd.DataFrame()
            for datafile in mast_files:
                masts_fluxes = pd.concat( [masts_fluxes, pd.read_pickle(datafile)] ) 
            masts_fluxes = masts_fluxes[~masts_fluxes.index.duplicated(keep='first')]
            masts_fluxes = masts_fluxes.sort_index()   
                
                
            # for horizontal length scales: filter only values that are reliable
            for column in inflow_fluxes[[col for col in inflow_fluxes.columns if ('ls_U' in col)  & ('flag' not in col) ]]:
                inflow_fluxes[column] = inflow_fluxes[column].where(inflow_fluxes['flag_'+column]==0)
            for column in masts_fluxes[[col for col in masts_fluxes.columns if ('ls_U' in col)  & ('flag' not in col) ]]:
                masts_fluxes[column] = masts_fluxes[column].where(masts_fluxes[masts_fluxes[column].name[:3]+'flag_'+masts_fluxes[column].name[3:]]==0)  
            for column in inflow_fluxes[[col for col in inflow_fluxes.columns if ('ls_w' in col)  & ('flag' not in col) ]]:
                inflow_fluxes[column] = inflow_fluxes[column].where(inflow_fluxes[column]<100)
            for column in masts_fluxes[[col for col in masts_fluxes.columns if ('ls_w' in col)  & ('flag' not in col) ]]:
                masts_fluxes[column] = masts_fluxes[column].where(masts_fluxes[column]<100)  
                             
                
            ## Combine wind and flux data
            inflow_fluxes = inflow_fluxes.resample("20min").nearest()
            inflow = inflow.merge(inflow_fluxes, left_index=True, right_index=True, how="outer")
            inflow.loc[inflow.duplicated(['H_S']), 'H_S'] = np.nan
            
            if len(masts)>0:
                masts_fluxes  = masts_fluxes.resample("20min").nearest()
                masts = masts.merge(masts_fluxes, left_index=True, right_index=True, how="outer")
                
                
            ## Prepare data
            
            # drop columns
            inflow = inflow[inflow.columns.drop(list(inflow.filter(regex='_corr')))]
            inflow = inflow[inflow.columns.drop(list(inflow.filter(regex='flag')))]
            inflow.drop(['Ri_b_Sonic', 'zL'], axis=1, inplace=True)
            if len(masts)>0:
               masts = masts[masts.columns.drop(list(masts.filter(regex='flag')))] 
                
            # remove repeated values
            for column in inflow.drop(inflow.filter(regex='flag').columns,axis=1).columns:  #   exclude flag columns
                index_double_value = inflow[column].dropna().diff()== 0
                inflow[column] = inflow[column].where(index_double_value==False)
                
            if len(masts) !=0:  
                
                for column in masts.drop(masts.filter(regex='flag').columns,axis=1).columns:  #   exclude flag columns
                    index_double_value = masts[column].dropna().diff()== 0
                    masts[column] = masts[column].where(index_double_value==False)            
                
          
            

            ### Resample all Sonic data to 1 min data
            res_freq = '1min'
            
            if len(inflow) !=0:  
                inflow_1min = resample_sonic(inflow, res_freq)  
            if len(masts) !=0:  
                masts_1min = resample_sonic(masts, res_freq) 
  
            # add max and std values of wind speed
            if len(inflow) !=0:  
                inflow_max = inflow.filter(regex='wspd').add_suffix('_max').resample(res_freq).max(numeric_only=True)
                inflow_std = inflow.filter(regex='wspd').add_suffix('_std').resample(res_freq).std(numeric_only=True)   
                inflow_1min = pd.concat([inflow_1min, 
                                        inflow_max,
                                        inflow_std], axis=1)                
            if len(masts) !=0:  
                masts_max = masts.filter(regex='wspd').add_suffix('_max').resample(res_freq).max(numeric_only=True)
                masts_std = masts.filter(regex='wspd').add_suffix('_std').resample(res_freq).std(numeric_only=True)   
                masts_1min = pd.concat([masts_1min, 
                                        masts_max,
                                        masts_std], axis=1)
                
 
            # dot the rest after resampling to avoid that wind dir recalculation is after renaming and the 15m wspd gets max and std dev     
            
            for df in [inflow, inflow_1min]:
                
                # rename parameters and Change coordinate system from North-West-Up to East-North-Up
                df.columns = df.columns.str.replace(r"U_ax_", "v_")
                df.columns = df.columns.str.replace(r"V_ax_", "u_")
                for col in df.filter(regex='u_').columns:
                    df[col]= - df[col]
                df.columns = df.columns.str.replace(r"W_ax_", "w_")
                df.rename(columns={"Temp": "Temp_2m"}, inplace=True)
                for col in df.filter(regex='TI_[0-9]m').columns:
                    df.rename(columns={col: 'TI_U_'+col[-2:]}, inplace=True)
                df.index.name='timestamp_UTC'
                
                df.rename(columns={"WS_15m": "wspd_15m"}, inplace=True) 
                
                # round to 3 digits
                df = df.round(3)
                

            if len(masts)>0:
                
                for df in [masts, masts_1min]:                
                    # rename parameters and Change coordinate system from North-West-Up to East-North-Up
                    df.columns = df.columns.str.replace(r"U_ax_", "v_")
                    df.columns = df.columns.str.replace(r"V_ax_", "u_")
                    for col in df.filter(regex='u_').columns:
                        df[col]= - df[col]
                    df.columns = df.columns.str.replace(r"W_ax_", "w_")
                    for col in df.filter(regex='TI_[0-9]m').columns:
                        df.rename(columns={col: col[:-3]+'_U_'+col[-2:]}, inplace=True)
                    df.index.name='timestamp_UTC'                
    
                    # round to 3 digits
                    df = df.round(3)  
            
            
            
            
            #%% Plot data
            

            
            if plot == 1:
            
                # Timeseries

                
                fig = plt.figure(figsize=(15,9))   
                plt.suptitle(year + month + day)
                ax1 = plt.subplot(5, 2, 1)
                ax1.set_ylabel('Temperature  ($^\circ$C)')

                
                ax2 = plt.subplot(5, 2, 2)
                ax2.set_ylabel('RH (%)')
                
                ax3 = plt.subplot(5, 2, 3)
                ax3.set_ylabel('Stability R_f')
                ax3.set_ylim(-0.2,0.2)
                
                ax4 = plt.subplot(5, 2, 4)
                ax4.set_ylabel('Heat flux (W m$^{-2}$)')
                
                ax5 = plt.subplot(5, 2, 5)
                ax5.set_ylabel('Wind speed 7m (m s$^{-1}$)')
                
                ax6 = plt.subplot(5, 2, 6)
                ax6.set_ylabel('Wind direction 7m ($^\circ$)')
                
                ax7 = plt.subplot(5, 2, 9)
                ax7.set_ylabel('length scale $w$ (m)')
    
                ax8 = plt.subplot(5, 2, 10)
                ax8.set_ylabel('length scale $U$ (m)')
                
                ax9 = plt.subplot(5, 2, 7)
                ax9.set_ylabel('TI 7m')
    
                ax10 = plt.subplot(5, 2, 8)
                ax10.set_ylabel('TKE 7m (m$^{2}$ s$^{-2}$)')
                                   
                ax1.plot(  inflow_1min.Temp_2m  ,"." , label = "Temp 2m")
                try:
                    ax1.plot(inflow_1min.Temp_3m,'8',label = 'Temp 3.5m', color="C0")
                    ax1.plot(inflow_1min.Temp_7m,'s',label = 'Temp 7m', color="C0")   
                except:
                    pass
                ax2.plot(  inflow_1min.RH  ,"." )
                ax3.plot(  inflow_1min.R_f  ,"." )
                ax4.plot(  inflow_1min.H_S  ,"." )    
                ax5.plot(  inflow_1min.wspd_7m  ,".",label='inflow' )
                ax6.plot(  inflow_1min.wdir_7m,"." )
                ax7.plot(  inflow_1min.ls_w_7m  ,"." )    
                ax8.plot(  inflow_1min.ls_U_7m  ,"." )
                ax9.plot(  inflow_1min.TI_U_7m  ,"." )    
                ax10.plot(  inflow_1min.TKE_7m  ,"." )   
                
                if len(masts) !=0: 
                    for height_col in [col for col in masts_1min.columns if ('wspd' in col)  & ('_m' not in col) & ('7m' in col) & ('_s' not in col)]:  
                        ax5.plot(masts_1min[height_col],'.', label = 'mast '+height_col[1])
                    for height_col in [col for col in masts_1min.columns if ('wdir' in col)  & ('_m' not in col) & ('7m' in col) & ('_s' not in col)]:  
                        ax6.plot(masts_1min[height_col],'.', label = '')    
                    for height_col in [col for col in masts_1min.columns if ('TKE' in col)  & ('_m' not in col) & ('7m' in col) & ('_s' not in col)]:  
                        ax10.plot(masts_1min[height_col],'.', label = '')  
                    for height_col in [col for col in masts_1min.columns if ('TI_U' in col)  & ('_m' not in col) & ('7m' in col) & ('_s' not in col)]:  
                        ax9.plot(masts_1min[height_col],'.', label = '')                              

                
                ax5.legend(loc=3)
                ax1.legend()             
                for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]:
                    ax.grid(True)
                    #ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
                
                fig.autofmt_xdate() 
                plt.tight_layout()  



            
   
            
                 
            #%% Save data files as Parwuet

                
                if save == 1:
                        
                    print ("Saving data ...")  
                        
                    
                    ## Parquet files 
                    # Define metadata and units
                    save_path = publication_path + 'inflow_mast/'
                    inflow_units = {'u_' :'m/s',
                                    'v_' :'m/s',
                                    'w_' :'m/s', 
                                    'Ts' :'degC',
                                    'wspd' :'m/s',
                                    'wdir' :'deg from North',                           
                                    'TKE' :'m2/s2',       
                                    'TI' :'-',
                                    'ls' :'m',                            
                                    'Temp' :'degC',                    
                                    'RH' :'%' ,
                                    'p' :'hPa' ,
                                    'H_S' :'W/m2' ,
                                    'Tau' :'kg m/s' ,
                                    'R_f' :'-' ,
                                    'Ri_b' :'-'}
                    inflow_metadata = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'author':'Ulrike Egerer',
                                'units': inflow_units}
                    complete_path = create_file_structure(file_path = save_path, resolution = '20Hz', year=year, month=month, day=day)
                    inflow.to_parquet(complete_path + 
                                      '/Inflow_Mast_20Hz_{}_{:0>2}h_to_{}_{:0>2}h.parquet'
                                      .format(inflow.index[0].date(),inflow.index[0].hour, 
                                              inflow.index[-1].date(),inflow.index[-1].hour), 
                                      #metadata=inflow_metadata
                                      )
                    complete_path = create_file_structure(file_path = save_path, resolution = res_freq, year=year, month=month, day=day)
                    inflow_1min.to_parquet(complete_path + 
                                      '/Inflow_Mast_{}_{}_{:0>2}h_to_{}_{:0>2}h.parquet'
                                      .format(res_freq, inflow.index[0].date(),inflow.index[0].hour, 
                                              inflow.index[-1].date(),inflow.index[-1].hour), 
                                      # metadata=inflow_metadata
                                      ) 
                    
                    if plot == 1:
                        
                       plt.savefig('Y:\Wind-data/Restricted/Projects/NSO/Daily_quicklooks' + 
                                         '/Mast_winds_{}_{}_{:0>2}h_to_{}_{:0>2}h.png'
                                         .format(res_freq, inflow.index[0].date(),inflow.index[0].hour, 
                                                 inflow.index[-1].date(),inflow.index[-1].hour), dpi=200) 
                       plt.close()
                       
                    if len(masts) !=0: 
                    
                        save_path = publication_path + 'wake_masts/'
                        wake_units = {  'u_' :'m/s',
                                        'v_' :'m/s',
                                        'w_' :'m/s', 
                                        'Ts' :'degC',
                                        'wspd' :'m/s',
                                        'wdir' :'deg from North',                           
                                        'TKE' :'m2/s2',       
                                        'TI' :'-',
                                        'ls' :'m'}
                        wake_metadata = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    'author':'Ulrike Egerer',
                                    'units': wake_units}
                        complete_path = create_file_structure(file_path = save_path, resolution = '20Hz', year=year, month=month, day=day)
                        masts.to_parquet(complete_path + 
                                          '/Wake_Masts_20Hz_{}_{:0>2}h_to_{}_{:0>2}h.parquet'
                                          .format(masts.index[0].date(),masts.index[0].hour, 
                                                  masts.index[-1].date(),masts.index[-1].hour), 
                                          # metadata=wake_metadata
                                          )
                        complete_path = create_file_structure(file_path = save_path, resolution = res_freq, year=year, month=month, day=day)
                        masts_1min.to_parquet(complete_path + 
                                          '/Wake_Masts_{}_{}_{:0>2}h_to_{}_{:0>2}h.parquet'
                                          .format(res_freq, masts.index[0].date(),masts.index[0].hour, 
                                                  masts.index[-1].date(),masts.index[-1].hour), 
                                          # metadata=wake_metadata
                                          )
                        
                        

    
                print ("ok")            
            
            
             
                  
            # ## Pickle files (for size comparison)
            # inflow.to_pickle(publication_path + 'Inflow_Mast_20Hz_{}_{}h_to_{}_{}h.pkl'.format(inflow.index[0].date(),inflow.index[0].hour, inflow.index[-1].date(),inflow.index[-1].hour))
            # inflow_1min.to_pickle(publication_path + 'Inflow_Mast_{}_{}_{}h_to_{}_{}h.pkl'.format(res_freq, inflow.index[0].date(),inflow.index[0].hour, inflow.index[-1].date(),inflow.index[-1].hour))  
            # masts.to_pickle(publication_path + 'Wake_Masts_20Hz_{}_{}h_to_{}_{}h.pkl'.format(masts.index[0].date(),masts.index[0].hour, masts.index[-1].date(),masts.index[-1].hour))
            # masts_1min.to_pickle(publication_path + 'Wake_Masts_{}_{}_{}h_to_{}_{}h.pkl'.format(res_freq, masts.index[0].date(),masts.index[0].hour, masts.index[-1].date(),masts.index[-1].hour))
           
      
            # # Netcdf (for file size comparison)
            # if len(masts) !=0: 
            #     dfs = [inflow, inflow_1min, masts, masts_1min]   # 
            #     names = ["Inflow_Mast_20Hz", "Inflow_Mast_1min", "Wake_Masts_20Hz", "Wake_Masts_1min"]
            # else:
            #     dfs = [inflow, inflow_1min]   # 
            #     names = ["Inflow_Mast_20Hz", "Inflow_Mast_1min"]  
                
                
            # for df, name in zip(dfs, names):
            
            #     try: xr.close()  
            #     except: pass

            #     xr = xarray.Dataset.from_dataframe(df)
            #     xr.attrs = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"), 'author':'Ulrike Egerer'}
                
            #     try:
            #         xr['RH'].attrs={'units':'%'}
            #         xr['p'].attrs={'units':'hPa'}
            #         xr['H_S'].attrs={'units':'W/m2'}
            #         xr['Tau'].attrs={'units':'kg m/s'}
            #         xr['R_f'].attrs={'units':'-'}
            #     except:
            #         pass
                
            #     for col in list(df.filter(regex='wspd')):
            #         xr[col].attrs={'units':'m/s'}
            #     for col in list(df.filter(regex='u_')):
            #         xr[col].attrs={'units':'m/s'}
            #     for col in list(df.filter(regex='v_')):
            #         xr[col].attrs={'units':'m/s'}
            #     for col in list(df.filter(regex='w_')):
            #         xr[col].attrs={'units':'m/s'}                        
                    
            #     for col in list(df.filter(regex='Temp')):
            #         xr[col].attrs={'units':'degC'}
            #     for col in list(df.filter(regex='Ts')):
            #         xr[col].attrs={'units':'degC'}
            #     for col in list(df.filter(regex='wdir')):
            #         xr[col].attrs={'units':'deg from North'}
            #     for col in list(df.filter(regex='TKE')):
            #         xr[col].attrs={'units':'m2/s2'}       
            #     for col in list(df.filter(regex='TI')):
            #         xr[col].attrs={'units':'-'}
            #     for col in list(df.filter(regex='ls')):
            #         xr[col].attrs={'units':'m'}
            #     for col in list(df.filter(regex='Ri_b')):
            #         xr[col].attrs={'units':'-'}                     

            #     #xr.info()
            #     xr.to_netcdf(publication_path + name+'_{}_{}h_to_{}_{}h.nc'.format(df.index[0].date(),df.index[0].hour, df.index[-1].date(),df.index[-1].hour))
            #     xr.close()  

            
            # read: 
            # test = xarray.open_dataset('../data_publish/Inflow_Mast_20Hz_{}_{}h_to_{}_{}h.nc'.
            #                            format(inflow.index[0].date(),inflow.index[0].hour, inflow.index[-1].date(),inflow.index[-1].hour))
            # test.to_dataframe()
                    
       
            
           
            
           
    
