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
import xarray as xr
import pickle
from numba import jit
import scipy.io

sys.path.append("../NSO_data_processing")

from Functions_masts import *
from Functions_general import *
from Functions_loads import *


# select dates
years   =  np.arange(2022,2024)   
months  =  np.arange(1,13)  
days    =  np.arange(1,32)  

# years   =  np.arange(2023,2024)   
#months  =  np.arange(2,13)  
# days    =  np.arange(1,32)  

# years   =  [2023] #
# months  =  [ 2] #
# days    =    [20] #np.arange(22,32)   #



# Define
resolution_loads = '20Hz'   # one out of: '20Hz', '1min_mat', '10sec_mat'   after Feb 7th 2023 only 20Hz data are available
resolution_winds = '20Hz'   # one out of: '20Hz', '1min'

loads_path = 'Y:\Wind-data/Restricted/Projects/NSO/Loads/Data/FastData_FT_formatted_new/'  
path_save =  'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Loads_v2/'   #'loads_data/'  # 
wind_path = 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v0/'   #  '../NSO_data_processing/data/'   # 

save = 1


#%% Read data for every single day



for year in years:

    for month in months:
        
        if year == 2022 and month<11: # no loads data before Nov 2022
            continue
        
        for day in days:
            
            # if month == 11 and day<25: # start of processing for new data
            #     continue
            
            try:      # Skip iterations in these time periods (all sensors not working)
                if (pd.to_datetime(str(year) + f'{month:02d}' + f'{day:02d}') > pd.to_datetime("2023-05-18")
                    ) and (pd.to_datetime(str(year) + f'{month:02d}' + f'{day:02d}') < pd.to_datetime("2023-06-01")):
                    continue
                elif (pd.to_datetime(str(year) + f'{month:02d}' + f'{day:02d}') > pd.to_datetime("2023-06-11")
                    ) and (pd.to_datetime(str(year) + f'{month:02d}' + f'{day:02d}') < pd.to_datetime("2023-07-01")):
                    continue
            except:
                pass

        
            # Read loads
            loads = read_raw_loads_at_date(year, month, day, resolution_loads, loads_path)

            if len(loads)==0:
                continue

            
            
            #%% Post-processing
            
            
            ### Correct time stamps
            
            loads.drop(["MS_Excel_Timestamp", 'R1_Enclosure_Temp', 'R2_Enclosure_Temp'], axis=1, inplace=True)   # excel timestamp is 7h behind Labview timestamp, labview timesamp is local time (7h behind UTC)
            #loads.set_index('LabVIEW_Timestamp', inplace=True)
            #loads.index = loads.LabVIEW_Timestamp +pd.to_timedelta(7, unit="H") # LabView time stamp is 7 hours behind UTC, which is Pacific Daylight Time (Summer time)()
            
            # Time conversion: before Feb 23, the GPS did not work and LabView time had a nominal offset of 7 hours, after that GPS provided UTC correctly.           
            loads.index = loads.LabVIEW_Timestamp.where(
                loads.LabVIEW_Timestamp > pd.to_datetime('2023-02-23 19:01:00'), loads.LabVIEW_Timestamp +pd.to_timedelta(7, unit="H"))

            loads.index.name = 'UTC'
            
            loads = loads.sort_index().loc['2022-11-19 00:30:00':]  # data on 18 Nov are influenced by installation
            
            
            ### Correct data
            
            loads = loads.rename(columns={"R1_SO_tilt": "R1_SO_Tilt"}) # wrong name in matlab data
            
            # remove unreasonable values
            for column in loads.filter(regex='Bending').columns:
                loads[column] = loads[column].where(loads[column]<30)
                loads[column] = loads[column].where(loads[column]>-20)
                
            for column in loads.filter(regex='Disp').columns:
                loads[column] = loads[column].where(loads[column]>70)
                loads[column] = loads[column].where(loads[column]<110)
                
            for column in loads.filter(regex='Torque').columns:
                loads[column] = loads[column].where(loads[column]<50)
                
            for column in loads.filter(regex='Tilt').columns:
                loads[column] = loads[column].where(loads[column]>-90)  
                
            loads.loc[:'2023-02-21 22:28:00','R4_Disp_Center'] = np.nan
            loads.loc['2023-02-26 01:15:00':'2023-02-26 02:00:00','R1_Disp_SW'] = np.nan
            loads.loc['2023-02-26 03:12:30':'2023-02-26 03:13:30','R1_Disp_SW'] = np.nan
            loads.loc['2023-02-26 02:59:30':'2023-02-26 02:59:46','R1_Disp_SW'] = np.nan
            loads.loc['2023-01-30 17:15:00':'2023-01-30 18:37:00','R1_Disp_SW'] = np.nan
            loads.loc['2023-01-01 12:10:00':'2023-01-01 12:46:00','R4_Disp_SW'] = np.nan
            loads.loc['2023-02-26 01:26:45':'2023-02-26 01:44:00','R4_Disp_Center'] = np.nan
            loads.loc['2023-02-26 01:31:00':'2023-02-26 02:18:40','R4_Disp_SE'] = np.nan
            loads.loc['2023-02-26 03:21:00':'2023-02-26 03:27:00','R4_Disp_SW'] = np.nan
            loads.loc['2023-02-26 02:59:00':'2023-02-26 03:04:06','R4_Disp_SW'] = np.nan
            loads.loc['2023-03-21 12:53:00':'2023-03-21 14:00:00','R4_Disp_SE'] = np.nan
            loads.R1_Disp_SW = loads.R1_Disp_SW.where(loads.R1_Disp_SW>90)
            loads.R4_Disp_SW = loads.R4_Disp_SW.where(loads.R4_Disp_SW>90)   

            if (day ==18) & (month == 5) and (year == 2023):
                loads = loads.loc[:'2023-05-19 04:26:00']
            if (day ==11) & (month == 6) and (year == 2023):
                loads = loads.loc[:'2023-06-12 01:30:00']
             
            
            # more data corrections
            loads.loc[:'2022-11-21 21:06:59','R4_SO_Tilt'] = loads.loc[:'2022-11-21 21:06:59','R4_SO_Tilt'] * -1
            loads.loc['2023-06-11 03:46:00':'2023-06-11 05:23:00','R2_Mid_Tilt'] = np.nan
            loads.loc['2023-06-09 03:09:00':'2023-06-09 03:41:00','R2_Mid_Tilt'] = np.nan
             
            ### Synchronization with wind time stamp from GPS, coefficients A and B are determined in 00_Synchronization.py; after '2023-02-23 19:01:00' the GPS sensor provided a correct time stamp
            
            loads['A'] = 0
            loads['B'] = 0
            loads['lag_fitted'] = np.nan
            loads['time'] = (loads.index - pd.to_datetime('2022-11-18 00:00:00')) / np.timedelta64(1,'s') 
            loads.loc[:'2022-11-28 00:00:00','A'] = 2.35771007e-06
            loads.loc[:'2022-11-28 00:00:00','B'] = 6.16246944e+01
            loads.loc['2022-11-28 00:00:00':'2023-01-05 00:00:00','A'] = 3.12347939e-06
            loads.loc['2022-11-28 00:00:00':'2023-01-05 00:00:00','B'] = 7.95930566e+01
            loads.loc['2023-01-05 00:00:00':'2023-01-13 16:30:00','A'] = 6.45083640e-06
            loads.loc['2023-01-05 00:00:00':'2023-01-13 16:30:00','B'] = -4.59459968e+01
            loads.loc['2023-01-13 16:30:00':'2023-02-05 00:00:00','A'] = 3.17934698e-06
            loads.loc['2023-01-13 16:30:00':'2023-02-05 00:00:00','B'] = -1.27003379e+01
            loads.loc['2023-02-05 00:00:00':'2023-02-20 00:52:00','A'] = 3.77645412e-06
            loads.loc['2023-02-05 00:00:00':'2023-02-20 00:52:00','B'] = -1.73116474e+01
            loads.loc['2023-02-20 00:52:00':'2023-02-23 19:01:00','A'] = 6.91208681e-06
            loads.loc['2023-02-20 00:52:00':'2023-02-23 19:01:00','B'] = -1.06307282e+02    
            # loads.loc['2023-02-23 19:01:00':'2023-12-31 00:00:00','A'] = 0
            # loads.loc['2023-02-23 19:01:00':'2023-12-31 00:00:00','B'] = 0  
            loads.lag_fitted = loads.B + loads.time*loads.A

            
            loads.index = loads.index + pd.to_timedelta(loads.lag_fitted, unit="S")
            loads.index = loads.index.round('50L')            
            loads.drop(["lag_fitted", 'time', 'A', "B"], axis=1, inplace=True) 

            
            # Remove outliers in the 20Hz files
            if resolution_loads == '20Hz':
                
                filter_window = '60S'
                for channel in loads.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64']).columns:
                    loads[channel] = loads[channel].where( np.abs(loads[channel] - loads[channel].rolling(filter_window, center=True, min_periods=1).median() ) 
                                                            <= (5* loads[channel].rolling(filter_window, center=True, min_periods=1).std() ) , np.nan)            
            
            # correct measurement location of bending moments 
            loads = correct_bending_moments(loads)
            
            # subtract moment offsets
            with open('Moment_offsets_2022-11-19_to_2023-06-12.pickle', 'rb') as handle:
                moment_offsets = pickle.load(handle)    # Defined in "01c_Check_data_errors.py"
                
            for column in ['R1_SO_Bending', 'R1_DO_Bending', 'R1_DO_Torque',
                           'R2_SO_Bending', 'R2_DO_Bending', 'R2_DO_Torque',
                           'R4_SO_Bending', 'R4_DO_Bending', 'R4_DO_Torque']:
                loads[column] = loads[column] - moment_offsets[column]
            
            
            # Read winds
            day_after = loads.index[0].date()+ pd.to_timedelta(1,"D")
            inflow  =  pd.concat( [     read_winds_at_dates([year],[month], [day], 
                                              path = wind_path,   flux_path = wind_path ,
                                              res= resolution_winds, read_fluxes = False, read_masts=False    ),
                                        read_winds_at_dates([day_after.year],[day_after.month], [day_after.day], 
                                              path = wind_path,   flux_path = wind_path ,
                                              res= resolution_winds, read_fluxes = False, read_masts=False    ) ])
            
            inflow_columns = ['wspd_7m', "p", "RH", "Temp", "wspd_3m", 'WS_15m']
            if inflow.empty:
                inflow = pd.DataFrame(columns=inflow_columns)
            else:
                inflow = inflow[inflow_columns].interpolate()
            inflow = inflow[inflow.index > loads.index[0] ]
            inflow = inflow[inflow.index < loads.index[-1] ]
            
            
            # Correct mirror displacements
            loads = pd.merge(loads, inflow.interpolate(limit_direction='both'), left_index=True, right_index=True, how="outer")
            
            with open('Displacement_offsets_2022-11-19_to_2023-06-26.pickle', 'rb') as handle:
                disp_offsets = pickle.load(handle)    # Defined in "01c_Check_data_errors.py"
            
            def func(x, a, b):
                return a * x + b
            
            for column in loads[[col for col in loads.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col) & ('_orig' not in col)]]:
                loads[column+"_orig"] = loads[column].copy()
                popt = disp_offsets[column]
                # offset = func(loads.Temp.interpolate(), *popt).interpolate() # do not use: Subtract temperature-dependent offset from displacements
                offset = func(15, *popt) # instead: find a constant calibration offset at a defined temperature of 15 deg C
                loads[column] = loads[column+"_orig"] - offset
                
                
            # Correct tilt angles
            
            if loads.index[0] <  pd.to_datetime('2022-12-22 00:00:00'):
                tilt_file = 'Tilt_offsets_pre_12_22_22.p'
            else:
                tilt_file = 'Tilt_offsets_post_12_22_22.p'                
            
            with open(tilt_file, 'rb') as handle:
                tilt_offsets = pickle.load(handle)    # Defined by Brooke Stanislawski
             
            for column in loads.filter(regex='Tilt$').columns:
                loads[column+"_orig"] = loads[column].copy()
                loads[column] = loads[column+"_orig"] - tilt_offsets['trough_angle_dev_{}'.format(column[0:6])]
  
    
            ## Add load coefficients
            wind_limit_coeff = 3
            loads = add_load_coefficients(loads, wind_speed_limit=wind_limit_coeff)
                

            ## correct offsets in accelerations
            with open('Accel_offsets_2022-11-19_to_2023-06-26.pickle', 'rb') as handle:
                accel_offsets = pickle.load(handle)   # Defined in "01c_Check_data_errors.py"
            
            def func(x, a, b, c, d):
                return a * np.sin(np.radians(x) * b +c) + d    
 
            for column in ['R1_SO_Accel_X', 'R1_SO_Accel_Y', 'R1_DO_Accel_X', 'R1_DO_Accel_Y',
                                      'R2_SO_Accel_X', 'R2_SO_Accel_Y', 'R2_DO_Accel_X', 'R2_DO_Accel_Y',
                                      'R4_SO_Accel_X', 'R4_SO_Accel_Y', 'R4_DO_Accel_X', 'R4_DO_Accel_Y']:      
                popt = accel_offsets[column]
                loads[column+"_orig"] = loads[column] 
                loads[column]   = (loads[column]  -popt[3]) / abs(popt[0])
            
            
            ## Check synchronization before dropping inflow parameters
            check_synchro = 0
            if check_synchro ==1:
                           
                plt.figure()
                plt.plot(loads.resample('60S').mean().index, loads.resample('60S').mean(numeric_only=True).Anemometer,".", ms=5, color='black', label='15m loads')
                plt.plot(inflow.resample('60S').mean().index, inflow.resample('60S').mean(numeric_only=True).WS_15m,".", ms=5,color='red', label='15m wind')
                #plt.plot(loads.MS_Excel_Timestamp+pd.to_timedelta(7, unit="H"), loads.Anemometer,"x", color='blue', label='15m loads')
                plt.legend(loc=1)
                plt.xlabel("Time (UTC)")
            
            if len(loads.WS_15m.dropna()) > 0:
                print ("Synchro of wind and loads data: R={}".format 
                        (loads.resample('60S').mean(numeric_only=True).Anemometer.corr(loads.resample('60S').mean(numeric_only=True).WS_15m)))
                        # Should be around 0.99

            loads.drop( inflow_columns + ['rho'], inplace=True, axis=1)
            
            
            # Add sun position
            angles = get_trough_angles(loads.index)
            loads = loads.merge(angles.projected_sun_angle, left_index=True, right_index=True, how="outer")


            ### Make 1 min means
            loads_1min = loads.resample('60S').mean(numeric_only=True)
         
            # add min, max and std values
            columns_no_stats = [col for col in loads.columns if ('orig' in col)] + ['projected_sun_angle']
            loads_1min_std = loads.drop(columns_no_stats, axis=1).add_suffix('_std').resample('60S').std(numeric_only=True)            
            loads_1min_min = loads.drop(columns_no_stats, axis=1).add_suffix('_min').resample('60S').min(numeric_only=True) 
            loads_1min_max = loads.drop(columns_no_stats, axis=1).add_suffix('_max').resample('60S').max(numeric_only=True) 
            
            loads_1min = pd.concat([loads_1min, 
                                    loads_1min_std, 
                                    loads_1min_min, 
                                    loads_1min_max], axis=1)
            

            
            
            #%% Save data files as pickle for faster reading

            
            if save == 1:
                
                print ("Saving data {} to {}...".format(loads.index[0], loads.index[-1])) 
               
                if len(loads) !=0:  
                    #plt.savefig(path_save+'Histogram_loads_'+resolution_loads+'_{}_{}h_to_{}_{}h.png'.format(loads.index[0].date(),loads.index[0].hour , loads.index[-1].date(), loads.index[-1].hour), dpi=200) 
                    plt.close('all')
                    loads.to_pickle(path_save+'Loads_'+resolution_loads+'_{}_{}h_to_{}_{}h.pkl'.format( loads.index[0].date(),loads.index[0].hour, loads.index[-1].date(),loads.index[-1].hour))
   
                    loads_1min.to_pickle(path_save+'Loads_'+'1min'+'_{}_{}h_to_{}_{}h.pkl'.format( loads.index[0].date(),loads.index[0].hour, loads.index[-1].date(),loads.index[-1].hour))
                    
                print ("ok")
               
               
               
        
            overview = 1
            if overview == 1:
                
                mpl.rcParams['lines.markersize'] = 1
                
             
                ## Look at time series
                fig = plt.figure(figsize=(17,10))   
                plt.suptitle("Loads at NSO, {} to {}".format(loads_1min.index[0] , loads_1min.index[-1] ))
                
                ax2 = plt.subplot(3, 2, 5)
                ax2.set_ylabel('Accelarations (g)')
                
                ax5 = plt.subplot(3, 2, 1, sharex=ax2)
                ax5.set_ylabel('Wind speed (m s$^{-1}$)')    
                
                ax7 = plt.subplot(3, 2, 3, sharex=ax2)
                ax7.set_ylabel('Bending moment (kNm)')
                
                ax8 = plt.subplot(3, 2, 4, sharex=ax2)
                ax8.set_ylabel('Torque moment (kNm)')    
            
                ax4 = plt.subplot(3, 2, 6, sharex=ax2)
                ax4.set_ylabel('Displacement (mm)')    
            
                ax3 = plt.subplot(3, 2, 2, sharex=ax2)
                ax3.set_ylabel('Tilt (deg)')
                
                
                for column in loads[[col for col in loads.columns if ('Bending' in col) & ('C' not in col)]]:
                    ax7.plot(loads_1min[column],".", label = loads[column].name) 
                ax7.legend(markerscale=5, loc='upper left')
                for column in loads[[col for col in loads.columns if ('Accel' in col) & ('_orig' not in col)]]:
                    ax2.plot(loads_1min[column],".", label = loads[column].name) 
                ax2.legend(markerscale=5, loc='upper left')
                for column in loads[[col for col in loads.columns if ('Torque' in col) & ('C' not in col)]]:
                    ax8.plot(loads_1min[column],".", label = loads[column].name) 
                ax8.legend(markerscale=5, loc='upper left')
                for column in loads[[col for col in loads.columns if ('Disp' in col) & ('_orig' not in col)]]:
                    ax4.plot(loads_1min[column],".", label = loads[column].name) 
                ax4.legend(markerscale=5, loc='upper left')
                for column in loads[[col for col in loads.columns if ('Tilt' in col) & ('_orig' not in col)]]:
                    ax3.plot(loads_1min[column],".", label = loads[column].name) 
                ax3.legend(markerscale=5, loc='upper left')
                ax5.plot(loads_1min.Anemometer,".", color='black', label='15m loads', ms=1)
                ax5.legend(markerscale=5, loc='upper left')
                
                for ax in [ax2, ax3, ax4, ax5, ax7, ax8]:
                    ax.grid(True)
                
                plt.tight_layout() 
                fig.autofmt_xdate() 
            
                if save ==1:
                    fig.savefig(path_save+'Overview_{}_to_{}.png'.format(loads_1min.index[0].date(), loads_1min.index[-1].date() ))
                    plt.close()


        # # check moment coefficienrs

        # plt.figure()
        
        # L_panel = 50          # m, length of panel with 6 mirror panels (check!)
        # L_segment = L_panel/6  # m, length of mirror panel (check!)
        # W = 5          # m, aperture width
        # Hc = 2.79083   # m, height of pivot axis

        
        # for column in loads[[col for col in loads.columns if ('Torque' in col)& ('C' in col)& ('R1' in col)]]:
        #       plt.plot(loads[column],".",ms=5, label = column)
        # for column in loads[[col for col in loads.columns if ('Torque' in col)& ('C' in col)& ('R1' in col)& ('_m'not in col)& ('_std' not in col)]]:
        #       plt.plot(loads.resample('60S').mean(numeric_only=True)[column],".",ms=10, label = column)
        # # plt.plot(loads.wspd_3m)
        # # plt.plot(rho(loads.p, loads.RH, loads.Temp).interpolate())
        
        # loads2 = add_load_coefficients(loads, wind_speed_limit=1)
        # for column in loads2[[col for col in loads2.columns if ('Torque' in col)& ('C' in col)& ('R1' in col)]]:
        #     plt.plot(loads2[column],".",ms=5, label = column)   
            
        # loads2 = add_load_coefficients(loads_1min, wind_speed_limit=1)
        # for column in loads2[[col for col in loads2.columns if ('Torque' in col)& ('C' in col)& ('R1' in col)& ('_m'not in col)& ('_std' not in col)]]:
        #     plt.plot(loads2[column],".",ms=5, label = column)    


   
        # # plt.legend()    
        
        # # plt.figure()
        # # plt.plot(loads.wspd_3m, loads.R1_DO_C_Torque,'.')

        
        # # plt.figure()
        # # plt.plot(loads.resample('60S').mean(numeric_only=True).wspd_3m)
        # # plt.plot(loads.resample('60S').mean(numeric_only=True).wspd_3m.rolling('60s', center=True, min_periods=0).mean())
        
        # # plt.figure()
        # # plt.plot(rho(loads.p, loads.RH, loads.Temp).interpolate().rolling('60s', center=True, min_periods=0).mean())






   
    

   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
            
            