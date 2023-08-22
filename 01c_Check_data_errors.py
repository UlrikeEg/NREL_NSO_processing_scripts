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
import os



sys.path.append("../NSO_data_processing")

from Functions_masts import *
from Functions_general import *
from Functions_loads import *

# os.chdir(__file__)



# select dates
years   =  np.arange(2022,2024)   #[2022] #
months  =  [11,12,1,2,3,4, 5, 6] #np.arange(1,13)   # [3] # 
days    =  np.arange(1,32)   #[5] # 


# years   =  [2023] #
# months  =  [2] # 
# days    =  [25]



# Define
resolution_loads = '1min'   # one out of: '20Hz', '1min'
resolution_winds = '1min' 


loads_path= 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Loads_v2/'   #    'loads_data/'   # 
wind_path = 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v0/'   #    '../NSO_data_processing/data/'  #



# Read loads
loads = read_processed_loads_at_dates(years, months, days, resolution_loads, loads_path=loads_path)

# Read winds
inflow, masts = read_winds_at_dates(years, months, days, read_masts=True, path = wind_path,   
                                    flux_path = wind_path , res= resolution_winds  )
inflow = inflow[inflow.index > loads.index[0] ]
inflow = inflow[inflow.index < loads.index[-1] ]
masts = masts[masts.index > loads.index[0] ]
masts = masts[masts.index < loads.index[-1] ]

for column in loads.filter(regex='Tilt').columns:
    loads[column] = loads[column].where(loads[column]>-90) 
loads.loc['2023-06-11 03:46:00':'2023-06-11 05:23:00','R2_Mid_Tilt'] = np.nan
loads.loc['2023-06-09 03:09:00':'2023-06-09 03:41:00','R2_Mid_Tilt'] = np.nan



# comnine all data
angles = get_trough_angles(loads.index)


all = pd.merge(loads[[col for col in loads.columns if ('_m' not in col) & ('_s' not in col)]], 
               inflow[[col for col in inflow.columns if ('_m' not in col) & ('_s' not in col)]]
               , left_index=True, right_index=True, how="inner")
all = pd.merge(all, masts[[col for col in masts.columns if ('_m' not in col) & ('_s' not in col)]]
               , left_index=True, right_index=True, how="outer")  
all = pd.merge(all, angles.trough_angle, left_index=True, right_index=True, how="inner")  

all = all[loads.index[0] : loads.index[-1]]


        
             




 
loads['R1_Tilt'] = loads[[col for col in loads.columns if 
                          ('R1' in col)  & ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col)]].mean(axis=1)
loads['R2_Tilt'] = loads[[col for col in loads.columns if 
                          ('R2' in col)  & ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col)]].mean(axis=1)
loads['R4_Tilt'] = loads[[col for col in loads.columns if 
                          ('R4' in col)  & ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col)]].mean(axis=1)


# Check moments
moments = 1
if moments == 1:
    
    
    mpl.rcParams['lines.markersize'] = 1

    # filter low wind days
    wind_limit = 1
    loads_filt = loads[loads.Anemometer < wind_limit]
    
    plt.figure(figsize = (12,6))
    plt.suptitle("Winds below {}m/s".format(wind_limit))
    ax1 = plt.subplot(1, 3, 1)
    plt.title('Row 1')
    ax1.set_ylabel('Moment (kNm)')
    ax1.set_xlabel('tilt (deg)')
    plt.grid(True)
    plt.plot(loads_filt.R1_SO_Tilt, loads_filt.R1_SO_Bending , ".", label = 'R1_SO_Bending')
    plt.plot(loads_filt.R1_DO_Tilt, loads_filt.R1_DO_Bending , ".", label = 'R1_DO_Bending')
    plt.plot(loads_filt.R1_DO_Tilt, loads_filt.R1_DO_Torque , ".", label = 'R1_DO_Torque')  
    ax1.legend()
    
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    plt.title('Row 2')
    ax2.set_ylabel('Moment (kNm)')
    ax2.set_xlabel('tilt (deg)') 
    plt.grid(True)
    plt.plot(loads_filt.R2_SO_Tilt, loads_filt.R2_SO_Bending , ".", label = 'R2_SO_Bending')
    plt.plot(loads_filt.R2_DO_Tilt, loads_filt.R2_DO_Bending , ".", label = 'R2_DO_Bending')
    plt.plot(loads_filt.R2_DO_Tilt, loads_filt.R2_DO_Torque , ".", label = 'R2_DO_Torque')
    ax2.legend()
    
    ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    plt.title('Row 4')
    ax3.set_ylabel('Moment (kNm)')
    ax3.set_xlabel('tilt (deg)')    
    plt.grid(True)
    plt.plot(loads_filt.R4_SO_Tilt, loads_filt.R4_SO_Bending , ".", label = 'R4_SO_Bending')
    plt.plot(loads_filt.R4_DO_Tilt, loads_filt.R4_DO_Bending , ".", label = 'R4_DO_Bending')
    plt.plot(loads_filt.R4_DO_Tilt, loads_filt.R4_DO_Torque , ".", label = 'R4_DO_Torque')
    ax3.legend()
    
    plt.tight_layout()
    
    
    
    
    # Find offsets for all moments during low-wind days  ---- determine offsets with Loads_v1 data!
    find_offsets = 0
    if find_offsets == 1:
    
    
        plt.figure()
        moment_offsets = {}  
        i = 1
    
            
        
    
        for column in ['R1_SO_Bending', 'R1_DO_Bending', 'R1_DO_Torque',
                       'R2_SO_Bending', 'R2_DO_Bending', 'R2_DO_Torque',
                       'R4_SO_Bending', 'R4_DO_Bending', 'R4_DO_Torque']:
        
            tilt_col = [col for col in loads.columns if 
                                      (column[:2] +'_Tilt' in col)  & ('_m' not in col) & ('_s' not in col)][0]
            
            test = loads_filt.where(loads_filt[tilt_col]<70).where(loads[tilt_col]>-70)
            
            plt.subplot(3,3,i)
            plt.title(column)
            #plt.plot(loads[tilt_col], loads[column],".", label = "all data", alpha=0.5)
            plt.plot(loads_filt[tilt_col], loads_filt[column],".", label = "all low-wind data", alpha=0.5)
            plt.plot(test[tilt_col], test[column],".", label = "all low-wind data +-70 deg", alpha=0.5)
            plt.plot(np.nan, np.nan, ".", ms = 10, color='black', label = "points for offset")
            plt.grid()
            plt.xlabel("tilt")
            plt.ylabel("moment")
        
            column_offsets = np.array([])
            for tilt, per in test.groupby(test[tilt_col].round(-1)):  
                if len(per)>10:
                    plt.plot(tilt, per[column].mean(),".", ms = 10, color='black', label = "")
                    column_offsets = np.append(column_offsets, per[column].mean())
    
            moment_offsets[column] = np.mean(column_offsets)
            plt.axhline(moment_offsets[column], color="black", label = "calibration offset")
            
            
            i = i+1
        
        plt.legend()
        
        with open('Moment_offsets_{}_to_{}.pickle'.format( loads.index[0].date()
                                                               , loads.index[-1].date()), 'wb') as handle:
            pickle.dump(moment_offsets, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        
        
        # subtract moment offsets
        for column in ['R1_SO_Bending', 'R1_DO_Bending', 'R1_DO_Torque',
                       'R2_SO_Bending', 'R2_DO_Bending', 'R2_DO_Torque',
                       'R4_SO_Bending', 'R4_DO_Bending', 'R4_DO_Torque']:
            loads_filt[column] = loads_filt[column] - moment_offsets[column]
            
        # plot again    
        plt.figure(figsize = (12,6))
        plt.suptitle("Winds below {}m/s".format(wind_limit))
        ax1 = plt.subplot(1, 3, 1)
        plt.title('Row 1')
        ax1.set_ylabel('Moment (kNm)')
        ax1.set_xlabel('tilt (deg)')
        plt.grid(True)
        plt.plot(loads_filt.R1_SO_Tilt, loads_filt.R1_SO_Bending , ".", label = 'R1_SO_Bending')
        plt.plot(loads_filt.R1_DO_Tilt, loads_filt.R1_DO_Bending , ".", label = 'R1_DO_Bending')
        plt.plot(loads_filt.R1_DO_Tilt, loads_filt.R1_DO_Torque , ".", label = 'R1_DO_Torque')  
        ax1.legend()
        
        ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
        plt.title('Row 2')
        ax2.set_ylabel('Moment (kNm)')
        ax2.set_xlabel('tilt (deg)') 
        plt.grid(True)
        plt.plot(loads_filt.R2_SO_Tilt, loads_filt.R2_SO_Bending , ".", label = 'R2_SO_Bending')
        plt.plot(loads_filt.R2_DO_Tilt, loads_filt.R2_DO_Bending , ".", label = 'R2_DO_Bending')
        plt.plot(loads_filt.R2_DO_Tilt, loads_filt.R2_DO_Torque , ".", label = 'R2_DO_Torque')
        ax2.legend()
        
        ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
        plt.title('Row 4')
        ax3.set_ylabel('Moment (kNm)')
        ax3.set_xlabel('tilt (deg)')    
        plt.grid(True)
        plt.plot(loads_filt.R4_SO_Tilt, loads_filt.R4_SO_Bending , ".", label = 'R4_SO_Bending')
        plt.plot(loads_filt.R4_DO_Tilt, loads_filt.R4_DO_Bending , ".", label = 'R4_DO_Bending')
        plt.plot(loads_filt.R4_DO_Tilt, loads_filt.R4_DO_Torque , ".", label = 'R4_DO_Torque')
        ax3.legend()
        
        plt.tight_layout()
        
    
    
    
    
    # plot wind dependence
    all_filt = all.where(all.wdir_7m<315).where(all.wdir_7m>225)#.where(all.R1_DO_Tilt>85)
    
    plt.figure(figsize = (12,6))
    plt.suptitle("Wind dir {} to {}, tilt {} to {}".format
                 (round(all_filt.wdir_7m.min()), 
                 round(all_filt.wdir_7m.max()),
                 round(all_filt.R1_DO_Tilt.min()), 
                 round(all_filt.R1_DO_Tilt.max())))
    ax1 = plt.subplot(1, 3, 1)
    plt.title('Row 1')
    ax1.set_ylabel('Moment (kNm)')
    ax1.set_xlabel('Wind speed (m/s)')
    plt.grid(True)
    plt.plot(all_filt.Anemometer, all_filt.R1_SO_Bending , ".", label = 'R1_SO_Bending')
    plt.plot(all_filt.Anemometer, all_filt.R1_DO_Bending , ".", label = 'R1_DO_Bending')
    plt.plot(all_filt.Anemometer, all_filt.R1_DO_Torque , ".", label = 'R1_DO_Torque')   
    ax1.legend()
    
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    plt.title('Row 2')
    ax2.set_ylabel('Moment (kNm)')
    ax2.set_xlabel('Wind speed (m/s)') 
    plt.grid(True)
    plt.plot(all_filt.Anemometer, all_filt.R2_SO_Bending , ".", label = 'R2_SO_Bending')
    plt.plot(all_filt.Anemometer, all_filt.R2_DO_Bending , ".", label = 'R2_DO_Bending')
    plt.plot(all_filt.Anemometer, all_filt.R2_DO_Torque , ".", label = 'R2_DO_Torque')
    ax2.legend()
    
    ax3 = plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    plt.title('Row 4')
    ax3.set_ylabel('Moment (kNm)')
    ax3.set_xlabel('Wind speed (m/s)')    
    plt.grid(True)
    plt.plot(all_filt.Anemometer, all_filt.R4_SO_Bending , ".", label = 'R4_SO_Bending')
    plt.plot(all_filt.Anemometer, all_filt.R4_DO_Bending , ".", label = 'R4_DO_Bending')
    plt.plot(all_filt.Anemometer, all_filt.R4_DO_Torque , ".", label = 'R4_DO_Torque')
    ax3.legend()
    
    plt.tight_layout()
    
    
    



# Check accelarations
accel = 0
if accel == 1:
    
    # filter low wind days
    wind_limit = 1
    loads_filt = loads[loads.Anemometer < wind_limit][:'2023-06-12 01:30:00']

    plt.figure()
    plt.plot(loads_filt.R1_SO_Tilt, loads_filt.R1_SO_Accel_X_orig , ".", label = 'R1_SO_Accel_X_orig')
    plt.plot(loads_filt.R1_SO_Tilt, loads_filt.R1_SO_Accel_Y_orig , "x", label = 'R1_SO_Accel_Y_orig')
    plt.plot(loads_filt.R1_DO_Tilt, loads_filt.R1_DO_Accel_X_orig , ".", label = 'R1_DO_Accel_X_orig')
    plt.plot(loads_filt.R1_DO_Tilt, loads_filt.R1_DO_Accel_Y_orig , "x", label = 'R1_DO_Accel_Y_orig')
    
    plt.plot(loads_filt.R2_SO_Tilt, loads_filt.R2_SO_Accel_X_orig , ".", label = 'R2_SO_Accel_X_orig')
    plt.plot(loads_filt.R2_SO_Tilt, loads_filt.R2_SO_Accel_Y_orig , "x", label = 'R2_SO_Accel_Y_orig')
    plt.plot(loads_filt.R2_DO_Tilt, loads_filt.R2_DO_Accel_X_orig , ".", label = 'R2_DO_Accel_X_orig')
    plt.plot(loads_filt.R2_DO_Tilt, loads_filt.R2_DO_Accel_Y_orig , "x", label = 'R2_DO_Accel_Y_orig')
    
    plt.plot(loads_filt.R4_SO_Tilt, loads_filt.R4_SO_Accel_X_orig , ".", label = 'R4_SO_Accel_X_orig')
    plt.plot(loads_filt.R4_SO_Tilt, loads_filt.R4_SO_Accel_Y_orig , "x", label = 'R4_SO_Accel_Y_orig')
    plt.plot(loads_filt.R4_DO_Tilt, loads_filt.R4_DO_Accel_X_orig , ".", label = 'R4_DO_Accel_X_orig')
    plt.plot(loads_filt.R4_DO_Tilt, loads_filt.R4_DO_Accel_Y_orig , "x", label = 'R4_DO_Accel_Y_orig')
    
    # plt.plot(loads_filt.R4_SO_Tilt, loads_filt.R4_SO_Accel_X - loads_filt.R2_DO_Accel_X , ".", label = 'R4_SO_Accel_X - R2_DO_Accel_X', color='grey')
    
    plt.ylabel('Accel (g)')
    plt.xlabel('tilt (deg)')
    plt.legend()
    plt.grid()
    
    
    # after offset corrections
    plt.figure()    
    
    def func(x, a, b, c, d):
       return a * np.sin(np.radians(x) * b +c) + d    
    
    accel_offsets = {}   
    for x_col, y_col in zip(['R1_SO_Tilt', 'R1_SO_Tilt', 'R1_DO_Tilt', 'R1_DO_Tilt', 
                             'R2_SO_Tilt', 'R2_SO_Tilt', 'R2_DO_Tilt', 'R2_DO_Tilt', 
                             'R4_SO_Tilt', 'R4_SO_Tilt', 'R4_DO_Tilt', 'R4_DO_Tilt'], 
                            ['R1_SO_Accel_X_orig', 'R1_SO_Accel_Y_orig', 'R1_DO_Accel_X_orig', 'R1_DO_Accel_Y_orig',
                             'R2_SO_Accel_X_orig', 'R2_SO_Accel_Y_orig', 'R2_DO_Accel_X_orig', 'R2_DO_Accel_Y_orig',
                             'R4_SO_Accel_X_orig', 'R4_SO_Accel_Y_orig', 'R4_DO_Accel_X_orig', 'R4_DO_Accel_Y_orig']):
        
        x_data = loads_filt[loads_filt[x_col]<160][x_col]    
        y_data = loads_filt[loads_filt[x_col]<160][y_col]        
    
        #plt.plot(x_data, y_data , ".", label = y_col)

        popt, pcov = curve_fit(func, x_data, y_data)
        accel_offsets[y_col[:-5]] = popt
        
        #plt.plot(x_data, func(x_data, *popt), ".", label=y_col+' fitted')
        plt.plot(x_data, (y_data-popt[3])/abs(popt[0]), ".", label=y_col+' offset')

    plt.ylabel('Accel (g)')
    plt.xlabel('tilt (deg)')
    plt.legend(markerscale=5)
    plt.grid()
    
    with open('Accel_offsets_{}_to_{}.pickle'.format( loads.index[0].date()
                                                           , loads.index[-1].date()), 'wb') as handle:
        pickle.dump(accel_offsets, handle, protocol=pickle.HIGHEST_PROTOCOL)




# Check diplacements
disp = 0
if disp == 1:
    
    for col in ['R1_Disp_NW_orig',
      'R1_Disp_NE_orig',
      'R1_Disp_SW_orig',
      'R1_Disp_SE_orig',
      'R1_Disp_Center_orig',
      'R4_Disp_NW_orig',
      'R4_Disp_NE_orig',
      'R4_Disp_SW_orig',
      'R4_Disp_SE_orig',
      'R4_Disp_Center_orig']:
        print (col)
        print (loads[col].mean())

    
    ### Plot initial state
    
    wind_limit = 3
    loads_filt = loads[loads.Anemometer < wind_limit]
    loads_tilt_filt = loads.where((loads.R1_Tilt<61) & (loads.R1_Tilt>53))
    
    plt.figure()
    plt.plot(loads.R1_Disp_NW_orig, '.', label = "unfiltered'", ms=4)
    plt.plot(loads_filt.R1_Disp_NW_orig, '.', label = 'wind below {}m/s'.format(wind_limit), ms=4)
    plt.plot(loads_tilt_filt.R1_Disp_NW_orig , '.', ms=4, label = 'mirror vertical')
    plt.plot(loads_filt.R1_Disp_NW_orig.where(
        (loads_filt.R1_Tilt<61) & (loads_filt.R1_Tilt>53)  )
             , '.', ms=7, label = 'wind below {}m/s and vertical mirror'.format(wind_limit))
    plt.plot(inflow.Temp/100+101.5, '.', label = 'Temperature (not scaled)', color='grey')
    plt.ylabel('R1_Disp_NW (mm)')
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.plot(loads.R1_Disp_Center_orig, '.', label = "unfiltered'", ms=4)
    plt.plot(loads_filt.R1_Disp_Center_orig, '.', label = 'wind below {}m/s'.format(wind_limit), ms=4)
    plt.plot(loads_tilt_filt.R1_Disp_Center_orig, '.', ms=4, label = 'mirror vertical')
    plt.plot(loads_filt.R1_Disp_Center_orig.where(
        (loads_filt.R1_Tilt<61) & (loads_filt.R1_Tilt>53)  )
             , '.', ms=7, label = 'wind below {}m/s and horizontal tilt'.format(wind_limit))
    plt.plot(inflow.Temp/50+74, '.', label = 'Temperature (not scaled)', color='grey')
    plt.ylabel('R1_Disp_Center (mm)')
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.plot(loads.R1_Tilt, loads.R1_Disp_NW_orig , ".", label = 'R1_Disp_NW')
    plt.plot(loads.R1_Tilt, loads.R1_Disp_Center_orig , ".", label = 'R1_Disp_Center')
    plt.plot(loads.R1_Tilt, loads.R4_Disp_NW_orig , ".", label = 'R4_Disp_NW')
    plt.plot(loads.R1_Tilt, loads.R4_Disp_Center_orig , ".", label = 'R4_Disp_Center')  
    plt.plot(loads_filt.R1_Tilt, loads_filt.R1_Disp_NW_orig , "x", label = 'R1_Disp_NW, low wind')
    plt.plot(loads_filt.R1_Tilt, loads_filt.R1_Disp_Center_orig , "x", label = 'R1_Disp_Center, low wind')
    plt.plot(loads_filt.R1_Tilt, loads_filt.R4_Disp_NW_orig , "x", label = 'R4_Disp_NW, low wind')
    plt.plot(loads_filt.R1_Tilt, loads_filt.R4_Disp_Center_orig , "x", label = 'R4_Disp_Center, low wind') 
    plt.xlabel('Tilt (deg)')
    plt.ylabel('Displacement (mm)')
    plt.legend()
    plt.grid()
    

    
    all = pd.merge(loads, inflow[['Temp', 'wspd_7m', 'wdir_7m']].interpolate(), left_index=True, right_index=True, how="outer")
    
    all_wind_filt = all.where(all.wspd_7m<wind_limit )
    all_filt = all.where((all.R1_Tilt<61) & (all.R1_Tilt>53)  ).where(all.wspd_7m<wind_limit )
    all_tilt_filt = all.where((all.R1_Tilt<110) & (all.R1_Tilt>80)  ).where(all.wdir_7m<315).where(all.wdir_7m>225)
    
    plt.figure()

    for column in all_tilt_filt[[col for col in all_tilt_filt.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('orig' in col)]]:
        plt.plot(all_tilt_filt.wspd_7m, all_tilt_filt[column], ".", label = column)   # - all_tilt_filt[column].mean()
    plt.xlabel('wind speed from west (m/s)')
    plt.ylabel('Displacement, at stow (mm)')
    plt.legend(markerscale=5)
    plt.grid()     
    
    
    ## Plot for paper
    mpl.rcParams['lines.markersize'] = 1   
    
    fig = plt.figure(figsize = (12,5))
    #plt.suptitle("Displacements before calibration offset, low wind conditions")
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.set_title("(a)")
    for column in loads_filt[[col for col in loads_filt.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('orig' in col)]]:
        plt.plot(all_wind_filt.R1_Tilt, all_wind_filt[column] , ".", label = column)
        plt.plot(57, all_filt[column].mean(), 'o', color='black', label = "", ms=5)  # calibration point
    ax1.set_ylabel('Displacement (mm)                                             ' )
    plt.grid()   
    ax1.set_ylim(98, 108)
    
    ax2 = plt.subplot(2, 2, 3)
    for column in loads_filt[[col for col in loads_filt.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('orig' in col)]]:
        plt.plot(all_wind_filt.R1_Tilt, all_wind_filt[column] , ".", label = column)
        plt.plot(57, all_filt[column].mean(), 'o', color='black', label = "", ms=5)  # calibration point
    plt.xlabel('Tilt ($^\circ$)')
    plt.grid()
    ax2.set_ylim(71, 81)
    
    
    ax3 = plt.subplot(2, 2, 2, sharey = ax1)
    ax3.set_title("(b)")
    for column in all[[col for col in all.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('orig' in col)]]:
        plt.plot(all_filt.Temp, all_filt[column], '.', label = column)   
    plt.grid()    
    
    ax4 = plt.subplot(2, 2, 4, sharey = ax2)
    for column in all[[col for col in all.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('orig' in col)]]:
        plt.plot(all_filt.Temp, all_filt[column], '.', label = column)   
    plt.xlabel('Temperature ($^\circ$ C)')
    plt.grid()


    # hide the spines between ax1 and ax2 and make slanted lines
    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()  
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    ax3.spines.bottom.set_visible(False)
    ax4.spines.top.set_visible(False)
    ax3.xaxis.tick_top()
    ax3.tick_params(labeltop=False)  # don't put tick labels at the top
    ax4.xaxis.tick_bottom()  
    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
          linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax3.plot([0, 1], [0, 0], transform=ax3.transAxes, **kwargs)
    ax4.plot([0, 1], [1, 1], transform=ax4.transAxes, **kwargs)

    
    ### Make linear temperature correction
    
    def func(x, a, b):
        return a * x + b
    
    disp_offsets = {}
    for column in all[[col for col in all.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col) & ('orig' in col)]]:
                    #['R1_Disp_NW']: #
        
        period = all.where((all[column[:2] + '_Mid_Tilt']<61) & (all[column[:2] + '_Mid_Tilt']>53)  ).where(all.Anemometer<3 )
        period = period[[column, "Temp"]].interpolate().dropna()
        popt, pcov = curve_fit(func, period.Temp, period[column])
        all[column[:-5]+"_offset"] = func(all.Temp.interpolate(), *popt).interpolate()
        #all[column[:-5]+"_corr"] = all[column] - all[column[:-5]+"_offset"]
        
        disp_offsets[column[:-5]] = popt

    # Save calibration values
    with open('Displacement_offsets_{}_to_{}.pickle'.format( loads.index[0].date()
                                                            , loads.index[-1].date()), 'wb') as handle:
        pickle.dump(disp_offsets, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
        #plt.plot(all.Temp, all[column[:-5]+"_offset"], '-', label = column[:-5]+"_offset")
        ax3.plot(15, func(15, *popt), 'o', color='black', label = "", ms=5)  # calibration point
        ax4.plot(15, func(15, *popt), 'o', color='black', label = "", ms=5)  # calibration point
      
    plt.legend(markerscale=8, loc='center left', bbox_to_anchor=(1, 0.85))
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)    
    fig.savefig('C:/Users/uegerer/Desktop/NSO/paper_plots/Displacement_cali.png', dpi=300)    
    
    
    ### Make one-point temperature correction (use this one)
           
    for column in all[[col for col in all.columns if ('Disp' in col)& ('_orig' not in col)  & ('_m' not in col) & ('_s' not in col) & ('_offset' not in col)& ('_corr' not in col)]]:
        popt = disp_offsets[column]
        # offset = func(loads.Temp.interpolate(), *popt).interpolate() # do not use: Subtract temperature-dependent offset from displacements
        offset = func(15, *popt) # instead: find a constant calibration offset at a defined temperature of 15 deg C
        all[column+"_corr"] = all[column+"_orig"] - offset
        
        

    
    ### Plot after correction
        
    all_wind_filt = all.where(all.Anemometer<2  )
    all_tilt_filt = all.where((all.R1_Tilt<110) & (all.R1_Tilt>80)  ).where(all.wdir_7m<315).where(all.wdir_7m>225)
        
        
    plt.figure(figsize = (16,9))
    mpl.rcParams['lines.markersize'] = 1
    
    plt.suptitle("Displacements after temperature correction")
                
    ax1 = plt.subplot(3, 2, 1)
    #plt.plot(loads.R1_Disp_NW, '.')
    plt.plot(all.R1_Disp_NW_orig - all.R1_Disp_NW_orig.mean(), '.', label = 'before temp correction')
    plt.plot(all.R1_Disp_NW_corr, '.', label = 'after temp correction')
    plt.plot(all.Temp/50-0.7, ".", color='grey', label = "Temperature (no scale)")
    plt.grid()
    plt.ylabel('R1_Disp_NW (mm)')
    plt.legend()
                
    ax2 = plt.subplot(3, 2, 2, sharex=ax1)
    plt.plot(all.R1_Disp_Center_orig - all.R1_Disp_Center_orig.mean(), '.', label = 'before correction')
    plt.plot(all.R1_Disp_Center_corr, '.', label = 'after correction')
    plt.plot(all.Temp/10-5, ".", color='grey', label = "Temp")
    plt.grid()
    plt.ylabel('R1_Disp_Center (mm)')
    
    ax3 = plt.subplot(3, 2, 3)
    plt.plot(all_wind_filt.R1_Tilt, all_wind_filt.R1_Disp_NW_orig -  all_wind_filt.R1_Disp_NW_orig.mean(), ".")   
    plt.plot(all_wind_filt.R1_Tilt, all_wind_filt.R1_Disp_NW_corr , ".")
    plt.xlabel('Tilt (deg)')
    plt.ylabel('Displacement (wind<2m/s) (mm)')    
    plt.grid()
    
    ax4 = plt.subplot(3, 2, 4)
    plt.plot(all_wind_filt.R1_Tilt, all_wind_filt.R1_Disp_Center_orig -  all_wind_filt.R1_Disp_Center_orig.mean(), ".", label = 'before correction')   
    plt.plot(all_wind_filt.R1_Tilt, all_wind_filt.R1_Disp_Center_corr , ".", label = 'after correction corr')
    plt.xlabel('Tilt (deg)')
    plt.ylabel('Displacement (wind<2m/s)  (mm)')  
    plt.grid()
    
    ax5 = plt.subplot(3, 2, 5)
    plt.plot(all_tilt_filt.wspd_7m, all_tilt_filt.R1_Disp_NW_orig -  all.R1_Disp_NW_orig.mean(), ".")   
    plt.plot(all_tilt_filt.wspd_7m, all_tilt_filt.R1_Disp_NW_corr , ".")
    plt.xlabel('wind speed from west (m/s)')
    plt.ylabel('Displacement (stow position) (mm)')    
    plt.grid()
    
    ax6 = plt.subplot(3, 2, 6)
    plt.plot(all_tilt_filt.wspd_7m, all_tilt_filt.R1_Disp_Center_orig -  all.R1_Disp_Center_orig.mean(), ".", label = 'before correction')   
    plt.plot(all_tilt_filt.wspd_7m, all_tilt_filt.R1_Disp_Center_corr , ".", label = 'after temp corr')
    plt.xlabel('wind speed from west (m/s)')
    plt.ylabel('Displacement (stow position) (mm)')  
    plt.grid()
    
    plt.tight_layout()
    
    # disp = all_tilt_filt.R1_Disp_Center_corr - all_tilt_filt.R1_Disp_NW_corr
    
    # plt.figure()
    # plt.plot(all_tilt_filt.wspd_7m, disp,".")

    
    
    plt.figure()
    plt.suptitle("Displacements after correction")
    for column in all[[col for col in all.columns if ('Disp' in col)  & ('orig' in col)& ('std' not in col)& ('_m' not in col)]]:
        plt.plot(all_wind_filt.R1_Tilt, all_wind_filt[column]- all_wind_filt[column].mean()  , ".", label = column)
    plt.xlabel('tilt (deg)')
    plt.ylabel('Displacement, low wind (mm)')
    plt.legend(markerscale=5)
    plt.grid()
    
    #all = all.drop(['R1_Disp_C_corr','R4_Disp_C_corr' ], axis=1)
    #all_tilt_filt = all_tilt_filt.drop(['R1_Disp_C_corr','R4_Disp_C_corr' ], axis=1)

    plt.rc('font', size=13) 
    plt.figure()
    plt.suptitle("Displacements after correction")
    for column in all_tilt_filt[[col for col in all_tilt_filt.columns if ('Disp' in col)  & ('corr' in col)& ('std' not in col)& ('_m' not in col)]]:
        plt.plot(all.where((all[column[:2] + '_Mid_Tilt']<125) & (all[column[:2] + '_Mid_Tilt']>115)).where(all.wdir_7m<315).where(all.wdir_7m>225).wspd_7m, 
                all.where((all[column[:2] + '_Mid_Tilt']<125) & (all[column[:2] + '_Mid_Tilt']>115)).where(all.wdir_7m<315).where(all.wdir_7m>225)[column] , 
                          ".", label = column)
    plt.xlabel('Wind speed from west (m/s)')
    plt.ylabel('Displacement (mm) at stow position')
    plt.grid()    
    plt.legend(markerscale=5, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    
    # plt.figure()
    # plt.title("Wind speed dependence at tilt > +-10deg")
    # plt.plot(loads_tilt_filt.Anemometer, loads_tilt_filt.R1_Disp_NW - loads_tilt_filt.R1_Disp_NW.mean(), ".", label = 'R1_Disp_NW')
    # plt.plot(loads_tilt_filt.Anemometer, loads_tilt_filt.R1_Disp_NW_max - loads_tilt_filt.R1_Disp_NW_max.mean() , ".", label = 'R1_Disp_NW max')
    # plt.plot(loads_tilt_filt.Anemometer, loads_tilt_filt.R1_Disp_Center - loads_tilt_filt.R1_Disp_Center.mean() , ".", label = 'R1_Disp_Center')
    # plt.plot(loads_tilt_filt.Anemometer, loads_tilt_filt.R1_Disp_Center_max - loads_tilt_filt.R1_Disp_Center_max.mean() , ".", label = 'R1_Disp_Center max')
    # #plt.plot(all.Anemometer.where((all.R1_Tilt<30) & (all.R1_Tilt>-30)), all.R1_Disp_NW_corr.where((all.R1_Tilt<30) & (all.R1_Tilt>-30)) , ".", label = 'R1_Disp_Center')
    # plt.xlabel('wind speed (m/s)')
    # plt.ylabel('Displacement, mean subtracted (mm)')
    # plt.legend()
    # plt.grid()  
    
    
    ### Test with baseline correction
    
    # loads["R1_Disp_NW_baseline"] = loads.R1_Disp_NW.where((loads.R1_Tilt<3) & (loads.R1_Tilt>-3)).where(loads.Anemometer<25)
    
    
    # plt.figure()
    # plt.plot(loads.R1_Disp_NW,".")
    # plt.plot(loads.R1_Disp_NW_baseline.interpolate(),".")
    # plt.plot(loads.R1_Disp_NW_baseline,".")
    
    
    
    
            
tilt_wind = 0
if tilt_wind == 1:
    
    # calculate deviations from nominal trough angle
    for column in all[[col for col in all.columns if ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col) & ('dev' not in col)]]: 
        # calculate deviation
        all[column + '_dev'] = all[column] - all.trough_angle
        # filter deviations
        all[column + '_dev'] = all[column + '_dev'].where((all[column + '_dev']<2) & (all[column + '_dev']>-2) )
    


    # filter for low wind and stow
    all_low_wind = all.where(all.wspd_3m<1 )    
    all_low_wind_stow = all_low_wind.where((all_low_wind.trough_angle<121) & (all_low_wind.trough_angle>119) )



    #  plot time series and histogram
    mpl.rcParams['lines.markersize'] = 3
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
    for column in all[[col for col in all.columns if ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col) & ('dev' not in col)]]:       
        
        ax1.plot(all_low_wind[column + '_dev'],".", label = all[column + '_dev'].name[:])       
        ax2.hist(all_low_wind[column + '_dev'].dropna(), bins=500, label = all_low_wind[column + '_dev'].name[:])
        
        all_low_wind[column + '_dev'] = all_low_wind[column + '_dev'] - all[column + '_dev'].median()
        all[column + '_dev'] = all[column + '_dev'] - all[column + '_dev'].median()
        ax4.hist(all_low_wind[column + '_dev'].dropna(), bins=500, label = all_low_wind[column + '_dev'].name[:])
        ax3.plot(all_low_wind[column + '_dev'],".", label = all[column + '_dev'].name[:])         
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel('deviation from nominal angle (deg)')
    ax2.set_xlabel('deviation from nominal angle (deg)')
    ax3.grid(True)
    ax4.grid(True)
    ax3.set_ylabel('deviation after median subtract (deg)')
    ax4.set_xlabel('deviation after median subtract (deg)')
    plt.tight_layout()
    
    
    all_low_wind_stow = all_low_wind.where((all_low_wind.trough_angle<121) & (all_low_wind.trough_angle>119) )

    
    # plot diurnal cycle of deviations depending on trough angle
    fig = plt.figure()
    i=1
    for column in all_low_wind[[col for col in all_low_wind.columns if ('Tilt_dev' in col)  ]]:
        plt.subplot(3, 3, i)
        plt.grid()  
        plt.xlabel("tilt")
        plt.ylabel(column)   
        smap = plt.scatter(all_low_wind[column[:-4]], all_low_wind[column], c=all_low_wind.index)
        N_TICKS = 8
        indexes = [all_low_wind.index[i] for i in np.linspace(0,all_low_wind.shape[0]-1,N_TICKS).astype(int)] 
        plt.ylim(-1,2)
        i = i+1
    cb = fig.colorbar(smap, orientation='vertical',
                      ticks= all_low_wind.loc[indexes].index.astype(int))  
    cb.ax.set_yticklabels([index.strftime('%d %b %Y') for index in indexes])    
    plt.tight_layout()
        
        
    # plot temperature influence

    plt.figure()
    for column in all_low_wind_stow[[col for col in all_low_wind_stow.columns if ('Tilt_dev' in col)  ]]:
        plt.plot(all_low_wind_stow.Temp, all_low_wind_stow[column],".", label = all_low_wind_stow[column ].name[:])
      #  plt.plot(all.where((all.trough_angle<121) & (all.R1_SO_Tilt>119)& (all.wspd_7m<1) ).Temp, all.where((all.trough_angle<121) & (all.R1_SO_Tilt>119)& (all.wspd_7m<1) )[column],".", label = all[column ].name[:])
    plt.legend()        
    plt.grid()
    plt.xlabel('Temperature (deg C)')
    plt.ylabel('tilt deviation')
        
   
    make_temp_correction = 0
    if make_temp_correction == 0:
    
        ### Make linear temperature correction
        

        def func(x, a, b):
            return a * x + b
        
        tilt_offsets = {}
        for column in all_low_wind[[col for col in all_low_wind.columns if ('Tilt_dev' in col)  & ('_m' not in col) & ('_s' not in col)]]:
                        #['R1_Tilt_SO']: #
            
            period = all_low_wind[[column, "Temp"]].interpolate().dropna()
            popt, pcov = curve_fit(func, period.Temp, period[column])
            all_low_wind[column+"_offset"] = func(all_low_wind.Temp.interpolate(), *popt).interpolate()
            all_low_wind[column+"_corr"] = all_low_wind[column] - all_low_wind[column+"_offset"]
            
            tilt_offsets[column] = popt
            
            plt.plot(all_low_wind.Temp, all_low_wind[column+"_offset"], '-', label = column+"_offset")  
    
        # with open('Tilt_offsets_{}_to_{}.pickle'.format( loads.index[0].date()
        #                                                        , loads.index[-1].date()), 'wb') as handle:
        #     pickle.dump(tilt_offsets, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
    
        # plot diurnal cycle of deviations after correction
        # plot diurnal cycle of deviations depending on trough angle
        fig = plt.figure()
        i=1
        for column in all_low_wind[[col for col in all_low_wind.columns if ('Tilt_dev_corr' in col)  ]]:
            plt.subplot(3, 3, i)
            plt.grid()  
            plt.xlabel("tilt")
            plt.ylabel(column)   
            smap = plt.scatter(all_low_wind[column[:-9]], all_low_wind[column], c=all_low_wind.index)
            N_TICKS = 8
            indexes = [all_low_wind.index[i] for i in np.linspace(0,all_low_wind.shape[0]-1,N_TICKS).astype(int)] 
            plt.ylim(-1,2)
            i = i+1
        cb = fig.colorbar(smap, orientation='vertical',
                          ticks= all_low_wind.loc[indexes].index.astype(int))  
        cb.ax.set_yticklabels([index.strftime('%d %b %Y') for index in indexes])    
        plt.tight_layout()
            
  

    
    # wind influence
    all_west = all.where((all.wdir_7m<300) & (all.wdir_7m>240) & (all.trough_angle<121) & (all.trough_angle>119))   
    plt.figure()
    for column in all_stow[[col for col in all_stow.columns if ('Tilt_dev' in col)  ]]:
        plt.plot(all_stow.wspd_3m, all_stow[column],".", label = all_stow[column ].name[:])
    plt.legend()        
    plt.grid()
    plt.xlabel('Western wind speed 3m (m/s)')
    plt.ylabel('tilt deviation (deg)')    
    
    

overview = 0
if overview == 1:
    
    mpl.rcParams['lines.markersize'] = 1
    
     
    ## Look at time series
    fig = plt.figure(figsize=(17,10))   
    plt.suptitle("Loads at NSO, {} to {}".format(loads.index[0] , loads.index[-1] ))
    
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
    
    
    for column in loads[[col for col in loads.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col)& ('C' not in col)]]:
        ax7.plot(loads[column],".", label = loads[column].name) 
    ax7.legend(markerscale=5, loc='upper left')
    for column in loads[[col for col in loads.columns if ('Accel' in col)  & ('_m' not in col) & ('_s' not in col) & ('_orig' not in col)]]:
        ax2.plot(loads[column],".", label = loads[column].name) 
    ax2.legend(markerscale=5, loc='upper left')
    for column in loads[[col for col in loads.columns if ('Torque' in col)  & ('_m' not in col) & ('_s' not in col)& ('C' not in col)]]:
        ax8.plot(loads[column],".", label = loads[column].name) 
    ax8.legend(markerscale=5, loc='upper left')
    for column in all[[col for col in all.columns if ('Disp' in col)  & ('_m' not in col) & ('_s' not in col)& ('_corr' in col) ]]:
        ax4.plot(all[column],".", label = all[column].name) 
    ax4.legend(markerscale=5, loc='upper left')
    for column in loads[[col for col in loads.columns if ('Tilt' in col)  & ('_m' not in col) & ('_s' not in col) 
                         & ('_orig' not in col) & ('R1_Tilt' not in col) & ('R2_Tilt' not in col) & ('R4_Tilt' not in col)]]:
        ax3.plot(loads[column],".", label = loads[column].name) 
    ax3.legend(markerscale=5, loc='upper left')
    ax5.plot(loads.Anemometer,".", color='black', label='15m loads', ms=1)
    ax5.plot(inflow.WS_15m,".", color='grey', label='15m winds', ms=1)
    ax5.plot(all.Anemometer.where(all.wdir_7m<290).where(all.wdir_7m>250),".", ms=2, color='red', label='15m loads west')
    ax5.legend(markerscale=5, loc='upper left')
    
    for ax in [ax2, ax3, ax4, ax5, ax7, ax8]:
        ax.grid(True)
        #ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    
    plt.tight_layout() 
    fig.autofmt_xdate() 
    
    save=0
    if save ==1:
        fig.savefig('Quicklookds_loads_processing/Overview_{}_to_{}.png'.format(inflow.index[0].date(), inflow.index[-1].date() ))

            
            
            
            
coeff = 0
if coeff == 1:            
   
        
        plt.figure()
        
        L_panel = 49          # m, length of panel with 6 mirror panels
        L_segment = 8  # m, length of mirror panel 
        W = 5          # m, aperture width
        Hc = 2.79083   # m, height of pivot axis
        # all['rho'] = rho(all.p, all.RH, all.Temp).interpolate()
        # wspd = all.wspd_3m.where(all.wspd_3m> 3)             
            
        # # # Bending moment coefficients
        # # for column in all[[col for col in all.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col)]]:
        # #   all[column[:6]+"C_"+ column[6:]] = all[column] *1000/ (all.rho/2 * wspd**2 * L_segment * W * Hc)   
        # # # Torque moment coefficients
        # # for column in all[[col for col in all.columns if ('Torque' in col)  & ('_m' not in col) & ('_s' not in col)]]:
        # #   all[column[:6]+"C_"+ column[6:]] = all[column] *1000/ (all.rho/2 * wspd**2 * L_panel* W**2)   
        # # # Drag force coefficients
        # # for column in all[[col for col in all.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col)]]:
        # #   fx = all[column] *1000 / Hc
        # #   all[column[:6]+"Cfx"] = fx / (all.rho/2 * wspd**2 * L_segment * W)  
        
        
        
        # for column in all[[col for col in all.columns if ('Torque' in col)  & ('_m' not in col) & ('_s' not in col)& ('C' not in col)& ('R1' in col)]]:
        #     test= all[column] *1000/ (all.rho/2 * wspd**2 * L_panel* W**2)   
        #     plt.plot(test, label = column + "test")
        
        
        for column in all[[col for col in all.columns if ('Torque' in col)& ('C' in col)& ('R1' in col)]]:
             plt.plot(all[column], label = column)
        # plt.plot(all.wspd_3m)
        # plt.plot(rho(all.p, all.RH, all.Temp).interpolate())
        
        all2 = add_load_coefficients(all, wind_speed_limit=3)


   
        for column in all[[col for col in all.columns if ('Torque' in col)& ('C' in col)& ('R1' in col)]]:
            plt.plot(all2[column], label = column)    

   
        plt.legend()    
        
        
        # Coefficient calculation is not consistent with processing routine!
        
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
            
            