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
import time
import itertools
import glob
import netCDF4 as nc
import xarray as xr
import pickle
import datetime
import matplotlib as mpl
mpl.rcParams['lines.markersize'] = 2
mpl.rcParams['lines.linewidth'] = 1
from windrose import WindroseAxes
import matplotlib.cm as cm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
pd.options.mode.chained_assignment = None  # default='warn'
import gc



from Functions_masts import *
from Functions_general import *



#%% Define

path =  './data/' # 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v1/'   
path_save = './data/' # 'Y:\Wind-data/Restricted/Projects/NSO/Processed_data/Met_masts_v1/'   

years   =  np.arange(2021,2024)   #
months  =  np.arange(1,13)   # 
days    =  np.arange(1,32)   # 

# years   =  [2021]   #
# months  =  [12]  # 
# days    =  [7]  # 


fs = 20 # [Hz], sampling frequency

window_fluxes = 60*20   # window for flux calc, in s

save  = 1
plot  = 1






#%% Read data            

print ("Read data")

for year in years:
    year = str(year)    # Convert the year to a string

    for month in months:
        
        if year == '2021' and month<10:    # no data before October 2021, skip iterations
            continue
        
        month = f'{month:02d}'    # Format the month as a two-digit string
    
        for day in days:
            day = f'{day:02d}'   # Format the day as a two-digit string
            

            print (year, month, day) 
            
            
            try:
                if pd.to_datetime(year + month + day) < pd.to_datetime("2021-01-01"):
                    continue   # Skip iterations before this date (for continuing data processing)
            except:
                pass

            # Find available data files       
            inflow_files = sorted(glob.glob(path +'Inflow_Mast_20Hz_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #   
            mast_files = sorted(glob.glob(path +'Wake_masts_20Hz_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #   
            
 
                 
            #%% Process inflow      
            
            
            # Read inflow data
            inflow = pd.DataFrame()    
            for datafile in inflow_files:
                inflow = pd.concat( [inflow, pd.read_pickle(datafile)]) 
            inflow = inflow.drop_duplicates().sort_index()    
                
            if len(inflow)==0:
                continue
            
            # Processing loop
            if len(inflow) !=0:  

                print ("Calculate inflow mast")
                
                
                # Correct Gill w-bug (https://www.licor.com/env/support/EddyPro/topics/w-boost-correction.html) for all Gill Sonics
                for col in inflow.columns:
                    if (('W_ax_7m' in col) or  
                        ('W_ax_5m' in col)  or  
                        ( ('W_ax_3m' in col) and (inflow.index[0] < pd.to_datetime("2022-11-16"))  )  
                        ):
                            inflow[col] = inflow[col].apply(apply_factors_for_w_bug)    

                
                
                inflow[['p', 'Temp', 'RH']] = inflow[['p', 'Temp', 'RH']].astype(float).interpolate()
                inflow['time'] = (inflow.index - inflow.index[0]) / np.timedelta64(1,'s')                 
                
                if 'Ts_3m' not in inflow:
                    inflow['Ts_3m'] = np.nan
                    inflow['U_ax_3m'] = np.nan
                    inflow['V_ax_3m'] = np.nan
                    inflow['W_ax_3m'] = np.nan
            
            
                # Initialize DataFrame for length scales (inflow)
                fluxes_inflow = pd.DataFrame()
                
                # create columns for length scales
                for height_col in inflow.columns:
                    if 'W_ax' in height_col:
                        ls_w = height_col.replace("W_ax", "ls_w")
                        fluxes_inflow[ls_w] = np.nan
                        ls_uE = height_col.replace("W_ax", "ls_uE")
                        fluxes_inflow[ls_uE] = np.nan
                        ls_vN = height_col.replace("W_ax", "ls_vN")
                        fluxes_inflow[ls_vN] = np.nan
                    
                    if 'wspd' in height_col:
                        ls_U = height_col.replace("wspd", "ls_U")
                        fluxes_inflow[ls_U] = np.nan

                    
                fluxes_inflow['H_S'] = np.nan 
                fluxes_inflow['Tau'] = np.nan 
                fluxes_inflow['R_f'] = np.nan  
                fluxes_inflow['zL'] = np.nan
                fluxes_inflow['L'] = np.nan
                
                
                # loop over 20 min segments
                for time, period in inflow.groupby( (window_fluxes/10.) * (inflow.time/(window_fluxes/10.)).round(-1)): # in window_fluxes

                    maxlegs = int(window_fluxes*fs/2) # maximum lag for autocorrelation function     
                    rho_Mast = rho(period.p, period.RH, period.Temp).mean()

                    if len(period) > window_fluxes*fs/2: 
                        
                        # create new line with time stamp
                        fluxes_inflow = pd.concat([fluxes_inflow, pd.DataFrame({    
                             },
                            index = [ period.index[int(len(period)/2)]] )  ]) 
                        
                        # 7m height for inflow characteristics
                        fluxes_inflow['H_S'].iloc[-1] = HS(wE = period.W_ax_7m, theta = period.Ts_7m, Rho  =  rho_Mast) 
                        fluxes_inflow['Tau'].iloc[-1] = tau(period.W_ax_7m, period.wspd_7m, rho_Mast)  
                        fluxes_inflow['R_f'].iloc[-1] = Ri_flux( period.Temp, period.Ts_7m, period.Ts_3m, 
                                                        period.U_ax_7m, period.U_ax_3m, period.V_ax_7m, period.V_ax_3m, 
                                                        period.W_ax_7m, 3.5 ) 
                        fluxes_inflow['zL'].iloc[-1] , fluxes_inflow['L'].iloc[-1]   = Obukhov_stability(Tref = period.Temp, Tv= period.Ts_7m, 
                                                                          U = period.U_ax_7m, V = period.V_ax_7m, W = period.W_ax_7m, z=7)
                
                        # loop over columns in mast with W component
                        for height_col in [col for col in inflow.columns if 'W_ax' in col]:    
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7:
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna())
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)                                
                                ws = height_col.replace("W_ax", "wspd")                          
                                ls_w = height_col.replace("W_ax", "ls_w")
                                fluxes_inflow[ls_w].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * period[ws].mean()
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_inflow[ls_w].iloc[-1] = np.nan
                                
   
                        # plt.figure()
                        # loop over columns in mast with U component
                        for height_col in [col for col in inflow.columns if 'wspd' in col]:             
                            U_mean = period[height_col].mean()
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7: 
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna()) 
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)                               
                                ls_U = height_col.replace("wspd", "ls_U")
                                fluxes_inflow[ls_U].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * U_mean
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_inflow[ls_U].iloc[-1] = np.nan
                                
                                # # Y = plt.acorr(U_corr.dropna(), maxlags = int(len(U_corr.dropna())/2)) 
                                # plt.plot(Y[0], Y[1])
                                # plt.plot(Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1], 1/np.e,".", ms=10)
                                    
                                                         
                        # loop over columns in mast with U_ax component - this is northward component, and becomes nV later
                        for height_col in [col for col in inflow.columns if 'U_ax' in col]:             
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7: 
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna()) 
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)      
                                ws = height_col.replace("U_ax", "wspd")   
                                ls_vN = height_col.replace("U_ax", "ls_vN")
                                fluxes_inflow[ls_vN].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * period[ws].mean()
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_inflow[ls_vN].iloc[-1] = np.nan


                                    
                        # loop over columns in mast with V_ax component - this is westward component, and becomes - uE later
                        for height_col in [col for col in inflow.columns if 'V_ax' in col]:             
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7: 
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna()) 
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)     
                                ws = height_col.replace("V_ax", "wspd")   
                                ls_uE = height_col.replace("V_ax", "ls_uE")
                                fluxes_inflow[ls_uE].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * period[ws].mean()
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_inflow[ls_uE].iloc[-1] = np.nan

            
                # remove repeated values
                for column in fluxes_inflow.drop(fluxes_inflow.columns,axis=1).columns: 
                    #fluxes_inflow.loc[fluxes_inflow.duplicated([column]), column]=np.nan   # this drops all duplicates even if they occurr not consequtively
                    index=fluxes_inflow[column].diff()== 0
                    fluxes_inflow[column].loc[index]=np.nan
                    
                if plot==1:   
                
                    # Timeseries of timescales and vertical wind
                    fig = plt.figure(figsize=(16,9))
                    
                    ax2 = plt.subplot(5, 1, 2)
                    plt.ylabel('Length scales $w$ (m)')       
                    for height_col in [col for col in fluxes_inflow.columns if ('ls_w' in col) & ('7m' in col)]:   
                        ax2.plot(fluxes_inflow[height_col],'-', label = height_col, lw=2)
                    plt.grid()
                    ax2.set_ylim(-0.1,15)  
                    
                    ax1 = plt.subplot(5, 1, 1, sharex=ax2)
                    ax1.set_ylabel('Length scales $U$ (m)')
                    for height_col in [col for col in fluxes_inflow.columns if ('ls_' in col) & ('ls_w' not in col) & ('7m' in col) & ('flag' not in col)]:   
                        ax1.plot(fluxes_inflow[height_col],'-', label = height_col, lw=2)
                    plt.grid()
                    #ax1.axhline(maxlegs/fs*inflow.wspd_3m.mean(), color='k', label='max leg limit')
                    ax1.legend(loc=1,  markerscale=4)
                    ax = ax1.twinx()
                    ax.set_ylim(0,10)
                    ax.set_yticklabels([]) 
                    ax.set_yticks([]) 
                    # for time in inflow.W_ax_7m.resample("H").mean().index[1:-1]:
                    #     angle = angles.trough_angle.loc[str(time)].mean()   
                    #     plt.plot(time, 1.5, marker=(2,0, -angle+90), markersize=30, linestyle='None', color='black', label='')
                    #     plt.plot(time, 0.5, markersize=20, marker="|", linestyle='None', color='black', label='')
                    ax3 = plt.subplot(5, 1, 3, sharex=ax2)
                    ax3.set_ylabel('Vertical wind (m/s)')
                    for height_col in [col for col in inflow.columns if 'W_ax' in col]:   
                        ax3.plot(inflow[height_col].resample("5min").mean(),'-', label = height_col, lw=2)
                    plt.grid()   
                    
                    ax4 = plt.subplot(5, 1, 4, sharex=ax2)
                    ax4.set_ylabel('$H_S$ (W/m$^2$)', color='C0')
                    plt.plot(fluxes_inflow.H_S)
                    plt.grid(True)  
                    ax4.set_xticklabels([])                    
                    ax5 = ax4.twinx() 
                    ax5.set_ylabel(r'$\tau$ (kg m$^{-1}$ s$^{-2}$)', color='C1')
                    ax5.plot(fluxes_inflow.Tau, color='C1')
                    
                    ax6 = plt.subplot(5, 1, 5)
                    ax6.set_ylabel('$R_f$', color='C0')
                    plt.plot(fluxes_inflow.R_f,".", ms=5)
                    plt.grid(True)  
                    ax6.set_xlim(ax2.get_xlim())                 
                    ax7 = ax6.twinx() 
                    ax7.set_ylabel(r'$L$', color='C1')
                    ax7.plot(fluxes_inflow.L[fluxes_inflow.L<100][fluxes_inflow.L>-100], ".", color='C1', label = "", ms=5)
                    ax7.plot(inflow.Temp.resample("5min").mean(), color='grey', label = "Temp")
                    ax7.set_ylim(ax6.get_ylim())  
                    ax7.legend()                   
      
                    ax2.legend()                       
                    ax1.legend()                       
                    ax3.legend()
     
                    plt.tight_layout()
                    plt.subplots_adjust( hspace=0.1)   
                        
      
                if save == 1:
       
                    fluxes_inflow.to_pickle(path_save+'Inflow_Mast_fluxes_{}_{}h_to_{}_{}h.pkl'.format(inflow.index[0].date(),inflow.index[0].hour, inflow.index[-1].date(),inflow.index[-1].hour))    

                    # delete all data before processing next day
                    del inflow, fluxes_inflow                       
                    
                
                
            #%% Process masts                   
            

                
            
            
            # Read mast files         
            masts = pd.DataFrame()
            for datafile in mast_files:
                masts = pd.concat( [masts, pd.read_pickle(datafile)] )   
            masts = masts.drop_duplicates().sort_index() 
            
            if len(masts) !=0:  
                
                print ("Calculate wake masts")
                
                
                # Correct Gill w-bug (https://www.licor.com/env/support/EddyPro/topics/w-boost-correction.html) for all Gill Sonics
                for col in masts.columns:
                    if (('W_ax_7m' in col)  or 
                        ('W_ax_5m' in col)  or 
                        ('W_ax_4m' in col)  or 
                        ('m3_W_ax_3m' in col) or 
                        ( ('m2_W_ax_3m' in col) and (masts.index[0] < pd.to_datetime("2022-11-16"))  ) or
                        ( ('m1_W_ax_3m' in col) and (masts.index[0] < pd.to_datetime("2022-11-16"))  )  
                        ):           
                            masts[col] = masts[col].apply(apply_factors_for_w_bug)    
          
           
                masts['time'] = (masts.index - masts.index[0]) / np.timedelta64(1,'s')
            
            
                # Initialize DataFrame for length scales wake masts
                fluxes_masts = pd.DataFrame()
                
                # create columns for length scales and flasgs
                for height_col in masts.columns:
                    if 'W_ax' in height_col:
                        ls_w = height_col.replace("W_ax", "ls_w")
                        fluxes_masts[ls_w] = np.nan   
                        ls_uE = height_col.replace("W_ax", "ls_uE")
                        fluxes_masts[ls_uE] = np.nan
                        ls_vN = height_col.replace("W_ax", "ls_vN")
                        fluxes_masts[ls_vN] = np.nan             
                        
                    if 'wspd' in height_col:
                        ls_U = height_col.replace("wspd", "ls_U")
                        fluxes_masts[ls_U] = np.nan  
                        



                
                # loop over 20 min segments
                for time, period in masts.groupby( (window_fluxes/10.) * (masts.time/(window_fluxes/10.)).round(-1)): # in window_fluxes
                    #print (period.index[int(len(period)/2)])
                
                    if len(period) > window_fluxes*fs/2: 
                        
                        # create new line with time stamp
                        fluxes_masts = pd.concat([fluxes_masts, pd.DataFrame({    
                            },
                            index = [ period.index[int(len(period)/2)]] )  ]) 
                        
                
                        # loop over columns in mast with W component
                        for height_col in [col for col in masts.columns if 'W_ax' in col]:    
                            U_corr = period[height_col].copy()                          
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7:
                                ws = height_col.replace("W_ax", "wspd")                            
                                ls_w = height_col.replace("W_ax", "ls_w")  
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna())
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)
                                fluxes_masts[ls_w].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * period[ws].mean()                           
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_masts[ls_w].iloc[-1] = np.nan
                
                        # loop over columns in mast with U component
                        for height_col in [col for col in masts.columns if 'wspd' in col]:   
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7:
                                U_mean = period[height_col].mean()
                                ls_U = height_col.replace("wspd", "ls_U")
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna())    
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)
                                fluxes_masts[ls_U].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * U_mean
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_masts[ls_U].iloc[-1] = np.nan
                                
                                
                        # loop over columns in mast with U_ax component - this is northward component, and becomes nV later
                        for height_col in [col for col in masts.columns if 'U_ax' in col]:             
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7: 
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna()) 
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)     
                                ws = height_col.replace("U_ax", "wspd")   
                                ls_vN = height_col.replace("U_ax", "ls_vN")
                                fluxes_masts[ls_vN].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * period[ws].mean()
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_masts[ls_vN].iloc[-1] = np.nan


                                    
                        # loop over columns in mast with V_ax component - this is westward component, and becomes - uE later
                        for height_col in [col for col in masts.columns if 'V_ax' in col]:             
                            U_corr = period[height_col].copy()
                            if len(U_corr.dropna()) > window_fluxes*fs*0.7: 
                                U_corr[U_corr.isna()==False] = scipy.signal.detrend(U_corr.dropna()) 
                                autocorr = np.correlate(U_corr.dropna(), U_corr.dropna(), mode='full')   # Compute autocorrelation
                                autocorr /= np.sqrt(np.dot(U_corr.dropna(), U_corr.dropna()) * np.dot(U_corr.dropna(), U_corr.dropna()))  # Normalize the result
                                lags = np.arange(-len(U_corr.dropna()) + 1, len(U_corr.dropna()))
                                Y = (lags, autocorr)     
                                ws = height_col.replace("V_ax", "wspd")   
                                ls_uE = height_col.replace("V_ax", "ls_uE")
                                fluxes_masts[ls_uE].iloc[-1] = Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1]*1/fs * period[ws].mean()
                                if Y[0][np.where(Y[1]==find_nearest(Y[1], value=1/np.e))][-1] > len(Y[0])/4 :   # if the value is close to the end of the autocorrelation function - skip the value
                                    fluxes_masts[ls_uE].iloc[-1] = np.nan

                        
                                
                            
                            
                # remove duplicate values
                for column in fluxes_masts.columns: 
                    index=fluxes_masts[column].diff()== 0
                    fluxes_masts[column].loc[index]=np.nan
                    
                if plot==1:   
                        
                    for height_col in [col for col in fluxes_masts.columns if ('ls_w' in col) & ('7m' in col)]:     
                        ax2.plot(fluxes_masts[height_col],'-', label = height_col)
                    for height_col in [col for col in fluxes_masts.columns if ('ls_U' in col) &  ('7m' in col)]:     
                        ax1.plot(fluxes_masts[height_col],'-', label = height_col)     
                    for height_col in [col for col in masts.columns if ('W_ax' in col)& ('7m' in col)]:     
                        ax3.plot(masts[height_col].resample("5min").mean(),'-', label = height_col, lw=1)  
                            
                    ax2.legend()                       
                    ax1.legend()                       
                    ax3.legend()
                     
                    
                if save == 1:
   
                    fluxes_masts.to_pickle(path_save+'Wake_Masts_fluxes_{}_{}h_to_{}_{}h.pkl'.format(masts.index[0].date(),masts.index[0].hour, masts.index[-1].date(),masts.index[-1].hour))  
                    
                    # delete all data before processing next day
                    del masts, fluxes_masts     
                
                                

            if save == 1:  
                
                fig.savefig(path_save+'Fluxes_{}-{}-{}.png'.format(year,month, day ))

                plt.close('all')
               
                
               
                
               



           
            
 








