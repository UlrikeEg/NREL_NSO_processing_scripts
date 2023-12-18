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
import pvlib
import os


#%% Read data



### Met tower

# @jit(nopython=True)
def read_sonic(datafile):
    
    ''' 
    - reads data from inflow mast Sonic data file
    - makes data corrections (outliers, faulty values)
    '''
    
    # print (datafile)
    
    ## read data
    data = pd.read_csv(datafile,
                    index_col = None,
                    header = 0,    
                    skiprows = [0, 2, 3], 
                    engine = 'c',
                    on_bad_lines='warn', 
                    na_values = {'NAN', '.',"37.18:q"},
                    #dtype = float,
                    dtype = {"TIMESTAMP": str}
                    # parse_dates=True
                        )
    
    # Convert time index
    data['date'] = pd.to_datetime(data.TIMESTAMP.str.slice(0,10))
    data.TIMESTAMP = data.TIMESTAMP.where(data.TIMESTAMP.str.slice(11,13)!='24',         # if hour is 24 make it 00 and one day later
                                        # data.TIMESTAMP.str.slice(0,8) 
                                        # + (data.TIMESTAMP.str.slice(8,10).astype(int)+1).astype(str) 
                                        (data.date + pd.to_timedelta(1, "D")).astype(str)
                                        + " 00" + data.TIMESTAMP.str.slice(13,22)    
                                          )
    data.drop(['date'], axis=1, inplace=True)
    data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
    
    data = data.set_index('TIMESTAMP')    
    data = data.astype(float)
    
    if 'RECORD' in data:
        data.drop(['RECORD'], axis=1, inplace=True)
        
    if 'Diag_CSAT3_3m' in data:
        data.drop(['Diag_CSAT3_3m'], axis=1, inplace=True)
    
    
    # plot for diagnostics of spikes
    # flag = 0
    # test = plt.figure()
    # for height_col in [col for col in data.columns if '_ax_' in col]:    
    #     if (data[height_col].max()>30) | (data[height_col].min()<-30):
    #         flag=1
    #         plt.plot(data[height_col], label=height_col+' '+datafile[-33:-27])
    # if flag==1:
    #     plt.legend(loc=1)
    #     plt.title('{}_to_{}.png'.format(data.index[0].date(), data.index[-1].date()))
    #     test.savefig('C:/Users/uegerer/Desktop/NSO/Faulty_mast_values/{}_{}h_to_{}_{}h_'.format(data.index[0].date(),data.index[0].hour, data.index[-1].date(), data.index[-1].hour )+datafile[-33:-27]+'.png')
    # plt.close(test)
        
    # filter spikes above 30 m/s (unrealistic, but present in data as spikes)
    for height_col in [col for col in data.columns if '_ax_' in col]:    
        data[height_col] = data[height_col].where(abs(data[height_col])<30)
        
    return data


def calc_wind(data):
    
    ''' 
    - adds wind direction and horizontal wind
    '''

    ## loop over every mast height for calculating wind direction and speed
    
    for height_col in [col for col in data.columns if 'U_ax_' in col]:    # loop over every mast height
        
        U = height_col
        V = height_col.replace("U_ax", "V_ax")
        col_dir = height_col.replace("U_ax", "wdir")
        col_spd = height_col.replace("U_ax", "wspd")

        # calculate wind direction and horizontal wind       
        data[col_dir] = np.degrees(np.arctan2( - data[V], data[U])) # this is the direction of the wind speed vector
        data[col_dir] = data[col_dir] + 180 # wind direction is 180 deg offset from wind vector
        data[col_spd] = np.sqrt(data[U]**2 + data[V]**2)    
        
    return data



def read_slow_data(datafile):
    
    ''' 
    - reads data from inflow mast PTU and cup anemometer (1Hz data)
    '''
    
    # print (datafile)
    
    # read data
    data = pd.read_csv(datafile,
                    index_col = None,
                    header = 0,    
                    skiprows = [0, 2, 3], 
                    engine = 'c',
                    on_bad_lines='warn', 
                    na_values = {'NAN'},
                    # dtype = float,
                    # parse_dates=True
                        )
    
    # Convert time index
    data['date'] = pd.to_datetime(data.TIMESTAMP.str.slice(0,10))
    data.TIMESTAMP = data.TIMESTAMP.where(data.TIMESTAMP.str.slice(11,13)!='24',         # if hour is 24 make it 00 and one day later
                                        # data.TIMESTAMP.str.slice(0,8) 
                                        # + (data.TIMESTAMP.str.slice(8,10).astype(int)+1).astype(str) 
                                        (data.date + pd.to_timedelta(1, "D")).astype(str)
                                        + " 00" + data.TIMESTAMP.str.slice(13,22)    
                                          )
    data.drop(['date'], axis=1, inplace=True)  
    data.TIMESTAMP = pd.to_datetime(data.TIMESTAMP)
    
    data = data.set_index('TIMESTAMP')    
    data = data.astype(float)

    if 'RECORD' in data:        
        data.drop(['RECORD'], axis=1, inplace=True)
    if 'Diag_CSAT3_3m' in data:        
        data.drop(['Diag_CSAT3_3m'], axis=1, inplace=True)


    return data


def resample_sonic(data, resample_freq):
    
    ''' 
    - resample datafile
    - calculate wind direction and horizontal wind again after resampling
    '''
    
    # resample
    data = data.astype(float)
    data = data.resample(resample_freq).mean()
    
    # calculate wind direction and horizontal wind again after resampling (mean of wind direction does not work!)
    data = calc_wind(data)


    return data


#%% Other functions

def get_trough_angles_txt(): # old
    angles = pd.read_csv('sun_angles.txt', parse_dates={'UTC': [0, 1]}).set_index('UTC')  # ,nrows=200
    angles.iloc[:,-1] = angles.iloc[:,-1].where(angles.iloc[:,-1]>0) # only elevations > 0, meaning over horizon
    angles['trough_angle'] = np.degrees( np.arctan2(sin(np.radians(angles.iloc[:,-1])), sin(np.radians(angles.iloc[:,-2])) )) 
    angles.trough_angle = angles.trough_angle.where(angles.trough_angle.isnull()==False, -30)
    return angles.trough_angle

def get_trough_angles_sol(): # old
    sol = pvlib.solarposition.get_solarposition(time=all.index, latitude=35.799554350079816, longitude=-114.98145396226035, altitude=545, temperature=all.Temp) 
    sol.elevation = sol.elevation.where(sol.elevation>0) # only elevations > 0, meaning over horizon
    sol['trough_angle'] = np.degrees( np.arctan2(sin(np.radians(sol.elevation)), sin(np.radians(sol.azimuth)) )) 
    sol.trough_angle = sol.trough_angle.where(sol.trough_angle.isnull()==False, -30)
    sol['trough_angle_fine'] = -sol.trough_angle + 90   # to make it the same definition like trough angles
    return sol



def get_trough_angles(times):
    lat, lon = 35.799554350079816, -114.98145396226035 #coordinates of NSO
    # times = pd.date_range(tstart, tend, freq='10T')

    solpos = pvlib.solarposition.get_solarposition(times, lat, lon, altitude=543) #, method='nrel_numba')
    # remove nighttime
    # solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]

    angles = pd.DataFrame()
    angles = sun_elev_to_trough_angles(solpos.apparent_elevation,solpos.azimuth)
    angles = angles.to_frame(name='trough_angle')
    anglesdf = solpos.merge(angles, left_index = True, right_index = True, how='inner')
    anglesdf['projected_sun_angle'] = anglesdf.trough_angle
    anglesdf.trough_angle[anglesdf['apparent_elevation'] < 0] = 120
    return anglesdf

def sun_elev_to_trough_angles(elev_angles, azimuth_angles):
    # trough_angles = np.degrees( np.arctan2(np.sin(np.radians(elev_angles)), np.sin(np.radians(azimuth_angles)) ))
    # print('trough angle = {:2f}'.format(trough_angles))
    # # print(trough_angles)
    # # trough_angles = trough_angles.where(trough_angles.isnull()==False, -30)
    # # print(trough_angles.where(trough_angles.isnull()==False, -30))
    # trough_angles = -trough_angles + 90
    # print('trough angle = {:2f}'.format(trough_angles))
    x, _, z = get_aimpt_from_sunangles(elev_angles, azimuth_angles)
    trough_angle = get_tracker_angle_from_aimpt(x,z)
    return trough_angle

def get_aimpt_from_sunangles(elev_angles, azimuth_angles):
    # trough_angles = sun_elev_to_trough_angles(elev_angles, azimuth_angles)
    # print('elev angle = {:2f}'.format(elev_angles))
    # print('azimuth angle = {:2f}'.format(azimuth_angles))
    # #print('trough angle = {:2f}'.format(trough_angles))
    # signed_elev_angles = 90 - trough_angles
    # x = factor * np.cos(np.radians(signed_elev_angles))
    # z = x * np.tan(np.radians(signed_elev_angles))
    x = np.cos(np.radians(elev_angles))*np.sin(np.radians(azimuth_angles))
    y = np.cos(np.radians(elev_angles)) * np.cos(np.radians(azimuth_angles))
    z = np.sin(np.radians(elev_angles))
    return x,y,z

def get_tracker_angle_from_aimpt(x,z):
    tracker_angle = np.degrees(np.arctan2(x,z))
    return tracker_angle


def create_file_structure(file_path, resolution, year, month, day):
    try:
         # Create the directory structure (resolution not needed anymore)
        if type(month)==str:
            directory = os.path.join(file_path, "year="+year+"/month="+month+"/day="+day)
        else:
            directory = os.path.join(file_path, f"year={year:04d}/month={month:02d}/day={day:02d}")

        # If the directory already exists, delete all files within it
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)

        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        return directory

    except Exception as e:
        print(f"Error: {e}")
                


#%%
if __name__ == "__main__":
    
    pass




