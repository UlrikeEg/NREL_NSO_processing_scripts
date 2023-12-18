"""Copyright (c) 2023 Alliance for Sustainable Energy, LLC"""

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
import pvlib

sys.path.append("../NSO_data_processing")

from Functions_masts import *
from Functions_general import *

           
unit_dict = {  
    "R1_DO_Bending": "kNm",
    "R1_DO_Torque": "kNm",
    "R1_SO_Bending": "kNm",
    "R1_DO_Accel_X": "g",
    "R1_DO_Accel_Y": "g",
    "R1_SO_Accel_X": "g",
    "R1_SO_Accel_Y": "g",
    "R1_Disp_NW": "mm",
    "R1_Disp_NE": "mm",
    "R1_Disp_SW": "mm",
    "R1_Disp_SE": "mm",
    "R1_Disp_Center": "mm",
    "R1_DO_Tilt": "$^\circ$",
    "R1_Mid_Tilt": "$^\circ$",
    "R1_SO_Tilt": "$^\circ$",
    "Anemometer": "m/s",
    "wspd_7m": "m/s",
    "wdir_3m": "$^\circ$",
    "wdir_5m": "$^\circ$",
    "wdir_7m": "$^\circ$",
    "R2_DO_Bending": "kNm",
    "R2_DO_Torque": "kNm",
    "R2_SO_Bending": "kNm",
    "R2_DO_Accel_X": "g",
    "R2_DO_Accel_Y": "g",
    "R2_SO_Accel_X": "g",
    "R2_SO_Accel_Y": "g",
    "R2_Disp_NW": "mm",
    "R2_Disp_NE": "mm",
    "R2_Disp_SW": "mm",
    "R2_Disp_SE": "mm",
    "R2_Disp_Center": "mm",
    "R2_DO_Tilt": "deg",
    "R2_Mid_Tilt": "$^\circ$",
    "R2_SO_Tilt": "$^\circ$",
    "R4_DO_Bending": "kNm",
    "R4_DO_Torque": "kNm",
    "R4_SO_Bending": "kNm",
    "R4_DO_Accel_X": "g",
    "R4_DO_Accel_Y": "g",
    "R4_SO_Accel_X": "g",
    "R4_SO_Accel_Y": "g",
    "R4_Disp_NW": "mm",
    "R4_Disp_NE": "mm",
    "R4_Disp_SW": "mm",
    "R4_Disp_SE": "mm",
    "R4_Disp_Center": "mm",
    "R4_DO_Tilt": "deg",
    "R4_Mid_Tilt": "$^\circ$",
    "R4_SO_Tilt": "$^\circ$",
    'R1_DO_C_Bending': "-",
    'R1_SO_C_Bending': "-",
    'R1_DO_C_Torque': "-",
    'R1_DO_Cfx': "-",
    'R1_SO_Cfx': "-",  
    'R2_DO_C_Bending': "-",
    'R2_SO_C_Bending': "-",
    'R2_DO_C_Torque': "-",
    'R2_DO_Cfx': "-",
    'R2_SO_Cfx': "-",  
    'R4_DO_C_Bending': "-",
    'R4_SO_C_Bending': "-",
    'R4_DO_C_Torque': "-",
    'R4_DO_Cfx': "-",
    'R4_SO_Cfx': "-",    
    "time": "s"} 
            
           
  #%% Read matlab data          
           
            
           
            
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict



def read_loads_data(loads_file, res_key ):

    matdata = loadmat(loads_file)['data_out']      
    
    
    # matdata.keys()               
    # matdata['Anemometer']['Props']['Units']
    # matdata['LabVIEW_Timestamp']['data']
    # matdata['Anemometer'].keys()  
    
    keys = list(matdata.keys())
    keys.remove('chanlist')
    keys.remove('Scan_Errors')
    keys.remove('Late_Scans')
    
    Loads = pd.DataFrame()  
    
    if '1min' in res_key:
        for key in keys:
            #print (key)
            Loads[key] = [matdata[key][res_key]]
    else:
        for key in keys:
            Loads[key] = matdata[key][res_key]
    
    return Loads


def read_raw_loads_at_date(year, month, day, resolution_loads, loads_path, print_units = 0):
    
    
    if resolution_loads == '20Hz':
        res_key = 'data'
    elif resolution_loads == '1min_mat':
        res_key = 'mean1min'
    elif resolution_loads == '10sec_mat':
        res_key = 'mean10sec'        
    # dict_keys(['name', 'Props', 'data', 'avtype', 'mean1min', 'max1min', 'min1min', 'stdev1min', 'mean10sec', 'max10sec', 'min10sec', 'stdev10sec'])
     

    year = str(year)            
    month = f'{month:02d}'       
    day = f'{day:02d}'
                  
    print (year, month, day)   
    
                
    ### Read loads files  
    
    loads_files = sorted(glob.glob(loads_path +  year + '-' + month + '-' + day  + '/NSO_Loads_' + year + '_' + month + '_' + day + '*_20Hz_Formatted_Data.mat'))

    loads = pd.DataFrame()
    
    for f in loads_files:
        loads = pd.concat( [loads, read_loads_data(loads_file = f, res_key = res_key)]) 

    
    if len(loads)>0: 
        loads.LabVIEW_Timestamp = pd.to_datetime(loads.LabVIEW_Timestamp, unit="s", origin='1904-01-01')
        loads.MS_Excel_Timestamp = pd.to_datetime(loads.MS_Excel_Timestamp, unit="D", origin='1899-12-30')
        
        
        print ('Read loads ok')       
        
        if print_units == 1:
            # Get units
            print ("Units: ")
            matdata = loadmat( loads_files[0])['data_out']     
            for key in list(matdata.keys())[4:-1]:
                print (key, ': ', matdata[key]['Props']['Units'])        
        
    else:
        print("No loads data avaialable")
            
    return loads


def read_processed_loads_at_dates(years, months, days, resolution_loads, loads_path):
    
    
    """ 
    resolution_loads  must be one of '20Hz', '1min', '10sec'  
    years, months and days must be list of numbers
    """
    
    loads_files = []

    for year in years:
        year = str(year)
    
        for month in months:
            
            if year == '2021' and month<10: # no data before October 2021
                continue
            
            month = f'{month:02d}'
        
            for day in days:
                day = f'{day:02d}'
                
                #print (year, month, day) 
                
                # Search loads files  
                loads_files = loads_files + sorted(glob.glob(loads_path + 'Loads_' + resolution_loads + '_' + year + '-' + month + '-' + day + '*.pkl'))


    # Read available data
    loads = pd.DataFrame()
    
    for f in loads_files:
        loads = pd.concat( [loads, pd.read_pickle(f)]) 
        print (f) 

    if len(loads)>0: 
        print ('Read loads ok')                
    else:
        print("No loads data avaialable")
        
    loads = loads[~loads.index.duplicated(keep='first')]
    loads = loads.sort_index()
            
    return loads
        



def  read_winds_at_dates(years, months, days, path = 'data/',   flux_path = 'data_fluxes/', res='20Hz', read_fluxes = False, read_masts=False  ):

    inflow_files = []
    mast_files = []
    inflow_files_fluxes = []
    mast_files_fluxes = []
    
    for year in years:
        year = str(year)
    
        for month in months:
            
            if year == '2021' and month<10: # no data before October 2021
                continue
            
            month = f'{month:02d}'
        
            for day in days:
                day = f'{day:02d}'
                  
                print (year, month, day) 
                inflow_files = inflow_files + sorted(glob.glob(path +'Inflow_Mast_'+res+'_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
                mast_files = mast_files + sorted(glob.glob(path +'Wake_masts_'+res+'_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #        
    
                inflow_files_fluxes = inflow_files_fluxes + sorted(glob.glob(flux_path +'Inflow_Mast_fluxes_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #
                mast_files_fluxes = mast_files_fluxes + sorted(glob.glob(flux_path +'Wake_Masts_fluxes_' + year + '-' + month + '-' + day + '_' + '*.pkl'))   #        
    
    ### Read data
    inflow = pd.DataFrame()    
    for datafile in inflow_files:
        inflow = pd.concat( [inflow, pd.read_pickle(datafile)]) 
        inflow = inflow[~inflow.index.duplicated(keep='first')]
    
    if read_masts == True:
        
        masts = pd.DataFrame()
        for datafile in mast_files:
            masts = pd.concat( [masts, pd.read_pickle(datafile)] )   
            masts = masts[~masts.index.duplicated(keep='first')]
        
    if read_fluxes == True:
        
        inflow_fluxes = pd.DataFrame()    
        for datafile in inflow_files_fluxes:
            inflow_fluxes = pd.concat( [inflow_fluxes, pd.read_pickle(datafile)]) 
            inflow_fluxes = inflow_fluxes[~inflow_fluxes.index.duplicated(keep='first')]
                    
        mast_fluxes = pd.DataFrame()    
        for datafile in mast_files_fluxes:
             mast_fluxes = pd.concat( [mast_fluxes, pd.read_pickle(datafile)]) 
             mast_fluxes = mast_fluxes[~mast_fluxes.index.duplicated(keep='first')]
               
    print ('Read winds ok')
    
    if (read_fluxes == True) and (read_masts == True):
        return inflow, masts, inflow_fluxes, mast_fluxes
    elif (read_fluxes == True) and (read_masts == False):
        return inflow, inflow_fluxes, mast_fluxes  
    elif (read_fluxes == False) and (read_masts == True):
        return inflow, masts 
    else:
        return inflow




def combine_meas_fluxes_angles(inflow, masts, inflow_fluxes, masts_fluxes, res_freq="10min"):
    inflow_fluxes = inflow_fluxes.resample(res_freq).mean()
    try:masts_fluxes  = masts_fluxes.resample(res_freq).mean()
    except: pass
    
    inflow = inflow.merge(inflow_fluxes, left_index=True, right_index=True, how="outer")
    try: masts = masts.merge(masts_fluxes, left_index=True, right_index=True, how="outer")
    except: pass
        
    # add trough angles
    angles = get_trough_angles(inflow.index)
    
    inflow = inflow.merge(angles.trough_angle, left_index=True, right_index=True, how="outer")
    
    return (inflow, masts)

#
def add_local_time_to_masts(data):
    data['UTC'] = data.index
    data.index = data.index.tz_localize('UTC').tz_convert('US/Pacific')
    return data

def correct_loads_time_lag(loads, lag):
    loads.index = loads.index + pd.to_timedelta(lag, unit="S")
    return loads


def correct_bending_moments(data):
    
    Hc = 2.79083   # m, height of pivot axis

    # correct bending moments for measurement location
    for column in data[[col for col in data.columns if ('DO_Bending' in col)  & ('_m' not in col) & ('_s' not in col)]]:
        data[column] = data[column] /(Hc - 0.48) * Hc
    for column in data[[col for col in data.columns if ('SO_Bending' in col)  & ('_m' not in col) & ('_s' not in col)]]:
        data[column] = data[column] /(Hc - 0.175) * Hc
            
    return data
    
    

def add_load_coefficients(data, wind_speed_limit=3):
    
    """ 
    'data' must be a dataframe with combined loads and wind data
    'wind_speed_limit' is the limit above coefficients are calculated
    """
    
    L_panel = 49          # m, length of panel with 6 mirror panels
    L_segment = 8  # m, length of mirror panel
    W = 5          # m, aperture width
    Hc = 2.79083   # m, height of pivot axis
    
    
    if 'rho' not in data.columns:
        
        try:
            data['rho'] = rho(data.p, data.RH, data.Temp).interpolate().rolling('60s', center=True, min_periods=0).mean()
        except Exception:
            pass
            # print("No data for air density calculation!")

    try:
        if 'wspd_4m' in data.columns:
            wspd = data.wspd_4m.rolling('60s', center=True, min_periods=0).mean()
        else:
            wspd = data.wspd_3m.rolling('60s', center=True, min_periods=0).mean()  
            
        wspd = wspd.where(wspd> wind_speed_limit)   
        
        # Bending moment coefficients
        for column in data[[col for col in data.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
            data[column[:6]+"C_"+ column[6:]] = data[column] *1000/ (data.rho/2 * wspd**2 * L_segment * W * Hc)   
        # Torque moment coefficients
        for column in data[[col for col in data.columns if ('Torque' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
            data[column[:6]+"C_"+ column[6:]] = data[column] *1000/ (data.rho/2 * wspd**2 * L_panel* W**2)   
        # Drag force coefficients
        for column in data[[col for col in data.columns if ('Bending' in col)  & ('_m' not in col) & ('_s' not in col) & ('C' not in col)]]:
            fx = data[column] *1000 / Hc
            data[column[:6]+"Cfx"] = fx / (data.rho/2 * wspd**2 * L_segment * W)  
            
        # Exclude unreasonably high values
        for column in data[[col for col in data.columns if ( ('C_' in col) or ('Cfx' in col))]]:
            data[column] = data[column].where(abs(data[column])<10)
        
    except Exception:
        pass
        #print("Check variables for moment coefficient calculation!")

    return data

def read_hosoya():
    hosoya = pd.read_csv('../NSO_additional_data/Hosoya_pressure_data.txt',delim_whitespace=True, index_col=None, header=0)
    # make the pitch angle definition consistent with our definition of howizontal = 0, facing west (into wind)=-90
    hosoya.Pitch = hosoya.Pitch - 90
    # for single collectors, the collector was hit by the wind from the back (Yaw=180) to get more pitch angles, make this definition consistent.
    hosoya.Pitch = hosoya.Pitch.where(hosoya.Yaw != 180, -hosoya.Pitch)
    for column in ['Cmy_Mean', 'Cmy_Max', 'Cmy_Min',  'Cfx_Mean',  'Cfx_Max', 'Cfx_Min']:
        hosoya[column] = hosoya[column].where(hosoya.Yaw != 180, -hosoya[column])
    save = hosoya.Cmy_Max
    hosoya.Cmy_Max = hosoya.Cmy_Max.where(hosoya.Yaw != 180, hosoya.Cmy_Min)
    hosoya.Cmy_Min = hosoya.Cmy_Min.where(hosoya.Yaw != 180, save)
    for column in []:
        hosoya[column] = hosoya[column].where(hosoya.Yaw == 180, -hosoya[column])
    save = hosoya.Cfx_Max
    hosoya.Cfx_Max = hosoya.Cfx_Max.where(hosoya.Yaw != 180, hosoya.Cfx_Min)
    hosoya.Cfx_Min = hosoya.Cfx_Min.where(hosoya.Yaw != 180, save)
    hosoya.Yaw = hosoya.Yaw.where(hosoya.Yaw != 180, 0)
    # make pitch angles between -180 and +180
    hosoya.Pitch = hosoya.Pitch.where(hosoya.Pitch >-180, hosoya.Pitch+360)
    return hosoya





























