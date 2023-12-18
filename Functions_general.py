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
import os
import math





    

#%% Thermodynamics

def rh2q(RH,T,p,w=0):
    """calculates the specific humidity (q) out of the relative humidity (%),\
    temperature (K) and pressure (Pa).
  
    option: 1=only q; 2 = q and water vapour mixing ratio(w)
    output [q]=g/kg, [w]=g/kg
    
    use: rh2q(Balloon.RH, Balloon.T_PT100+273.15 ,Balloon.p*100) 
    """
    # constants
    Rd=287.6 # J/(kg*K)
    Rv=461.5 # J/(kg*K)
    # unit conversion
    T_c=T - 273.15 # K
    rh=RH/100      # 0..1
  
    # saturation vapour pressure Pa
    es = 611.20*np.exp(17.67*T_c/(T_c+243.5)) # Bolton 1980 equation 10 ## over water
    #es = 611.20*np.exp(22.46*T_c/(T_c+272.62)) # over ice (Wikipedia)
    e=rh*es                                # Pa 
    w_air=(e*Rd)/(Rv*(p-e))                # [kg/kg]
    q_air=w_air/(w_air+1)                  # [kg/kg]
    q_air=q_air*1000                        # [g/kg]
    w_air=w_air*1000                        # [g/kg]

    if(w==1):
        return (q_air, w_air) 
    else:
        return q_air
    
def es(T_c):
    """calculates the saturation vapour pressure (Pa) from temperature (deg C).
    """
    es = 611.20*np.exp(17.67*T_c/(T_c+243.5)) # Bolton 1980 equation 10 ## over water
    #es = 611.20*np.exp(22.46*T_c/(T_c+272.62)) # over ice (Wikipedia)
    return es

def T_from_es(es):
    """calculates the temperature (deg C) from  saturation vapour pressure (Pa).
    """
    T = (243.5 * math.log(es/611.2)) / (17.67 - math.log(es/611.2))
    return T
    
    
        
def q2rh(q,T,p):
    p = p*100
    T = T+273.15
    q = q/1000
    RH = 0.263 * p * q * (np.exp( (17.67*(T-273.15)) / (T-29.65) ))**(-1)

    return RH


    
def Tv(RH,T,p): # Wikipedia/ Wendisch+Brenguier S. 63
    """calculates Tv
    use: Tv(Balloon.RH_LinP, Balloon.T_Thermo+273.15, Balloon.p*100) 
    input:
        Rh in proz
        T in K
        p in Pa
    output:
        Tv in deg C
    """
    return T * (1+ 0.6087 * rh2q(RH,T,p)/1000 )-273.15

def T_from_Tv(RH,Tv,p,T): # Wikipedia/ Wendisch+Brenguier S. 63
    """calculates Tv
    use: T_from_Tv(Balloon.RH_EE08, Balloon.Tv, Balloon.p*100, Balloon.T_comp) 
    input:
        Rh in proz
        Tv in K
        p in Pa
        T in K
    output:
        T in deg C
    """
    return Tv / (1+ 0.6087 * rh2q(RH,T,p)/1000 ) - 273.15

def q_from_T_Tv(Tv, T):
    q = (Tv/T - 1)/0.6087
    return q*1000

def R_f(RH, p, T): # Wiki
    '''
    input:
     RH in percent
     p in hPa
     T in degC
    output:
     R_f in J/(kg K)
    use:
     R_f(data.RH_EE08, data.p, data.T_Thermo)   
    '''
    RH = RH/100
    p = p*100
    es = 611.20*np.exp(17.67*T/(T+243.5)) # Bolton 1980 equation 10
    R_f = 287.058 / ( 1- (RH* es/p* (1- 287.058/461.523) ) )
    return R_f 
    
def h_to_p(h, p0=101325.0, T0=288.15): # Wendisch+Brenguier S. 11
    '''
    h in m
    p in Pa
    '''
    L0=0.0065
    g=9.80665
    R=287.05
    p = p0 * ( 1 - (L0 * h)/T0 ) ** (g / ( R * L0)) 
    return p

def p_to_h(p, p0=101325.0, T0=288.15, h0=0): # Wendisch+Brenguier S. 11
    '''
    h in m
    p in Pa
    '''
    L0=0.0065
    g=9.80665
    R=287.05    
    h = h0 + T0/L0 * ( 1 - (p/p0) ** (R*L0/g))
    return h

def p_to_h_ICAO(p, p0=101325.0, T0=0., h0=0): # ICAO
    '''
    h in m
    p in Pa
    '''
    g=9.80665
    RL=287.05    # ?
    h = h0 + RL * (T0 + 273.15) / g * np.log(p0 / p)
    return h


def theta(T, p, p0=100000.0): # Wiki
    # dry theta!
    '''
    T in K
    p in Pa
    p0 in pa
    theta in deg C
    '''
    cp = 1005
    R=287.05  
    theta = T * (p0/p) ** (R/cp) - 273.15
    return theta

def rho(p, RH, T): # Wendisch+Brenguier S. 19
    '''
    p in hPa
    T in deg C
    RH in percent
    '''
    p = p*100
    rho = p / ( R_f(RH, p, T) * (T+273.15) )
    return rho


#%% Turbulence functions
   


    
def variance_window(data_column, fs, window=50):
    ''' calculates variance for the a specified column in time periods of window length''' 
    ''' data_column: fluctuations already have to be filtered'''
    ''' 
    input: 
        data_column: columns of a pandas dataframe
        fs: sampling frequency
        window: length of the window in seconds
    output:
        data_var: column with the length of data_column -> variance for window around this value
    '''

    data_column = data_column.dropna()
    if len(data_column) < round_up_to_odd(window*fs):   # if not enough data points       
        return np.nan
                 
    # variance
    data_var = data_column.rolling(int(fs*window), min_periods=10, center=True).var()
    
    return data_var

def tke(u,v,w):
    ''' calculates TKE for the wind vector u,v,w in the whole time period of u,v,w'''
    ''' return a single value'''
    # u,v,w must be  pandas columns
#    if len(u) % 2 == 0:
#        u = u[:-1] # make odd number
#        v = v[:-1] # make odd number
#        w = w[:-1] # make odd number
#    window = len(u)

    u_fluc = scipy.signal.detrend(u)
    v_fluc = scipy.signal.detrend(v)
    w_fluc = scipy.signal.detrend(w)
    
    tke = 0.5 * ( u_fluc.var() + v_fluc.var() + w_fluc.var() )   
    return   tke

def tke_window(u_fluc ,v_fluc ,w_fluc, fs, window=50):
    ''' calculates running TKE for the wind vector u,v,w in time periods of window length'''
    ''' u,v,w have to be fluctuations of the wind velocity '''
    ''' returns a columns with the length of u,v,w'''
    # u,v,w must be  pandas columns
    # window in seconds
    if len(u_fluc) < int(window*fs+1):   # if not enough data points       
        return np.nan
    
    tke = 0.5 * ( u_fluc.rolling(int(window*fs), min_periods=10, center=True).var() 
                + v_fluc.rolling(int(window*fs), min_periods=10, center=True).var() 
                + w_fluc.rolling(int(window*fs), min_periods=10, center=True).var() )  
    return   tke

def TI_window(U_fluc, U_mean, fs, window=50):
    ''' calculates running turbulence intensity for the horizontal wind U in time periods of window length'''
    ''' U is the wind velocity '''
    ''' returns a columns with the length of U'''
    # u,v,w must be  pandas columns
    # window in seconds
    if len(U_mean) < int(window*fs+1):   # if not enough data points       
        return np.nan
    
    TI = U_fluc.rolling(int(window*fs), min_periods=10, center=True).std() / U_mean.rolling(int(window*fs), min_periods=10, center=True).mean()
                
    return   TI


def TI_time_window(U_fluc, U_mean, time_window="10min"):
    ''' calculates running turbulence intensity for the wind component U_fluc in time periods of window length'''
    ''' index must be datetime format '''
    ''' U is the wind velocity component '''
    ''' returns a columns with the length of U'''
    
    TI = U_fluc.rolling(time_window, min_periods=10, center=True).std() / U_mean.rolling(time_window, min_periods=10, center=True).mean()
                
    return   TI

def tke_time_window(u_fluc ,v_fluc ,w_fluc, time_window="10min"):
    ''' calculates running TKE for the wind vector u,v,w in time periods of window length'''
    ''' u,v,w have to be fluctuations of the wind velocity '''
    ''' returns a columns with the length of u,v,w'''
    # u,v,w must be  pandas columns
    # window is a datetime index
    
    tke = 0.5 * ( u_fluc.rolling(time_window, min_periods=10, center=True).var() 
                + v_fluc.rolling(time_window, min_periods=10, center=True).var() 
                + w_fluc.rolling(time_window, min_periods=10, center=True).var() )  
    return   tke






def Ri_bulk( thetav_upper, thetav_lower, U_upper, U_lower, V_upper, V_lower, deltaz ):
    """
    input:
        thetav_upper : potential virtual temperature at upper level [deg C] - single value
        thetav_lower : potential virtual temperature at lower level [deg C] - single value
        U_upper, U_lower : U at upper and lower level [m/s] - single value
        V_upper, V_lower : V at upper and lower level [m/s] - single value
        deltaz : height difference between lower and upper level [m] - single value
    output:
        bulk Richardson number [no unit]
    """
    Ri = 9.81/ ( (thetav_upper+thetav_lower)/2 +273.15 )   *  ((thetav_upper - thetav_lower) / deltaz)   / ( 
                                                ((U_upper - U_lower) / deltaz)**2  +  ((V_upper - V_lower) / deltaz)**2  )  
    #Ri = Ri.rolling(20, center=True, min_periods=10).median()
    Ri = Ri.where(Ri<100)
    Ri = Ri.where(Ri>-100)
    
    return Ri


def Ri_flux(Tref, thetav_upper, thetav_lower, U_upper, U_lower, V_upper, V_lower, W_upper, deltaz):
    """
    input:
        thetav_upper : potential virtual temperature at upper level [deg C] - pd.DataFrame column
        thetav_lower : potential virtual temperature at lower level [deg C] - pd.DataFrame column
        U_upper, U_lower : U at upper and lower level [m/s] - pd.DataFrame column
        V_upper, V_lower : V at upper and lower level [m/s] - pd.DataFrame column
        W_upper          : W at upper           level [m/s] - pd.DataFrame column
        deltaz : height difference between lower and upper level [m] - single value
    output:
        flux Richardson number [no unit]
    """
    
    #Temp = ( (thetav_upper+thetav_lower)/2 +273.15 ).mean()   # do not use because virtual temperatures are not calibrated.
    Temp = Tref.mean()+273.15 
    horiz = np.sqrt(U_upper**2 + V_upper**2)      
    U_grad = (U_upper.mean() - U_lower.mean()) / deltaz
    V_grad = (V_upper.mean() - V_lower.mean()) / deltaz
    
    # detrend
    for X in [thetav_upper,U_upper,  V_upper, horiz, W_upper]:
        X[X.isna()==False] = scipy.signal.detrend(X.dropna())
    
    Rif = 9.81/ Temp   * W_upper.cov(thetav_upper)   / ( 
                         (  W_upper.cov(U_upper) * U_grad)  +  (   W_upper.cov(V_upper) *  V_grad ) )  

    if (Rif > -50) and (Rif<50): 
        return Rif    
    else:
        return np.nan


def Obukhov_stability( Tref, Tv, U, V, W, z):
    """
    input:
        Tref: reference potential virtual temperature (absolute value)  [deg C] - pd.DataFrame column
        Tv: virtual temperature, fast response for covariance  [deg C] - pd.DataFrame column
        U, V, W : wind speed components [m/s] - pd.DataFrame column
        z : height level [m] - single value
    output:
        Obukhov stability parameter z/L [no unit]
        Obukhov length [m]
        
    """
    
    
    k = 0.4     # (AMS glossary)  
    
    Temp = ( Tref.mean() +273.15 )
    
    horiz = np.sqrt(U**2 + V**2)  
    
    # interpolate single nans
    horiz = horiz.interpolate(limit=2)
    W     = W.interpolate(limit=2)
    Tv    = Tv.interpolate(limit=2)
    
    # detrend
    for X in [horiz, W, Tv]:
        X[X.isna()==False] = scipy.signal.detrend(X.dropna())  # detrend eliminates the trend and calculates fluctuations
    
    u_star = np.sqrt( abs( horiz.cov(W) ))

    zL = - z * k *  9.81  * W.cov(Tv)  / Temp  / u_star**3  
    
    L = z / zL
    
    return zL, L    
    
    
#%% Turbulent Fluxes


    
def HS(wE, theta, Rho):# Wendisch+Brenguier S. 69
    '''
    Sensible heat flux
    use Sonic: HS(-data_for_flux.w_E, data_for_flux.theta, 
                  rho(data_for_flux.p, data_for_flux.RH_EE08, data_for_flux.T_comp).mean())
    use Mast: HS(data_for_flux.verticalwind, data_for_flux.TV, rho_mast)
    
    to be clarified:
        - use hanning window and if yes, where?
    '''
    T = theta
    df = pd.DataFrame({'x':wE, 'y':T}).interpolate(limit=2).dropna(how='any')
    wE = scipy.signal.detrend(df.x)
    T = scipy.signal.detrend(df.y)
#    wE = wE * signal.hann(len(wE)) * len(wE) / sum(signal.hann(len(wE)))
#    Tv = Tv * signal.hann(len(T)) * len(T) / sum(signal.hann(len(T)))
    wT = (wE - wE.mean()) * (T - T.mean())
    HS = Rho * 1006 * wT.mean()

    return HS

def BF(wE, theta_v):# AMS glossary
    '''
    Buoyancy flux
    use Sonic: BF(-data_for_flux.w_E, data_for_flux.theta_v)
    '''
    Tv = theta_v.mean()+273.15
    df = pd.DataFrame({'x':wE, 'y':theta_v}).interpolate(limit=2).dropna(how='any')
    wE = scipy.signal.detrend(df.x)
    theta_v = scipy.signal.detrend(df.y)
    
    wT = (wE - wE.mean()) * (theta_v - theta_v.mean())
    BF = 9.81/ Tv * wT.mean() # unit different from W/m2
    return BF

def HL(wE, q, Rho):
    '''
    Latent heat flux
    use Sonic: HL(-data_for_flux.w_E, data_for_flux.q, 
                  rho(data_for_flux.p, data_for_flux.RH_EE08, data_for_flux.T_comp).mean())
    use Mast: HL(data_for_flux.verticalwind, data_for_flux.q, rho_mast)
    '''    
    df = pd.DataFrame({'x':wE, 'y':q}).interpolate(limit=2).dropna(how='any')
    wE = scipy.signal.detrend(df.x)
    q = scipy.signal.detrend(df.y)
    wq = (wE - wE.mean()) * (q - q.mean())
    HL = Rho * 2.5 * 10**6 * wq.mean()/1000
    return HL        
    
def tau(wE, u, Rho):  
    '''
    Momentum flux
    use Sonic: tau(-data_for_flux.w_E, data_for_flux.longitudinalwind, 
                  rho(data_for_flux.p, data_for_flux.RH_EE08, data_for_flux.T_comp).mean())
    use Mast: tau(data_for_flux.verticalwind, data_for_flux.horizontalwind, rho_mast)
    '''
    df = pd.DataFrame({'x':wE, 'y':u}).interpolate(limit=2).dropna(how='any')
    wE = scipy.signal.detrend(df.x)
    u = scipy.signal.detrend(df.y)
    wu = (wE - wE.mean()) * (u - u.mean())
    tau = - Rho * wu.mean()
    return tau

def ogive(signal1, signal2, freq, time_series = 0):
    '''
    calculates and plots cospectrum of two signals and ogive
    input:
        signal1 and signal 2: columns of pandas DataFrame for calculating cospectra
        freq: Frequency in Hz
    returns:
        X: frequency axis
        Co: Cospectra per frequency
        Og: Ogive (cumulative cospectum)
    '''
  
    freq = float(freq)
    
    # Define signals
    fftsig1 = signal1.interpolate()
    fftsig2 = signal2.interpolate()
    
    # plot time series
    if time_series == 1:
        times = fftsig1.index
        fig1, ax1 = plt.subplots()
        ax1.plot(times,fftsig1.values, label ='wE original')
        ax1.plot(times,fftsig2.values, label ='theta_v original')
        
    # Detrend
    fftsig1 = sp.signal.detrend(fftsig1)
    fftsig2 = sp.signal.detrend(fftsig2)

    if time_series == 1:
        ax1.plot(times,fftsig1, label ='wE detrended')
        ax1.plot(times,fftsig2, label ='theta_v detrended')
        ax1.grid(True)
        ax1.set_xlabel("Time (s)", fontsize='x-large')
        ax1.set_ylabel("Signal", fontsize='x-large')
        ax1.legend(loc='best', fontsize='large')  
        ax1.set_title('Time series', fontsize='x-large')
    
    N = len(fftsig1) 
    
    # frequency axis
    X = fftfreq(N, 1/freq)
    X = X[1:int(N/2)]
    df = X[2] - X[1] # frequency interval
    
    # FFT
    Y1 = fft(fftsig1)/N
    Y2 = fft(fftsig2)/N       
    
    # Cospectrum
    Co = (Y1.real * Y2.real + Y1.imag * Y2.imag)     # Stull p. 330        
    Co = 2* Co[1:int(N/2)] /df                            # see also:
                                                        # https://de.mathworks.com/matlabcentral/fileexchange/53545-ogive-optimization-toolbox
                                                        # https://acp.copernicus.org/articles/15/2081/2015/acp-15-2081-2015.pdf
    # Ogive 
    Og = np.cumsum(Co[::-1]*df)[::-1]          # [::-1] reverses the list to integrate beginning with highest frequency

    return X, Co, Og 


#%% Other

def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y



def round_up_to_odd(f):
        return int( math.ceil(f / 2.) * 2 +1)
    
def round_up_to_even(f):
        return int( math.ceil(f / 2.) * 2 )


def find_fs(data):
    fs = 1/np.median(np.diff(data.time.values))
    return fs

 

    
