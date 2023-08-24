import sys
import numpy as np
import glob
import pyarrow.parquet as pq

sys.path.append("../NSO_data_processing")
from Functions_masts import *




#%% Define

lidar_path =       'Y:\Wind-data/Restricted/Projects/NSO/Lidar_Data/'  
turbulence_path =  'Y:\Wind-data/Restricted/Projects/NSO/Lidar_Turbulence/'    
publication_path = 'Y:\Wind-data/Restricted/Projects/NSO/Data_publish/NSO/lidar/'  



save  = 1
plot  = 0



#%% Functions

def read_raw_lidar(datafile):
    lidar = pd.read_csv(datafile,
                        sep = '\t',
                        skiprows = 5,
                        header = 0,
                        index_col = None,
                        parse_dates = [3]
                        )
    return lidar
                 


def prep_lidar_data(lidar):
    
    ## Prepare data
    
    # rename columns with spaces 
    lidar.rename(columns={"Range gate": "Range_gate", 
                          "Ray time": "Ray_time"},
                 inplace=True)
    
    lidar.set_index("Ray_time", inplace=True)
    lidar = lidar.drop_duplicates().sort_index() 
    
    # Create column with filtered Doppler
    INT = lidar.Intensity.where(lidar.Intensity > 1) # otherwise the SNR becomes -inf
    SNR = 10 * np.log10(INT - 1)
    lidar['Doppler_filtered'] = lidar.Doppler.where(SNR>-19)
    lidar['Doppler_filtered'] = lidar.Doppler_filtered.where(lidar.Doppler_filtered <35)
    lidar['Doppler_filtered'] = lidar.Doppler_filtered.where(lidar.Doppler_filtered >-35)
    
    return lidar

def plot_lidar_data(lidar):
               
    # Timeseries
    plt.figure()   
    plt.suptitle(year + month + day)
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_ylabel('Doppler velocity (m/s)')

    ax1.plot(lidar.Doppler,".")
    ax1.plot(lidar.Doppler_filtered,".")
    ax1.grid(True)

    # Dependence on Azimuth
    fig = plt.figure()   
    plt.suptitle(year + month + day)
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_ylabel('Doppler velocity (m/s)')
    ax1.set_xlabel('Azminuth (deg)')

    ax1.plot(lidar.Az, lidar.Doppler_filtered,".")
   
    ax1.grid(True)
    
def save_processed_lidar_data(lidar, complete_path):
                    
    ## Parquet files 
    # Define metadata and units
    lidar_metadata = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'author':'Ulrike Egerer'}
    
    # Save
    lidar.to_parquet(complete_path + 
                      '/Lidar_{}_{:0>2}-{:0>2}-{:0>2}_to_{}_{:0>2}-{:0>2}-{:0>2}.parquet'
                      .format(lidar.index[0].date(),lidar.index[0].hour,lidar.index[0].minute,lidar.index[0].second, 
                              lidar.index[-1].date(),lidar.index[-1].hour,lidar.index[-1].minute, lidar.index[0].second), 
                      #metadata=lidar_metadata
                      )


    plt.close('all')




#%% Process 360 degree scans           

print ("Processing 360 degree scans...")


years   =  [2022]   # np.arange(2022,2024)   #
months  =  [6,7,8,9]
days    =  np.arange(1,32)   # 

# years   =  [2022] #
# months  =  [6] # 
# days    =  [10] # 


for year in years:
    year = str(year)

    for month in months:
  
        month = f'{month:02d}'
    
        for day in days:
            
            if year == '2022' and month=='06' and day <10: # do not publish data before that date
                continue     
            
            day = f'{day:02d}'         

            print (year, month, day) 

            
            ## Read lidar files of each day
            lidar_files = sorted(glob.glob(lidar_path + year + month + "/" +  year + month + day + "/*/*.scn"))   #
            
            
            ## Create publishing file structure for this day
            complete_path = create_file_structure(file_path = publication_path, resolution = '10min', year=year, month=month, day=day)
   
    
            ## Loop over each scan file and save seperately
            for datafile in lidar_files:     
                
                lidar = read_raw_lidar(datafile)
                 
                if len(lidar)==0:
                    continue
    
                lidar = prep_lidar_data(lidar)    

                
                if plot == 1:
                    
                    plot_lidar_data(lidar) 

                
                if save == 1:

                    save_processed_lidar_data(lidar, complete_path)

    
            print ("Day processed")            

            
           
#%% Process turbulence stare scans           

print ("Processing turbulence stare scans...")          

turbulence_daily_files = sorted(glob.glob("./lidar_turbulence_file_names/*"))   #

for turbulence_daily_file in turbulence_daily_files:
    
    year  = '20' + turbulence_daily_file[30:32]
    month = turbulence_daily_file[34:36]
    day   = turbulence_daily_file[36:38]
    
    print (year, month, day) 
    
    complete_path = create_file_structure(file_path = publication_path, resolution = '2s', year=year, month=month, day=day)

    lidar_turbulence_files =  open(turbulence_daily_file, 'r').readlines()
    
    ## Loop over each scan file and save seperately
    for datafile in lidar_turbulence_files:  
        
        Datafile = datafile[:-1].replace("/Volumes/NSO/Lidar_Turbulence/", turbulence_path)
        
        lidar = read_raw_lidar(Datafile)
        
        
        
        if len(lidar)==0:
            continue

        lidar = prep_lidar_data(lidar) 
        
        #  print (lidar.index[0])

        
        if plot == 1:
            
            plot_lidar_data(lidar) 

        
        if save == 1:

            save_processed_lidar_data(lidar, complete_path)
    
    
     

           
            
           
            
           
            
           
            
           
            
           
           
    
