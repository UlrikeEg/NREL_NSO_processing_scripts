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

years   =  [2022]   # np.arange(2022,2024)   #
months  =  [6,7,8,9]
days    =  np.arange(1,32)   # 

# years   =  [2022] #
# months  =  [6] # 
# days    =  [10] # 


save  = 1
plot  = 0




#%% Read data            

print ("Processing ...")

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
            
            # Create publishing file structure for this day
            complete_path = create_file_structure(file_path = publication_path, resolution = '10min', year=year, month=month, day=day)
   
            # Loop over each scan file and save seperately
            for datafile in lidar_files:
                lidar = pd.read_csv(datafile,
                                    sep = '\t',
                                    skiprows = 5,
                                    header = 0,
                                    index_col = None,
                                    parse_dates = [3]
                                    ) 
                 
                if len(lidar)==0:
                    continue
    
                      
                    
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
               
                
      
                
                
                
                #%% Plot data
                
    
                
                if plot == 1:
                
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
    
    
    
                
       
                
                     
                #%% Save data files as Parwuet

                
                if save == 1:

                    
                    ## Parquet files 
                    # Define metadata and units
                    lidar_metadata = {'creation_date':pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                                'author':'Ulrike Egerer'}
                    
                    # Save
                    lidar.to_parquet(complete_path + 
                                      '/Lidar_{}_{:0>2}-{:0>2}_to_{}_{:0>2}-{:0>2}.parquet'
                                      .format(lidar.index[0].date(),lidar.index[0].hour,lidar.index[0].minute, 
                                              lidar.index[-1].date(),lidar.index[-1].hour,lidar.index[-1].minute), 
                                      #metadata=lidar_metadata
                                      )

                    
                    # if plot == 1:
                        
                    #    fig.savefig('Y:\Wind-data/Restricted/Projects/NSO/Daily_quicklooks' + 
                    #                      '/Lidar_winds_{}_{}h_to_{}_{}h.png'
                    #                      .format(lidar.index[0].date(),lidar.index[0].hour, 
                    #                              lidar.index[-1].date(),lidar.index[-1].hour), dpi=200) 
                    plt.close('all')
                       
 

    
            print ("Day processed")            

            
           
            
           
            
           
            
           
            
           
            
           
            
           
            
           
            
           
           
    