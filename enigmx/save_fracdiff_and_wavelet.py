"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import os
import numpy as np
import pandas as pd
from enigmx.utils import simpleFracdiff
from Lambda.variables.transform import Variables

#base paramters for Wavelet Features Computation
data_dir = 'D:/data_basic_stacked/' 
wav_kernels = ["MORLET", "DOG", "PAUL"]
path_save = 'D:/data_intermediate_stacked/'

#1) list stocks definition (getting zarr file names) | Include. WAVELET
def intermediate_stacked_operation(data_dir, wav_kernels, path_save, 
                                   column_for_fracdiff='time'): #otorgar el nombre de la columna fracdiff y lo convierte
    for i,file in enumerate(os.listdir(data_dir)):
        
        if file.endswith(".csv"):
            
            #stock name to open basic stacked data
            stock_file_name = os.path.basename(file)
            
            #basic stacked datacsv
            info_pandas = pd.read_csv(data_dir + stock_file_name)
            
            #fracdiff computation using 'time' daily vwap series
            fracdiff_ = simpleFracdiff(info_pandas[column_for_fracdiff])
            
            #insert 'nan' as 1st time series element due to the slicing
            fracdiff = np.insert(fracdiff_, 0, np.nan)
            
            #append of fractional differentiation over central pandas
            info_pandas["fracdiff"] = fracdiff
            
            #wavelet list computation based on kernel type 
            wavelets_ = [
                Variables(
                    data = info_pandas[['datetime','time']],
                    Wav_base = Wav_base 
                    )._extract_features() for Wav_base in wav_kernels
                ]
            
            #insert central pandas in wavelets list
            wavelets_.insert(0, info_pandas)
            
            #concadenate all information
            result_dataset = pd.concat(
                                wavelets_, 
                                axis=1
                                )
            
            #save csv including wavelets and fracdiff
            result_dataset.to_csv(
                path_save + stock_file_name[:-18] + "_INTERMEDIATE_STACKED.csv",
                index=False
                )
            
    print("Stocks saved")
