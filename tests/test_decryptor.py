"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
########################### MAIN WHITEWHOLE TEST #############################

import whitehole as wh

dcrypt= wh.Decryptor(repo_path= 'C:/Users/HELI/data/zarr/', 
          symbol='FB', 
          date="2018-12-31", 
          full_day=True, 
          save=True,
          storage_path='C:/Users/HELI/data/csvdata/')

dcrypt.run_decryptor()


