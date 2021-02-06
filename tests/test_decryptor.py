"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
########################### MAIN WHITEWHOLE TEST #############################

from enigmx.decryptor import Decryptor

stock = 'F'
path = 'C:/Users/HELI/data/zarr3/'
dates = ['2020-08-01','2020-08-02']

dc = Decryptor(stock,path,dates)
print(dc.days)



