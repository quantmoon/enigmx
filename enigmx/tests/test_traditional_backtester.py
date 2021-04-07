"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

########################### MAIN BACKTESTER TEST #############################

import enigmx as qz
from enigmx.backtester.singularity import UserStrategy


nc_dir= 'D:\data_repository'
symbol_list= ['A'] 
start_date = "2019-01-08"
end_date = "2019-01-10"
heartbeat = 3600 #This value is in seconds
api_key = "bt4a2lv48v6ue5eg959g"

bt = qz.Backtest(nc_dir, 
                 symbol_list, 
                 start_date, 
                 end_date, 
                 vwap = True, 
                 strategy=UserStrategy)

bt.add(frequency = 5)
bt.add(bartype = 'time')
bt.add(heartbeat = heartbeat)


bt.simulate_trading()