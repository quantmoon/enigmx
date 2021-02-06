"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pickle
import pandas as pd
from enigmx.combinatorial_backtest import *
from enigmx.utils import dataPreparation

#0) path of backtest data
csv_path = "D:/data_single_stacked/A_COMPLETE.csv"

#i) paths of exogenous and endogenous model 
exogenous_path = "D:/data_split_stacked/exogenous_model.pkl"
endogenous_path = "D:/data_split_stacked/endogenous_model.pkl"


### STEPS

# 1) Reading Data
backtest_data = pd.read_csv(csv_path).iloc[1:,0:14].dropna()

# 2) Loading Exogenous Model and Endogenous Model
exogenousModel = pickle.load(open(exogenous_path, 'rb'))
endogenousModel = pickle.load(open(endogenous_path, 'rb'))


# 3) Data Refinement 
dataset = dataPreparation(backtest_data)

# 3) Setting Combinatorial Purged K-Fold Instance
combinatorial_backtest_instance = CPKFCV(
    data = dataset, N = 11, k = 2
    )

# 4) Get Combinatorial Purged K-Fold Instance Predictions using models
backtestPaths = combinatorial_backtest_instance.getResults(
    exogenous_model=exogenousModel, endogenous_model = endogenousModel
    )

#for idx in range(1, len(backtestPaths[0])):
#    print(all(backtestPaths[0][idx-1] == backtestPaths[0][idx]))
    
#print("**"*10)

#PROBLEM! All paths are the same | even the real labels
#for idx in range(1, len(backtestPaths[1])):
#    print(all(backtestPaths[1][idx-1] == backtestPaths[1][idx]))
    
    