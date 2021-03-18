"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None  
from enigmx.combinatorial_backtest import CPKFCV
from enigmx.utils import dataPreparation


class EnigmxBacktest(object):
    
    def __init__(self, 
                 base_path, 
                 data_file_name, 
                 feature_sufix = 'feature',
                 label_name = 'barrierLabel',
                 exogenous_model_name = "exogenous_model.pkl",
                 endogenous_model_name = "endogenous_model.pkl",
                 features_sufix = 'feature',
                 timeIndexName = "close_date",
                 timeLabelName = "horizon",
                 y_true = True):
        
        self.base_path = base_path
        self.data_file_name = data_file_name
        
        
        self.feature_sufix = feature_sufix
        self.label_name = label_name
        
        self.exogenous_model_name = exogenous_model_name
        self.endogenous_model_name = endogenous_model_name
        
        self.csv_path = "{}{}".format(
            self.base_path, self.data_file_name
            )
        self.exogenous_path = "{}{}".format(
            self.base_path, self.exogenous_model_name
            )
        self.endogenous_path = "{}{}".format(
            self.base_path, self.endogenous_model_name
            )
        
        self.features_sufix = features_sufix
        self.timeIndexName = timeIndexName
        self.timeLabelName = timeLabelName
        self.y_true = y_true
        
    def __infoReader__(self):
        
        self.exogenousModel = pickle.load(open(self.exogenous_path, 'rb'))
        self.endogenousModel = pickle.load(open(self.endogenous_path, 'rb'))     
        
        self.dfStacked = pd.read_csv(
            self.csv_path, 
            ).dropna()
        
        self.dfStacked[self.timeIndexName] = self.dfStacked[self.timeIndexName].astype('datetime64[ns]')
        
        self.dfStacked = self.dfStacked.set_index(self.timeIndexName)        
            
        
        self.dataset = dataPreparation(
            data_csv = self.csv_path, 
            feature_sufix= self.feature_sufix,
            label_name= self.label_name, 
            timeIndexName = self.timeIndexName,
            timeLabelName = self.timeLabelName            
            )
        
    def __computeCombinatorialBacktest__(self, n, k):
        
        self.__infoReader__()
        
        combinatorial_backtest_instance = CPKFCV(
            data = self.dataset, N = n, k = k
            )        
        
        backtestPaths = combinatorial_backtest_instance.getResults(
            exogenous_model = self.exogenousModel, 
            endogenous_model = self.endogenousModel
            )
        
        # predicción categórica simple
        self.predCat = backtestPaths[0] 
        # predicción betsize
        self.predBetsize = backtestPaths[1] 
        # true label
        self.trueLabel = backtestPaths[2] 
        # index event
        self.indexEvent = backtestPaths[3]
        
    def get_combinatorial_backtest(self, 
                                   n_trials, 
                                   k_partitions, 
                                   df_format = True):
        
        self.__computeCombinatorialBacktest__(n = n_trials, k = k_partitions)
        
        if df_format:
            
            list_frame_trials = []
            
            for idx, partition in enumerate(self.indexEvent):
                
                dataTemp = self.dfStacked
                
                data = dataTemp.reset_index(drop=False) 

                data["predCat"] = self.predCat[idx]
                data["predBetSize"] = self.predBetsize[idx]
                data["trial"] = idx

                list_frame_trials.append(data)
                
            generalFrame = pd.concat(list_frame_trials).reset_index(drop=True)
            
            return generalFrame
            
        else:
            return self.predCat, self.predBetsize, self.trueLabel, self.indexEvent
        

                
#################################### TEST ####################################

instance = EnigmxBacktest(
    "D:/feature_importance/", 
    "STACKED_ENDO_VOLUME_MDA.csv", 
    feature_sufix='bar_cum')


results = instance.get_combinatorial_backtest(n_trials = 11, k_partitions = 2)

results.to_csv("D:/feature_importance/backtest_results.csv",index=False)