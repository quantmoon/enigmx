"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from enigmx.betsize import betsize_livetrading
from enigmx.utils import (
    nCk, 
    paths,
    keep_same, 
    consecutive,
    master_sets,
    purge_embargo, 
    split_map_CPCV, 
    dataPreparation,  
    groups_from_test,
    list_for_train_test
    )

def check_data(idxs, base_dataset):
    base_dataset = base_dataset.reset_index(drop=True).values
    featuresMatrix, labelsVector = (
            base_dataset[idxs,:-1].astype(np.float32), 
            base_dataset[idxs,-1:]
        )
    
    return featuresMatrix, labelsVector

class GeneralModel(object):
    
    def __init__(self, MA, MB):
        
        self.MA = MA 
        self.MB = MB
    
    def __dataPreparation__(self, idxs, basedata):
        """
        Contraint:
        If 'base_dataset': pd.DataFrame:
            'label':= idx[-1]
        If 'base_dataset': np.ndarray:
            'label':= idx[-1] by row dimension

        Columns pandas/array data should be only features.
        """
        assert isinstance(basedata, 
                          (pd.DataFrame, np.ndarray)
                         ), "Wrong format for arg2*"    
        
        if type(basedata) == pd.DataFrame:    
            basedata = basedata.reset_index(drop=True).values

        #select features and labels
        featuresMatrix, labelsVector = (
                basedata[idxs,:-1].astype(np.float32), 
                basedata[idxs,-1:]
            )        
        
        return featuresMatrix, labelsVector
    
    def fit(self, idxs, basedata):
        #split division to define X and Y (training set)
        X, y = self.__dataPreparation__(idxs, basedata)
        
        
        #print("DATA QUE SE OBTIENE EN  FIT")
        #print(X)
        #print(np.unique(y))
        #print(" ")
        
        #fitting/update training process in exogenous model (side)
        self.MA = self.MA.fit(X,y)
    
    def predict(self, idxs, basedata, y_true=False):
        
        #split division to define X and Y (testing set)
        X, y_real = self.__dataPreparation__(idxs, basedata)
        
        
        #print("DATA 'X' QUE SE OBTIENE EN  PREDICT")
        #print(X.shape)
        #print(" ")
        
        #prediction process in exogenous model (side)
        predictionA = self.MA.predict(X)
        
    
        
        #print("PREDICCIONES 'A' EN PREDICT")
        #print(predictionA)
        #print(" ")
        
        #prediction process in endogenous model (size | metalabelling)
        predictionB = betsize_livetrading(
            X, predictionA.reshape(-1,1), self.MB
            )
        
        
        
        #for PA, PB in zip(predictionA, predictionB):
        #if np.unique(predictionA).shape[0]>1: 
        print(np.unique(predictionA, return_counts=True), np.unique(predictionB))
        
        #if 'y_true', returns Y' (predictions) and Y(real label events)
        if y_true: #useful to calculate the performance metric
            return predictionB, y_real
        else:
            return predictionB
        
class CPKFCV(object):
    """
    Description:
        Combinatorial Purged K-Fold Main Class.
        
    Inputs: 
        1. 'data': main dataframe features-labels dataset
        2. N: number of desired general groups
        3. k: number of groups for sub-test set (mostly, 2)
        
    Methods:
        1. '__IndexStructureGeneration__': 
                This method develop three variables:
                1.1. 'masterSets': 
                1.2. 'split_map':
                1.3. 'n_paths': calculated number of paths from (k & N)
            
            Aditional input: 
                * 'embargo_level': int representing embargo intensity
                
        2. 'getResults':
                Get the predictions for each path based on 
                '__IndexStructureGeneration__' method results.
                
            Aditional input:
                * 'exogenous_model': base Machine Learning model | side pred.
                * 'endogenous_model': complementary ML model | size pred.
                * 'nonfeature': useless var list to drop from feature matrix.
                
    """
    def __init__(self, data, N, k):
        self.data = data
        self.N = N
        self.k = k
        
    def __IndexStructureGeneration__(self, embargo_level=10):
        
        #kFold using N  
        kf = KFold(n_splits = self.N, shuffle = False)
        
        #pieces: groups division (equals to N)
        pieces = []
        for train_indices, test_indices in kf.split(self.data):
            pieces.append(test_indices) 
        
        #model_split: total subdatasets along groups and splits
        model_split = int(nCk(self.N,self.k))
        
        #total paths based on k(subtest-group), N & model_split
        n_paths = int(model_split * self.k/self.N)
        
        #item 1 | idxs to construct paths along each group/splits: N elements 
        split_map = split_map_CPCV(self.N, self.k) 
        
        #item 2 | idxs to construct 
        masterSets = master_sets(
            self.N, self.k, split_map, pieces, self.data, embargo_level
        )
        return masterSets, split_map, n_paths
    
    def getResults(self, exogenous_model, endogenous_model, 
                   nonfeature=['horizon'], y_true = True):
        
        #get useful dataset
        featuresLabelData = self.data.drop(nonfeature,axis=1)
        
        #items: [0] masterSets, [1] split_map, [2] n_paths        
        items = self.__IndexStructureGeneration__()
        
        #predictions and y_real | elements to compute performance
        updatedPreds = []
        y_reals = []
        #temp = []
        
        #predicciones_sin_split = []

        #Updating Models Process using 'train' subset (idx[0])         
        for idxs_arrays_in_tuple in items[0]:
            
            #definition of general model: side model + size model
            model = GeneralModel(
                exogenous_model, endogenous_model
            )
            
            #fitting general model with subset of train
            model.fit(
                idxs_arrays_in_tuple[0], featuresLabelData 
            )
            
            #fitting general model with subset of test | predictions + y_real
            predictions, y_real = model.predict(
                idxs_arrays_in_tuple[1], featuresLabelData, y_true
            )
            
            #predicciones_sin_split.append(predictions)
            
            #split side_size_predictions & y_real 
            splitted_predictions = np.array_split(predictions, self.k)
            splitted_y_reals = np.array_split(y_real, self.k)
            
            #include splitted information to each base lists
            updatedPreds.append(splitted_predictions)
            y_reals.append(splitted_y_reals)
            
            #print("CHECK DATA")
            #temp.append(check_data(
            #    idxs_arrays_in_tuple[0], featuresLabelData,
            #) )
        
        #print("CHECKING FETURES DATA EQUALITY") 
        #for idx in range(1,len(temp)):
        #    print("Checking Features & Labels Equality")
            
        #    print( np.unique(temp[idx-1][0] ==  temp[idx][0]))
        #    print( np.unique(temp[idx-1][1] ==  temp[idx][1]))
        #print(" ")
            
        #print("Imprimiendo Split Maps")
        #for mapeo in items[1]:
        #    print(mapeo)
            
        #print("Checking Predictions base model equality")
        #for idx in range(1,len(predicciones_sin_split)):
            
        #    print("Prediccion")
        #    print(predicciones_sin_split[idx])
        #    print(predicciones_sin_split[idx-1])
           
        #    print( predicciones_sin_split[idx-1] == predicciones_sin_split[idx] )
        
        #print("*******************")
            
    
            
        #constuction of final paths (N elements) | predictions
        final_prediction_paths = [
            np.concatenate(path) for path in paths(
                updatedPreds, items[1], items[2])
        ]
        
        #constuction of final paths (N elements) | y_real 
        final_y_reals_paths = [
            np.concatenate(path) for path in paths(
                y_reals, items[1], items[2])
        ]
        
        return final_prediction_paths, final_y_reals_paths