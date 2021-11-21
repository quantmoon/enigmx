"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
import pandas as pd
from time import time
from sklearn.model_selection import KFold
from enigmx.betsize import betsize_livetrading

from keras.models import Sequential

from keras.wrappers.scikit_learn import KerasClassifier
from enigmx.utils import (
    nCk, 
    paths,
    master_sets,
    split_map_CPCV,  
    transformYLabelsVector,
    decodifierNNKerasPredictions
    )

from enigmx.classModelTunning import list_heuristic_elements

class GeneralModel(object):
    """
    Clase general para utilizado de Modelo Macro.
    
    Modelo Macro: modelo exógeno (side) + modelo endógeno (size)
    
    Recrea la estructura de modelo sklearn (fit y predict).
    
    Clase GeneralModel:
        Inputs obligatorios:
            - 'MA' (sklearn, keras, pytorch o pers.): modelo exógeno.
            - 'MB' (RandomForestClassifier o SVC): modelo endógenos.
            
    Métodos centrales:
        - 'fit': recrea el aprendizaje del modelo sobre el modelo exógeno.
            * 'idxs': indices lst para distinción entre features matrix y label vector
            * 'basedata': dataframe o numpy array conteniendo la data para el modelo
        - 'predict': ejecuta la predicción doble (m. exógeno y m. endógeno)
            * 'idxs': indices lst para distinción entre features matrix y label vector 
            * 'basedata': dataframe o numpy array conteniendo la data para el modelo
            * 'y_true': bool para retornar o no el vector del yLabel verdadero
                        útil para calcular métricas de performance.
                        
    Métodos accesitarios:
        - '__dataPreparation__': permite dividir un dataset en featuresMatrix y labelsVector
            * 'idxs': indices lst para distinción entre features matrix y label vector
            * 'basedata': dataframe o numpy array conteniendo la data para el modelo
    """
    def __init__(self, MA, MB):
        
        # modelo exogeno
        self.MA = MA 
        
        # modelo endogeno
        self.MB = MB
    
    def __dataPreparation__(self, idxs, basedata):
        """
        Constraint:
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
                basedata[idxs,:-5].astype(np.float32), 
                basedata[idxs,-1:]
            )        

        # assert len(np.unique(labelsVector)) > 2, \
        #     'Only 2 of 3 labels exist. Increase sample to get more labels.'
    
        # revisamos si el modelo ingresado es una NN de Keras
        if type(self.MA) in (KerasClassifier, Sequential):
            labelsVector = transformYLabelsVector(labelsVector)

        return featuresMatrix, labelsVector
    
    def fit(self, idxs, basedata):
        """
        Método de entrenamiento para Modelo Macro. 
        
        Trabaja solo sobre MA (modelo exógeno)
        """
        #split division to define X and Y (training set)
        X, y = self.__dataPreparation__(idxs, basedata)

        #fitting/update training process in exogenous model (side)
        print(':::: >> Fitting Exogenous Model...')
        
        y=y.astype('int') 

        self.MA.fit(X,y)
    
    def predict(self, idxs, basedata, y_true=False):
        """
        Método de predicción para modelo Macro.
        """
        #split division to define X and Y (testing set)
        X, y_real = self.__dataPreparation__(idxs, basedata)
        
        #prediction process in exogenous model (side)
        predictionA = self.MA.predict(X)
        
        # revisamos si el modelo ingresado es una NN de Keras
        if type(self.MA) in (KerasClassifier, Sequential):
             
            # reduce 3D predictions to single prediction | HAVE TO DO THIS IN LIVETRADING IN CASE USE NN models
            predictionA = np.argmax(predictionA, axis=-1)
            
            # Transform values from 3Label vector to original (-1,0,1) labels
            predictionA  = decodifierNNKerasPredictions(
                predictionA 
                )
        
        #prediction process in endogenous model (size | metalabelling)
        predictionB = betsize_livetrading(
            X, predictionA.reshape(-1,1), self.MB
            )
        
        #if 'y_true', returns Y' (predictions) and Y(real label events)
        if y_true: #useful to calculate the performance metric
            return predictionA, predictionB, y_real 
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
                
                Returns tuple of information:
                    * side predictions by path (N-1) in array format
                    * size predictions by path (N-1) in array format
                    * true label index by path (N-1) in array format
                    * event idx by path (N-1) in array format
                
            Aditional input:
                * 'exogenous_model': base Machine Learning model | side pred.
                * 'endogenous_model': complementary ML model | size pred.
                * 'nonfeature': useless var list to drop from feature matrix.
                
    """
    def __init__(self, data, N, k, embargo_level = 5):
        self.data = data
        self.N = N
        self.k = k
        self.embargo_level = embargo_level
        
    def __IndexStructureGeneration__(self):
        
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
        
        #item 2 | idxs to construct | error here
        masterSets = master_sets(
            self.N, self.k, split_map, pieces, self.data, self.embargo_level
        )
        return masterSets, split_map, n_paths
    
    def getResults(self, exogenous_model, endogenous_model, 
                   nonfeature=['horizon'], y_true = True):
        

        #get useful dataset
        featuresLabelData = self.data.drop(nonfeature, axis=1)
        
        #items: [0] masterSets, [1] split_map, [2] n_paths        
        items = self.__IndexStructureGeneration__()
        
        #predictions and y_real | elements to compute performance
        updatedPreds = []
        y_reals = []
        predCat = []
        indexPredEvent = []
        
        
        #Updating Models Process using 'train' subset (idx[0])         
        for idx,idxs_arrays_in_tuple in enumerate(items[0]):
            
            
            #definition of general model: side model + size model
            t = time()
            model = GeneralModel(
                exogenous_model, endogenous_model
            )
            
            #fitting general model with subset of train
            model.fit(
                idxs_arrays_in_tuple[0], featuresLabelData 
            )
            
            #fitting general model with subset of test | predictions + y_real
            predictionCat, predictions, y_real = model.predict(
                idxs_arrays_in_tuple[1], featuresLabelData, y_true
            )
            
            #split side_size_predictions & y_real 
            splitted_predCat = np.array_split(predictionCat, self.k)
            splitted_predictions = np.array_split(predictions, self.k)
            splitted_y_reals = np.array_split(y_real, self.k)
            splitted_index_y = np.array_split(idxs_arrays_in_tuple[1], self.k)
            
            #include splitted information to each base lists
            predCat.append(splitted_predCat)
            updatedPreds.append(splitted_predictions)
            y_reals.append(splitted_y_reals)
            indexPredEvent.append(splitted_index_y)
    
            print(f"Getting backtest of path {idx}:", time()-t)
        #constuction of final paths (N elements) | simple categorical pred.
        final_prediction_catpath = [
            np.concatenate(path) for path in paths(
                predCat, items[1], items[2])
        ]        
        
        #constuction of final paths (N elements) | betsize predictions
        final_prediction_paths = [
            np.concatenate(path) for path in paths(
                updatedPreds, items[1], items[2])
        ]
        
        #constuction of final paths (N elements) | true label (y)
        #values in each path are the same
        final_y_real_paths = [
            np.concatenate(path) for path in paths(
                y_reals, items[1], items[2])
        ]
        
        #constuction of final paths (N elements) | true index label (y-idx)
        #values in each path are the same
        final_index_y_paths = [
            np.concatenate(path) for path in paths(
                indexPredEvent, items[1], items[2])
        ]
        
        
        return (
            final_prediction_catpath, 
            final_prediction_paths, 
            final_y_real_paths, 
            final_index_y_paths
            )
