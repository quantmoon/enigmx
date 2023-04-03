"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scaling(series):
    """
    Function that scales values with a MinMax method.
    Range values for transformation defined between (0, 1).
    
    Take into consideration that this scaling it's not backward.
    It uses all the data available, no matter it belongs to a timeframework.
    
    Inputs:
        - pd.Series with datavector of values.
    Output:
        - pd.Series with datavector scaled between 0 to 1.
    """
    scalerK = MinMaxScaler()
    scaled_values = scalerK.fit_transform(
        series.values.reshape(-1,1)
    )
    return pd.Series(scaled_values.reshape(-1))

class __QuantopianClass__:
    
    """
    Main Quantopian Class for heuristic models.
    
    This function is the base ingestable object that should be introduced
    in the models computation remotely as a unique param.
    
    Base Input:
        - dataframe with variables information for heuristic computation.
                
    Methods:
        - '__generalOperatorProcess__':
            Allow dataframe selection based on 'length' and 'variableName'.
            
        Methods for variable selection:
            > 'bar_volume'
            > 'basic_volatility'
            > 'open_price'
            > 'high_price'
            > 'low_price'
            > 'close_price'
            > 'open_date'
            > 'high_date'
            > 'low_date'
            > 'close_date'
    """
    
    def __init__(self, dataframe):
        
        # take the dataframe with 'base variables'
        self.dataframe = dataframe
        
        # main error message in case length is irrational
        self.errorLengthMessage = 'Heuristic Length Variable should be >0. Please check'
        
    def __generalOperatorProcess__(self, variableName, length):
        
        assert length>0, self.errorLengthMessage
        if length == 1:
            data = self.dataframe[variableName] 
            
        else:
            data = self.dataframe[variableName].rolling(length)
            
        return data 

    def bar_volume(self, length = 1):
        return self.__generalOperatorProcess__(
            'bar_cum_volume', length
        )
    
    def basic_volatility(self, length = 1):
        return self.__generalOperatorProcess__(
            'basic_volatility', length
        ) 
            
    def open_price(self, length = 1):
        return self.__generalOperatorProcess__(
            'open_price', length
        )     
    
    def high_price(self, length = 1):
        return self.__generalOperatorProcess__(
            'high_price', length
        )   
    
    def low_price(self, length = 1):
        return self.__generalOperatorProcess__(
            'low_price', length
        )
    
    def close_price(self, length = 1):
        return self.__generalOperatorProcess__(
            'close_price', length
        )   

    
class EnigmxHeuristic:
    
    """
    Class that recieves the model created and formated as a 'QuantopianClass'.
    
    This representation works as the base sklearn structure.

    Base Inputs: 
        - 'model_function': model created as a simple function.
        
        Important — this model should be created with a unique param, like e.g.:
        
        '
        def example_model(quantopian):
            critic = np.sqrt(quantopian.bar_volume() /  quantopian.open_price())
            critic = critic.apply(lambda x: 1 if x >100 else (-1 if x<20 else 0))
            return critic:
        '    
        
        where 'quantopian' or another unique string is the base param.
        
        
    Main methods:
        
        - 'fit':
            inputs:
                - x: features matrix 
                - y: label vector 
            output:
                - same base model (tunning or fitting do not exist)
        
        - 'predict':
            inputs: 
                - x: 'features' matrix (pd.Dataframe version with bars info)
            outputs:
                - 'predictions': vector with predictions
    """
    
    def __init__(self, model_function):
        
        # ingesting model 
        self.model_function = model_function
        
        # error shape of predictions
        self.errorShape = "Predictions should be returned as 1D. \
            Check your output-model shape."
        
        # lenght predictions inconsistency
        self.errorInconsistency = "Predictions should return more than one case. \
            Not a single prediction returned."
        
        # object of predictions is not a DataFrame, Series or NumpyNdarray
        self.errorPredictionsType = "Predictions object output type is not allowed \
            Onlty pd.DataFrame, Series or np.ndarray could be used."
            
        # error if categories are not only -1, 0, 1
        self.errorLabelCategories = "Predictions labels are different than estimated \
            Label categories allowed are [-1, 0, 1]."
        
    def __checkingInconsistencies__(self):
        
        # check if preditcions are 1D 
        assert len(self.predictions.shape) == 1,  self.errorShape
        
        # check that sufficient predictions will be returned
        assert self.predictions.shape[0] >= 1, self.errorInconsistency 
        
        # check if predictions object is dataframe, series or array
        assert isinstance(
            self.predictions, (pd.Series, pd.DataFrame, np.ndarray)
            ), self.errorPredictionsType
        
        # check if categories for predictions are different than -1, 0 & 1
        assert np.all(np.unique(self.predictions) == [-1, 0, 1]), self.errorLabelCategories
        
    
    def fit(self, x, y):
        # fit method for model > returns the same model 
        return self.model_function
        
    def predict(self, x):
        
        # open quantopian instance for lecture of heuristic
        quantopianInstance = __QuantopianClass__(
            dataframe=x
        )
        
        # compute predictions
        self.predictions = self.model_function(
            quantopian = quantopianInstance
            )
        
        # checking for possible errors > stop in that case
        self.__checkingInconsistencies__()

        # check if predictions are series or dataframe -> transfotm to numpy
        if isinstance(self.predictions, (pd.Series, pd.DataFrame)):
            return self.predictions.values
        else:
            return self.predictions
