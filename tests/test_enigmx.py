"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import ray
ray.init(include_dashboard=(False),ignore_reinit_error=(True))

from enigmx.classEnigmx import EnigmX

from keras.layers import Dense
from keras.models import Sequential

from sklearn.tree import DecisionTreeClassifier
from enigmx.utils import EquitiesEnigmxUniverse

##############################################################################

# NN Keras model
def kerasModel(num_features):
    model = Sequential()
    model.add(
        Dense(8, input_dim = num_features, activation='relu') 
    )
    model.add(Dense(3, activation='softmax')) 
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model

# dict with models, and params
dict_models = {
    'decisionTree': (
        DecisionTreeClassifier(), 
        {'max_leaf_nodes': list(range(2, 20)), 
         'min_samples_split': [2, 3, 4]} 
        ), 
    'keras': (
        kerasModel, 
        dict(
            batch_size = [10, 20, 40, 60, 80, 100], 
            epochs = [10, 50, 100]
            ) 
        )
     }

##############################################################################

main_path = 'C:/data/'

# EnigmX instance definition
instance = EnigmX(bartype = 'VOLUME', 
                  method = 'MDA', 
                  base_path = main_path,
                  cloud_framework = False
                  ) 

# feature importance

instance.get_feature_importance(    
                       model = DecisionTreeClassifier(), 
                       list_stocks = EquitiesEnigmxUniverse(main_path), 
                       score_constraint = 0.3,
                       server_name = "WINDOWS-NI805M6",
                       database = "BARS_FEATURES",
                       uid = '',
                       pwd = '',
                       driver = "{SQL Server}",
                       kendall_threshold = 0.00000000000001
                      )
    

# get multi process for tunning and backtest
#instance.get_multi_process(
#    code_backtest = '001', 
#    dict_exo_models = dict_models,
#    endogenous_model_sufix= 'rf',    
#    trials = 11, 
#    partitions = 2, 
#    )