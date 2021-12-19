"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

# import ray
# ray.init(include_dashboard=(False),ignore_reinit_error=(True), num_cpus=2)

from enigmx.classEnigmx import EnigmX
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from sklearn.tree import DecisionTreeClassifier

from enigmx.tests.stocks import stocks

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier

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
    'randomForest' :(
        RandomForestClassifier(),
        {'max_leaf_nodes': list(range(2, 20)), 
         'min_samples_split': [2, 3, 4],
         'max_features' : [1],
         'max_samples': [20],
         'n_estimators' : [10]} 
        ),
    'stochasticGradient' : 
        (SGDClassifier(), 
         {'loss':['log', 'modified_huber'],
          'penalty':['l2', 'l1', 'elasticnet'],
          'max_iter':list(range(1,10))}),
     }

##############################################################################

main_path = '/var/data/data/'
#code = input('Ingresa el n° de serie de este intento: ')
code = '_estacionario_08'
#variables = input('Por favor ingresa las variables con las que se va a construir el modelo (solo separados por comas): ')
variables = "feature_technical_EMA_10_signal, feature_technical_sar_signal, feature_technical_bollinger_band_integer, feature_technical_bollinger_volatility_compression"
# EnigmX instance definition
instance = EnigmX(bartype = 'VOLUME', 
                  method = 'MDI', 
                  base_path = main_path,
                  cloud_framework = True,
                  server_name = "34.71.157.141",
                  stationary_stacked = True,
                  features_database = "BARS_FEATURES",
                  uid = "sqlserver",
                  pwd = 'quantmoon21',
                  driver = ("{ODBC DRIVER 17 for SQL Server}"),
                  ) 

# feature importance
instance.get_feature_importance(    
                      model = RandomForestClassifier(max_features=1, random_state=0), 
#                      model = SGDClassifier(loss='log'), #para MDA
                      list_stocks = stocks,
                      #list_stocks = ['INFN','KRA','LCII','LUNA'],
                      score_constraint = 0.3, #activar 
                      server_name = "34.71.157.141",
                      database = "BARS_FEATURES",
                      uid = ['sqlserver'],
                      pwd = ['quantmoon2021'],
                      driver = [("{ODBC DRIVER 17 for SQL Server}")],
                      trial = code,
                      pval_kendall = 0.1,
                      k_min = 10,
                      n_samples = 15,
                      cutpoint = 0.8
                      )
   

# get multi process for tunning and backtest
# instance.get_multi_process(
#     code_backtest = 100, 
#     dict_exo_models = dict_models,
#     endogenous_model_sufix= 'rf',    
#     trials = 11, 
#     partitions = 2, 
#     cloud_instance = False,
#     variables = variables
#     )

# # extraemos las métricas del combinatorial
#instance.get_metrics(
#    code_backtest = '100'
#    )