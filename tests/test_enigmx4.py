"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import ray
ray.init(include_dashboard=(False),ignore_reinit_error=(True), num_cpus=3)

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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neural_network import MLPClassifier

from enigmx.utils import EquitiesEnigmxUniverse

##############################################################################

# NN Keras model
def kerasModel(num_features):
    # def. sequential model
    model = Sequential()
    # first embedding
    model.add(Dense(8, input_length=num_features, activation='relu'))
    # cetnral layer arch.
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    # exit layer
    model.add(Dense(8, input_dim = num_features, activation='relu'))
    # final exit 
    model.add(Dense(3, activation='softmax'))
    
    # Compile model
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    return model
# dict with models, and params
dict_models = {
     'svm' :
	(SVC(), 
	 {'C' : [.8,1],
	  'kernel' : ['poly','rbf','sigmoid']})
     }
    

##############################################################################

main_path = '/var/data/data/'
#code = input('Ingresa el n° de serie de este intento: ')
#code = '_estacionario_08'
#variables = input('Por favor ingresa las variables con las que se va a construir el modelo (solo separados por comas): ')
variables = "feature_alpha53,feature_alpha41,feature_alpha46,feature_technical_slow_stochastic,feature_technical_choppiness,feature_microstructural_vpin,feature_microstructural_amihud,feature_microstructural_roll,feature_technical_atr"
# EnigmX instance definition
instance = EnigmX(bartype = 'VOLUME', 
                  method = 'MDA', 
                  base_path = main_path,
                  cloud_framework = True,
                  server_name = "34.71.157.141",
                  stationary_stacked = True,
                  features_database = "BARS_FEATURES",
                  uid = ['sqlserver'],
                  pwd = ['quantmoon2021'],
                  driver = [("{ODBC DRIVER 17 for SQL Server}")],
                  ) 

# feature importance
#instance.get_feature_importance(    
#                      model = RandomForestClassifier(max_features=1, random_state=0), 
#                      model = SGDClassifier(loss='log'), #para MDA
#                      list_stocks = stocks,
#                      #list_stocks = ['INFN','KRA','LCII','LUNA'],
#                      score_constraint = 0.3, #activar 
#                      server_name = "34.71.157.141",
#                      database = "BARS_FEATURES",
#                      uid = ['sqlserver'],
#                      pwd = ['quantmoon2021'],
#                      driver = [("{ODBC DRIVER 17 for SQL Server}")],
#                      trial = code,
#                      pval_kendall = 0.1,
#                      k_min = 10,
#                      n_samples = 15,
#                      cutpoint = 0.8
#                      )
   

# get multi process for tunning and backtest
instance.get_multi_process(
     code_backtest = 2, 
     dict_exo_models = dict_models,
     endogenous_model_sufix= 'rf',    
     trials = 20, 
     partitions = 2, 
     cloud_instance = True,
     variables = variables
     )
 # extraemos las métricas del combinatorial
#instance.get_metrics(
#    code_backtest = '100'
#    )