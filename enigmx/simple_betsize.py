"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
import pandas as pd
from sklearn.svm import NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


class simpleBetSize:
    def __init__(self,
                 array_predictions, 
                 array_labels,
                 endogenous_model = 'rf',
                 test_size = 0.25):
        
        self.array_predictions = array_predictions #new X
        self.array_labels = array_labels
        self.endogenous_model = endogenous_model
        self.test_size = test_size
        
    def __randomGridVariablesRF__(self): #AGREGAR AQUI MÁS PARÁMETROS!!!
        #parameters for RandomForest RandomGridSearch
          # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(
            start = 100, 
            stop = 1000, 
            num = 1) #10
        ]
        
          # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
    
          # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(
            start = 10, 
            stop = 100, 
            num = 10)
        ]
        max_depth.append(None)
    
          # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
          # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
          # Method of selecting samples for training each tree
        bootstrap = [True, False]
    
          #return random grid dictionary
        return {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}      
    
    def __randomGridVariablesSVM__(self):
        #parameters for SVM RandomGridSearch
        list_nus = np.linspace(0.01,0.99,2)
        list_kernels = ["rbf", "sigmoid"]  
        list_coef0 = [0.0,.1,.5,.9]      
        
        return {'nu': list_nus,
                'kernel': list_kernels,
                'coef0': list_coef0}    
        
    def __endogenousModel__(self):

        if isinstance(self.array_predictions, (pd.Series, pd.DataFrame)):
            self.array_predictions = self.array_predictions.values.reshape(
                -1,1
                )
                
        if len(self.array_predictions.shape) == 1:
            self.array_predictions = self.array_predictions.reshape(
                -1,1
                )
        
        new_array_features = self.array_predictions
        new_array_labels = (self.array_labels!=0)*1    
        
        train_test_object = train_test_split(
            new_array_features, 
            new_array_labels, 
            test_size=self.test_size, 
            random_state=0
            )
        
        (
            new_x_train_res, new_x_test, 
            new_y_train_res, new_y_test
            ) = train_test_object
        
        #uploading to __init__ variables
        self.new_x_test = new_x_test
        self.new_y_test = new_y_test

        if self.endogenous_model == 'rf':
            #RF Dict of parameters  
            random_grid_dictionary = self.__randomGridVariablesRF__()
            
            #Random Forest Classifier 
            rf = RandomForestClassifier()
            
            #RandomGridSearch over RF 
            rf_random = RandomizedSearchCV(
              estimator = rf, 
              param_distributions = random_grid_dictionary, 
              n_iter = 50, 
              cv = 3, 
              verbose=2, 
              random_state=42, 
              n_jobs = -1
            )
            
            rf_random.fit(new_x_train_res, new_y_train_res)
            
            model_selected = rf_random.best_estimator_       
           
            #otherwise, the endogenous model is a SVM
        
        else:
            
            #SVM Dict of parameters  
            random_grid_dictionary = self.__randomGridVariablesSVM__()
          
            #Nu Support Vector Machine
            svm = NuSVC(probability = True)
          
            #RandomGridSearch over NuSVC
            svm_random = RandomizedSearchCV(
                  estimator = svm, 
                  param_distributions = random_grid_dictionary, 
                  n_iter = 25, 
                  cv = 3, 
                  verbose=2, 
                  random_state=42, 
                  n_jobs = -1
                  )      
    
            svm_random.fit(new_x_train_res, new_y_train_res)
    
            model_selected = svm_random.best_estimator_
    
        return model_selected           
    
    def get_betsize(self):
        return self.__endogenousModel__()
            
            
            
            
            
        
    
