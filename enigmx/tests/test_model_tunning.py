"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pickle
from enigmx.betsize import *
from sklearn.tree import DecisionTreeClassifier
from enigmx.utils import dataPreparation_forTuning
from enigmx.model_hypertuning import clfHyperFit


class ModelConstruction(object):
    
    def __init__(self, 
                 stacked_path, 
                 feature_sufix = "feature",                 
                 label_name = "barrierLabel",
                 feat_bartype = "VOLUME", 
                 feat_method = "MDA",
                 timeIndexName = "close_date",
                 datetimeAsIndex = True):
        
        
        assert stacked_path[-1] == '/', "Check path ingested. Should be end in '/'"
        
        self.stacked_path = stacked_path
        self.feature_sufix = feature_sufix
        self.label_name = label_name
        self.feat_bartype = feat_bartype
        self.feat_method = feat_method
        
        self.timeIndexName = timeIndexName
        self.datetimeAsIndex = datetimeAsIndex
    
        
    def __featuresLabelConstruction__(self):
        
        csv_path = "{}STACKED_{}_{}_{}.csv".format(
            self.stacked_path, 
            self.dataReq,
            self.feat_bartype.upper(), 
            self.feat_method.upper()
            )
        
        X, y, t1 = dataPreparation_forTuning(
            csv_path = csv_path, 
            label_name = self.label_name, 
            features_sufix = self.feature_sufix, 
            timeIndexName = self.timeIndexName, 
            set_datetime_as_index = self.datetimeAsIndex 
            )
    
        
        return X, y, t1
    
    def __modelInitialization__(self):
        
        X, y, t1 = self.__featuresLabelConstruction__()
        
        exogenous_model = clfHyperFit(
            feat=X, lbl=y, t1=t1, 
            param_grid=self.dict_params, pipe_clf=self.model
            )
        
        return exogenous_model 

        
    def get_exogenous_model(self, 
                         model, 
                         dic_params,                          
                         save_as_pickle = True, 
                         exogenous_pickle_file_name = "exogenous_model.pkl"):
        
        #tipo de dato requerido
        self.dataReq = "EXO"
        
        #modelo ingestado
        self.model = model
        
        #parametros para tunning del modelo exógeno
        self.dict_params = dic_params        
        
        exogenous_model = self.__modelInitialization__()
        
        if save_as_pickle:
            
            filepath = r'{}{}'.format(
                self.stacked_path, 
                exogenous_pickle_file_name  
                )

            pickle.dump(exogenous_model, open(filepath, 'wb'))
            
            print("---Process Finished! '{}' file was saved in '{}'.\n".format(
                exogenous_pickle_file_name , self.stacked_path)
                )
        
        return exogenous_model
        
    def get_endogenous_model(self, 
                             endogenous_model_sufix,
                             save_as_pickle = True,
                             rebalance = True,
                             test_size = 0.25,
                             balance_method = 'smote', 
                             data_explorer = False, 
                             confusion_matrix = False,
                             dollar_capital = None, 
                             exogenous_pickle_file_name = "exogenous_model.pkl",
                             endogenous_pikcle_file_name = "endogenous_model.pkl"):
        
        # selección del tipo de dato para entrenamiento
        self.dataReq = "ENDO"
        
        # partición del tipo de dato 
        X, y, t1 = self.__featuresLabelConstruction__()
        
        # conversión de dataframe en arrays 
        X, y = X.values, y.values
        
        # generación del nombre/path de modelo exógeno
        exogenous_model_path = r'{}{}'.format(
                self.stacked_path, 
                exogenous_pickle_file_name 
                )        
        
        # 1) Open Exogenous Model
        exogenous_model = pickle.load(open(exogenous_model_path, 'rb'))        
        
        # 2) Predict using Exogenous Model
        exogenous_predictions = exogenous_model.predict(X)
        
        # 3) Ingest Predictions, Features & Real Labels in BetSize instance
        betsizeInstace = BetSize(
            array_features = X, 
            array_predictions = exogenous_predictions, 
            array_labels = y, 
            endogenous_model = endogenous_model_sufix
            )
        
        # 4) Endogenous Metalabelling Model Generation from BetSize Instance
        endogenous_model = betsizeInstace.get_betsize(
            data_explorer = False,
            confusion_matrix = False,
            dollar_capital = None
            )        
        
        if save_as_pickle:        
            
            # 5) Saving Endogenous Metalabelling Model using Pickle
            filepath = r'{}{}'.format(
                self.stacked_path, 
                endogenous_pikcle_file_name
                )
            pickle.dump(endogenous_model, open(filepath, 'wb'))      
            
            print("---Process Finished! '{}' file was saved in '{}'.\n".format(
                endogenous_pikcle_file_name , self.stacked_path)
                )            
        
        return endogenous_model

#################################### TEST ####################################
    
#definition of model for tuning
model_for_tuning = DecisionTreeClassifier()

#definition of parameters for tuning based on model
params_for_tuning = {
    'max_leaf_nodes': list(range(2, 20)), 
    'min_samples_split': [2, 3, 4]
    }

csv_path = "D:/feature_importance/"    
featuresufix = 'bar_cum'

instance = ModelConstruction(stacked_path=csv_path, 
                                      feature_sufix=featuresufix)

print(
instance.get_exogenous_model(model = model_for_tuning, 
                          dic_params = params_for_tuning, 
                          save_as_pickle=True)
)

print(
instance.get_endogenous_model(endogenous_model_sufix = 'rf')
)