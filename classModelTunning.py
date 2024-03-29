"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pickle
#import marshal
#import dill as pickle
import numpy as np

from enigmx.betsize import BetSize
from enigmx.simple_betsize import simpleBetSize

from enigmx.model_hypertuning import clfHyperFit
from enigmx.utils import (
    transformYLabelsVector,
    dataPreparation_forTuning, 
    decodifierNNKerasPredictions,
    data_heuristic_preparation_for_tunning
    )

from keras.models import load_model
from keras.models import Sequential
from enigmx.quantopian import EnigmxHeuristic
from keras.wrappers.scikit_learn import KerasClassifier

list_heuristic_elements = [
    'bar_cum_volume',
    'horizon',
    'basic_volatility',
    'close_price',
    'high_price',
    'low_price',
    'open_price',
    ]

class ModelConstruction(object):
    
    """
    Clase para Tunning y construcción de modelos Exógeno y Endógeno.
    
    Clase ModelConstruction:
        Inputs obligatorios:
             - 'stacked_path' (str): dirección path local alojamiento data stacked.
        
        Inputs accesitarios:
            - 'feature_sufix' (str): sufijo identificador de los features.
            - 'label_name' (str): nombre identificador del label/etiqueta.
            - 'feat_bartype' (str): nombre de barra usada (lectura de CSV).
            - 'feat_method' (str): proceso de Feature Importance usado ('MDA' o 'MDI')
            - 'timeIndexName' (str): nombre de la columna temporal a usarse como index.
            - 'datetimeAsIndex' (bool): inicialización de datetime como índice.
            
    Métodos centrales de clase ModelConstruction:
        - 'get_exogenous_model':
            #################### PARÁMETROS OBLIGATORIOS #################### 
            - model (modelo sklearn o propio, e.g., 'DecisionTreeClasfier()')
            - dic_params (dict for model tunning)
            #################### PARÁMETROS ACCESITARIOS #################### 
            - save_as_pickle (bool)
            - exogenous_pickle_file_name (str): nombre con el que se guardará el modelo en pickle
        
            Output: 
                OP1: guardado de modelo en 'stacked_path'
                OP2: modelo entrenado (side).
        
        - 'get_endogenous_model' (BetSize)
            #################### PARÁMETROS OBLIGATORIOS #################### 
            - endogenous_model_sufix (str): nombre del modelo endógeno útil para guardado pickle.
            
            #################### PARÁMETROS ACCESITARIOS #################### 
            - save_as_pickle (bool)
            - rebalance (bool): rebalanceo de los labels para BetSize
            - test_size (float): tamaño de test para val. de betsize.
            - balance_method (str): método de balanceo (unique: 'smote')
            - data_explorer (bool): activar si se desea exploración de valores BetSize (no model)
            - confusion_matrix (bool): plotear la matriz de confusión
            - dollar_capital (float, defatul None): cant. dinero invertible.
            - update_training_exomodel (bool): actualizar el modelo exógeno luego del betsize.
            - exogenous_pickle_file_name (str): nombre para lect. del modelo exógeno.
            - endogenous_pickle_file_name (str): nombre para guardar el modelo endógeno.
            
    Métodos accesitarios:
        - '__featuresLabelConstruction__': obtención de 'X','y' y 't1'.
        - '__modelInitialization__': inicialización de modelo
    """
    
    def __init__(self, 
                 stacked_path, 
                 feature_sufix = "feature",                 
                 label_name = "barrierLabel",
                 feat_bartype = "VOLUME", 
                 feat_method = "MDA",
                 timeIndexName = "close_date",
                 datetimeAsIndex = True,
                 exo_openning_method_as_h5 = False,
                 heuristic_model = False,
                 list_heuristic_elements = list_heuristic_elements):
        
        # revisión de correcto formato del path
        assert stacked_path[-1] == '/', "Check path ingested. Should be end in '/'"
        
        self.stacked_path = stacked_path
        self.feature_sufix = feature_sufix
        self.label_name = label_name
        self.feat_bartype = feat_bartype
        self.feat_method = feat_method
        
        self.timeIndexName = timeIndexName
        self.datetimeAsIndex = datetimeAsIndex
        
        self.exo_openning_method_as_h5 = exo_openning_method_as_h5
        self.heuristic_model = heuristic_model
        
        self.list_heuristic_elements = list_heuristic_elements
    
        
    def __featuresLabelConstruction__(self):
        
        # construcción del csv path
        csv_path = "{}STACKED_{}_{}_{}.csv".format(
            self.stacked_path, 
            self.dataReq,
            self.feat_bartype.upper(), 
            self.feat_method.upper()
            )
        
        if self.heuristic_model:
            X = data_heuristic_preparation_for_tunning(
                csv_path = csv_path,
                list_heuristic_elements = self.list_heuristic_elements,
                )
            
            y = dataPreparation_forTuning(
                csv_path = csv_path, 
                label_name = self.label_name, 
                features_sufix = self.feature_sufix, 
                timeIndexName = self.timeIndexName, 
                set_datetime_as_index = self.datetimeAsIndex 
                )[1]
            
            return X, y

        else:
            # obtención de matriz de features, vector de label y de timeIndex
            X, y, t1 = dataPreparation_forTuning(
                csv_path = csv_path, 
                label_name = self.label_name, 
                features_sufix = self.feature_sufix, 
                timeIndexName = self.timeIndexName, 
                set_datetime_as_index = self.datetimeAsIndex 
                )
            return X, y, t1
    
    def __modelInitialization__(self):
        
        # revisar si el modelo ingestado es heuristico
        if self.heuristic_model:             
            # retorna simplemente la logica del modelo (no existe entrenamiento)
            print('::::: >> Detected Heuristic Model-Function Ingested...\n')
            
            # ingresa modelo a la class EnigmXHeuristic para dotarle de fit y predict
            return EnigmxHeuristic(self.model) 

        else:
            # ejecuta el proceso de entrenamiento o tunning para sklearn o keras
            X, y, t1 = self.__featuresLabelConstruction__()
            
            # si el modelo ingestado es una funcion no sklearn (keras)
            if hasattr(self.model, '__call__') and self.exo_openning_method_as_h5:
                print('::::: >> Detected Keras Model-Function Ingested...\n')
                
                self.model = KerasClassifier(
                    build_fn = self.model, 
                    verbose = 0, 
                    num_features = X.shape[1])
                
                print('::::: >> KerasClassifier Object Created...\n')
                
            # si es un objecto Sklearn..
            else:
                print('::::: >> Sklearn Model Detected...\n')
                print('::::: >> No KerasClassifier Object Created...\n')
                
                pass
            
            # obtención del modelo exógeno
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
        
        # tipo de dato requerido
        self.dataReq = "EXO"
        
        # modelo ingestado
        self.model = model
        
        # parametros para tunning del modelo exógeno
        self.dict_params = dic_params        
        
        # obtenemos modelo exógeno tuneado
        exogenous_model = self.__modelInitialization__()
        
        # si se desea guardar como pickle
        if save_as_pickle:
            
            # path para guardar modelo como pickle
            filepath = r'{}{}'.format(
                self.stacked_path, 
                exogenous_pickle_file_name  
                )
            
            # si es una red neuronal 
            if type(exogenous_model) in (KerasClassifier, Sequential):
                exogenous_model.model.save('{}.h5'.format(filepath)) 
            
            # si no es una red neuronal 
            else:
                # si es un modelo heuristico
                if self.heuristic_model:
                    
                    pickle.settings['recurse'] = True

                    pickle.dump(exogenous_model, open(filepath, 'wb'))

                # si es un modelo sklearn
                else:
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
                             update_training_exomodel = True,
                             exogenous_pickle_file_name = "exogenous_model.pkl",
                             endogenous_pickle_file_name = "endogenous_model.pkl"):
        
        # selección del tipo de dato para entrenamiento
        self.dataReq = "ENDO"
        
        # revisamos si el modelo es heuristico
        if self.heuristic_model:
            X, y = self.__featuresLabelConstruction__()
            
            y = y.values
                    
        # si no es heuristico
        else:
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
        if self.exo_openning_method_as_h5:
            # keras model
            exogenous_model = load_model(exogenous_model_path + '.h5')
        
        else:
            # sklearn model o heuristico
            if self.heuristic_model:
                
                objectPickle = open(exogenous_model_path, 'rb')
                
                exogenous_model = pickle.load(objectPickle)
                
            else:
                exogenous_model = pickle.load(open(exogenous_model_path, 'rb'))   
                
        # 2) Predict using Exogenous Model
        exogenous_predictions = exogenous_model.predict(X)
        
        # Step: check if exogenous model is a Keras-NN
        if type(exogenous_model) in (KerasClassifier, Sequential):
            
            # Getting predictionCategory by each dimension of array
            exogenous_predictions = np.argmax(exogenous_predictions,axis=-1)
            
            # Transform values from 3Label vector to original (-1,0,1) labels
            exogenous_predictions = decodifierNNKerasPredictions(
                exogenous_predictions
                )
            
            # Preparing encoder to encrypt values for NN re-training
            # update 'y' values for retraining the 
            yforUpdating = transformYLabelsVector(y)
            
            updated_exogenous_model = exogenous_model.fit(X, yforUpdating)
            
        # if model is not a Keras NN (a sklearn Model instead)
        else:
            # just updated the training for the model 
            updated_exogenous_model = exogenous_model.fit(X, y)
        
        print('::>> BetSize computation... ')
        
        # 3) check if model ingested is heuristic to use simplified version of betsize
        if self.heuristic_model:
            betsizeInstace = simpleBetSize(
                array_predictions = exogenous_predictions, 
                array_labels = y,
                endogenous_model = endogenous_model_sufix,
                )          
            
        else:
            # 3) Ingest Predictions, Features & Real Labels in BetSize instance
            betsizeInstace = BetSize(
                array_features = X, 
                array_predictions = exogenous_predictions, 
                array_labels = y, 
                endogenous_model = endogenous_model_sufix,
                )        

        # 4) Endogenous Metalabelling Model Generation from BetSize Instance
        # BaseParams: data_explorer = False | confusion_matrix = False | dollar_capital = None
        endogenous_model = betsizeInstace.get_betsize()        
        
        print('::>> BetSize finished...')
                        
        if save_as_pickle:        
            
            # 5) Saving Endogenous Metalabelling Model using Pickle
            filepath = r'{}{}'.format(
                self.stacked_path, 
                endogenous_pickle_file_name 
                )
            pickle.dump(endogenous_model, open(filepath, 'wb'))      
            
            print("\n---Process Finished! '{}' file was saved in '{}'.\n".format(
                endogenous_pickle_file_name, self.stacked_path)
                )            
            
            if update_training_exomodel:
                                
                filepath = r'{}{}'.format(
                    self.stacked_path, 
                    exogenous_pickle_file_name
                    )
                
                # check if updated exogenous model is keras
                try:
                    type(updated_exogenous_model.model)
                    
                # except AttributeError due the 'model' attribute
                except AttributeError:
                    is_keras = False
                    
                # if 'model' attribute exists...
                else:
                    is_keras = True

                #if type(updated_exogenous_model.model) in (KerasClassifier, Sequential):
                if is_keras:
                    updated_exogenous_model.model.save('{}.h5'.format(filepath))    
                
                else:                             
                    # re-escribiendo el archivo "exogenous_model.pkl"
                    pickle.dump(updated_exogenous_model, open(filepath, 'wb'))    
            
                print("\n---Process Finished! Updated '{}' file was saved in '{}'.\n".format(
                    exogenous_pickle_file_name, self.stacked_path)
                    )                        
            
        return endogenous_model, updated_exogenous_model
