"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import pickle
import datetime
import pandas as pd
pd.options.mode.chained_assignment = None  
from enigmx.combinatorial_backtest import CPKFCV
from enigmx.utils import dataPreparation

class EnigmxBacktest(object):
    
    """
    Combinatorial Purged K-Fold Cross Validation Backtest (clase)
    
    Permite computar el CPKFCV en lugar del tradicional back-forward backtest.
    
    Clase EnigmxBacktest:
        Inputs obligatorios:
            - 'base_path' (str): dirección path local alojamiento data y modelos.
            - 'data_file_name' (str): nombre del csv de data para backtest.
        
        Inputs accesitarios:
            - 'feature_sufix' (str): sufijo identificador de los features.
            - 'label_name' (str): nombre identificador del label/etiqueta.
            - 'exogenous_model_name' (str): nombre de modelo exógeno (inc 'pkl') 
            - 'endogenous_model_name' (str): nombre de modelo endógeno (inc 'pckl')
            - 'timeIndexName' (str): nombre de la columna temporal a usarse como index.
            - 'timeLabelName' (str): nombre de la columna horizon del label.
            - 'y_true' (bool): booleano para retornar valores y verdaderos.
            
    Método central de clase EnigmxBacktest:
        - 'get_combinatorial_backtest':
            #################### PARÁMETROS OBLIGATORIOS #################### 
            - n_trials (int)
            - k_partitions (int)
            #################### PARÁMETROS ACCESITARIOS #################### 
            - embargo_level (int > 2, 5 por defecto)
            - max_leverage (int >=1, 2 por defecto)
            - df_format (booleano, True por defecto)
            - save (booleano, True por defecto)
            
            Output: 
                OP1: guardado de archivos en 'base_path'
                OP2: tupla de arrayas con info. de trials.
            
    Métodos accesitarios:
        - '__infoReader__': lectura de archivos y preparación de datos.
        - '__computeCombinatorialBacktest': cómputo de backtest combinatorial
            - n: trials (int)
            - k: partitions (int)
            - embargo_level (int): nivel de embargo entre samples (predf. = 5)
    """
    
    def __init__(self, 
                 base_path, 
                 data_file_name, 
                 feature_sufix = 'feature',
                 label_name = 'barrierLabel',
                 exogenous_model_name = "exogenous_model.pkl",
                 endogenous_model_name = "endogenous_model.pkl",
                 timeIndexName = "close_date",
                 timeLabelName = "horizon",
                 y_true = True):
        
        # ingesta de parámetros a la clase
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
        
        self.timeIndexName = timeIndexName
        self.timeLabelName = timeLabelName
        self.y_true = y_true
        
    def __infoReader__(self):
        
        # carga de modelos desde formato pickle
        self.exogenousModel = pickle.load(open(self.exogenous_path, 'rb'))
        self.endogenousModel = pickle.load(open(self.endogenous_path, 'rb'))     
        
        # carga de csv con data ciega stacked para backtest
        self.dfStacked = pd.read_csv(
            self.csv_path, 
            ).dropna()
        
        # cambio de formato de col de tiempo de str a datetime
        self.dfStacked[self.timeIndexName] = self.dfStacked[self.timeIndexName].astype('datetime64[ns]')
        
        # seteo de datetime como índice
        self.dfStacked = self.dfStacked.set_index(self.timeIndexName)        
        
        # dataset útil para el combinatorial 
        self.dataset = dataPreparation(
            data_csv = self.csv_path, 
            feature_sufix= self.feature_sufix,
            label_name= self.label_name, 
            timeIndexName = self.timeIndexName,
            timeLabelName = self.timeLabelName            
            )
        
    def __computeCombinatorialBacktest__(self, n, k, embargo_level = 5):
        
        assert embargo_level >= 5, "Level of embargo should be >= 5" 
        
        self.__infoReader__()
        
        print("<<<::::: RUNNING COMBINATORIAL PURGED KFOLD CV :::::>>>")

        print("::::::::::::::::: PATHS COMPUTING NOT INC. INFO |>>> ")        
        combinatorial_backtest_instance = CPKFCV(
            data = self.dataset, N = n, k = k, embargo_level = embargo_level
            )        
        
        print("::::::::::::::::: PATHS COMPUTING INC. INFO |>>> ")
        backtestPaths = combinatorial_backtest_instance.getResults(
            exogenous_model = self.exogenousModel, 
            endogenous_model = self.endogenousModel
            )
        
        print("::::::::::::::::: VECTORIZED REPRESENTATION |>>> ")
        # predicción categórica simple
        self.predCat = backtestPaths[0] 
        # predicción betsize
        self.predBetsize = backtestPaths[1] 
        # true label
        self.trueLabel = backtestPaths[2] 
        # index event
        self.indexEvent = backtestPaths[3]
        
    def getCombinatorialBacktest(self, 
                                   n_trials, 
                                   k_partitions, 
                                   embargo_level = 5,
                                   max_leverage = 2,
                                   df_format = True,
                                   save = True):
        
        # llamado de cómputo del combinatorial backtest
        self.__computeCombinatorialBacktest__(
            n = n_trials, k = k_partitions, embargo_level = embargo_level
            )
        
        print(":::::>>> ENDING COMBINATORIAL PURGED KFOLD CV <<<:::::")
        
        if df_format:
            
            list_frame_trials = []
            
            # iteración por partición para la predicción
            for idx, partition in enumerate(self.indexEvent):
                
                dataTemp = self.dfStacked
                
                data = dataTemp.reset_index(drop=False) 
                
                # variables resultantes
                data["predCat"] = self.predCat[idx]
                data["predBetSize"] = self.predBetsize[idx]
                data["leverage"] = data.predBetSize * max_leverage
                data["trial"] = idx
                
                # ordenamiento de la data de forma temporal
                data = data.sort_values(by='close_date')
                
                list_frame_trials.append(data)
            
            # unión de todos los trial (n)
            generalFrame = pd.concat(list_frame_trials).reset_index(drop=True)
            
            if save: 
                
                trialTime = datetime.datetime.now() 
                iDTime = "{}{}{}{}".format(
                    format(trialTime.day, '02'), format(trialTime.month, '02'), 
                    format(trialTime.hour, '02'), format(trialTime.minute, '02')
                    )
                print("::::::::::::::::::::: BACKTEST CODE: {}".format(iDTime))
                
                self.iDTime = iDTime
                
                generalFrame.to_csv(
                    "{}BACKTEST_RESULTS_{}.csv".format(self.base_path, self.iDTime), 
                    index=False
                    ) 
                
                # return general frame with results + id name
                return generalFrame, self.iDTime
            
        else:
            return self.predCat, self.predBetsize, self.trueLabel, self.indexEvent

