"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
import pandas as pd
from enigmx.classModelTunning import ModelConstruction, list_heuristic_elements
from enigmx.classFeatureImportance import featureImportance 
from enigmx.classCombinatorialPurged import EnigmxBacktest
from enigmx.metrics import metricsByPath, baseCriticsDicforBacktestMetrics 


        
class EnigmX:
    
    """
    Clase EnigmX principal. 
    
    Permite calcular los procesos globales del EnigmX Worflow:
        - Feature Importance (clase 'featureImportance')
        - Model Construction (clase 'ModelConstruction')
        - EnigmX Combinatorial Purged KFold Backtest (clase 'EnigmxBacktest')
        
    Clase EnigmX:
        Inputs obligatorios:
            - 'bartype' (str): tipo de barra.
            - 'method' (str): método de feature importance ('MDA' o 'MDI')
            - 'base_path' (str): path local para almacenamiento/lectura de files.
        
        Inputs accesitarios:
            - 'feature_sufix' (str): sufijo identificador de los features.
            - 'label_name' (str): nombre identificador del label/etiqueta.
            - 'time_label_name' (str): nombre identificador del time del label.
            - 'time_label_index_name' (str): nombre de columna para idx.
            
        Diccionarios de datos genéricos por método. 
        Datos predefinidos como "kwargs" en cada método:
            
            - KfeatureImportanceParams (diccionario de Feature Importance)
                - "pca_comparisson: True"
                - "top_features: 10"
                - "score_constraint": 0.6,
                - "server_name":"DESKTOP-N8JUB39", 
                - "database":'BARS_FEATURES',
                - "rolling_window": 10, 
                - "rolling_type":'gaussian', 
                - "rolling_std": 5, 
                - "rolling": True,
                - "depured": True,
                - "global_range": True,
                - "col_weight_type": 'weightTime',
                - "pca_min_var_expected": 0.05,
                #############Constant Method Params###################
                - "filtering_featimp": True,
                - "save_featimp": True,
                - "split_featimp": True,
                - "pct_split_featimp": 0.6,                
                
            - KModelConstructionParams (diccionario de Construcción de Modelos)
                - "datetimeAsIndex": True,
                ############# Constant Method Exogenous Model Params #############
                - "exo_model": None,
                - "exo_dic_params": None,
                - "save_exo_as_pickle": True,
                - "exogenous_pickle_file_name": "exogenous_model.pkl",
                ############# Constant Method Endogenous Model Params #############
                - "endogenous_model_sufix": 'rf', #predefined changable
                - "save_endo_as_pickle": True,
                - "rebalance": True, 
                - "test_size": 0.25,
                - "balance_method": "smote",
                - "data_explorer": False,
                - "confusion_matrix": False,
                - "dollar_capital": None,
                - "update_training_exomodel": True,
                - "endogenous_pickle_file_name": "endogenous_model.pkl"            
            
            - KCombinatorialBacktestParams (diccionario de Combinatorial Backtest)
                - "data_file_name": "STACKED_BACKTEST_{}_{}.csv".format(
                    self.bartype.upper(),
                    self.method.upper(),
                    ),
                ############# Constant Combinatorial Backtest Params #############
                - "features_sufix": self.features_sufix,
                - "label_name": self.label_name,
                - "exogenous_pickle_file_name": "exogenous_model.pkl",
                - "endogenous_pickle_file_name": "endogenous_model.pkl",
                - "y_true": True,
                ############# Constant Method Combinatorial Backtest Params #############
                - "embargo_level": 5, #predefined changable
                - "max_leverage": 2,
                - "df_format": True, 
                - "save_combinatorial": True
                
        Métodos de clase EnigmX:
            
            - 'get_feature_importance': 
                - 'model': sickit learn model o modelo para Feat Importance
                - 'list_stocks': lista con nombre de acciones (str)
                    - **kwargs - diccionario 'KfeatureImportanceParams'
                    
            - 'get_model_tunning': 
                - 'exo_process': booleano para tunning de modelo exógeno.
                - 'endo_process': booleanos para tunning de modelo endógeno.
                    - **kwargs - diccionario 'KModelConstructionParams'
                        - añadir modelo* (None en diccionario base)
                    
            - 'get_combinatorial_backtest'
                - 'trials': int con el valor de 'n'
                - 'partitions': particiones con el valor de 'k'
                    - **kwargs - diccionario 'KCombinatorialBacktestParams'
                
        Outputs:
            - Elementos guardados en dirección 'base_path'
            
        Important:
            - 'uid' & 'pwd' are defined for SQL-Cloud in 
               'get_feature_importance'.
               
               Change if you run this in local-device.
    """
    
    def __init__(self, 
                 bartype, 
                 method, 
                 base_path,
                 cloud_framework = True,
                 features_sufix = "feature", 
                 label_name = "barrierLabel",
                 time_label_name = "horizon",
                 time_label_index_name = "close_date"):
        
        # definimos parámetros generales para los 3 procesos
        self.bartype = bartype
        self.method = method
        self.base_path = base_path
        self.features_sufix = features_sufix
        self.label_name = label_name
        self.time_label_name = time_label_name
        self.time_label_index_name = time_label_index_name 
        self.cloud_framework = cloud_framework
        
        # diccionario con los parámetros generales del Feature Importance
        KfeatureImportanceParams = {
            "pca_comparisson": True,
            "pval_kendall": 0.5,
            "score_constraint": 0.6, 
            "driver": "{ODBC Driver 17 for SQL Server}", #change for local SQL
            "uid":"sqlserver", #change for local SQL
            "pwd":"J7JA4L0pwz0K56oa",#change for local SQL
            "server_name":"34.67.233.155", #change for local SQL
            "database":'BARS_FEATURES',
            "rolling_window": 10, 
            "rolling_type":'gaussian', 
            "rolling_std": 5, 
            "rolling": True,
            "depured": True,
            "global_range": True,
            "col_weight_type": 'weightTime',
            "pca_min_var_expected": 0.05,
            "select_sample" : True,
            "combinations_on" : 30,
            "n_samples" : 10,
            #############Constant Method Params###################
            "filtering_featimp": True,
            "save_featimp": True,
            "split_featimp": True,
            "pct_split_featimp": 0.6,
            }
        
        self.KfeatureImportanceParams = KfeatureImportanceParams        
        
        # diccionario con los parámetros generales del Model Tunning
        KModelConstructionParams = {
            "datetimeAsIndex": True,
            ############# Constant Method Exogenous Model Params #############
            "exo_model": None,
            "exo_dic_params": None,
            "save_exo_as_pickle": True,
            "exogenous_pickle_file_name": "exogenous_model.pkl",
            ############# Constant Method Endogenous Model Params #############
            "endogenous_model_sufix": 'rf', #predefined changable
            "save_endo_as_pickle": True,
            "rebalance": True, 
            "test_size": 0.25,
            "balance_method": "smote",
            "data_explorer": False,
            "confusion_matrix": False,
            "dollar_capital": None,
            "update_training_exomodel": True,
            "endogenous_pickle_file_name": "endogenous_model.pkl",
            "exo_openning_method_as_h5": False,
            "heuristic_model": False, 
            "list_heuristic_elements": list_heuristic_elements
            }
        
        self.KModelConstructionParams = KModelConstructionParams
        
        # diccionario con los parámetros generales del Combinatorial Backtest
        KCombinatorialBacktestParams = {
            "data_file_name": "STACKED_BACKTEST_{}_{}.csv".format(
            self.bartype.upper(),
            self.method.upper(),
            ),
            ############# Constant Combinatorial Backtest Params #############
            "features_sufix": self.features_sufix,
            "label_name": self.label_name,
            "exogenous_pickle_file_name": "exogenous_model.pkl",
            "endogenous_pickle_file_name": "endogenous_model.pkl",
            "y_true": True,
            ############# Constant Method Combinatorial Backtest Params #############
            "embargo_level": 5, #predefined changable
            "max_leverage": 2,
            "df_format": True, 
            "save_combinatorial": True,
            "dict_critics_for_metrics": baseCriticsDicforBacktestMetrics,
            "exo_openning_method_as_h5": False,
            "heuristic_model": False,
            "list_heuristic_elements": list_heuristic_elements
            }
        
        self.KCombinatorialBacktestParams = KCombinatorialBacktestParams
        
        # diccionario con los parámetros generales para el Multi Tunning Backtest
        multiTuningParams= {
            **self.KCombinatorialBacktestParams, 
            **self.KModelConstructionParams
            }
        
        self.multiTuningParams = multiTuningParams          
    
    # método feature importance: model y list_stocks como parámetros obligatorios    
    def get_feature_importance(self, model, list_stocks, **kwargs):
        
        assert len(list_stocks) >= 1, "Empty 'list_stocks' is not allowed." 
        
        for (prop, default) in self.KfeatureImportanceParams.items():
            setattr(self, prop, kwargs.get(prop, default))
        
        # instancia feature importance
        instance = featureImportance(
            model = model, 
            method = self.method, 
            driver = self.driver, 
            uid = self.uid, 
            pwd = self.pwd,
            list_stocks = list_stocks,
            pca_comparisson = self.pca_comparisson,
            cloud_framework = self.cloud_framework, #cloud activation
            pval_kendall = self.pval_kendall,
            score_constraint= self.score_constraint,
            server_name = self.server_name,
            database = self.database,
            rolling_window = self.rolling_window,
            rolling_type = self.rolling_type,
            rolling_std = self.rolling_std,
            pictures_pathout = self.base_path,
            bartype = self.bartype,
            rolling = self.rolling,
            depured = self.depured,
            global_range = self.global_range,
            features_sufix = self.features_sufix,
            col_weight_type = self.col_weight_type,
            col_t1_type = self.time_label_name,
            col_label_type = self.label_name,
            pca_min_var_expected = self.pca_min_var_expected
            )        
        
        # resultado del feature importance (dataframe)
        valueResultFeatImp = instance.get_relevant_features(
            filtering = self.filtering_featimp,
            save = self.save_featimp,
            split = self.split_featimp, 
            pct_split = self.pct_split_featimp
            )
        
        # si no se pide guardar, retornar dataframe
        if not self.save_featimp:
            return valueResultFeatImp
        
    # método model tunning: activación boleana y modelo como entradas
    def get_model_tunning(self, exo_process, endo_process, **kwargs):
        
        for (prop, default) in self.KModelConstructionParams.items():
            setattr(self, prop, kwargs.get(prop, default))        
        
        # instancia model tunning
        instance = ModelConstruction(
            stacked_path = self.base_path,
            feature_sufix = self.features_sufix,
            label_name = self.label_name,
            feat_bartype = self.bartype,
            feat_method = self.method,
            timeIndexName = self.time_label_index_name,
            datetimeAsIndex = self.datetimeAsIndex,
            exo_openning_method_as_h5 =  self.exo_openning_method_as_h5,
            heuristic_model = self.heuristic_model
            )
    
        # si se activa tunning para modelo exógeno    
        if exo_process:
            assert self.exo_model != None, "'exo_model' is not defined."
            
            # modelo exógeno retornado 
            valueResultExoModel = instance.get_exogenous_model(
                model = self.exo_model,
                dic_params = self.exo_dic_params,
                save_as_pickle = self.save_exo_as_pickle,
                exogenous_pickle_file_name = self.exogenous_pickle_file_name
                )

            # no se desea guardar, retorna el modelo
            if not self.save_exo_as_pickle:
                return valueResultExoModel
        
        # si se activa tunning para modelo endógeno
        if endo_process:
            
            # modelo endógeno retornado
            valueResultEndoModel = instance.get_endogenous_model(
                endogenous_model_sufix = self.endogenous_model_sufix,
                save_as_pickle = self.save_endo_as_pickle,
                rebalance = self.rebalance,
                test_size = self.test_size,
                balance_method = self.balance_method, 
                data_explorer = self.data_explorer, 
                confusion_matrix = self.confusion_matrix,
                dollar_capital = self.dollar_capital, 
                update_training_exomodel = self.update_training_exomodel,
                exogenous_pickle_file_name = self.exogenous_pickle_file_name,
                endogenous_pickle_file_name= self.endogenous_pickle_file_name
                )
            
            # no se desea guardar, retorna el modelo
            if not self.save_endo_as_pickle:
                return valueResultEndoModel
    
    # método combinatorial purgked kfold cv: 'n' y 'k' como valores obligatorios
    def get_combinatorial_backtest(self, trials, partitions, **kwargs): 
        
        for (prop, default) in self.KCombinatorialBacktestParams.items():
            setattr(self, prop, kwargs.get(prop, default))        
                
        # definición de instancia base del backtest
        instance = EnigmxBacktest(
            base_path = self.base_path,
            data_file_name = self.data_file_name,
            feature_sufix = self.features_sufix,
            label_name = self.label_name,
            exogenous_model_name = self.exogenous_pickle_file_name,
            endogenous_model_name = self.endogenous_pickle_file_name,
            timeIndexName = self.time_label_index_name,
            timeLabelName = self.time_label_name,
            y_true = self.y_true,
            exo_openning_method_as_h5 = self.exo_openning_method_as_h5,
            heuristic_model = self.heuristic_model
            )
        
        # resultado del combinatorial purged KFold CV (dataframe)
        resultValueCombinatorial = instance.getCombinatorialBacktest(
            n_trials = trials,
            k_partitions = partitions,
            embargo_level = self.embargo_level,
            max_leverage = self.max_leverage,
            df_format = self.df_format,
            save = self.save_combinatorial
            )
        
        print(":::> Computing Backtest Statistics... ")
        
        # retorna el df con la prediccion del backtest y el codigo serial para identf.
        combinatorialDataframe, backtestCode = resultValueCombinatorial 
        
        # transforma el label index name a formato string
        combinatorialDataframe[self.time_label_index_name] \
                = combinatorialDataframe[self.time_label_index_name].astype(str)  
        
        # transforma el df combinatorial base a un df de statistical metrics
        dfMetrics = metricsByPath(
                combinatorialDataframe, 
                crits = self.dict_critics_for_metrics
                )
        
        # si se activa la opcion de guardado en local
        if self.save_combinatorial:
            
            print(":::> Saving Metrics csv...")
            
            dfMetrics.to_csv(
                        "{}BACKTEST_METRICS_{}.csv".format(
                            self.base_path, backtestCode
                            ), 
                        index=False
                        ) 
            
            print("||| :::: EnigmX Process Ended :::: |||")
        
        # si no se elige guardar, retorna la tupla de información
        else:
            return dfMetrics, combinatorialDataframe
        
    @ray.remote
    def __combinedGeneralMultiModel__(self, 
                                 ############## Ray value Objects #############
                                 modelName, 
                                 tupleInfoModel, 
                                 ##############################################
                                 endogenous_model_sufix,
                                 endogenous_pickle_file_name,
                                 trials,
                                 partitions, 
                                 ############ hyperparams for tunning #########
                                 datetimeAsIndex,
                                 save_exo_as_pickle,
                                 save_endo_as_pickle,
                                 rebalance,
                                 test_size,
                                 balance_method,
                                 data_explorer,
                                 confusion_matrix,
                                 dollar_capital,
                                 update_training_exomodel,
                                 ############ hyperparams for backtest ########
                                 features_sufix,
                                 label_name,
                                 y_true,              
                                 embargo_level,
                                 max_leverage,
                                 df_format, 
                                 dict_critics_for_metrics
                                 ):
        """
        Function intermediadora que reune los prorcesos de Tunning + Backtest.
        
        Unifica el proceso de parallelizacion que se realice por c/u modelos.
        
        Ver method 'get_multi_process' para conocer mayores detalles.
        """
        
        # selecting model and params from tuple in iteration
        tempModel, paramsModel = tupleInfoModel[0], tupleInfoModel[1]

        # setting exogenous PickleFileName for lecture
        exogenousPickleFileName = modelName + '_exo_model'
            
        # setting endogenous PickleFileName for lecture
        endogenousPickleFileName = endogenous_model_sufix + '_joined_with_' \
            + modelName + '_' + endogenous_pickle_file_name
        
        # define statements for process model and data based on model type
        if 'keras' in modelName:
            self.exo_openning_method_as_h5 = True
            
        if 'heuristic' in modelName:
            self.heuristic_model = True
            
        # initializing model tuning
        self.get_model_tunning(
                datetimeAsIndex = datetimeAsIndex,
                ############# Constant Method Exogenous Model Params #############
                exo_model = tempModel,
                exo_dic_params = paramsModel,
                save_exo_as_pickle = save_exo_as_pickle,
                exogenous_pickle_file_name = exogenousPickleFileName,
                ############# Constant Method Endogenous Model Params #############
                endogenous_model_sufix = endogenous_model_sufix,
                save_endo_as_pickle =  save_endo_as_pickle,
                rebalance = rebalance, 
                test_size = test_size,
                balance_method = balance_method,
                data_explorer = data_explorer,
                confusion_matrix = confusion_matrix,
                dollar_capital = dollar_capital,
                update_training_exomodel = update_training_exomodel,
                endogenous_pickle_file_name = endogenousPickleFileName,
                ##################### Params that can't be changed #####################
                exo_process = True, 
                endo_process = True,
                #################### Params for Model Understanding ####################
                exo_openning_method_as_h5 =  self.exo_openning_method_as_h5,
                heuristic_model = self.heuristic_model
            )
            
        # initializing combinatorial backtest
        dfMetrics, dfBacktest = self.get_combinatorial_backtest(
                ############# Constant Combinatorial Backtest Params #############
                features_sufix = features_sufix,
                label_name = label_name,
                exogenous_pickle_file_name = exogenousPickleFileName,
                endogenous_pickle_file_name = endogenousPickleFileName,
                y_true = y_true,
                ############# Constant Method Combinatorial Backtest Params #############                
                embargo_level = embargo_level,
                max_leverage = max_leverage,
                df_format = df_format, 
                save_combinatorial = False, # don't save combinatorial, only return
                dict_critics_for_metrics = dict_critics_for_metrics,
                ############## Params ingested in the recent function #################
                trials= trials,
                partitions= partitions,
                exo_openning_method_as_h5 =  self.exo_openning_method_as_h5,
                heuristic_model = self.heuristic_model
                )
            
        # setting name of models
        dfMetrics['model_name'] = modelName
        dfBacktest['model_name'] = modelName
        
        return dfMetrics, dfBacktest
    
    
    def get_multi_process(self, 
                          code_backtest,
                          dict_exo_models, 
                          endogenous_model_sufix,
                          trials, 
                          partitions, 
                          **kwargs):
        """
        Descripción General:
        
        El presente método reúne los procesos de Tunning y Backtest en uno solo.
        Esto permite generar un proceso iterativo para muchos modelos.
        
        Por tanto, el presente método es una combinación de los sig. sub métodos:
            - 'get_model_tunning'
            - 'get_combinatorial_backtest'
            
        La aplicacion de estos se encuentra paralelizada a traves de Ray.
       
        El proceso de los mismos permitirá que se guarden los pasos intermedios.
        Estos pasos intermedios son:
            - 'get_model_tunning' -> guarda dos modelos en formato 'pkl'.
            - 'get_combinatorial_backtest' -> guarda CSV resultante de 
                                              cada meta-modelo 
                                              ('pkl' exógeno y 'pkl' endógeno)
        
        C/u de estos sub-métodos toma el 'CSV' respectivo de su dataset
        almacenados en espacio local/cloud para computar sus procesos req.
        
        Finalmente, el presente metodo guardara 2 cvs:
            1. CSV de las metricas por modelo
            2. CSV de los resultados del backtest por modelo
            
        Estos archivos se guardaran en el 'base_path' definido.
        
        La iteración de esta etapa será ejecutada con RAY para una optimización
        en el tunning, almacenado y predicción que puedan realizar los modelos.
        
        Inputs:
            - 'dict_exo_models': diccionario para tunning conteniendo la sig.info,
                ... = {
                    'NOMBRE DE MODELO 1': (modelObject1, paramsTunningDict1)
                    'NOMBRE DE MODELO 2': (modelObject2, paramsTunningDict2)
                    ...
                    'NOMBRE DE MODELO N': (modelObjectN, paramsTunningDictN)
                    }
            - 'endogenous_model_sufix': sufijo del modelo endógeno seleccionado
            - 'trials': valor 'n' para el Combinatorial Backtest
            - 'partitions': valor 'K' para el Combinatorial Backtest

        Output:
            - 'csv' jerárquico en path local/cloud para metrics.
            - 'csv' jerárquico en path local/cloud para backtest.
        """        
        
        print('\n      ::::: >>>> MultiProcess for CPKF as Backtesting started...')
        
        # ingesting ModelTunning Params + BacktestParams
        for (prop, default) in self.multiTuningParams.items():
            setattr(self, prop, kwargs.get(prop, default))
        
        # compute parallelized multiProcess for many models: get dfBacktest & dfMetrics
        listModelBacktestMetrics = [
            self.__combinedGeneralMultiModel__.remote(self, 
                                 ############## Ray value Objects #############
                                 modelName = model_name, 
                                 tupleInfoModel = tuple_info_model, 
                                 ##############################################
                                 endogenous_model_sufix = endogenous_model_sufix,
                                 endogenous_pickle_file_name = self.endogenous_pickle_file_name,
                                 trials = trials,
                                 partitions = partitions, 
                                 ############ hyperparams for tunning #########
                                 datetimeAsIndex = self.datetimeAsIndex,
                                 save_exo_as_pickle = self.save_exo_as_pickle,
                                 save_endo_as_pickle = self.save_endo_as_pickle,
                                 rebalance = self.rebalance,
                                 test_size = self.test_size,
                                 balance_method = self.balance_method,
                                 data_explorer = self.data_explorer,
                                 confusion_matrix = self.confusion_matrix,
                                 dollar_capital = self.dollar_capital,
                                 update_training_exomodel = self.update_training_exomodel,
                                 ############ hyperparams for backtest ########
                                 features_sufix = self.features_sufix,
                                 label_name = self.label_name,
                                 y_true = self.y_true,              
                                 embargo_level = self.embargo_level,
                                 max_leverage = self.max_leverage,
                                 df_format = self.df_format, 
                                 dict_critics_for_metrics = self.dict_critics_for_metrics
                                 ) 
        for model_name, tuple_info_model in dict_exo_models.items()
        ]
        
        # get datasets from ray objects: tuple info for each Obj
        list_datasets =  ray.get(listModelBacktestMetrics)
        
        # concadenation of metrics results
        dataset_metrics = pd.concat([
            infoTupleModels[0] for infoTupleModels in list_datasets
            ]).reset_index(drop=True)
        
        # concadenation of backtest results
        dataset_backtests = pd.concat([
            infoTupleModels[1] for infoTupleModels in list_datasets
            ]).reset_index(drop=True)
        
        # setting time as string to preserve millisecond information
        dataset_backtests[self.time_label_index_name] \
                = dataset_backtests [self.time_label_index_name].astype(str)          
        
        # saving metrics csv's 
        dataset_metrics.to_csv(
            '{}METRICS_TRIAL_{}.csv'.format(self.base_path, code_backtest), 
            index=False
            ) 
        
        # saving backtests results csv's
        dataset_backtests.to_csv(
            '{}BACKTEST_TRIAL_{}.csv'.format(self.base_path, code_backtest), 
            index=False
            )        