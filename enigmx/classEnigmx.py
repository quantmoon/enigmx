"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
from enigmx.classModelTunning import ModelConstruction
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
            "top_features": 10,
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
            "endogenous_pickle_file_name": "endogenous_model.pkl"
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
            "dict_critics_for_metrics": baseCriticsDicforBacktestMetrics
            }
        
        self.KCombinatorialBacktestParams = KCombinatorialBacktestParams
    
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
            top_features = self.top_features,
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
            datetimeAsIndex = self.datetimeAsIndex
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
            y_true = self.y_true
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
        
        if self.save_combinatorial:
            print(":::> Computing Backtest Statistics... ")
            
            combinatorialDataframe, backtestCode = resultValueCombinatorial 
            
            combinatorialDataframe[self.time_label_index_name] \
                = combinatorialDataframe[self.time_label_index_name].astype(str)  
            
            dfMetrics = metricsByPath(
                combinatorialDataframe, 
                crits = self.dict_critics_for_metrics
                )
            
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
            return resultValueCombinatorial
