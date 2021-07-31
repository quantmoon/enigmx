"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import sys
import ray
import pandas as pd
import random
from enigmx.utils import enigmxSplit, kendall_evaluation
from enigmx.features_algorithms import FeatureImportance
from scipy.stats import kendalltau
from itertools import combinations
from enigmx.purgedkfold_features import plotFeatImportance
from time import time


def sorts(i):
    i.sort()
    return i

class featureImportance(object):
    
    """
    Clase madre featureImportance para Selección de Features.
    
    Clase FeatureImportance:
        Inputs Obligatorios:
            - 'model' (modelo sklearn o propio, e.g., 'DecisionTreeClasfier()') 
                    Este modelo debe estar en relación con el siguiente paaram:
                        'method'.
            - 'method' (str): proceso de Feature Importance selecc. ('MDA' o 'MDI')
            - 'list_stocks' (lst): lista de str con stock-names 
            - 'pca_comparisson' (bool): filtrado de los top N con base al PCA
        
        Inputs accesitarios:
            - 'top_features' (int): top N features seleccionados
            - 'score_constraint' (float): 0.0<val<1 detiene proceso si val > modelKFold score  
            - 'server_name' (str): nombre del servidor SQL.
            - 'database' (str): nombre de la base de datos SQL para ext. datos.
            - 'rolling_window' (int): ventana para normalización de stacked.
            - 'rolling_type' (str): método de normalización de stakced.
            - 'rolling_std' (int): param extra por si norm method es gaussian.
            - 'pictures_pathout' (str): path para guardar las imágenes de control.
            - 'bartype' (str): tipo de barra usada. Útil para llamado de tablas.
            - 'rolling' (bool): activar el proceso de norm para stacked.
            - 'depured' (bool): activar proceso de depurado | true: selecc solo features
            - 'global_range' (bool): stacked global df. Útil para llamado desde SQL.
            - 'feature_sufix' (str): sufijo identificador de los features.
            - 'col_weight_type' (str): nombre identificar de la columna de sample weights.
            - 'col_t1_type' (str): nombre de columna de timeHorizon del label.
            - 'col_label_type' (str): nombre de la columna de label. 
            - 'pca_min_var_expected' (float): valor pca (set 0.05) de variación mín. esperado.
        
    Métodos centrales de clase featureImportance:
        - 'get_relevant_features': 
            #################### PARÁMETROS ACCESITARIOS ####################
            - filtering (bool): devolver el df con features filtrado.
            - save (bool): guardar el csv en el 'picture_pathout'.  
            - split (bool): dividir el general stacked df en 3 sub df.
                * dataframe para tunning del modelo exógeno
                * dataframe para tunning dl modelo endógeno
                * dataframe para combinatorial purged KFold Backtest
    
    Métodos accesitarios:
        - '__instanceOverture__': asignación de params a instancia FeatureImportance
                                  y obtención de matriz stacked con feat filtrados.
    """
    
    def __init__(self, 
                 model, 
                 method, 
                 list_stocks,
                 pca_comparisson,
                 cloud_framework = True,
                 pval_kendall = 0.5,
                 score_constraint = 0.6,
                 server_name = "DESKTOP-N8JUB39", 
                 driver="{ODBC Driver 17 for SQL Server}",
                 uid="sqlserver",
                 pwd="J7JA4L0pwz0K56oa",
                 database = 'BARS_FEATURES',
                 rolling_window = 10, 
                 rolling_type = 'gaussian', 
                 rolling_std = 5, 
                 pictures_pathout = "D:/feature_importance/",
                 bartype='VOLUME',
                 rolling = True,
                 depured = True,
                 global_range = True,
                 features_sufix = "feature",
                 col_weight_type = 'weightTime',
                 col_t1_type  = 'horizon',
                 col_label_type = 'barrierLabel', 
                 pca_min_var_expected = 0.05,
                 select_sample = True,
                 n_samples = 10,
                 clustered_features = False,
                 k_min = 5
                 ):
        
        # ingesta de parámetros
        self.model = model
        self.list_stocks = list_stocks
        self.method = method.upper()
        self.score_constraint = score_constraint #ACTIVAR!
        self.driver = driver
        self.uid = uid
        self.pwd = pwd

        self.pval_kendall = pval_kendall
        self.pca_comparisson = pca_comparisson
        
        self.server_name = server_name
        self.database = database 
        self.rolling_window = rolling_window
        self.rolling_type = rolling_type
        self.rolling_std = rolling_std
        self.rolling_type = rolling_type
        self.pictures_pathout = pictures_pathout
        
        self.bartype = bartype
        self.depured = depured
        self.rolling = rolling
        self.global_range = global_range
        self.features_sufix = features_sufix
        self.col_weight_type = col_weight_type
        self.col_t1_type = col_t1_type
        self.col_label_type = col_label_type
        self.pca_min_var_expected = pca_min_var_expected
        
        self.cloud_framework = cloud_framework #cloud activation
        self.select_sample = select_sample
        self.n_samples = n_samples
        self.clustered_features = clustered_features
        
        self.k_min = k_min
        self.kendalls = {}
        
        self.ErrorKendallMessage = 'Any valid kendall value exists in the set of trials'
        
    def __getSubsamples__(self, standardMatrix):
        total_samples = [] 
        for i in range(self.k_min,len(standardMatrix.columns) + 1): 
            for j in range(self.n_samples): 
                samples = random.sample(list(standardMatrix.columns.values),i)
                total_samples.append(tuple(samples))
                total_samples = [list(tup) for tup in total_samples]

        listSamples1 = [sorts(i) for i in total_samples]
        listSamples2 = [list(tupl) for tupl in {tuple(item) for item in listSamples1}]
        
        return listSamples2
    

  
    def __instanceOverture__(self):
        
        # instancia feature importance base
        instance = FeatureImportance(
                server_name = self.server_name,
                database_name = self.database, 
		        driver = self.driver,
		        uid = self.uid,
		        pwd = self.pwd,
                list_stocks = self.list_stocks, 
                bartype = self.bartype,
                depured = self.depured, 
                rolling = self.rolling, 
                global_range = self.global_range,
                features_sufix = self.features_sufix,
                window = self.rolling_window,
                win_type = self.rolling_type,
                add_parameter= self.rolling_std,
                col_weight_type = self.col_weight_type,
                col_t1_type = self.col_t1_type,
                col_label_type = self.col_label_type
                )
        
        print("----------Process {} started---------- \n".format(self.method))

        # extrae la matriz de features estacionaria y estandarizada, el vector de labels y el df stacked
        featStandarizedMatrix, labelsDataframe, original_stacked = instance.__checkingStationary__(
            self.pictures_pathout
            ) 
        print("         :::: >>> Running Iteration over samples for Kendall Test...")
            
        samples = self.__getSubsamples__(featStandarizedMatrix)

        itterIdx = range(len(samples))
        
        ray_object_list = [
                     __subsampleProcess__.remote(
                                          self,sample, idx,featStandarizedMatrix,instance,labelsDataframe)
                     for sample,idx in zip(samples,itterIdx)
                     ]
 
        list_correlations = ray.get(ray_object_list)
        print(list_correlations)

        kendalls = {}

        for idx, i in enumerate(list_correlations):
            kendalls[idx] = i

#        for sample in samples:
#            itterIdx += 1             
#            print(f'Running combination        : {itterIdx}')
#           
#            t1 = time()
#            featImpRank, featPcaRank, scoreNoPurged, scorePurged, imp = instance.get_feature_importance(
#                featStandarizedMatrix[sample], labelsDataframe, 
#                self.pictures_pathout, self.method, self.model, itterIdx
#                )

             # si no pasa el score_constraint del 'scorePurged', no usas ese sample
            
#            kendallCorrelation, pValKendall = kendalltau(featImpRank,featPcaRank)

#            print(f'Temporal Kendall Correlation             : {kendallCorrelation}')                
#            print(f'Temporal Kendall p-value                 : {pValKendall}')
#            print(f'Lenght of features tested at Kendall     : {len(sample)}\n')
            
#            kendalls[itterIdx] = [sample, kendallCorrelation, pValKendall]
                
#            print("Tiempo para hacer el árbol:",time()-t1)
        
        
        kendalls = pd.DataFrame.from_dict(kendalls, orient='index')
        
        print("Kendalls",kendalls)

        kendalls.columns = ['best_feature_combination', 'kendall_correlation', 'kendall_pval']
       
        kendalls.to_csv(self.pictures_pathout+'kendall_values.csv', index = False)
       
        groupedPvalCond = kendalls.loc[kendalls.kendall_pval < self.pval_kendall]
       
        if groupedPvalCond.shape[0] == 0:
            sys.exit("{}".format(self.ErrorKendallMessage))
       
        else:
            bestSetFeatures = groupedPvalCond.loc[groupedPvalCond.kendall_correlation.idxmax()]            
       
            kendallCorrSelected, pvalKendallSelected, combFeaturesSelected =  (
                                        bestSetFeatures.loc['kendall_correlation'], 
                                        bestSetFeatures.loc['kendall_pval'], 
                                        bestSetFeatures.loc['best_feature_combination']
                                    )
            
        print(':::::: >>> Kendall Test :')
        print(f'     Best Kendall Correlation calculated is     : {kendallCorrSelected}')
        print(f'     Best Kendall PValue calculated is          : {pvalKendallSelected}')
        print(f'     Lenght of Best Useful Features Combination : {len(combFeaturesSelected)}')
            

        featImpRank, featPcaRank, scoreNoPurged, scorePurged, imp = instance.get_feature_importance(
                featStandarizedMatrix[combFeaturesSelected], labelsDataframe,
                self.pictures_pathout, self.method, self.model, itterIdx
                )


        plotFeatImportance(
                self.pictures_pathout,    
                imp,
                0,
                scorePurged,
                method=self.method, 
                tag='First try',
                simNum= self.method + '_' + type(self.model).__name__,
                model=type(self.model).__name__
                )
                
        print("Feature Importance picture Saved in {}".format(
                self.pictures_pathout)
                    )
        # retorna el dataframe stacked (roleado original) y los nombres de los features
        return original_stacked, combFeaturesSelected 
        
    def get_relevant_features(self, 
                              filtering = True, 
                              save = False, 
                              split = True, 
                              pct_split = 0.6):
        
        # revisa que el pct_split no sea menor a 0.6
        assert pct_split >= 0.6, "Percentage of 'splits' should be 0.6 (60%) as min."
        
        # retorna el dataframe stackeado de features (rolleado = escalado )
        stacked, list_features_tested = self.__instanceOverture__()
        
        # si se activa el proceso de filtrado
        if filtering:
            
            # stacked roleado y escalado proveniente del 'features_algorithms.py'
            stacked = stacked.reset_index(drop=True)

            # extraemos las columnas que no están en la lista global de features
            stackedDiff = stacked[
                stacked.columns.difference(
                    list_features_tested
                    ).values
                ]                     
            
            # elimina los features marginales no utiles dejando la info general principal
            stackedNoFeatures = stackedDiff[stackedDiff.columns.drop(
                list(stackedDiff.filter(regex=self.features_sufix))
                )]
            
            # stacked df filtrado por features seleccionados con col. no-features
            stacked = pd.concat([stackedNoFeatures, stacked[list_features_tested]],axis=1) 
            
            # conversión a dtypes de las fechas de las columnas
            colDates = stacked.dtypes.where(
                        stacked.dtypes == "datetime64[ns]"
                        ).dropna().index.values
                                
            # si se activa proceso de split
            if split:
                
                # obtención de df aleatorio para modelo exogeno, endogeno y backtest
                df_exo, df_endo, backtest = enigmxSplit(
                        df = stacked, 
                        pct_average = pct_split
                        )
                
                # ordenamiento temporal
                df_exo.sort_values(
                    by=['close_date']
                    )
                df_endo.sort_values(
                    by=['close_date']
                    )
                backtest.sort_values(
                    by =['close_date']
                    )
                
                # si se decide guardar
                if save:              
                    
                    # conversión de fecha como string para evitar pérdida de info
                    df_exo[colDates] = df_exo[colDates].astype(str)                
                    df_endo[colDates] = df_endo[colDates].astype(str)
                    backtest[colDates] = backtest[colDates].astype(str)
                    
                    # si no esta activado para cloud, asigna path local primero
                    if not self.cloud_framework:
                        
                        exo_path = "{}STACKED_EXO_{}_{}.csv".format(
                                      self.pictures_pathout,self.bartype, self.method
                                      )                        
                        endo_path = "{}STACKED_ENDO_{}_{}.csv".format(
                                      self.pictures_pathout, self.bartype, self.method
                                      )
                        backtest_path = "{}STACKED_BACKTEST_{}_{}.csv".format(
                                      self.pictures_pathout, self.bartype, self.method
                                      )                        
                    
                    # si esta activado para cloud, no asignes path local alguno
                    else: 
                        
                        exo_path = "STACKED_EXO_{}_{}.csv".format(
                                      self.bartype, self.method
                                      )                        
                        endo_path = "STACKED_ENDO_{}_{}.csv".format(
                                      self.bartype, self.method
                                      )
                        backtest_path = "STACKED_BACKTEST_{}_{}.csv".format(
                                      self.bartype, self.method
                                      )
                        
                    # almacenamientos
                    df_exo.to_csv(exo_path, index=False, 
                                  date_format='%Y-%m-%d %H:%M:%S'
                                  )
                    
                    df_endo.to_csv(endo_path, index=False, 
                                  date_format='%Y-%m-%d %H:%M:%S'
                                  )          
                    
                    backtest.to_csv(backtest_path, index=False, 
                                  date_format='%Y-%m-%d %H:%M:%S'
                                  )          
                    
                    print("Process finished! 'exo', 'endo' & 'backtest' df saved at {}...".format(
                        self.pictures_pathout)
                        )
                    
                # caso contrario, retorna solo split para modelo exogeno y endogeno
                else:
                    return df_exo, df_endo
            
            # caso contrario, si no se elije split
            else:
                
                # si se elije guardar
                if save: 
                    
                    # conversión de string
                    stacked[colDates] = stacked[colDates].astype(str)                
                    
                    # se guarda stacked filtered df 
                    stacked.to_csv(
                        "STACKED_{}_{}.csv".format(
                                self.bartype, self.method
                                ), index=False, date_format='%Y-%m-%d %H:%M:%S'
                            )
                    
                    print("Stacked Dataframe saved at {}...".format(
                            self.pictures_pathout)
                            )
                
                # caso contrario, retorna stacked filtered df                
                else:
                    return stacked
        
        # caso contrario, retorna los nombres de features seleccionados
        else:
            return list(list_features_tested) 


@ray.remote
def __subsampleProcess__(self,sample,itterIdx,featStandarizedMatrix,instance,labelsDataframe):

    print(f'Running combination        : {itterIdx}')

    t1 = time()
    featImpRank, featPcaRank, scoreNoPurged, scorePurged, imp = instance.get_feature_importance(
                featStandarizedMatrix[sample], labelsDataframe,
                self.pictures_pathout, self.method, self.model, itterIdx
                )

     # si no pasa el score_constraint del 'scorePurged', no usas ese sample

    kendallCorrelation, pValKendall = kendalltau(featImpRank,featPcaRank)

    print(f'Temporal Kendall Correlation             : {kendallCorrelation}')
    print(f'Temporal Kendall p-value                 : {pValKendall}')
    print(f'Lenght of features tested at Kendall     : {len(sample)}\n')

    print("Tiempo para hacer el árbol:",time()-t1)

    return  [sample, kendallCorrelation, pValKendall]





