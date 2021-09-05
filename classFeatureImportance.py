"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import sys
import ray
import click
import random
import pandas as pd
import numpy as np
from time import time
from scipy.stats import kendalltau
from enigmx.features_algorithms import FeatureImportance
from enigmx.purgedkfold_features import plotFeatImportance
from enigmx.featuresclustering import ClusteredFeatureImportance
from enigmx.utils import enigmxSplit, baseFeatImportance, clickMessage1 #,kendall_evaluation
from enigmx.rscripts import regression_intracluster, regression_intercluster #,convert_pandas_to_df

##############################################################################
########################## COMPLEMENTARY FUNCTIONS ###########################
##############################################################################

@ray.remote
def __iterativeFeatImpBySample__(sample, 
                         itterIdx, 
                         featStandarizedMatrix, 
                         instance, 
                         labelsDataframe, 
                         pictures_pathout,
                         method, 
                         model):

    print(f'Running combination        : {itterIdx}')

    t1 = time()
    featImpRank, featPcaRank, scoreNoPurged, scorePurged, imp = \
        instance.get_feature_importance(
                    featStandarizedMatrix[sample], labelsDataframe,
                    pictures_pathout, method, model, itterIdx
                )

     # si no pasa el score_constraint del 'scorePurged', no se emplea el sample

    kendallCorrelation, pValKendall = kendalltau(featImpRank,featPcaRank)

    print(f'Temporal Kendall Correlation             : {kendallCorrelation}')
    print(f'Temporal Kendall p-value                 : {pValKendall}')
    print(f'Lenght of features tested at Kendall     : {len(sample)}\n')

    print(f"Tiempo para el featImportance de IDX '{itterIdx}':",time()-t1)
    print("***"*30)

    return  [sample, kendallCorrelation, pValKendall]

def __getSubsamples__(standardMatrix, k_min, n_samples):
    total_samples = [] 
    for i in range(k_min, len(standardMatrix.columns) + 1): 
        for j in range(n_samples): 
            samples = random.sample(list(standardMatrix.columns.values),i)
            total_samples.append(tuple(samples))
            
    total_samples = [list(tup) for tup in total_samples]
    listSamples1 = [sorts(i) for i in total_samples]
    listSamples2 = [list(tupl) for tupl in {tuple(item) for item in listSamples1}]
    
    return listSamples2


def mainClusteringCombinatorialFeatImp(standarized_features_matrix, 
                                       df_labels,
                                       cluster, 
                                       featimp_instance,
                                       cluster_idx, 
                                       n_samples, 
                                       k_min,
                                       picturesPathout,
                                       method_featimp,
                                       model_featimp):
    
    # redefiniendo n_samples en caso sea menor a 1 
    if n_samples < 2:
        n_samples = 2
    
    # particiona el dataframe solo con los features del cluster
    featClusteringSelection = standarized_features_matrix[cluster]
    
    
    # obtiene una lista combinatorias por cluster 
    samplesInnerCluster = __getSubsamples__(
        standardMatrix=featClusteringSelection,
        k_min=k_min,
        n_samples=n_samples
        )
    
    
    # define un código C_S, que itendifica cada Cluster (C) y cada Sample (S)
    itterIdxBySample = [
        str(cluster_idx) + '_' + str(sampleIdx) 
        for sampleIdx in list(range(len(samplesInnerCluster)))
        ]
    
    # iteración paralelizada por sample del cluster
    ray_featImp_by_cluster = [
            __iterativeFeatImpBySample__.remote(
                    sample, # lista de nombres de los features
                    idx, # idx del cluster
                    featClusteringSelection, # features-matrizx stand & statio.
                    featimp_instance, # instancia del featImp no clusterizado (base)
                    df_labels, # labels-information dataframe 
                    picturesPathout, 
                    method_featimp, 
                    model_featimp
                )            
            for sample, idx in zip(samplesInnerCluster, itterIdxBySample)
            ]
    
    # checking 
    resultado_final_temp = ray.get(ray_featImp_by_cluster)
    
    return resultado_final_temp

def sorts(i):
    i.sort()
    return i


##############################################################################
######################### BASE FEAT IMPORTANCE CLASS #########################
##############################################################################

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
                 clustered_features = True,
                 k_min = 5,
                 trial = '001',
                 residuals = False, #add out
                 silh_thres = 0.65, #add out 
                 n_best_features = 7, #add out
                 global_featImp = True, #add out
                 ):
        
        # ingesta de parámetros
        self.model = model
        self.list_stocks = list_stocks
        self.method = method.upper()
        self.score_constraint = score_constraint #ACTIVAR!
        self.driver = driver
        self.uid = uid
        self.pwd = pwd
        self.trial = trial

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

        self.residuals = residuals
        self.silh_thres = silh_thres
        
        self.n_best_features = n_best_features
        self.global_featImp = global_featImp 
        
        self.ErrorKendallMessage = 'Any valid kendall value exists in the set of trials'
        

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

        # extrae matriz de features estacionaria-estandarizada, labelsDf y df stacked
        featStandarizedMatrix, labelsDataframe, original_stacked = \
            instance.__checkingStationary__(
                self.pictures_pathout
            ) 
        
        print("         :::: >>> \
              Running Iteration over samples for Kendall Test...")
              
        
        
        # si se utiliza el clustering featImp 
        if self.clustered_features: 
            
            print("      :::>>>> Clustering Feature Importance Initialized...")
            
            #Selecciona features numéricos y discretos, todos los discretos van a formar un cluster independiente
            discrete_feat = [x for x in featStandarizedMatrix.columns if x.split('_')[-1] == 'integer']
            numerical_feat = [x for x in featStandarizedMatrix.columns if x not in discrete_feat]
            

            # creamos instancia de featImportance clusterizado
            instance_clustered_featimp = ClusteredFeatureImportance(
                                        # parámetros obligatorios
                                        feature_matrix = featStandarizedMatrix, 
                                        model = self.model, 
                                        method = self.method,
                                        # parámetros optativos (valores predef.)
                                        max_number_clusters = None,
                                        number_initial_iterations = 10,
                                        additional_features = discrete_feat,
                                        numerical_features = numerical_feat
                                    )
            
            # obtenemos el featImportance clusterizado y los clusters
            clusteredFeatImpRank, clusters = \
                instance_clustered_featimp.get_clustering_feature_importance(
                                        labels = labelsDataframe["labels"]
                                    )
            
            # obtiene numeral 'C_#' de c/cluster manteniendo el importance-order
            clustersIdx_sorted_by_importance = [
                int(string.split("_")[-1]) 
                for string in clusteredFeatImpRank.index.values
                ]
            
            
            # filtración de los features que pasen det. silhouette score
            features_and_silh = instance_clustered_featimp.silh.where(
                instance_clustered_featimp.silh < self.silh_thres
                ).dropna()
            
            # lista vacía para almacenar los features a transformar
            features_to_transform =[]
            
            # búsqueda de features a transformar por cada cluster
            for features in clusters.values():
                features_to_transform.append(
                    [
                        feature for feature in features 
                        if feature in features_and_silh.index
                        ]
                    )
            
            # lista de lista con los features de c/ cluster
            clusters_values = list(clusters.values())
            clusters_values.append(discrete_feat)
            
            print("          ------------------>>>> Computing residuals... ")
            
            # si se selecciona la transformación por residuos
            if self.residuals:
                
                # computa las regresiones intercluster (enfoque MDLP)
                # featStandarizedMatrix = regression_intercluster(
                #     featStandarizedMatrix, 
                #     features_to_transform, 
                #     clusters_values
                #     )
                
                # computa las regresiones intracluster
                featStandarizedMatrix = regression_intracluster(
                    featStandarizedMatrix, 
                    clusters_values
                    )
                
                # inserting same IDX in features matrix as labelsDF
                featStandarizedMatrix.index = labelsDataframe.index
                
                # si ejecutar el featImp sobre toda la matriz de feat
                if self.global_featImp: 
                    
                    # func. featImp estandar | puede reemp. anteriores (sin PCA)
                    importanceRank, scoreNoPurged, scorePurged, stackedImp = \
                        baseFeatImportance(
                            features_matrix = featStandarizedMatrix,
                            labels_dataframe = labelsDataframe,
                            random_state = 42,
                            method = self.method,
                            model_selected = self.model,
                            pct_embargo = 0.01, #agregar como variable para llamar desde fuera
                            cv = 5, #agregar como variable para llamar desde fuera
                            oob = False,
                        )
                        
                    # reordenamiento featImpRank | de mayor a menor rank-val
                    importanceRankSorted = importanceRank.sort_values(
                        ascending=False
                        )
                    
                    # selección de top n best features según importance Rank
                    featureSelected = importanceRankSorted[
                        :self.n_best_features
                        ].index.values 
                    
                    # definimos variables para el return general de la clase
                    imp, combFeaturesSelected = stackedImp, featureSelected

                # ejecutar featImp x cluster para elegir al mejor feat de c/u
                else:
                    
                    # lista vacía para almacenar el mejor features de c/ cluster
                    list_features_selected = []
                    
                    # iteración por grupo de features de cada cluster
                    for idxClust, clusterFeatures in enumerate(clusters_values):
                        
                        # anula 1er elemento denominado 'intercepto' (no es feature)
                        clusterFeatures = clusterFeatures[1:]

                        # matriz temporal de features
                        tempClusterFeatMatrix = featStandarizedMatrix[clusterFeatures]
                        
                        # proceso temporal de featImp inner cluster... (sin PCA)
                        importanceRankTemp, scoreNoPurgedTemp, scorePurgedTemp, stackedImpTemp = \
                            baseFeatImportance(
                                features_matrix = tempClusterFeatMatrix,
                                labels_dataframe = labelsDataframe,
                                random_state = 42,
                                method = self.method,
                                model_selected = self.model,
                                pct_embargo = 0.01, #agregar como variable para llamar desde fuera
                                cv = 5, #agregar como variable para llamar desde fuera
                                oob = False,
                            )
                        
                        # reordenamiento featImpRank | de mayor a menor rank-val
                        importanceRankSortedTemp = importanceRankTemp.sort_values(
                            ascending=False
                            ) 
                        
                        # selección de top n best features según importance Rank
                        featureSelectedTemp = importanceRankSortedTemp[
                            :self.n_best_features
                            ].index.values 
                        
                        # almacenamos features elegidos del cluster
                        list_features_selected.extend(featureSelectedTemp)        
                        
                        # guardado de imágenes del featImp rank temporal
                        plotFeatImportance(
                                self.pictures_pathout,    
                                stackedImpTemp,
                                0,
                                scorePurgedTemp,
                                method=self.method, 
                                tag='First try',
                                simNum= self.method + '_' + type(self.model).__name__ + 
                                'cluster' + str(idxClust),
                                model=type(self.model).__name__
                                )                        
                    
                    # definimos variables para el return general de la clase
                    imp, combFeaturesSelected = None, list_features_selected 
            
            # si no se utiliza la transformación por residuos en el clusterizado
            else:
 
                #Iteración por grupo de features de cada cluster:
                for idxClust, clusterFeatures in enumerate(clusters_values):


                    #matriz temporal de features
                    tempClusterFeatMatrix = featStandarizedMatrix[clusterFeatures]

                    #proceso de feature importances dentro del cluster:
                    importanceRankTemp, scoreNoPurged, scorePurgedTemp, stackedImpTemp = \
                        baseFeatImportance(
                                features_matrix = tempClusterFeatMatrix,
                                labels_dataframe = labelsDataframe,
                                random_state = 42,
                                method = self.method,
                                model_selected = self.model,
                                pct_embargo = 0.01, #agregar como variable para llamar desde fuera
                                cv = 5, #agregar como variable para llamar desde fuera
                                oob = False,
                            )

                    # reordenamiento featImpRank | de mayor a menor rank-val
                    importanceRankSortedTemp = importanceRankTemp.sort_values(
                            ascending=False
                            ) 
                        
                        
                    # guardado de imágenes del featImp rank temporal
                    plotFeatImportance(
                                self.pictures_pathout,    
                                stackedImpTemp,
                                0,
                                scorePurgedTemp,
                                method=self.method, 
                                tag=self.trial,
                                simNum= self.method + '_' + type(self.model).__name__ + 
                                'cluster' + str(idxClust) + ' || Try: ' + self.trial,
                                model=type(self.model).__name__
                                )                        
                                    
                print(" ...... El proceso se detendrá, la decisión de los features elegidos se realizará discrecionalmente >>>")

                #Se plotea la importancia entre clusters
                plotFeatImportance(
                                self.pictures_pathout,
                                clusteredFeatImpRank,
                                0,
                                0,
                                method=self.method,
                                tag=self.trial,
                                simNum= self.method + '_' + type(self.model).__name__ +
                                ' || Try: ' + self.trial,
                                model=type(self.model).__name__
                                )
 

                #Retorna el stacked
                return original_stacked
                    
        
        # si no se utiliza el clustering featImp (featImp con Kendall Tau Corr)
        else: 
            
            # obtiene combinatorias según un min de feat y n samples
            samples = __getSubsamples__(
                featStandarizedMatrix, 
                self.k_min, 
                self.n_samples
                )
            
            # estima los valores de iteración según el total de samples
            itterIdx = range(len(samples))
        
            # computa el featImp iterativo por sample
            ray_object_list = [
                     __iterativeFeatImpBySample__.remote(
                                          sample, 
                                          idx,
                                          featStandarizedMatrix,
                                          instance,
                                          labelsDataframe,
                                          self.pictures_pathout,
                                          self.method,
                                          self.model
                                          )
                     for sample,idx in zip(samples,itterIdx)
                     ]
        
            print("     :::>>>> Getting Kendall Tau List Corr by Sample...")
            # lista de correlaciones por sample
            list_correlations = ray.get(ray_object_list)
            
            # definición de diccionario 'kendall' para guardar val. corr por idx
            kendalls = {}
    
            # itera por c/ kendall tau corr en la lista de corr para asignar idx
            for idx, i in enumerate(list_correlations):
                kendalls[idx] = i
            
            # define un dataframe con los valores de corr
            kendalls = pd.DataFrame.from_dict(kendalls, orient='index')
            
            print("  ----- >>> Building & Saving KendallTau dataframe...")
            
            # defining kendalls df columns
            kendalls.columns = [
                'best_feature_combination', 'kendall_correlation', 'kendall_pval'
                ]
           
            # saving kendalls to csv
            kendalls.to_csv(
                self.pictures_pathout+'kendall_values.csv', index = False
                )
           
            # select only combinations that pass pval kendall threshold
            groupedPvalCond = kendalls.loc[
                kendalls.kendall_pval < self.pval_kendall
                ]
           
            # si no existe una combinación con un kendallTau aceptable
            if groupedPvalCond.shape[0] == 0:
                sys.exit("{}".format(self.ErrorKendallMessage))
           
            # caso contrario, selecciona los index de las combinaciones aceptables
            else:
                # obtiene el mejor conjunto de features con un 'loc'
                bestSetFeatures = groupedPvalCond.loc[
                    groupedPvalCond.kendall_correlation.idxmax()
                    ]            
                
                # obtiene valores finales: correlación, pval y mejor comb de features
                kendallCorrSelected, pvalKendallSelected, combFeaturesSelected =  (
                                            bestSetFeatures.loc['kendall_correlation'], 
                                            bestSetFeatures.loc['kendall_pval'], 
                                            bestSetFeatures.loc['best_feature_combination']
                                        )
                
            print(':::::: >>> Kendall Test :')
            print(f'     Best Kendall Correlation calculated is     : {kendallCorrSelected}')
            print(f'     Best Kendall PValue calculated is          : {pvalKendallSelected}')
            print(f'     Lenght of Best Useful Features Combination : {len(combFeaturesSelected)}')
                
            # ejecuta el featImp con la mejor comb. de features
            featImpRank, featPcaRank, scoreNoPurged, scorePurged, imp = \
                instance.get_feature_importance(
                    featStandarizedMatrix[combFeaturesSelected], labelsDataframe,
                    self.pictures_pathout, self.method, self.model, itterIdx
                    )
            
        print("************** Returning to General Stage | FeatImp Class >>>")

        # guardado de imágenes
        if imp is not None:
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
                
            print("Saving Picture!!!! ---> Feature Importance picture Saved in {}".format(
                self.pictures_pathout)
                )
        
        print("Final Return | FeatImp :::: >>> Useful features combination:")
        print(combFeaturesSelected)
        
        # retorna el df stacked (roleado orig.) y los nombres de los features
        return original_stacked
        
    def get_relevant_features(self, 
                              filtering = True, 
                              save = False, 
                              split = True, 
                              pct_split = 0.6):
        
        # revisa que el pct_split no sea menor a 0.6
        assert pct_split >= 0.6, "Percentage of 'splits' should be 0.6 (60%) as min."
        
        # df stackeado de feats (rolleado = escalado) + la comb. elegida de features
        stacked = self.__instanceOverture__()
        
        #En este punto el proceso se queda en standby, esperando la lista de features seleccionados:
        list_features_tested = input("Por favor ingresa la lista de features, sin corchetes y separada por comas, luego presiona enter: ")
        list_features_tested = list_features_tested.split(",")
 
        print("Gracias, ahora voy a guardar los csv's del stacked con los features elegidos", flush = True)

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
