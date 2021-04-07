"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import sys
import numpy as np
import pandas as pd
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier
    )
from enigmx.utils import EquitiesEnigmxUniverse, enigmxSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier  
from enigmx.tests.test_features import FeatureImportance, PCA_QM


class featureImportance(object):
    
    def __init__(self, 
                 model, 
                 method, 
                 list_stocks,
                 pca_comparisson,
                 top_features = 2,
                 score_constraint = 0.6,
                 server_name = "DESKTOP-N8JUB39", 
                 database = 'BARS_WEIGHTED',
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
                 pca_min_var_expected = 0.05):
        
        self.model = model
        self.list_stocks = list_stocks
        self.method = method.upper()
        self.score_constraint = score_constraint
        
        self.top_features = top_features
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
        
    def __instanceOverture__(self):
                
        instance = FeatureImportance(
                server_name = self.server_name,
                database_name = self.database, 
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

        featImpDf, scorePurged, scoreNoPurged, stack =instance.get_feature_importance(
            self.pictures_pathout, self.method, self.model
            )
        
        stackedPackage = instance.__getStacked__()
        pcaIng = stackedPackage[0][stackedPackage[1]]               
        
        if self.score_constraint > scoreNoPurged: 
            print("Warning! NoPurgedScore < ScoreConstraint \n")
        
        if self.score_constraint > scorePurged:
            print("RequirementError: \
                  scorePurged for method {} is < scorePurged".format(
                    self.method
                    ))             
            sys.exit("Process Interrupted")
            
        else:
            if self.pca_comparisson: 
                
                print("\n---------- PCA Comparisson activated----------\n")
                
                pcaDf = PCA_QM(pcaIng).get_pca(
                        number_features = len(stackedPackage[1]), 
                        path_to_save_picture = self.pictures_pathout,
                        min_var_exp = self.pca_min_var_expected,
                        feature_selection = True
                        )
                            
                bestArr1 = pcaDf.sort_values(
                    by=['mean'], ascending = False
                    ).head(self.top_features).index.values
                
                bestArr2 = featImpDf.sort_values(
                    by=['mean'], ascending = False
                    ).head(self.top_features).index.values
                
                bestAllPcaFeatImp = np.intersect1d(bestArr1, bestArr2)
                
                if bestAllPcaFeatImp.shape[0] == 0:
                    print("RequirementError: \
                          No features matches in PCA vs {} comparisson".format(
                        self.method
                        ))
                    sys.exit("Process Interrupted")
        
                else:
                    bestPcaVal = pcaDf.loc[bestAllPcaFeatImp][['mean']]
                    bestMETVal = featImpDf.loc[bestAllPcaFeatImp][['mean']]                
    
                    dfMatrix = pd.concat([bestPcaVal, bestMETVal],axis=1)
                    dfMatrix.columns = ['ePCA', 'e{}'.format(self.method)]
                    
                    return dfMatrix, stack, stackedPackage[1]
                            
            else:
                dfMatrix = featImpDf.rename(
                    columns={'mean':'e{}'.format(self.method)}
                    )
                
                return dfMatrix, stack, stackedPackage[1]
        
    def get_relevant_features(self, 
                              filtering = True, 
                              save = False, 
                              split = True, 
                              pct_split = 0.5):
        
        if filtering:
            dfMatrix, stacked, listaGlobalFeat = self.__instanceOverture__()

            stacked = stacked.reset_index(drop=True)
            
            stackedDiff = stacked[
                stacked.columns.difference(
                    list(listaGlobalFeat)
                    ).values
                ]            

            stackedSel = stacked[list(dfMatrix.index.values)]

            stackedFilteredDf = pd.concat([stackedDiff, stackedSel],axis=1) 
            
            colDates = stackedFilteredDf.dtypes.where(
                        stackedFilteredDf.dtypes == "datetime64[ns]"
                        ).dropna().index.values
                                
            
            if split:
                
                df_exo, df_endo = enigmxSplit(
                        df = stackedFilteredDf, 
                        pct_average = pct_split
                        )
                
                df_exo.sort_values(
                    by=['close_date']
                    )
                
                df_endo.sort_values(
                    by=['close_date']
                    )
                
                if save:              
                
                    df_exo[colDates] = df_exo[colDates].astype(str)                
                    df_endo[colDates] = df_endo[colDates].astype(str)
                    
                    df_exo.to_csv(self.pictures_pathout +
                                  "STACKED_EXO_{}_{}.csv".format(
                                      self.bartype, self.method
                                      ), index=False, 
                                  date_format='%Y-%m-%d %H:%M:%S'
                                  )
                    
                    df_endo.to_csv(self.pictures_pathout +
                                  "STACKED_ENDO_{}_{}.csv".format(
                                      self.bartype, self.method
                                      ), index=False, 
                                  date_format='%Y-%m-%d %H:%M:%S'
                                  )                 

                    print("Process finished! 'exo' & 'endo' dataframes at {}...".format(
                        self.pictures_pathout)
                        )
                    
                else:
                    return df_exo, df_endo
                    
            else:
                
                if save: 
                    
                    stackedFilteredDf[colDates] = stackedFilteredDf[colDates].astype(str)                
                    
                    stackedFilteredDf.to_csv(self.pictures_pathout +
                                                 "STACKED_{}_{}.csv".format(
                                self.bartype, self.method
                                ), index=False, date_format='%Y-%m-%d %H:%M:%S'
                            )
                    
                    print("Stacked Dataframe saved at {}...".format(
                            self.pictures_pathout)
                            )
                else:
                    return stackedFilteredDf

        else:
            return list(dfMatrix.index.values)


#################################### TEST ####################################
          
instance = featureImportance(
    model = GaussianNB()  , 
    method = 'MDA', 
    list_stocks = EquitiesEnigmxUniverse[10:20],
    pca_comparisson = True,
    score_constraint=0.2,
    )        


instance.get_relevant_features(filtering = True, 
                               save = True, 
                               split = True, 
                               pct_split = 0.5
                               )