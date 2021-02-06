"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
        
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from enigmx.utils import global_list_stocks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from purgedkfold_features import featImportances, plotFeatImportance

#Main Feature Importance Class
class FeatureImportance(object):
    
    def __init__(self, 
                 path_individual_csv, 
                 depured = True, 
                 rolling = True,
                 window = None,
                 win_type = None,
                 add_parameter = None):
        
        self.path_individual_csv = path_individual_csv
        self.depured = depured
        self.rolling = rolling
        self.window = window
        self.win_type = win_type
        self.add_parameter = add_parameter
        
    def __getStacked__(self):
        """
        Open individual csv and merge them to generate stacked.
        """
        #open single stacked csv
        list_stocks = global_list_stocks(self.path_individual_csv,
                                         common_path = '.csv', 
                                         drop_extension = 13)
                
        #csv dataframes single stacked
        list_df = [pd.read_csv(
                self.path_individual_csv + stock + '_COMPLETE' + '.csv'
                ).dropna() for stock in list_stocks]
        
        #list_df = list_df[:6] #delete this when reloading process!!!

        #rolling smoothing over single equity df
        if self.rolling:
            
            columns_for_rolling = list_df[0].columns[9:]
 
            list_df = [
                pd.concat(
                    [
                        dataframe.iloc[:,:9],
                        dataframe[columns_for_rolling].rolling(
                            self.window, win_type=self.win_type
                            ).sum(std=self.add_parameter)
                        ], axis=1
                                ).dropna()  for dataframe in list_df
                        ]

        df_ = pd.concat(list_df).sort_index(
            kind='merge'
            )
        
        return df_
    
    def __splitStacked__(self):
        """
        Generate x_train, y_train, x_test & y_test for stacked df.
        """
        df_global_stacked = self.__getStacked__()
        
        #basic depuration
        if self.depured:
            df = df_global_stacked.iloc[:,9:]
        else:
            df = df_global_stacked
        
        #pured stacked df
        stacked = df[
            df.columns.drop(
                list(
                    df.filter(
                        regex='_WAVELET10'
                        )
                    )
                )
            ]
        
        #dataframe creation for each label
        y = df_global_stacked['tripleBarrier'].to_frame('labels')
        y['t1'] = df_global_stacked['datetime']
        y['w'] = 1/len(y)
        y.set_index(df_global_stacked['datetime'],inplace=True)
        
        #stacked segmentation
        x = stacked
        x.set_index(df_global_stacked['datetime'],inplace=True)
        
        #data split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=42)
        
        return x_train, x_test, y_train, y_test
    
    def get_feature_importance(self, 
                               pathOut, 
                               method, 
                               model_selected, 
                               pctEmbargo = 0.01, 
                               cv=5, 
                               oob=False):
        
        x_train, x_test, y_train, y_test = self.__splitStacked__()
    
        if method == 'MDA':
            if type(model_selected).__name__=='RandomForestClassifier':
                raise ValueError(
                    "Only {} is not allowed to implement 'MDA'".format(
                        'RandomForestClassifier'
                        )
                    )      
                
        if method == 'MDI': 
            if type(model_selected).__name__!='RandomForestClassifier':
                raise ValueError(
                    "Only {} is allowed to implement 'MDI'".format(
                        'RandomForestClassifier'
                        )
                    )
            
        imp,oos,oob = featImportances(x_train, 
                                      y_train, 
                                      model_selected,
                                      method=method,
                                      sample_weight = y_train['w'],
                                      pctEmbargo=0.01,
                                      cv=5,
                                      oob=False)
        
        model_selected.fit(x_train,y_train['labels'])
            
        print("Score sin el PurgedKFold:",
                  model_selected.score(x_test, y_test['labels']))
        print("Score con el PurgedKFold:",
                  oos)
        
        print("Saving picture...")
        plotFeatImportance(
                        pathOut,    
                        imp,
                        0,
                        oos,
                        method=method, 
                        tag='First try',
                        simNum= method + '_' + type(model_selected).__name__,
                        model=type(model_selected).__name__)
        
#Main PCA Calculation for Feature Selection
class PCA_QM(object):
    """
    This class is over PCA process over stacked file.
    """
    def __init__(self, 
                 stacked_df):
        
        self.stacked_df = stacked_df

        
    def get_pca(self,
               number_features = 9,
               min_var_exp = 0.05,
               feature_selection = True,
               save_and_where = None):
         
        if type(save_and_where) != tuple and type(save_and_where) != None:
            raise ValueError(
                "Only (BOOLEAN, PATH_STRING) tuple for 'save_and_where'"
                )
            
        scaler = StandardScaler()
        scaler.fit(self.stacked_df)
        X = scaler.transform(self.stacked_df)
        
        pca = PCA(n_components=min(X.shape))
        
        X_new = pca.fit_transform(X)   

        pc_s = []
            
        for i in pca.explained_variance_ratio_:

            if i > min_var_exp:
                pc_s.append(i)
        
        pc_s = np.array(pc_s)   

        if feature_selection:
            
            Fea_imp = abs(pca.components_[:pc_s.shape[0],:])
            
            In_sort = np.sort(Fea_imp, axis=1)[:,::-1]
            
            Matriz_FI = np.zeros(
                (Fea_imp.shape[0], 
                 number_features,)
                )
            
            for i in range(Fea_imp.shape[0]):
            
                for j in range(number_features):
                
                    Matriz_FI[i,j] = np.argwhere(
                        Fea_imp[i,:] == In_sort[i,j]
                        )[0][0]
            
            
            features_ = []
            pca_values_ = []
            for i in range(0,Matriz_FI.shape[0]): 
                features_.append(
                    self.stacked_df.iloc[:, Matriz_FI[i]].columns
                    )
                pca_values_.append(
                    list(Fea_imp[i][Matriz_FI[i].astype(int)])
                    )

            if save_and_where[0]:
                
                #pandas dataframe
                imp = pd.DataFrame(
                    {
                        'mean': list(
                            itertools.chain(*pca_values_)
                            ),
                        'std': [np.nan]
                        }, index=list(itertools.chain(*features_))
                    )
                
                #plot and save picture
                plotFeatImportance(
                            save_and_where[1],    
                            imp,
                            number_features,
                            min_var_exp,
                            method='PCA', 
                            tag='First try',
                            simNum='PCA')
                return "PCA picture Saved in {}".format(save_and_where[1])
            
            else:
                #return feature names selected by PCA
                return list(set(itertools.chain(*features_)))
        
        else:
            #reduced matrix
            return X_new

def fitting_to_pca(
        path_of_stacked, 
        columns_to_avoid = 9,
        unnecesary_feature='_WAVELET10'):
    
    """
    This functions was developed to avoid all the 'WAVELET10'.
    If you don't remove it from the base dataset, it will cause a problem
    over the PCA by overlaping the features names.
    
    Also, it is important to remove it as long as it does not have significant
    values. Mostly, all of them are zero.
    """
    
    df = pd.read_csv(path_of_stacked, index_col=False)
    df_matrix = df.iloc[:,columns_to_avoid:]
    
    df_matrix = df_matrix[df_matrix.columns.drop(
        list(
            df_matrix.filter(
                regex=unnecesary_feature
                )
            )
        )
        ]
    return df_matrix
