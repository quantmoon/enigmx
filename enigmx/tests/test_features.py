"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
        
import itertools
import numpy as np
import pandas as pd

from enigmx.interface import EnigmXinterface
from enigmx.dbgenerator import databundle_instance 
    
from sklearn.decomposition import PCA
from enigmx.utils import global_list_stocks
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from enigmx.purgedkfold_features import featImportances, plotFeatImportance

#Main Feature Importance Class
class FeatureImportance(object):
    
    def __init__(self, 
                 server_name, 
                 database_name, 
                 list_stocks,
                 bartype = 'VOLUME',
                 depured = True, 
                 rolling = True,
                 global_range = True,
                 features_sufix = 'feature',
                 window = None,
                 win_type = None,
                 add_parameter = None, 
                 col_weight_type = 'weightTime',
                 col_t1_type  = 'horizon',
                 col_label_type = 'barrierLabel'):
        
        self.server_name = server_name
        self.database_name = database_name
        self.list_stocks = list_stocks
        self.bartype = bartype 
        self.depured = depured
        self.rolling = rolling
        self.global_range = global_range
        self.features_sufix = features_sufix
        self.window = window
        self.win_type = win_type
        self.add_parameter = add_parameter
        self.col_weight_type = col_weight_type
        self.col_t1_type = col_t1_type
        self.col_label_type = col_label_type
        
    def __getStacked__(self):
        """
        Open individual csv and merge them to generate stacked.
        """
        
        #abrir la instancia sql base, la conexión y el cursor
        SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = self.database_name,
                    #boleano para crear la tabla
                    create_database = False, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )                    
        
        #lista con tablas/dataframes por acción extraídas de SQL
        list_df = [
            SQLFRAME.read_table_info(
                statement = "SELECT * FROM [{}].[dbo].{}_GLOBAL".format(
                    self.database_name, stock + "_" + self.bartype
                    ), 
                dbconn_= dbconn, 
                cursor_= cursor, 
                dataframe=True
                )
            for stock in self.list_stocks
            ]
        
        #proceso de rolling para uniformizar la dist. de los features
        if self.rolling:
            
            #columnas para rolling 
            columns_for_rolling = list_df[0].filter(
                like=self.features_sufix
                ).columns     
            
            #temporal | BORRAR AL COMPUTAR FEATURES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            columns_for_rolling = ["vwap", "fracdiff", "bar_cum_volume"]
        
            #hace rolling solo sobre las columnas de feature
            list_df = [
                pd.concat(
                    [   #extrae aquellas columnas que no haran rolling
                        dataframe[dataframe.columns.difference(
                            columns_for_rolling
                            ).values],
                        #computa el rolling solo en columnas de features
                        dataframe[columns_for_rolling].rolling(
                            self.window, win_type=self.win_type
                            ).sum(std=self.add_parameter)
                        ], axis=1
                                ).dropna()  for dataframe in list_df
                        ]
        
        #dataset final concadenado
        df_ = pd.concat(list_df).sort_index(
            kind='merge'
            )
        
        return df_, columns_for_rolling
    
    def __splitStacked__(self): #AQUI SE EXTRAEN EL TEST Y TRAIN Y SELEC LOS FEATURES
        """
        Generate x_train, y_train, x_test & y_test for stacked df.
        """
        #obtener el dataframe de los valores stacked + columnas de features
        df_global_stacked, features_col_name = self.__getStacked__()
        
        #depuracion del dataframe de entrada, seleccionando solo los features
        if self.depured: 
            df = df_global_stacked[features_col_name]
        else:
            df = df_global_stacked
        
        #creación de la información de las etiquetas
        y = df_global_stacked[self.col_label_type].to_frame('labels')
        
        #definición de 't1': horizonte temporal máx. de c/ barra (barrera vert.)
        y['t1'] = df_global_stacked[self.col_t1_type]
        
        #definición de 'w': pesos para cada observación
        y['w'] = df_global_stacked[self.col_weight_type]
        y.set_index(df_global_stacked['close_date'],inplace=True)
                
        #stacked segmentation
        x = df
        x.set_index(df_global_stacked['close_date'],inplace=True)

        #data split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=42)
        
        return x_train, x_test, y_train, y_test, df_global_stacked
    
    def get_feature_importance(self, 
                               pathOut, 
                               method, 
                               model_selected, 
                               pctEmbargo = 0.01, 
                               cv=5, 
                               oob=False):
        
        x_train, x_test, y_train, y_test, dfStacked = self.__splitStacked__()
    
        if method == 'MDA':
            if type(model_selected).__name__=='RandomForestClassifier':
                raise ValueError(
                    "{} model is not allowed to implement 'MDA'".format(
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
        
        score_sin_cpkf = model_selected.score(x_test, y_test['labels'])
        
        print("Score sin el PurgedKFold:", score_sin_cpkf)
        print("Score con el PurgedKFold:", oos)
        
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
        
        
        print("Feature Importance picture Saved in {}".format(
            pathOut)
            )
        
        return imp, score_sin_cpkf, oos, dfStacked
        
#Main PCA Calculation for Feature Selection
class PCA_QM(object):
    """
    This class is over PCA process over stacked file.
    """
    def __init__(self, 
                 stacked_df):
        
        self.stacked_df = stacked_df.reset_index(drop=True)

        
    def get_pca(self,
               number_features,
               path_to_save_picture,
               min_var_exp = 0.05,
               feature_selection = True):
        
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

            pca_values_ = np.max(Fea_imp, 0)
            features_ = self.stacked_df.columns.values
            #pandas dataframe
            imp = pd.DataFrame(
                    {
                        'mean': pca_values_, 
                        'std': [np.nan]*len(pca_values_)
                        }, 
                    index=features_
                    )
                    
                
                
            #plot and save picture
            plotFeatImportance(
                            path_to_save_picture,    
                            imp,
                            number_features,
                            min_var_exp,
                            method='PCA', 
                            tag='First try',
                            simNum='PCA')
            print("PCA picture Saved in {}".format(path_to_save_picture))
            return imp

        else:
            #reduced matrix
            return X_new
