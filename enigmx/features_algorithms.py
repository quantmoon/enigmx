"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from enigmx.dbgenerator import databundle_instance 
from sklearn.model_selection import train_test_split
from enigmx.purgedkfold_features import featImportances, plotFeatImportance

#Main Feature Importance Class
class FeatureImportance(object):
    """
    Clase hija computable Feature Importance.
    
    Ejecuta todos los cómputos directos del feature importance.
    
    Para revisión de parámetros necesarios, ver:
        enigmx.classFeatureImportance (clase madre)
        
    """
    def __init__(self, 
                 server_name, 
                 database_name, 
                 driver,
                 uid,
                 pwd,
                 list_stocks,
                 bartype = 'VOLUME',
                 depured = True, 
                 rolling = True,
                 global_range = True,
                 features_sufix = 'feature',
                 referential_base_database = 'TSQL',
                 window = None,
                 win_type = None,
                 add_parameter = None, 
                 col_weight_type = 'weightTime',
                 col_t1_type  = 'horizon',
                 col_label_type = 'barrierLabel'):
        
        self.driver = driver
        self.uid = uid
        self.pwd = pwd
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
        
        self.referential_base_database = referential_base_database
        
    def __getStacked__(self):
        """
        Open individual csv and merge them to generate stacked.
        """
        
        #abrir la instancia sql base, la conexión y el cursor
        SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = self.database_name,driver = self.driver, uid = self.uid, pwd = self.pwd,
                    #boleano para crear la tabla
                    create_database = False, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range,
                    #referential database for SQL initial Loc
                    referential_base_database = self.referential_base_database
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
            columns_for_rolling = list(list_df[0].filter(
                like=self.features_sufix
                ).columns.values)     
        
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
    
    def __splitStacked__(self): 
        """
        Generate x_train, y_train, x_test & y_test for stacked df.
        
        Se extraen el test y train, y se seleccionan los features.
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
        """
        Método central para el proceso de feature importance.
        """
        
        # extracción de data útil
        x_train, x_test, y_train, y_test, dfStacked = self.__splitStacked__()
        
        # si el método seleccionado es 'MDA'
        if method == 'MDA':
            # verificación de modelo utilizado
            if type(model_selected).__name__=='RandomForestClassifier':
                raise ValueError(
                    "{} model is not allowed to implement 'MDA'".format(
                        'RandomForestClassifier'
                        )
                    )      
        
        # si el método seleccionado es 'MDI
        if method == 'MDI': 
            # verificación de modelo utilizado
            if type(model_selected).__name__!='RandomForestClassifier':
                raise ValueError(
                    "Only {} is allowed to implement 'MDI'".format(
                        'RandomForestClassifier'
                        )
                    )
            
        # importance rank, score con cpkf, y mean val (NaN)
        imp,oos,oob = featImportances(x_train, 
                                      y_train, 
                                      model_selected,
                                      method=method,
                                      sample_weight = y_train['w'],
                                      pctEmbargo=pctEmbargo,
                                      cv=cv,
                                      oob=False)
        
        # fit del modelo seleccionado para el featImp
        model_selected.fit(x_train,y_train['labels'])
        
        # score sin combinatorial purged kfold (socre del modelo)
        score_sin_cpkf = model_selected.score(x_test, y_test['labels'])
        
        print("Score sin el PurgedKFold:", score_sin_cpkf)
        print("Score con el PurgedKFold:", oos)
        
        print("Saving picture...")
        # guardado imagen de feature importance como prueba de control
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

    # método principal para obtención del PCA
    def get_pca(self,
               number_features,
               path_to_save_picture,
               min_var_exp = 0.05,
               feature_selection = True):
        
        # transformación escalar
        scaler = StandardScaler()
        scaler.fit(self.stacked_df)
        X = scaler.transform(self.stacked_df)
        
        # computación pca
        pca = PCA(n_components=min(X.shape))
        
        # transformación de X dims.
        X_new = pca.fit_transform(X)   

        pc_s = []
        
        # si pca está por encima del ratio de varianza esperado
        for i in pca.explained_variance_ratio_:

            if i > min_var_exp:
                pc_s.append(i)
        
        pc_s = np.array(pc_s)   

        # selección de features con base a PCA
        if feature_selection:
            
            # valor de features importance
            Fea_imp = abs(pca.components_[:pc_s.shape[0],:])

            pca_values_ = np.max(Fea_imp, 0)
            features_ = self.stacked_df.columns.values
            
            # pandas dataframe de importance
            imp = pd.DataFrame(
                    {
                        'mean': pca_values_, 
                        'std': [np.nan]*len(pca_values_)
                        }, 
                    index=features_
                    )
                    
            # plot and save picture
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
            # reduced matrix
            return X_new
