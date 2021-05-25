"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pickle
import datetime
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from enigmx.utils import simpleFracdiff
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from enigmx.dbgenerator import databundle_instance 
from sklearn.model_selection import train_test_split
from enigmx.purgedkfold_features import featImportances, plotFeatImportance


# PCA computation Snippet (8.5 page 119)
def get_eVec(dot,varThres):
    # compute eVec from dot prod matrix, reduce dimension
    eVal,eVec=np.linalg.eigh(dot)
    idx=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[idx],eVec[:,idx]
    #2) only positive eVals
    eVal=pd.Series(eVal,index=['PC_'+str(i+1) for i in range(eVal.shape[0])])
    eVec=pd.DataFrame(eVec,index=dot.index,columns=eVal.index)
    eVec=eVec.loc[:,eVal.index]
    #3) reduce dimension, form PCs
    cumVar=eVal.cumsum()/eVal.sum()
    dim=cumVar.values.searchsorted(varThres)
    eVal,eVec=eVal.iloc[:dim+1],eVec.iloc[:,:dim+1]
    return eVal,eVec
#----------------------------------------------------------------------------#
def orthoFeats(dfZ,varThres=.95):
    # Given a dataframe dfX of features, compute orthofeatures dfP
    dot=pd.DataFrame(np.dot(dfZ.T,dfZ),index=dfZ.columns,columns=dfZ.columns)
    eVal,eVec=get_eVec(dot,varThres)
    dfP=np.dot(dfZ,eVec)
    return dfP, eVal
#----------------------------------------------------------------------------#


######################### Main Feature Importance Class #######################
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
                    bartype_database = self.database_name,
                    #nombre de driver, usuario y password
                    driver = self.driver, 
                    uid = self.uid, 
                    pwd = self.pwd,
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
        
            #hace rolling solo sobre las columnas de feature de forma individual por accion
            list_df = [
                pd.concat(
                    [   #extrae aquellas columnas que no haran rolling
                        dataframe[dataframe.columns.difference(
                            columns_for_rolling
                            ).values],
                        #computa el rolling solo en columnas de features
                        dataframe[columns_for_rolling].rolling(
                            self.window, win_type=self.win_type
                            ).sum(std=self.add_parameter) #sum? #std?
                        ], axis=1
                                ).dropna()  for dataframe in list_df
                        ]
            
        #dataset final concadenado
        df_ = pd.concat(list_df).sort_index(
            kind='merge'
            )
        
        return df_, columns_for_rolling
    
    def __organizationDataProcess__(self): 
        """
        Inputs : matriz de features stackedada y nombres de features.
        
        Outputs : devuelve organizados el df de features y el df de labels
        """
        #obtener el dataframe de los valores stacked (rolleado) + columnas de features
        df_global_stacked, features_col_name = self.__getStacked__()
        
        #seleccionamos los features para su escalamiento
        elementsToScale = df_global_stacked[features_col_name]
        
        # definimos el objeto de escalamiento general de los features
        self.scalerObj = StandardScaler() 
        
        # fiteamos el objeto de escalamiento con todo los features del stacked
        self.scalerObj.fit(elementsToScale)
        
        # transformamos los features del stacked a valores escalados
        elementsScaled = self.scalerObj.transform(elementsToScale)
                
        # redefinimos los valores de los features con sus valores escalados
        df_global_stacked[features_col_name] = elementsScaled 
        
        #depuracion del dataframe de entrada, seleccionando solo los features
        if self.depured: 
            df = df_global_stacked[features_col_name]
        else:
            df = df_global_stacked
        
        #creación de la información de las etiquetas ('y' como df)
        y = df_global_stacked[self.col_label_type].to_frame('labels')
        
        #definición de 't1': horizonte temporal máx./barra (barrera vert.)
        y['t1'] = df_global_stacked[self.col_t1_type]
        
        #definición de 'w': pesos para cada observación en y como df
        y['w'] = df_global_stacked[self.col_weight_type]
        y.set_index(df_global_stacked['close_date'],inplace=True)
                
        #segmentacion y definition del stacked
        x = df
        x.set_index(df_global_stacked['close_date'],inplace=True)

        #retorna dataframe de features y dataframe de etiquetas
        return x, y, df_global_stacked
    
    def __checkingStationary__(self, pval_adf = '5%'):
        
        # extrae matriz de features, vector de labels del split stacked, y el global stacked
        df_base_matrix, yVectorArray, df_global_stacked = self.__organizationDataProcess__()
        
        # generamos copia del dataset
        xMatrixDf = df_base_matrix.copy()
        
        # contador de features no estacionarios transformados
        featuresTransformed = 0        
        
        for featuresName, featuresData in xMatrixDf.iteritems():
            
            # extrae el vector de la caracteristica sobre la que se itera
            temporalFeatArray = featuresData.values

            # calcula el Dickey-Fuller
            critical_values_adf = adfuller(temporalFeatArray)
            
            # desagregando estadistico y valor critico del dickey-fuller
            statVal, criticalVal = critical_values_adf[0], critical_values_adf[4][pval_adf]
                        
            # si el valor del estadistico no es menor al del valor critico 
            if not (statVal < criticalVal): 
                print("-----> Warning! '{}' is not stationary... ".format(
                        featuresName
                        )
                    )
                print(":::: Stationary Transformation Initialized >>>")
                # aplica transformacion estacionaria con diferenciacion fraccional
                newTemporalFeatArray = simpleFracdiff(
                    temporalFeatArray
                    )
                
                # reasigna la informacion estacionarizada al dataframe
                xMatrixDf[featuresName] = newTemporalFeatArray
                
                # agrega un valor al contador de features modificacods
                featuresTransformed+=1
        
        print(':::::::::::::'*5)
        print(f'\n::: >>> Stationary Process Checked!\
              Features Transformed {featuresTransformed} over {xMatrixDf.shape[1]}')
        print(':::::::::::::'*5)            
              
        # estandarizacion Mean/Std de los valores de la matriz de features
        dfStandarized = xMatrixDf.sub(
            xMatrixDf.mean(), axis=1
            ).div(xMatrixDf.std(),axis=1)   
        
        return dfStandarized, yVectorArray, df_global_stacked
    
    def get_feature_importance(self, 
                               pathOut, 
                               method, 
                               model_selected,
                               pctEmbargo = 0.01, 
                               cv=5, 
                               oob=False, 
                               variance_explained = 0.95):
        """
        Método central para el proceso de feature importance.
        """
        
        # extrae la matriz de features estacionaria y estandarizada, el vector de labels y el df stacked
        featStandarizedMatrix, labelsDataframe, dfStacked = self.__checkingStationary__() 
        
        # ejecuta el proceso de ortogonalizacion 
        orthogonalized_features_matrix, pca_egienvalues = orthoFeats(
            dfZ = featStandarizedMatrix,
            varThres = variance_explained
            )
        
        # convertimos la matriz ortogonalizada de array a dataframe
        new_orthogonalized_feat_matrix = pd.DataFrame(
            orthogonalized_features_matrix, 
            index=labelsDataframe.index, 
            columns=['PC_{}'.format(i) for i in range(
                1, orthogonalized_features_matrix.shape[1]+1
                )
                ]
            )         
        
        # extrae data de train/test desde la matriz ortogonalizada y el vector de etiquetas                
        x_train, x_test, y_train, y_test = train_test_split(
            new_orthogonalized_feat_matrix , labelsDataframe, random_state=42
            )
        
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
            
        # importance values, score con cpkf, y mean val (NaN)
        imp,oos,oob = featImportances(x_train, 
                                      y_train, 
                                      model_selected,
                                      method=method,
                                      sample_weight = y_train['w'],
                                      pctEmbargo=pctEmbargo,
                                      cv=cv,
                                      oob=False)
        
        # computamos el importance rank 
        featureImportanceRank = imp['mean'].rank()
        
        # computamos el PCA rank utilizando los eigenvalues
        pcaImportanceRank = pca_egienvalues.rank()

        print("       >>>> Complementary research...")
        # fit del modelo seleccionado para el featImp
        model_selected.fit(x_train,y_train['labels'])
        
        # score sin combinatorial purged kfold (socre del modelo)
        score_sin_cpkf = model_selected.score(x_test, y_test['labels'])
        
        print("FeatImportance Score without PurgedKFold:", score_sin_cpkf)
        print("FeatImportance Score with PurgedKFold:", oos)
        
        print("Saving featImportance picture...")
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
        
        nowTimeID = str(datetime.datetime.now().time())[:8].replace(':','')
        
        print('        >>> Saving Scaler Object... at ', pathOut)
        pickle.dump(self.scalerObj, open('{}/scaler_{}.pkl'.format(pathOut, nowTimeID),'wb'))

        # retorna featuresRank (0), pcaRank (1), accuracy CPKF, accuracy con CPKF, y el stacked
        return featureImportanceRank, pcaImportanceRank, score_sin_cpkf, oos, dfStacked