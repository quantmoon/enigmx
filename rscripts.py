import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import sys

importr("dplyr")
robjects.r("options(warn=-1)")


def convert_pandas_to_df(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(df)
    return r_from_pd_df

def convert_df_to_pandas(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        pd_from_df = robjects.conversion.rpy2py(df)
    return pd_from_df



def adf_test(datos):

    importr("tseries")
    robjects.globalenv['datos'] = datos
    robjects.r("""
    get_adfs <- function(datos){
    feature = pval = NULL
    for (i in names(datos)[2:dim(datos)[2]]){
      t <- proc.time()
      serie <- ts(datos[i]) 
      adx <- adf.test(serie)
      time <- proc.time() - t
      cat(time[3]," ",i," ",adx$p.value, "\n")
      if (adx$p.value > 0.05){
      feature = c(feature,i)
      pval = c(pval,adx$p.value)
      }
     }
    df <- data.frame(feature = feature, pval = pval)
    return (df)

    }""")

    get_adfs = robjects.globalenv['get_adfs']

    df = get_adfs(datos)
    try:
        features = list(df[0])
    except:
        features = []

    return features


def regression_intracluster(matrix,clusters, path = 'D:/data_enigmx/'):

    matrix = convert_pandas_to_df(matrix)

    robjects.r("""
        get_residuals_values <- function(matriz,cluster){
       
        matriz <- matriz[,cluster]
        coefficients = residual_matriz = NULL
        idx = 1
        for (var in cluster){
          fit <- lm(matriz[,var]~ .,data = select(matriz,-var))
          residual_matriz <- cbind(residual_matriz,residuals(fit))
                 #var_coef <- c(fit$coefficients)
          var_coef <- append(fit$coefficients,0,after = idx)
          coefficients <- rbind(coefficients,var_coef)
          idx <- idx + 1
           }
        residual_matriz <- data.frame(residual_matriz)
        coefficients <- data.frame(coefficients)
        return (list(matriz,coefficients))
        }
        """) 

    get_residuals = robjects.globalenv['get_residuals_values'] 
    df = pd.DataFrame()
        
    for idx,cluster in enumerate(clusters):
        #features = [feature for feature in cluster if feature in features_to_transform[idx]]
        #if len(features) == 0 : continue
        #referent = silhouettes.loc[features].sort_values(ascending=False).index[0]
        #cluster = [x for x in cluster if x != referent]

        res = robjects.vectors.StrVector(cluster)
        #feat_to_trans = robjects.vectors.StrVector(features_to_transform[idx])
        residual_matrix,coefficients = get_residuals(matrix,res)
        residual_matrix = convert_df_to_pandas(residual_matrix)
        coefficients = convert_df_to_pandas(coefficients)
        coefficients.index = cluster
        cluster.insert(0,'Intercepto')
        coefficients.columns = cluster
        coefficients.to_csv(f'{path}cluster_{idx}_coefficients_intra.csv')
        df = pd.concat([df,residual_matrix], axis = 1)

    return df


def regression_intercluster(matrix,features_to_transform,clusters, path = 'D:/data_enigmx/'):

    matrix = convert_pandas_to_df(matrix)

    robjects.r("""
     get_residuals_values <- function(matriz,feats,cluster){
        coefficients = NULL
        for (var in feats){
          fit <- lm(matriz[,var]~ .,data=select(matriz,-cluster))
          matriz[var] <- residuals(fit)
          coefficients <- rbind(coefficients,fit$coefficients)
           }
        coefficients <- data.frame(coefficients)
        return (list(matriz,coefficients))
        }
        """)

    get_residuals = robjects.globalenv['get_residuals_values']
    df = pd.DataFrame()

    for idx,cluster in enumerate(clusters):

        res = robjects.vectors.StrVector(cluster)
        feat_to_trans = robjects.vectors.StrVector(features_to_transform[idx])
        residual_matrix,coefficients = get_residuals(matrix,feat_to_trans,res)
        residual_matrix = convert_df_to_pandas(residual_matrix)
        coefficients = convert_df_to_pandas(coefficients)
        coefficients.index = features_to_transform[idx]
        coefficients.to_csv(f'{path}cluster_{idx}_coefficients.csv')
        df = pd.concat([df,residual_matrix], axis = 1)

    return df