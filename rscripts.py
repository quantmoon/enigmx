import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import pandas as pd
import sys


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


def get_residual_matrix(matrix,features_to_transform,silhouettes,clusters,one_vs_all=False):

    importr("dplyr")

   
    if one_vs_all:

        robjects.r("""
        get_residuals_values <- function(matriz,cluster,y,features_to_transform){
        cluster_matriz <- select(matriz,c(cluster,y))
        coefficients = NULL
        for (var in features_to_transform){
          fit <- lm(cluster_matriz[,var]~ cluster_matriz[,y],data = cluster_matriz)
          cluster_matriz[var] <- residuals(fit)
          coefficients <- rbind(coefficients,fit$coefficients)
           }
        coefficients <- data.frame(coefficients)

        return (list(cluster_matriz,coefficients))
        }
        """)
        get_residuals = robjects.globalenv['get_residuals_values']
        df = pd.DataFrame()
        
        for idx,cluster in enumerate(clusters):
            features = [feature for feature in cluster if feature in features_to_transform[idx]]
            if len(features) == 0 : continue
            referent = silhouettes.loc[features].sort_values(ascending=False).index[0]
            cluster = [x for x in cluster if x != referent]
            res = robjects.vectors.StrVector(cluster)
            feat_to_trans = robjects.vectors.StrVector(features_to_transform[idx])
            residual_matrix,coefficients = get_residuals(matrix,res,referent,feat_to_trans)
            residual_matrix = convert_df_to_pandas(residual_matrix)
            coefficients = convert_df_to_pandas(coefficients)
            coefficients.index = features_to_transform[idx]
            print(coefficients)
            coefficients.to_csv(f'/var/data/csvs/cluster_{idx}_coefficients.csv')
            df = pd.concat([df,residual_matrix], axis = 1)

        return df


    else:
        robjects.r("""
        get_residuals_values <- function(matriz,cluster,features_to_transform){
        cluster_matriz <- select(matriz,cluster)
        coefficients = NULL
        for (var in features_to_transform){
          fit <- lm(cluster_matriz[,var]~ .,data=cluster_matriz)
          cluster_matriz[var] <- residuals(fit)
           coefficients <- rbind(coefficients,fit$coefficients)
           }
        coefficients <- data.frame(coefficients)
   
        return (list(cluster_matriz,coefficients))
        }
        """)

        get_residuals = robjects.globalenv['get_residuals_values']
        df = pd.DataFrame()

        idx = 0

        for cluster in clusters:

            res = robjects.vectors.StrVector(cluster)
            feat_to_trans = robjects.vectors.StrVector(features_to_transform[idx])
            residual_matrix,coefficients = get_residuals(matrix,res,feat_to_trans)
            residual_matrix = convert_df_to_pandas(residual_matrix)
            coefficients = convert_df_to_pandas(coefficients)
            coefficients.index = features_to_transform[idx]
            print(coefficients)
            coefficients.to_csv(f'/var/data/csvs/cluster_{idx}_coefficients.csv')
            df = pd.concat([df,residual_matrix], axis = 1)
            idx += 1

        return df


