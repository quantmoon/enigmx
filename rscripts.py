import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter

def convert_pandas_to_df(df):
    with localconverter(robjects.default_converter + pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(df)
    return r_from_pd_df

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
