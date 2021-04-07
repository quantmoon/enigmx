"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
import numpy as np
import pandas as pd
from numba.typed import List
from statsmodels.regression.linear_model import OLS
from numba import njit,float64,jit,float32,int64,typeof,char

# FUNCION 1: Recibe de insumo un DataFrame en tiempo (Yt) 
# y devuelve sus retornos (Yt, Yt-1,... Yt-n) para n lags
def lagDF(df0,lags): #Obtiene los n lags y los anida a la derecha del series
    df = np.ndarray(shape=(len(df0)-lags))
    for lag in range(0,lags+1):
        arr = np.roll(df0,lag)[lags:]
        df = np.column_stack((df,arr))
    df = df[:,1:]
    #df1 va a ser un dataframe con yt, yt-1, yt-2
    #según n lags para las series que recibe de insumo
    return  df


#FUNCION 2: Da la forma de la regresión : diff(Yt) = Yt-1 + diff(Yt-nlags)
#Recibe la matriz a dividir y arroja el "y" y los "x" para la regresión
def getYX(series,constant,lags): 
    series_=np.ediff1d(series)[1:] #diff(Yt)
    x=lagDF(series_,lags) # lag(diff(Yt))
    series = series.reshape(-1,1)
    #Modifica x[0]=diff(Yt), de forma que sea igual a Yt-1
    x[:,0]=series[-x.shape[0]-1:-1,0] 
    y=series_[-x.shape[0]:] #Y = diff(Yt)
    if constant!='nc': #CONSTANTES
        x=np.append(x,np.ones((x.shape[0],1)),axis=1)
        if constant[:2]=='ct':
            trend=np.arange(x.shape[0]).reshape(-1,1)
            x=np.append(x,trend,axis=1)
        if constant=='ctt':
            x=np.append(x,trend**2,axis=1)
    # va a arrojar "y" y "x" 
    # según la forma de la regresión diff(yt) = yt-1+diff(yt-1) en n lags            
    return y,x

# FUNCION 3: Obtiene los criterios AIC y BIC para encontrar 
# el número de rezagos óptimo que reduce la correlación de los errores
# ARGUMENTOS: maxlags, el num máximo de lags sobre el que probar la regresión
def find_nlags(series, constant, lags):
    y, x = getYX(series, constant, lags)
    results = pd.DataFrame({"lag": [], "bic":[], "aic":[]})
    for lag in range(lags + 1):
        model = OLS(y, x[:, :lag+1]).fit()
        result = {"lag": [], "bic":[], "aic":[]}
        result["lag"] = lag+1
        result["bic"] = model.bic
        result["aic"] = model.aic
        results = results.append(result, ignore_index=True)
    nlag = int(results.bic.argmin())
    return nlag

@njit('f8[:,:],i8,unicode_type,i8') 
def get_allsadf(close,minSL,constant,lags): 
    """
    Obtención de SADF's para toda la serie
    Parámetros:
    close: Serie de tiempo principal (en escala logarítmica)
    minSL: 
    constant: constante para el cálculo del sadf
    lags: número de series lagueadas para el cálculo de los betas
    """
    
    bsadfs = List() #Lista de SADF's
    #empieza en 6 de forma arbitraria a calcular el sadf para c/punto 
    for i in range(6, len(close)): 
        series_=np.ediff1d(close[:-(i-5)])[1:] #diff(Yt)
        series_ = series_.reshape(series_.shape[0],1)
        
        df = series_[lags:] 
        #generación de array multidimensional con los valores laguedos
        for lag in range(0,lags+1):
            arr = np.roll(series_,lag)[lags:]
            df = np.column_stack((df,arr))
        x = df[:,1:]
        temp = close[:-(i-5)]
        #Modifica x[0]=diff(Yt), de forma que sea igual a Yt-1
        x[:,0] = temp[-x.shape[0]-1:-1,0] 
        y=series_[-x.shape[0]:] 
        if constant!='nc': #CONSTANTES --- esta parte no me queda muy clara
            x=np.append(x,np.ones((x.shape[0],1)),axis=1)
            #En libro pone "if constant[:2]=='ct'" 
            #<- como si fuera una lista las constantes ¿?
            if constant=='ct': 
                trend=np.arange(x.shape[0]).reshape(-1,1)
                x=np.append(x,trend,axis=1)
            if constant=='ctt':
                x=np.append(x,trend**2,axis=1)
     
        #para el bucle
        startPoints,bsadf,allADF=range(
            0,y.shape[0]+lags-minSL
            ),float64(-100),List()
        #cálculo de todos los ADF desde diferentes inicios
        for start in startPoints: 
            y_,x_=np.ascontiguousarray(y[start:]),np.ascontiguousarray(
                x[start:, ]
                )
            
            try:
                #x transpuesta * y
                xy=np.ascontiguousarray(np.dot(x_.T,y_)) 
                #x tanspuestas por x
                xx=np.dot(x_.T,x_) 
                if xx.shape==():
                    xxinv=1/xx
                else:
                    #inversa de x'x
                    xxinv=np.ascontiguousarray(np.linalg.inv(xx)) 
                bMean_=np.dot(xxinv,xy) #fórmula de beta de mínimos cuadrados
                err=y_-np.dot(x_,bMean_) #errores = y observado - y estimado
                #Error estándar de la función, denominador del ADF
                bVar_=np.dot(err.T,err)/(x_.shape[0]-x_.shape[1])*xxinv 
                Mean_ = bMean_[0]
                Std_= bVar_[0, 0]**.5
                allADF.append((Mean_/Std_)[0]) #ADF
                if allADF[-1] > bsadf: #Se busca el mayor ADF de la serie
                    bsadf = allADF[-1]
            except:
                pass

        bsadfs.append(bsadf)
        
    return bsadfs


def generalSADFMethod(original_frame, main_column_name, lags = None):
    """
    Main  function of the SADF Process.
    
    Adds column to base original_frame with "SADF" data.
    
    Inputs:
        - original_frame (dataframe)
        - main_column_name (str): column to apply the SADF
        - lags (None | int): number of lags to apply the SADF
    """
    
    assert(
        type(original_frame) == pd.DataFrame
        ), "'original_frame' param is not a pandas."
    
    # get the array price series
    price_array_series = np.log(
        original_frame[main_column_name].values
    ).reshape(-1,1)
    
    # if there is no lags, work time series as datapoints goes by
    if lags is None:
        lags = find_nlags(price_array_series, constant="nc", lags = 1)
    
    # list with points of super augmented dickey fuller 
    bsadfs = get_allsadf(price_array_series, 1 ,'nc', lags)
    
    # find differences in legnth of original df and SADF list
    if original_frame.shape[0]!= len(bsadfs):
        difference = abs(original_frame.shape[0] - len(bsadfs))
        _ = [-1]*difference
        _.extend(bsadfs)
    
    # add new SADF column to original frame containing list points
    original_frame["SADF"] = _
    
    # return reformed original frame
    return original_frame

def gettingSADF(etf_df, 
                lags = None, 
                main_value_name = 'value'):
    
    # función canalizadora del método SADF general
    
    sadf_frame = generalSADFMethod(etf_df, main_value_name, lags = lags)

    return sadf_frame