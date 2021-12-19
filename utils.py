"""
author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import talib
import sys
import os
import zarr
import math
import pyodbc
import urllib
import sqlalchemy
import numpy as np
import pandas as pd
from time import time
from numba import njit,float64,int64
from numba import typeof
import numba
from scipy import stats
from functools import reduce
from numba.typed import List
from datetime import datetime
from keras.utils import np_utils
import pandas_market_calendars as mcal
from fracdiff import StationaryFracdiff
from sklearn.preprocessing import LabelEncoder
from enigmx.protofeatures import protoFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from enigmx.purgedkfold_features import featImportances
#from enigmx.dbgenerator import databundle_instance


#general class to personalize ValueError Messages
class UnAcceptedValueError(Exception):
    """
    General Error Class.
    """
    def __init__(self,data):
        self.data = data    
    def __str__(self):
        return repr(self.data)

#VWAP general class | get a dataframe as input
def compute_vwap(ds):
    """
    The following code compute the VWAP
    (Volume Weighted Average Price)
    
    The main formula is:
    
    VWAP = ∑(Price * Volume)/∑(Volume)
    
    It takes a pd.DataFrame of ticks 
    as a parameter with some specific
    value types.
    
    Example 'df_ticks':
        
     	timestamp 	price 	volume
    0 	1586937600079 	283.82 	1000
    1 	1586937609055 	284.41 	216
    
    Format 'df_ticks':
    
    timestamp      int64
    price        float64
    volume         int64
    dtype: object
    
    IMPORTANT
    ---------
    This functions was created to use
    the output dataframe from the 
    "get_tick_pricing()" function.
    
    Another dataframe you'd like to use 
    have to fit the conditions described above.
    """
    return np.sum(ds.price * ds.vol) / np.sum(ds.vol)

def compute_vwap_alternative(ds):
    return np.sum(ds.value * ds.vol) / np.sum(ds.vol)

#computation of autocorrelations in a features matrix
def autocorr(x, lag=1):
    """
    Get autocorrelation from a numpy.ndarray data series ('x')
    based on some 'lag' window interval.
    """
    return np.corrcoef(np.array([x[:-lag],
                                 x[lag:]]))


nyse = mcal.get_calendar('NYSE')

#get trading days based on pandas mlcalendar library        
def sel_days(init,last):
    """
    Gets trading days based on NYSE Calendar.
    """
    early = nyse.schedule(start_date=init, end_date=last)
    dts = list(early.index.date)
    dt_str = [date_obj.strftime('%Y-%m-%d') for date_obj in dts]
    #trnsform as datetime.date() each string date
    return dt_str

#OPEN ZARR | Version 0.0: no useful simple version
def open_zarr_ga(stock,path):         
    new_path = path+stock+".zarr"
    zarrds = zarr.open_group(new_path)
    return zarrds

#OPEN ZARR | Version 1.0: simple open zarr (date by date)
def open_zarr(path, date_):
    
    zarrds= zarr.open_group(path)
    
    infoArrays_ = [
        (
            zarrds.timestamp[idx], 
            zarrds.value[idx], 
            zarrds.vol[idx] 
        ) 
        for idx, date_element in enumerate(
            zarrds.date
        ) if date_element in date_
    ]
    
    infoArrays = [
        (
            tupleArrays[0][tupleArrays[0]!=0], 
            tupleArrays[1][tupleArrays[1]!=0],
            tupleArrays[2][tupleArrays[2]!=0]
        ) 
        for tupleArrays in infoArrays_
    ]
    
    #return time, price, vol
    return infoArrays[0][0], infoArrays[0][1], infoArrays[0][2]

#OPEN ZARR | Version 1.1: global simple open zarr (range of dates)        
def open_zarr_global(path, range_dates):
    
    zarrds = zarr.open_group(path)
    
    start_date, end_date = range_dates[0], range_dates[-1]
    init_last = [start_date, end_date]
    
    idxs_ = [
            idx for idx, date_element in enumerate(
                                zarrds.date
                                ) 
             if date_element in init_last
            ]
    
    idxs_[-1] = idxs_[-1] + 1 
    prices = zarrds.value[idxs_[0]:idxs_[-1]].flatten()
    volume = zarrds.vol[idxs_[0]:idxs_[-1]].flatten()
    ts_ = zarrds.timestamp[idxs_[0]:idxs_[-1]].flatten()
    
    return ts_[ts_!=0], prices[prices!=0], volume[volume!=0]

#OPEN ZARR | Versión 1.2.: alternative singular open zarr (date by date)
def getTickPrices(path, date_):
    
    zarrds= zarr.open_group(path)
    
    infoArrays_ = [
        (
            zarrds.timestamp[idx],
            zarrds.value[idx]
        )   
        
        for idx, date_element in enumerate(
            zarrds.date
        ) if date_element in date_
    ][0]
    
    return (infoArrays_[0][infoArrays_[0]!=0],
            infoArrays_[1][infoArrays_[1]!=0])

#OPEN ZARR | Version 2.0: vectorized version (use range of dates)
def open_zarr_general(zarrDates, range_dates, zarrObject, ref_stock_name = None):    
    
    if range_dates not in zarrDates:
        print('Date selected is not in range_dates from zarr vector for {}'.format(
            ref_stock_name)
            )
        print(':::::>>> IndexError will raise...')
        
    idxs_ = [
            np.where(zarrDates == range_dates[0])[0][0],
            np.where(zarrDates == range_dates[-1])[0][0]+1
            ]

    #these 3 process are the longest 
    prices = zarrObject.value[idxs_[0]:idxs_[-1]]    
    volume = zarrObject.vol[idxs_[0]:idxs_[-1]]
    ts_ = zarrObject.timestamp[idxs_[0]:idxs_[-1]]

    return ts_[ts_!=0], prices[ts_!=0], volume[ts_!=0]

#Volatility from VolumeBar | no-vectorized
def __volatility__(volbar, window=1, span=100):
        
        #get prices data from volume bars
        prices = volbar.set_index(
                'datetime'
            ).iloc[:,[1]] 
    
        #define delta based on window length
        delta = pd.Timedelta(
            days=window #could be hours or seconds
        )
        
        #define timeframework to compute returns
        df0 = prices.index.searchsorted(
            prices.index - delta
        )
        
        #drop no-included datapoints
        df0 = df0[df0 > 0]
        
        #align data of p[t-1] to timestamps of pt[t]
        df0 = pd.Series(
            prices.index[df0-1],    
            index=prices.index[
                prices.shape[0]-df0.shape[0]:
            ]
        )
        
        #define p[t]
        finalValues = prices.loc[df0.index]
        
        #define p[t-1]
        initialValues = prices.loc[df0.values]
        
        #computing returns
        returns = finalValues.values/initialValues.values - 1
        
        #create a uniform timestamp dataset
        df0 = pd.DataFrame(
            {
                "volatility":returns.flatten(), 
            }, 
            #already in the right datetime
            index=finalValues.index 
        )
        
        #applying STD over EWMA from a defined span
        df0 = df0.ewm(span=span).std()

        #daily resampling to calculate volatility mean
        df0 = df0.resample(
            '{}{}'.format(window, 'D')
        ).mean()
            
        return df0.dropna()

#Define TripleBarrier Time Horizons based on prices
def get_horizons(prices, window=1):
    
    #delta based on window == volatility volbar
    delta=pd.Timedelta(days=window) #change to days for include in spyder
    
    #t1 price series
    t1 = prices.index.searchsorted(
        prices.index + delta
    )

    t1 = t1[t1 < prices.shape[0]]
    
    t1 = prices.index[t1]
    
    t1 = pd.Series(
        t1, 
        index=prices.index[:t1.shape[0]]
    )
    return t1

#Get Barrier Values: horizontal upper, vertical & horizontal lower | No Sigma
def getBarrierValues(path, date_, upperBound, lowerBound):
    
    timeTicks, tickPrices = getTickPrices(path, date_)
    
    conditionPrices_ = np.select(
        [
            tickPrices>upperBound,
            tickPrices<lowerBound,
            ], [
                tickPrices, tickPrices, 
            ]
        )

    try: 
        tickPrices[conditionPrices_!=0][0]
    except IndexError:
        price_ = 0 
        tsprice_ = 0
    else:
        price_ = tickPrices[conditionPrices_!=0][0]
        tsprice_ = datetime.fromtimestamp(
            timeTicks[conditionPrices_!=0][0] / 10**3
            )   
            
    return price_, tsprice_

#Get Barrier Values: horizontal upper, vertical & horizontal lower | Sigma inc.
def getBarrierValuesNew(path, date_, upperBound, lowerBound, sigma):
    
    if sigma > 0.5:
        raise ValueError(
            "'sigma' must be less than 0.5 only."
            )
    
    timeTicks, tickPrices = getTickPrices(path, date_)
    
    
    #AÑADIR CONDICIONALES: if conditionupper and conditionlower is empty.
    
    #calculate money difference over upper/lower barrier
    difference = upperBound - lowerBound
    
    #label '1.0'
    conditionUpper = tickPrices > upperBound
    
    #label '0.75
    conditionMidUpper = (
        tickPrices > upperBound - difference*sigma
        ) & (
            tickPrices <= upperBound
            )
    
    #label '0.25'
    conditionMidLower = (
        lowerBound <= tickPrices
        ) & (
            tickPrices < lowerBound + difference*sigma
            )
    
    #label '0.0'
    conditionLower = tickPrices < lowerBound
    
    #data selection
    conditionPrices_ = np.select(
        [
            conditionUpper, 
            conditionMidUpper,
            conditionMidLower,
            conditionLower
            ], [
                tickPrices, 
                tickPrices, 
                tickPrices, 
                tickPrices
            ]
        )
    
    try: 
        tickPrices[conditionPrices_!=0][0]
    except IndexError:
        price_ = 0 
        tsprice_ = 0
    else:
        price_ = tickPrices[conditionPrices_!=0][0]
        tsprice_ = datetime.fromtimestamp(
            timeTicks[conditionPrices_!=0][0] / 10**3
            )   
            
    return price_, tsprice_

#Get Barrier Coordinates: get barrier values by date | no-vectorized                                                 
def getBarrierCoords(path, 
                     initHorizon, 
                     endHorizon, 
                     upperBound, 
                     lowerBound, 
                     sigma):
    """
    General Triple Barrier Workflow:
        
    getBarrierCoords <-- getBarrierValues <-- getTickPrices
    
    """

    list_dates = sel_days(
        initHorizon, 
        endHorizon
        )[:-1]
      
    result = [
        getBarrierValues(
            path, date_, upperBound, lowerBound
            ) for date_ in list_dates
        ][0] #just matter the first one
    
    return result

#Get Barrier Coordinates: get barrier values by date | vectorized
barrierCoords = np.frompyfunc(
            getBarrierCoords, 5, 1
            )

#ZARR DATA VERIFICATION | Check if zarr price-vol-time data has same shape
def check_zarr_data(infoTuple):
    
    timestamps, price_, vol_ = infoTuple

    ts_dt = (
        timestamps*10e5
        ).astype(
            'int64'
            ).astype(
                'datetime64[ns]'
                ) - np.timedelta64(5, 'h')

    if price_.shape != vol_.shape:
            difference = abs(
                    price_.shape[0] - vol_.shape[0]
                    )
            
            if price_.shape[0] > vol_.shape[0]:
                price_ = price_[:-difference]
                ts_dt = ts_dt[:-difference]
                
            else:
                price_ = price_[:-difference]
                ts_dt = ts_dt[:-difference] 
                
    return ts_dt, price_, vol_    

#Back/Forward fill for one dimension arrays in case of NaN values appear
def forwardFillOneDimension(array):
    prev = np.arange(array.shape[0])
    idx = np.argwhere(
             pd.isnull(array),
        )
    prev[idx] = 0
    prev = np.maximum.accumulate(prev)
    return array[prev]

#Get Mean from each array of list/array of arrays | Vectorized form
def __timeVectorizedNoVwap__(array):
    tempArray = np.mean(array)
    return tempArray

timeVectorizedNoVwap = np.frompyfunc(
    __timeVectorizedNoVwap__,1,1
    )

#Get VWAP from each array of list/array of arrays | Vectorized form
def __vectorizedVwap__(arrayprice, arrayvol):
    return np.sum(arrayprice * arrayvol) / np.sum(arrayvol)

vectorizedSimpleVwap = np.frompyfunc(
        __vectorizedVwap__, 2, 1
    )

#Get Complex VWAP from each array of list/arrays of arrays | Vectorized form 
def __vectorizedComplexVwap__(arrayNd):
    return np.sum(arrayNd[:,0]*arrayNd[:,1])/np.sum(arrayNd[:,1])

vectorizedComplexVwap = np.frompyfunc(
    __vectorizedComplexVwap__, 1, 1
    )

#MatrixCombination (vstack) from two list/arrays of arrays | Vectorized form
def __MatrixCombination__(array1, array2):
    return np.vstack(
            (
                array1, 
                array2
                )
            ).T 

matrixCombination = np.frompyfunc(
    __MatrixCombination__, 2, 1
    )

#Numba Iterative VWAP from two list of arrays
@njit
def iterativeVwap(list_arrayPrice, list_arrayVol):
    result = []
        
    for idx in range(len(list_arrayPrice)):

        if np.sum(list_arrayVol[idx]) == 0:
            vwap_ = np.sum(
                list_arrayPrice[idx]*list_arrayVol[idx]
                )
            result.append(vwap_)
        else:
            vwap_ = np.sum(
                list_arrayPrice[idx]*list_arrayVol[idx]
                )/np.sum(
                    list_arrayVol[idx]
                    )
            result.append(vwap_)
            
    return result


#Tick Bar Construction Function | Vectorized
def __tickBarConstruction__(arrayTime, 
                            arrayPrice, 
                            arrayVol, 
                            num_time_bars):
    
    #ticks per bar
    num_ticks_per_bar = arrayPrice.shape[0] / num_time_bars
    
    #rounded ticks per bar
    num_ticks_per_bar = round(
            num_ticks_per_bar, -1
        ) * .25
    
    #grpId based on arrayPrice length
    grpId = np.arange(
        0, arrayPrice.shape[0]
        )//num_ticks_per_bar
    
    #idx definition for slicing 
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]
    
    #split time with idx
    groupTime = np.split(
        arrayTime, 
        idx
        )

    #split price with idx
    groupPrice = np.split(
        arrayPrice, 
        idx
        )
    
    #split vol with idx
    groupVol = np.split(
                arrayVol, idx
                )        
    
    #ungrouping subsets of bar time 
    one_single_list_time = list(zip(*groupTime))
    
    #get first item of groupTime to set initBarTime
    groupTimeFirst = one_single_list_time[0]
    
    #compute vwap from Price and Vol
    vwap = iterativeVwap(groupPrice, groupVol)

    return np.arange(1,len(vwap)+1), groupTimeFirst, vwap    
    
tickBarConstruction = np.frompyfunc(
    __tickBarConstruction__, 4, 3
    )    
    
#Volume Bar Construction Function | Vectorized
def __volumeBarConstruction__(arrayTime, 
                              arrayPrice, 
                              arrayVol, 
                              num_time_bars):

    #volume cumsum
    cumsumVol = np.cumsum(arrayVol)
    
    #total volume
    total_vol = cumsumVol[-1] 

    #vol per bar
    vol_per_bar = total_vol / num_time_bars

    #vol per bar rounded
    vol_per_bar = round(vol_per_bar, -1) * .2

    #grpId based on cumsumVol matrix
    grpId = cumsumVol//vol_per_bar
    
    #idx position using grpId
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]
    
    #split time with idx
    groupTime = np.split(
        arrayTime, 
        idx
        )

    #split price with idx
    groupPrice = np.split(
        arrayPrice, 
        idx
        )
    
    #split vol with idx
    groupVol = np.split(
                arrayVol, idx
                )        
    
    #get first item of groupTime to set time
    groupTimeFirst  = list(zip(*groupTime))[0]
    
    #compute vwap from Price and Vol
    vwap = iterativeVwap(groupPrice, groupVol)
     
    return np.arange(1,len(vwap)+1), groupTimeFirst, vwap

volumeBarConstruction = np.frompyfunc(
    __volumeBarConstruction__, 4, 3
    )

#Get Information (time, price & volume) from Barrier direction | vectorized
def __getVectInfoArrays__(dayIdxListVect, direction):
    
    zarrObject = zarr.open(direction)
    
    prices = zarrObject.value[dayIdxListVect]    
    volume = zarrObject.vol[dayIdxListVect]
    ts_ = zarrObject.timestamp[dayIdxListVect]
    
    return ts_[ts_!=0], prices[ts_!=0], volume[ts_!=0]

getVectInfoArrays = np.frompyfunc(
    __getVectInfoArrays__, 2, 3
    )

#Simple Fractional Differentiation Function
def simpleFracdiff(priceArray, 
                   window_fracdiff=2):
    """
    This functions compute a simnple fractional differentiation
    without any vectorization process.
    """
    
    assert window_fracdiff >= 2, "Window fracdiff no greater/equal than 2."
                                
    #transformation 1D to 2D single col array
    X_ = np.vstack(priceArray)
            
    #fracdiff instance
    fracdiff_ = StationaryFracdiff(
                    window=window_fracdiff,
                    pvalue=0.01
                    )
    

    #fracdiff calculation
    fracdiff_results = fracdiff_.fit_transform(X_)
            
    #where fracdiff result is not nan
    fracdiff_results = fracdiff_results[
                ~np.isnan(
                    fracdiff_results
                    )
                ]

    #length to insert zero values based on window
    fracdiff_results = np.r_[[0]*(window_fracdiff-1), fracdiff_results]
    
    return fracdiff_results 

#Get Float Range Values from start-stop-step values
def float_range(start, stop, step):
    lst=[]
    while start < stop:
        lst.append(start)
        start += step
    return lst

####Funciones para generar el df con precio y volumen####

#load data from symbol and based on a path and dates
def load_data(symbol,path,dates,drop_dup = False):
    lista = [] 
    zarrds = open_zarr_ga(symbol,path)
    for date in dates:
        X = construct_df_global(zarrds,date,drop_dup = drop_dup)
        new_ts = [datetime.fromtimestamp(i) for i in X['ts']/1000]
        X['ts'] = new_ts#ts_idx
        X.set_index('ts',inplace=True)
        lista.append(X)
    result = pd.concat(lista)
    return result

#construct a DataFrame from zarr objects and selected dates
def construct_df_global(zarrds,date_,drop_dup = False):
    arr = np.array(zarrds.date)
    idx = np.where(arr == date_)[0][0]
    prices =  zarrds.value[idx]
    prices = prices[prices>0]
    volume = zarrds.vol[idx]
    volume = volume[:len(prices)]
    timestamp = zarrds.timestamp[idx]
    timestamp = timestamp[:len(prices)]
    df = pd.DataFrame({
            'ts':timestamp,
            'price':prices,
            'vol':volume,
            })
    if drop_dup == True:
        df = df.drop_duplicates()
    else:
        df = df.copy()
    return df

# Numba | get absolute values from array differentiation    
@njit
def get_b(arr):
    arr = np.ediff1d(arr)
    arr = np.abs(arr)/arr
    return arr

# Numba | fill all nans with the forward value
@njit
def fill_nans(arr):
    a = 1
    for idx,i in enumerate(arr):
        if np.isnan(i):
            arr[idx] = a
        else:
            a = i
    return arr

# Numba | Get product from 2 arrays
@njit
def get_prod(arr1,arr2):
    return arr1*arr2


#######----------------------------------------------------------------####### 
####### Main IMBALANCE BARS Function #######

def input_(df, init_b ,tipo):
    '''
    init_b: valor inicial del tick direction, puede ser 1 o -1, no deberia tener gran impacto
    tipo: se refiere al tipo de barra imbalance, puede ser:
    DIB:= DOLLAR IMBALANCE BAR
    VIB:= VOLUME IMBALANCE BAR
    TIB:= TICK IMBALANCE BAR
    '''
    if init_b not in [-1,1]:
        raise ValueError ("init_b has to be -1 or 1")
    if tipo not in ["DIB","VIB","TIB"]:
        raise ValueError ("tipo not DIB, VIB or TIB")
    
    price = np.array(df['price'],dtype=np.float64)
    
    b = get_b(price)
    b = np.insert(b, 0, init_b, axis=0)
    b = fill_nans(b)   
    
    
    if tipo=="DIB":
        vol = np.array(df['vol'],dtype=np.float64)
        v = get_prod(vol,price)
        
        df['vol$'] = v
        df['b']  = b
        df['bv'] = get_prod(b,v)
    
    elif tipo=="VIB":
        v = df['vol']
        df['b']  = b
        df['bv'] = b*v
    elif tipo == "TIB":
        df['b'] = b
    
    #in" ")
    return df

def init_values(df,tipo,num_bars):
    
    X = input_(df,init_b = 1,tipo=tipo)
    df = X.copy()
    X["ones"] = 1
    
    '''
    Assumptions:
    1)  P(b_t=1) and P(b_t=-1) are stable. 
        What I mean is that a dayly probability is a good proxi for a 1/n day probability
    2)  The mean across all days in my sample of the count of the number of ticks in a day, 
        divided by the number of bars I expect to have in a day is a good initial guess for E(T) 
    3)  The mean across all days in my conditional sample (b=i) of the mean of dollar vol by ticks 
        in a day, is a good initial guess for E(v|b=i)     
    '''
    if tipo == "DIB" or tipo == "VIB":
        T = X[["bv","ones"]].resample('D').sum()
    elif tipo == "TIB":
        T = X[["b","ones"]].resample('D').sum()
    T= T[T["ones"]!=0]
    ET_init= T["ones"].mean()/num_bars

    ## conditional resampling: b==1 and b==-1
    X1= X[X["b"]==1]
    X2= X[X["b"]==-1]
    
    ## the number of bars I expect per day
    ## num_bars = 5
    if tipo=="DIB":
        ## resampling to compute E(T)

        T1 = X1[["vol$","ones"]].resample('D').sum()
        VOL1 = X1[["vol$","ones"]].resample('D').mean()
        T1 = T1[T1["ones"]!=0]
        VOL1 =VOL1[VOL1["ones"]!=0]
        T2 = X2[["vol$","ones"]].resample('D').sum()
        VOL2 = X2[["vol$","ones"]].resample('D').mean()
        T2 = T2[T2["ones"]!=0]
        VOL2 =VOL2[VOL2["ones"]!=0]

        ## computing probabilities
        total = T1.ones+T2.ones
        T1["ones"] = T1["ones"]/total
        T2["ones"] = T2["ones"]/total

        ## computing initial guess:
        Ev1 = VOL1["vol$"].mean()
        pb1 = T1["ones"].mean()
        Ev2 = VOL2["vol$"].mean()
        pb2 = T2["ones"].mean()

        ## 
        Ebv_init = pb1*(Ev1) - pb2*(Ev2)
        Eb_init = 2*pb1-1

    else:
        ## resampling to compute E(T)
        
        T1 = X1[["vol","ones"]].resample('D').sum()
        VOL1 = X1[["vol","ones"]].resample('D').mean()
        T1 = T1[T1["ones"]!=0]
        VOL1 =VOL1[VOL1["ones"]!=0]
        T2 = X2[["vol","ones"]].resample('D').sum()
        VOL2 = X2[["vol","ones"]].resample('D').mean()
        T2 = T2[T2["ones"]!=0]
        VOL2 =VOL2[VOL2["ones"]!=0]

        ## computing probabilities
        total = T1.ones+T2.ones
        T1["ones"] = T1["ones"]/total
        T2["ones"] = T2["ones"]/total

        ## computing initial guess:
        Ev1 = VOL1["vol"].mean()
        pb1 = T1["ones"].mean()
        Ev2 = VOL2["vol"].mean()
        pb2 = T2["ones"].mean()

        ## 
        Ebv_init = pb1*(Ev1) - pb2*(Ev2)
        Eb_init = 2*pb1-1        
        
    return X, ET_init, Eb_init, Ebv_init


#Compute Daily Bars for each Day | Imbalance Bars
@njit
def compute_Ts_DIB(bvs_val, ET_init, Ebv_init,alpha_1,alpha_2,len_Ts,num_days):
    Ts = List()
    ETs = List()
    Ebvs = List()    
    thresholds = List()
    ET_init = float64(ET_init)
    ETs.append(ET_init)   
    Ebvs_init = float64(Ebv_init)
    Ebvs.append(Ebvs_init)   
    i_prev, E_T, Ebv  = 0, ET_init, Ebv_init  
    n = bvs_val.shape[0]
    #bvs_val = bvs.values.astype(np.float64)
    cur_theta = 0 
    #thresholds[0] = np.abs(E_T * Ebv)
    for i in range(0, n):
        cur_theta += bvs_val[i]
        abs_theta = np.abs(cur_theta)        
        threshold = np.abs(E_T * Ebv)
        #thresholds[i] = float64(threshold)
        
        if abs_theta >= threshold:
            thresholds.append(float64(threshold))
            cur_theta = 0
            T = np.float64(i+1 - i_prev)
            Ts.append(T)
            if len(Ts)>len_Ts*num_days:
                val = List()
                val2 = List()
                val.append(float64(0))
                val2.append(float64(0))
                break
            #E_T = _ewma(np.array([ETs[-1],T]), window=100)[-1]
            E_T = (1-alpha_1)*ETs[-1]+alpha_1*T
            ETs.append(E_T)
            ### aca esto no me gusta, pero para probar ###
            #bv_new = _ewma(bvs_val[i_prev:i+1], window=100)[-1]
            bv_new = bvs_val[i_prev:i+1].mean()
            #Ebv = _ewma(np.array([Ebvs[-1],bv_new]), window=np.int64(100))[-1]
            Ebv = (1-alpha_2)*Ebvs[-1]+alpha_2*bv_new
            Ebvs.append(Ebv)
            # Ebvs = np.append(Ebvs,Ebv)
            #i_s = np.append(i_s,i)
            i_prev = i+1
            #
            # window of 3 bars
            #Ebv =  _ewma(np.array(Ebvs), window=np.int64(ET_init))[-1] #ET_init*3))[-1] )
    if len(Ts)>len_Ts*num_days:
        return val,val2
    else:
        return Ts,thresholds
 
def inputs(prices,volume,tipo,init_b =1):
    b = get_b(prices)
    b = np.insert(b, 0, init_b, axis=0)
    b = fill_nans(b)       
    if tipo=="DIB":
        vol = np.array(volume,dtype=np.float64)
        v = get_prod(vol,prices)
        bv = get_prod(b,v)
    return bv

#select values based on each date
def get_arrs(zarrds,arr,date_):
    idx = np.where(arr == date_)[0][0]
    prices = zarrds.value[idx]
    prices = prices[prices>0]
    volume = zarrds.vol[idx]
    volume = volume[:len(prices)]
    return prices,volume 

#imbalance hyperdict values    
def get_hyperp_dict(df,symbol_list):
    """
    Función que obtiene los hyperparámetros para el cálculo del 
    imbalance feature
    df: pandas proveniente del csv de los hiperparámetros
    symbol_list: lista de acciones
    """
    hyperp_dict = {}
    for symbol in symbol_list:
        if symbol in df.columns:
            alpha_1,alpha_2,num_bars = df[symbol][0],df[symbol][1],df[symbol][2]
            hyperp_dict[symbol] = [alpha_1,alpha_2,num_bars]
        else:
            hyperp_dict[symbol] = [0.0001,0.0001,15]
    return hyperp_dict

#imbalance feature values
def get_init_values(df,symbol_list,hyperp_dict):
    """
    Función que obtiene los valores iniciales para el cálculo de la 
    imbalance feature
    df: pandas con todo el consolidado de ticks de la acción    
    symbol_list: lista de acciones
    hyperp_dict: diccionario con los hiperparámetros

    """
    #En caso demore mucho, descomentar la siguiente línea:
    #df = df.iloc[-22:]
    init_vals = {}
    for symbol in symbol_list:
        ET_init, Eb_init, Ebv_init = init_values(
            df, tipo="DIB", num_bars=hyperp_dict[symbol][2]
            )
        init_vals[symbol] = [ET_init, Eb_init, Ebv_init]
    return init_vals                

#MAIN IMBALANCE COMPUTATION
def imb_feat(prices,
             volume,
             init_vals, #init vals is imbalance dictionary
             tipo = 'DIB',
             drop_dup=False):
    """
    Función de computo de las barras
    prices,volume: array de precio y volumen correspondiente al día
    init_vals: valores computados en get_init_values (esperanzas de thresholds)
    hyperp_dict: alpha_1, alpha_2, num_bars -> valores para el cálculo de las 
                 ewmas y el número de barras esperado
    tipo: imbalance de ticks, dollar, volume (Actualmente solo para dollar)
    """
    
    alpha_1 = init_vals[0]
    alpha_2 = init_vals[1]
    num_bars = init_vals[2] 
    ET_init = init_vals[3]
    Ebv_init = init_vals[5]
    
    #Generación del array "bv"
    array = inputs(prices, volume, tipo)
                        
    #Función para computar las barras
    Ts,thres = compute_Ts_DIB(array,
                              ET_init, 
                              Ebv_init,
                              alpha_1,
                              alpha_2,
                              num_bars*5,
                              1)   
    return len(Ts)

#Get imbalance parameters
def get_imbalance_parameters(imbalance_parameters_csv):
    "Read the csv file that contains the imbalance parameters stored"
    df = pd.read_csv(imbalance_parameters_csv)
    dict1={}
    for i in df.columns:
        dict1[i] = list(df[i])
    dict1.pop("Unnamed: 0")    
    return dict1

#############################################################################
####################### VOLATILITY COMPUTATION METHODS ######################
#############################################################################
#ALTERNATIVE 1 | Rolling Standard Deviation by indexing time
def doubleReformedVolatility(prices, window=1, span=100): 
    """
    La siguiente función calcula el Rolling STD por factor de agrupamiento 
    temporal.
    
    En cada agrupamiento 'D' (diario), calcula el retorno de cada valor 
    de la serie (close prices).
    
    Luego, estima la EWMA con base a determinado SPAN sobre la que 
    calcula la desviación estándar. 
    
    Posteriormente, aplica una métrica de resumen (media).
    
    Por último, hace un lagg de una unidad de tiempo para la utilización
    del valor de volatilidad obtenido.
    """

    ewma_over_intraday_bars = prices.groupby(
        pd.Grouper(freq='D')).apply(
            lambda x: 
                x.pct_change().fillna(0).ewm(span=span).std().fillna(0).mean()
            ).dropna()

    ewma_over_intraday_bars = ewma_over_intraday_bars.shift(window).dropna()
    return ewma_over_intraday_bars


#ALTERNATIVE 2 | Computation of volatility | Reformed Goal Marcos Version
def reformedVolatility(prices, window = 1, span=100):
    """
    La sig. función de volatilidad toma el 1er y último close de c/ periodo
    (diario, en este caso).
    
    Computa el retorno total de dicho periodo.
    
    Estima el Rolling STD con la EWMA para dicho periodo.
    
    EWMA simple explanation:
        https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
        
    This method does not require shift over certain window.
    """
    resampled_data = prices.resample(
            '{}{}'.format(window, 'D')
        )
    
    firstDailyPrices = resampled_data.first()
    lastDailyPrices = resampled_data.last()
    
    returns = lastDailyPrices/firstDailyPrices - 1
    volatility = returns.dropna().ewm(span=span).std().fillna(0)
    
    #Generación de una sma de volatilidad
    sma100 = talib.SMA(volatility, span)

    #Si la media móvil tiene un valor más de 3 veces más alto que el valor anterior
    # se capea a solo 3 veces el valor anterior
    for idx,i in enumerate(sma100):
        if idx == 0:
            pass
        else:
            if i > 3*sma100[idx-1]:
                volatility[idx+1] = 3*volatility[idx]

    return volatility
    

#ALTERNATIVE 3 | Computation of volatility | Simplified MLDF Version
def getDailyVolatility(prices, window=1, span=100):
    """
    Volatility computation on a rolling window.
    
    Get daily volatility from intraday points.
    
    Takes 'prices' as Dataframe|Series with datetime index. 
    
    Use span an window horizons.
    
    EWMA simple explanation:
        https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
    
    This method requires shift over certain window.
    """
    
    #define delta based on window length
    delta = pd.Timedelta(
            days=window #could be hours or seconds
        )

    #define timeframework to compute returns
    df0 = prices.index.searchsorted(
            prices.index - delta
        )
    
    #drop no-included datapoints
    df0 = df0[df0 > 0]

    #align data of p[t-1] to timestamps of pt[t]
    df0 = pd.Series(
            prices.index[df0-1],    
            index=prices.index[
                prices.shape[0]-df0.shape[0]:
            ]
        )
    
    #define p[t]
    finalValues = prices.loc[df0.index]
    
    #define p[t-1]
    initialValues = prices.loc[df0.values]
      
    #computing returns: avoid broadcasting error
    try: 
        returns = finalValues.values/initialValues.values - 1
    except ValueError:
        _ = [
            finalValues.values.shape[0], 
            initialValues.values.shape[0]
            ] 
        idx_ = _.index(min(_))
        if idx_ == 1:
            finalValues = finalValues[:-abs(_[0]-_[1])]
        
        returns = finalValues.values/initialValues.values - 1
                   
    #create a uniform timestamp dataset
    df0 = pd.DataFrame(
            {
                "volatility":returns.flatten(), 
            }, index= finalValues.index 
            #already in the right datetime
        )
    
    #applying STD over EWMA from a defined span___________________________
    df0 = df0.ewm(span=span).std().fillna(0)
        
    #daily resampling to calculate volatility mean
    df0 = df0.resample(
            '{}{}'.format(window, 'D')
        ).mean().dropna()
    
   #creating pd.Series with org. dates
    original_dates = prices.resample(
        '{}{}'.format(window, 'D')
        ).first().dropna() #have close prices, but useless
    
    #Series org.Dates combination with Series window resampled volatility
    combination_series = df0.volatility.combine(
        original_dates, min #min: select volatility and no close prices
        )
    
    #lag N periods forward to align volatilities | avoid look-ahead-bias
    lagged_no_bias_series = combination_series.shift(window).fillna(0)
    
    return lagged_no_bias_series


#Computation of volatility feature based on prices | MLDP Snippet 3.1 reformed
#lagg error
def volatility_feature(prices, window=1, span=100):
    """
    Volatility computation on a rolling window.
    
    Get daily volatility from intraday points.
    
    Takes 'prices' as Dataframe|Series with datetime index. 
    
    Use span an window horizons.
    
    EWMA simple explanation:
        https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc324.htm
    """
    
    #define delta based on window length
    delta = pd.Timedelta(
            days=window #could be hours or seconds
        )

    #define timeframework to compute returns
    df0 = prices.index.searchsorted(
            prices.index - delta
        )
    
    #drop no-included datapoints
    df0 = df0[df0 > 0]

    #align data of p[t-1] to timestamps of pt[t]
    df0 = pd.Series(
            prices.index[df0-1],    
            index=prices.index[
                prices.shape[0]-df0.shape[0]:
            ]
        )
    
    #define p[t]
    finalValues = prices.loc[df0.index]
    
    #define p[t-1]
    initialValues = prices.loc[df0.values]
      
    #computing returns: avoid broadcasting error
    try: 
        returns = finalValues.values/initialValues.values - 1
    except ValueError:
        _ = [
            finalValues.values.shape[0], 
            initialValues.values.shape[0]
            ] 
        idx_ = _.index(min(_))
        if idx_ == 1:
            finalValues = finalValues[:-abs(_[0]-_[1])]
        
        returns = finalValues.values/initialValues.values - 1
                   
    #create a uniform timestamp dataset
    df0 = pd.DataFrame(
            {
                "volatility":returns.flatten(), 
            }, index= finalValues.index 
            #already in the right datetime
        )
    
    #applying STD over EWMA from a defined span
    df0 = df0.ewm(span=span).std().fillna(0)
        
    #daily resampling to calculate volatility mean
    df0 = df0.resample(
            '{}{}'.format(window, 'D')
        ).mean()
    
    #drop nan values (weekends no-data)
    df0 = df0.dropna()
        
    #get just volatility values
    result_ = df0.volatility.values         
    
    #inser first item as zero (no-shiftting)
    result_volatilities = np.insert(result_, 0, 0)
        
    return result_volatilities


#############################################################################

#Comutation of BarVolume from Time, Price & Vol arrays
def simpleBarVolume(arrayTime, 
                    arrayPrice, 
                    arrayVol, 
                    num_time_bars,
                    alpha = 1/1e3):

    #volume cumsum
    cumsumVol = np.cumsum(arrayVol)
    
    #total volume
    total_vol = cumsumVol[-1] 

    #vol per bar
    vol_per_bar = total_vol / num_time_bars

    #vol per bar rounded
    vol_per_bar = round(
        vol_per_bar, -2
        ) * alpha

    #grpId based on cumsumVol matrix
    grpId = cumsumVol//vol_per_bar
    
    #idx position using grpId
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]

    return idx.shape[0]+1

#Computation of Tick Bars from Time, Price & Vol arrays
def simpleBarTick(arrayTime, 
                  arrayPrice, 
                  arrayVol, 
                  num_time_bars,
                  alpha = 1/1e3):
    
    #ticks per bar
    num_ticks_per_bar = arrayPrice.shape[0] / num_time_bars
    
    #rounded ticks per bar
    num_ticks_per_bar = round(
            num_ticks_per_bar, -2
        ) * alpha
    
    #grpId based on arrayPrice length
    grpId = np.arange(
        0, arrayPrice.shape[0]
        )//num_ticks_per_bar
    
    #idx definition for slicing
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]
         
    return idx.shape[0]+1

#Get List of Stocks from path direction
def global_list_stocks(data_dir, 
                       common_path = '.zarr', 
                       drop_extension = 5):
    
    """
    Función para obtener todas la lista de acciones de un repositorio local.
    
    Lee los nombres de los archivos existentes que tienen un 
    string en común (prefijo, infijo, sufijo).
    
    Elimina los strings innecesarios y se queda con el nombre de las acciones.
    
    Inputs:
        - data_dir (str): local path para buscar.
        - common_path (str): tipo de su-pre-in fijo para buscar.
        - drop_extension (int): # de eliminación de los últimos #
        
    outputs:
        - lista de acciones (lista de strings)
    """
    
    list_stocks = []
    for i,file in enumerate(os.listdir(data_dir)):
        if file.endswith(common_path):
            list_stocks.append(os.path.basename(file)[:-drop_extension])
    return list_stocks

# nominación de lista de acciones para ingesta directa de path 
EquitiesEnigmxUniverse = global_list_stocks

#Data Tunning Preparation| CAMBIARLO POR SQL 
def dataPreparation_forTuning(driver,
                              uid,
                              pwd,
                              server_name,
                              referential_base_database,
                              label_name = "barrierLabel",
                              features_sufix = "feature",
                              timeIndexName = "close_date",
                              set_datetime_as_index = True,
                              set_instance = None,
                              stationary_stacked = None,
                              cutpoint = None,
                              variables = None,
                              step = ''):
    """
    Función para preparar un csv para la utilización con los modelos.
    
    Inputs:
        - csv_path (str): dirección local donde se encuentra el csv
        - label_name (str): nombre del label en dicho pandas extraído del csv-path.
        - features_sufix (str): sufijo de las columnas de features en el df.
        - timeIndexName (str): nombre de la columna temporal a usarse como Idx de evento.
        - set_datetime_as_index (bool): setear el idx de evento como idx del dataframe.
        
    Output:
        - Tuple:
            * X : matriz de features
            * y : vector de label
            * t1: vector de timeIndex 
    """
    
    if not set_datetime_as_index:
        raise ValueError(
            "Only True available for datatime as index until now."
            )

    SQLFRAME, dbconn, cursor = set_instance(
                    driver = driver, uid = uid, pwd = pwd,
                    #nombre del servidor SQL Local
                    server = server_name,
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = "BARS_STACKED",
                    #boleano para crear la tabla: si la tabla está creada, debe >
                    create_database = False,
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = True,
                    #referential SQL Database name just for initialization
                    referential_base_database = referential_base_database
                    )


    #Se cambia el nombre del punto de corte para que la matriz se pueda leer de la base de datos
    cutpoint = '_'.join(str(cutpoint).split('.'))

    if stationary_stacked:
            matriz = f"STACKED_{step}_STATIONARY_{cutpoint}"
    else:
            matriz = f"STACKED_{step}_{cutpoint}"

    # extrae matriz de features estacionaria-estandarizada de la base de datos, así como serie de labels
    dfStacked = SQLFRAME.read_table_info(
             statement = f"SELECT * FROM [BARS_STACKED].[dbo].{matriz}",
            dbconn_= dbconn,
            cursor_= cursor,
            dataframe=True
            )
 
    
    
    y = SQLFRAME.read_table_info(
            statement = f"SELECT * FROM [BARS_STACKED].[dbo].LABELS_{step}",
            dbconn_= dbconn,
            cursor_= cursor,
            dataframe=True
            )

    #dfStacked = dfStacked.iloc[:1000]
    #y = y.iloc[:1000]

    
    #Asignación de índice del stacked
    dfStacked.index = dfStacked[timeIndexName]
    dfStacked.drop(timeIndexName,axis = 1,inplace = True)
    y.index = y[timeIndexName]
    y.drop(timeIndexName,axis = 1,inplace = True)
    y = y['labels']

    # array conteniendo los nombres de features ingresados
    variables = [variable.strip() for variable in variables.split(',')]

    X = dfStacked.copy()
    X = X[variables]
    
#    # selección únicamente de label
#    y = dfStacked[label_name]
    
    # ordenamiento serie-temporal según timeIndexName
    X = X.sort_index()
    y = y.sort_index()

    dfStacked = dfStacked.sort_index()

    # Para calcular purga y embargo
    if step == 'BACKTEST':
        X['horizon'] = dfStacked['horizon']
        X['horizon'] = X['horizon'].astype('datetime64[ns]')
        X['barrierTime'] = dfStacked['barrierTime'].astype('datetime64[ns]')
        X['close_price'] = dfStacked['close_price']
        X['barrierPrice'] = dfStacked['barrierPrice']
        X['bidask_spread'] = dfStacked['bidask_spread']

    

    # obtención de vector de timeIndex
    t1 = pd.Series(data=y.index, index=y.index)
    
    return X, y, t1


def data_heuristic_preparation_for_tunning(csv_path,
                                           list_heuristic_elements):
    # obtención del df stacked desde path
    dfStacked = pd.read_csv(
        csv_path, 
        ).dropna()
    
    # array conteniendo los elementos de la data según nombre de columna 
    dataframe = dfStacked[list_heuristic_elements]

    return dataframe
    
#############################################################################
#############################################################################
###################FUNCTIONS FOR COMBINATORIAL PURGED K-FOLD################# 
#############################################################################
#############################################################################

#SI MODIFICA FEATURES EN BETSIZE_TEST O HYPERPARAMETER_TUNING_TEST
#DEBE AGREGAR/QUITAR DICHOS FEATURES AQUÍ

#no useful features
keep_same = {
    'datetime', 'upper', 'lower',
    'horizon','priceAchieved','tripleBarrier', 
    'timeAchieved', 'time','special_time'
}

def dataPreparation(data_csv, 
                    feature_sufix = 'feature',
                    label_name='barrierLabel', 
                    timeIndexName = "close_date",
                    timeLabelName = "horizon"): 
    
    """
    Función para preparar un csv para la utilización con los modelos.
    
    Inputs:
        - csv_path (str): dirección local donde se encuentra el csv
        - label_name (str): nombre del label en dicho pandas extraído del csv-path.
        - features_sufix (str): sufijo de las columnas de features en el df.
        - timeIndexName (str): nombre de la columna temporal a usarse como Idx de evento.
        - set_datetime_as_index (bool): setear el idx de evento como idx del dataframe.
        
    Output:
        - Tuple:
            * X : matriz de features
            * y : vector de label
            * t1: vector de timeIndex 
    """    
    
    dfStacked = pd.read_csv(
        data_csv, 
        ).dropna()
    
    dfStacked[timeIndexName] = dfStacked[timeIndexName].astype('datetime64[ns]')
    dfStacked[timeLabelName] = dfStacked[timeLabelName].astype('datetime64[ns]')
    
    dfStacked = dfStacked.set_index(timeIndexName)
    
    featureColumnsName = list(dfStacked.filter(
                like= feature_sufix
                ).columns.values)       
    
    featureColumnsName.append(timeLabelName)
    
    X = dfStacked[featureColumnsName]

    y = dfStacked[label_name]
    
    X[label_name] = y
     
    return X

def dataPreparationOld(data, 
                    label_name='tripleBarrier', 
                    add_suffix = True, 
                    no_suffix = keep_same):
    """
    Prepara los datos de la carpeta 'single_stacked'
    para el combinatorial purged k-fold CV.
    
    1. Añade a los features el sufijo '_feature'
    2. Convierte 'horizon' y 'datetime' de string a datetime format
    3. Convierte la fecha ('datetime') en index
    4. Genera un dataframe de tipo:
    
           Index  ####_feature ####_feature ...    horizon target
      ####-##-##  %%%%%%%%%%%% %%%%%%%%%%%%     ####-##-## label#
      
    Recordar que este dataset debe tener siempre dos columnas:
    - 'horizon'
    - 'target' (esta última se genera en esta función de 'tripleBarrier')
    
    'target' se posiciona al final de este dataframe.
    """
    
    Y = data[label_name].values
      
    index = pd.to_datetime(data["datetime"]).values
    horizon =  pd.to_datetime(data["horizon"]).values
    
    data.columns = [
        '{}{}'.format(c, '' if c in keep_same else '_feature') 
              for c in data.columns
    ]
    
    #choose only features ('column names with feature-string')  
    X = data.filter(like='feature')
    X["index"],X["horizon"],X["target"] = (
            index, horizon, Y
        )
    X.set_index("index", inplace=True)    
    return X

def nCk(N,k): 
    '''
    Calcula la combinatoria de n sobre k
    
    Parametros:

    N:= Número total de particiones sobre nuestros datos
    k:= Número de particiones utilizadas para predecir y formar los paths

    Output:
    
    Número total de posibles combinaciones train-test
    '''
    f = math.factorial 
    return f(N) / f(k) / f(N-k)

def split_map_CPCV(N,k):
    '''
    Retorna una matrix que identifica cada paths 
    como la combinación de N particiones 
    de todos los posibles ordenamientos del KFold
    Parametros:

    N:= Número total de particiones sobre nuestros datos
    k:= Número de particiones utilizadas para predecir y formar los paths

    Output:

    Matrix con elementos del 0 al N-1, 
    donde los números iguales y diferentes de zero forman un path
    '''
    model_split = int(nCk(N,k))
    
    split_map = np.zeros([N,model_split])  
    col = 0
    for base in range(N):
        for other in range(base +1,N):
            split_map[base,col] = 1
            split_map[other,col] = 1
            col += 1

    for row in range(N):
        for i in range(1,model_split):
            val = split_map[row,i]
            prev_val = np.max(split_map[row,:i])
        
            if val ==0:
                continue
            elif val == 1:
                split_map[row,i] = prev_val+1
    
    return split_map

def purge_embargo(X,train_indices,test_indices,embargo):
    '''
    Ejecuta el Purge y el Embargo en el training set
    Parametros(Input):
    X            : PANDAS DATAFRAME INDEXADO CON FECHAS, 
                   contine los features y la ventana de evaluacion (horizon)
    train_indices: los indices obtenidos con KFold
    test_indices : los indices obtenidos con KFold
    embargo      : Horizonte de embargo (en dias)
    
    Output:
    los indices del training despues del Purge y Embargo
    '''
    sub = X[['horizon']]
   
    train = sub.iloc[train_indices,:]
    test = sub.iloc[test_indices,:]
    
    start = test.index.values[0]
    end = test.index.values[-1]

    train_1 = train[train.horizon < start]
    train_2 = train[train.index > (end + np.timedelta64(embargo,'D'))]
    
    new_train = pd.concat([train_1,train_2])
    train.loc[:, 'new_index'] = train_indices.copy()
    train = pd.merge(train,new_train,
                     how='left',
                     left_index  = True, 
                     right_index = True).dropna()
    
    adj_train_indices = np.array(train.new_index)
    
    return adj_train_indices   


def master_sets(N,k,split_map,pieces,X,embargo):
    '''
    Retorna el vector de indices de todas las posibles combinaciones 
    de train-test set despues de aplicar el purged y embargo
    
    Parametros:

    N         := Número total de particiones sobre nuestros datos
    k         := Número de particiones utilizadas para predecir y 
                 formar los paths
    split_map := Matrix cuyos elementos iguales y mayores que zero 
                 forman un path
    pieces    := Vector cuyos elementos son los indices de cada partición 
                 train-test
    X         := PANDAS DATAFRAME INDEXADO CON FECHAS,
                 contine los features y la ventana de evaluacion (horizon)
    embargo   := Horizonte del embargo a aplicar (en días)

    '''
    model_split = int(nCk(N,k))

    model_index_splits = []

    for split in range(model_split):
        non_zero = np.nonzero(split_map[:,split])
        
        train = []
        test = []
    
        for piece in range(len(pieces)):
            if np.isin(piece,non_zero):
                test.append(pieces[piece])
            else:
                train.append(pieces[piece])
    
        model_index_splits.append((train,test))

    master_sets = []

    for i in range(len(model_index_splits)):
        
        sets = model_index_splits[i]
        train = sets[0]
        test = sets[1]

        train_indices = np.concatenate(train)
    
        adjusted_trains = []
    
        for i in range(len(test)):
            test_indices = test[i]
            adj_train_indices = purge_embargo(
                X, train_indices, test_indices, embargo
            )
            adjusted_trains.append(adj_train_indices)
        
        master_train = reduce(np.intersect1d, adjusted_trains)
        master_test = np.concatenate(test)
        master_sets.append((master_train, master_test))

    return master_sets 

##################Funciones para obtener los idxs  de cada grupo ############

def consecutive(data, stepsize=1):
    '''
    Detecta discontinuidades en un array
    '''
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)    

def groups_from_test(master_sets,N):
    '''
    Extrae los indices de cada grupo (pre purging y embargoing) 
    usando la lista con grupos de test (los rojos)
    la clave aqui fue notar que cada path 
    identifica completamente N-1 grupos
    '''
    groups_list = []
    for i in range(1,N-1):
        X = consecutive(master_sets[i][1])
        if i == 1:
            groups_list.append(X[0])
            a = X[0][-1]+1
            b = X[1][0]
            groups_list.append(np.arange(a,b))
            groups_list.append(X[1])
        else:
            groups_list.append(X[1])
    return groups_list

def paths(test_predictions,split_map,n_paths):
    '''
    Esta función agrupa los elementos de nuestro vector de predicciones
    para formar los paths y calcular las métricas correspondientes 
    
    INSUMOS:
    
    test_predictions := vector con todas las predicciones
    split_map        := matrix con el ordenamiento de los paths
    n_paths          := número de paths
    
    OUTPUT:
    
    vector con todos los paths 
    '''
    paths = []
    
    for y in range(1,n_paths+1):
        A = np.where(split_map==y)[1]
        counts = np.bincount(A)
        a = np.where(counts > 1)[0]
        a = a[0]
        Y = []
        for k in range(len(A)):
            
            if k < len(A)-1:
                if A[k] < a: 
                    j = 0
                elif A[k] == A[k+1]:
                    j = 0
                else: 
                    j = 1
            else: 
                j = 1
            i = A[k]
            Y.append(test_predictions[i][j])
        paths.append(Y)
    return paths

##########Func. accesitarias para particionar un vector según grupos#########

def list_for_train_test(master_sets,split,groups,train = True):
    '''
    Extrae los indices de cada grupo para el TRAIN 
    (post purging y embargoing). 
    La clave aqui fue notar que cada grupo en el TRAIN 
    está contenido en algun grupo de test
    '''
    i = split
    if train == True:
        Y = master_sets[i][0]
    else:
        Y = master_sets[i][1]
    lista_train = [] 
    for X in groups:
        Z = np.intersect1d(X,Y)
        if Z.shape[0] == 0:
            continue
        else:
            lista_train.append(Z)
    return lista_train

##############################################################################
##############################################################################
################### NEW INFORMATION DRIVEN BARS FUNCTION COMPUTATION #########
##############################################################################
##############################################################################

#New Tick Bar Construction Function | No Vectorized
def __newTickBarConstruction__(arrayTime, 
                               arrayPrice, 
                               arrayVol, 
                               alpha_calibration = 1e3):
    
    #ticks per bar calibrated by alpha value (int>1)
    #num_ticks_per_bar = arrayPrice.shape[0] / alpha_calibration
    num_ticks_per_bar = alpha_calibration
    
    #rounded ticks per bar
    num_ticks_per_bar = round(
            num_ticks_per_bar, -1
        ) #* .25
    
    #grpId based on arrayPrice length
    grpId = np.arange(
        0, arrayPrice.shape[0]
        )//num_ticks_per_bar
    
    #idx definition for slicing 
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]
    
    #split time with idx
    groupTime = np.split(
        arrayTime, 
        idx
        )

    #split price with idx
    groupPrice = np.split(
        arrayPrice, 
        idx
        )
    
    #split vol with idx
    groupVol = np.split(
                arrayVol, idx
                )        
     
    return groupTime, groupPrice, groupVol

#New Volume Bar Construction Function | No Vectorized
def __newVolumeBarConstruction__(
                                 arrayTime, 
                                 arrayPrice, 
                                 arrayVol, 
                                 arrayTickRule, 
                                 alpha_calibration=1e3): #agregar parametro tickrule spliteado

    #volume cumsum
    cumsumVol = np.cumsum(arrayVol)
    
    #total volume
    #total_vol = cumsumVol[-1] 

    #vol per bar
    #vol_per_bar = total_vol / alpha_calibration
    vol_per_bar = alpha_calibration

    #vol per bar rounded
    vol_per_bar = round(vol_per_bar, -1) #* .25
    
    #grpId based on cumsumVol matrix
    grpId = cumsumVol//vol_per_bar
    
    #idx position using grpId
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]
    
    #split time with idx
    groupTime = np.split(
        arrayTime, 
        idx
        )

    #split price with idx
    groupPrice = np.split(
        arrayPrice, 
        idx
        )
    
    #split vol with idx
    groupVol = np.split(
                arrayVol, idx
                )        
    
    #split array tick with idx
    groupTickRule = np.split(
                arrayTickRule, idx
                )
    
    ### groupTickRule
    
    return groupTime, groupPrice, groupVol, groupTickRule #, groupTickRule agrupado 

def __newDollarBarConstruction__(arrayTime, 
                                 arrayPrice, 
                                 arrayVol, 
                                 alpha_calibration=1e3):
    """
    New dollar bar construction vectorized base version.
    
    Deprecated
    """
    #volume cumsum
    cumsumDol = np.cumsum(arrayPrice * arrayVol)
    
    #total volume
    #total_dol = cumsumDol[-1] 

    #vol per bar
    #dol_per_bar = total_dol / alpha_calibration
    dol_per_bar = alpha_calibration

    #vol per bar rounded
    dol_per_bar = round(dol_per_bar, -1) #* .2

    #grpId based on cumsumVol matrix
    grpId = cumsumDol//dol_per_bar
    
    #idx position using grpId
    idx = np.cumsum(
        np.unique(
            grpId, return_counts=True
            )[1]
        )[:-1]
    
    #split time with idx
    groupTime = np.split(
        arrayTime, 
        idx
        )

    #split price with idx
    groupPrice = np.split(
        arrayPrice, 
        idx
        )
    
    #split vol with idx
    groupDol = np.split(
                arrayVol, idx
                )        
     
    return groupTime, groupPrice, groupDol 

#@njit
def OHLC_BAR(list_arrayPrice, list_arraytime, list_arrayVol, list_array_tick_rule):
    """
    No vectorized version.
    
    OHLC + VOLATILITY + VOLUME computation.
    
    Includes arrayTime values for OHLC prices.

    """
    barOHLC_plus_time_info = []
    
    #for each array of price (arrays in price same as in Time and Vol)
    for idx in range(len(list_arrayPrice)):
        
        #array 1d of prices
        subset_info = list_arrayPrice[idx]
        #array 1d of time
        subset_info_time = list_arraytime[idx]
        #array 1d of tickRule
        subset_array_tick_rule = list_array_tick_rule[idx]
        #array 1d of volume
        subset_info_volume = list_arrayVol[idx]
        
        open_ = subset_info[0]
        high_ = np.max(subset_info)
        high_index = np.where(subset_info==high_)[0][0]
        low_ = np.min(subset_info)
        low_index = np.where(subset_info==low_)[0][0]
        close = subset_info[-1]
        
        #basic volatility as simple standard deviation of bar prices
        basic_volatility = np.std(subset_info)
        
        volume_in_bar = np.cumsum(subset_info_volume)[-1]
        
        # calculamos los PROTO FEATURES (7 valores)
        proto_features_elements = protoFeatures(
            price_vector = subset_info, 
            volume_vector = subset_info_volume, 
            tick_rule_vector = subset_array_tick_rule 
            ).get_proto_features()
        
        barOHLC_plus_time_info.append(
            [
                open_, high_, low_, close,
                subset_info_time[0],
                subset_info_time[high_index], 
                subset_info_time[low_index],
                subset_info_time[-1], 
                basic_volatility, 
                volume_in_bar, 
                # protofeature 'feat_buyInitTotal'
                proto_features_elements[0],
                # protofeature 'feat_sellInitTotal'
                proto_features_elements[1],
                # protofeature 'feat_signVolSide'
                proto_features_elements[2],
                # protofeature 'feat_accumulativeVolBuyInit'
                proto_features_elements[3], 
                # protofeature 'feat_accumulativeVolSellInit'
                proto_features_elements[4],
                # protofeature 'feat_accumulativeDollarValue'
                proto_features_elements[5],
                # protofeature 'feat_hasbrouckSign'
                proto_features_elements[6]
                ]
            )

    return barOHLC_plus_time_info 

def __OHLCBARVEC__(arrayPrice, arrayTime, arrayVol):
    
    """
    Vectorized version.
    
    OHLC + VOLATILITY + VOLUME computation.
    
    Includes arrayTime values for OHLC prices.
    """
            
    open_ = arrayPrice[0]
    high_ = np.max(arrayPrice)
    high_index = np.where(arrayPrice==high_)[0][0]
    low_ = np.min(arrayPrice)
    low_index = np.where(arrayPrice==low_)[0][0]
    close = arrayPrice[-1]

    #basic volatility as simple standard deviation of bar prices    
    basic_volatility = np.std(arrayPrice)
    
    volume_in_bar = np.cumsum(arrayVol)[-1]
    
    information_list = [
        open_, high_, low_, close, 
        arrayTime[0],
        arrayTime[high_index], 
        arrayTime[low_index],
        arrayTime[-1], 
        basic_volatility,
        volume_in_bar
        ]
    
    return information_list
    
OHLC_BAR_VEC = np.frompyfunc(
    __OHLCBARVEC__, 2, 1
    )    

def infoBarGenerator(grp_time, 
                     grp_prices,
                     grp_vols,
                     grp_tick_rule,
                     bartype):
    
    """
    Takes the group list of arrays information from 
    'time', 'prices' and 'vols' and, based on the 
    string 'bartype', compute OHLC Bars (inc. 'VOLATILITY' + 'BARVOLUME')
    and VWAP.
    """
    
    grp_prices = List(grp_prices)
    grp_vols = List(grp_vols)
    
    if bartype == 'tick':
        #compute vwap from Price and Vol by bar
        vwap = iterativeVwap(grp_prices, grp_vols)     
        
        #List of lists of values:
        #[O, H, L, CL, O-Date, H-Date, L-Date, C-Date, volatility, bar volume]
        # incluye tambien los 7 protofeatures ('feat_...')         
        OHLCBars = OHLC_BAR(
            grp_prices, grp_time, grp_vols
            ) #ad '_VEC' to func. name for vectorized version
        
        #returns object
        return OHLCBars, vwap
    
    elif bartype=='volume':
        #compute vwap from Price and Vol by bar
        vwap = iterativeVwap(grp_prices, grp_vols)     
        
        #List of lists of values:
        #[O, H, L, CL, O-Date, H-Date, L-Date, C-Date, volatility, bar volume] 
        # incluye tambien los 7 protofeatures ('feat_...') 
        OHLCBars = OHLC_BAR(
            grp_prices, grp_time, grp_vols, grp_tick_rule #groupTickRule agrupado 
            ) #ad '_VEC' to func. name for vectorized version
        
        #returns object        
        return OHLCBars, vwap
    
    elif bartype=='dollar':
        #compute vwap from Price and Vol by bar
        vwap = iterativeVwap(grp_prices, grp_vols)     
        
        #List of lists of values:
        #[O, H, L, CL, O-Date, H-Date, L-Date, C-Date, volatility, bar volume] 
        # incluye tambien los 7 protofeatures ('feat_...')         
        OHLCBars = OHLC_BAR(
            grp_prices, grp_time, grp_vols
            ) #ad '_VEC' to func. name for vectorized version
        
        #returns object        
        return OHLCBars, vwap
    
    else:
        raise ValueError("Not recognized bartype string input")

##############################################################################
##############################################################################
############### PRE-ELEMENTS FOR TRIPLE BARRIER COMPUTATION ##################
##############################################################################
##############################################################################
        
def getTripleBarrierOHLC(df):
    """
    Función que permite computar los elementos básicos previos de una Triple Barrera.
    
    Inputs:
        - df (pd.DataFrame): pandas dataframe con valores OHLC de la barra.
        
    Output:
        - dataframe con las sig. columnas nuevas:
            * uppber_barrier
            * lower_barrier
            * barrier_price
            * barrier_date
    """
    #upper and loser barriers column definition
    df["upper_barrier"] = df.close * (1 + df.volatility)
    df["lower_barrier"] = df.close * (1 - df.volatility)
    
    #drop when is zero volatility and reset index 
    data = df.query("volatility!=0").reset_index(drop=True)
    
    #list of barrier prices & dates to save over group iteration
    price_barriers_touched = []
    date_barriers_touched = []
    
    #group iteration by each "horizon" value
    for idx, day_bound in enumerate(data["horizon"].values):
        
        #subdataset created from each close date to each day limit/bound
        df_masked = data[
            data.close_date.between(data.close_date, day_bound)
        ]
        
        #definition of upper and lower barrier from index 'idx'
        upper_barrier, lower_barrier = (
            df_masked.upper_barrier.iloc[idx], 
            df_masked.lower_barrier.iloc[idx]
        ) 
                
        #subdataset division one row further: cannot search in same bar
        check_ohlc_prices, check_ohlc_dates = (
            df_masked.iloc[idx+1:,:4].reset_index(drop=True), 
            df_masked.iloc[idx+1:,4:8].reset_index(drop=True)
        )
        
        #get first ocurrencies of upper and lower barrier
        first_ocurrence_upper = np.column_stack(
            np.where(check_ohlc_prices>upper_barrier)
            )
        first_ocurrence_lower = np.column_stack(
            np.where(check_ohlc_prices<lower_barrier)
            )
        
        ############### First Conditional Groups: Positioning ############### 

        #upper and lower barrier is touched in the same row/event
        if (first_ocurrence_upper.shape[0] != 0 and 
            first_ocurrence_lower.shape[0] != 0):

            #row and column coords
            upper_coords = first_ocurrence_upper[0]
            lower_coords = first_ocurrence_lower[0]

        #only upper barrier is touched in a row/event
        if (first_ocurrence_upper.shape[0] != 0 and 
            first_ocurrence_lower.shape[0] == 0): 

            #row and column coords
            upper_coords = first_ocurrence_upper[0]
            lower_coords = [np.inf, np.inf]

        #only lower barrier is touched in a row/event
        if (first_ocurrence_upper.shape[0] == 0 and 
            first_ocurrence_lower.shape[0] != 0):

            #row and column coords
            upper_coords = [np.inf, np.inf] 
            lower_coords = first_ocurrence_lower[0]

        #there is no barrier touched in some row/event
        if (first_ocurrence_upper.shape[0] == 0 and 
            first_ocurrence_lower.shape[0] == 0):
            #no rows and no columns
            upper_coords, lower_coords = [], []

        ############### Second Conditional Groups: Finding ##################
        
        #if any barrier is no touched, set date/price values as 0    
        if len(upper_coords) == 0 and len(lower_coords) ==0:
            first_date_barrier_touched, first_price_barrier_touched = 0, 0

        #check if upper barrier is touched first
        elif upper_coords[0]<lower_coords[0]:

            row_index, column_index = upper_coords[0], upper_coords[1]

            first_date_barrier_touched = \
                check_ohlc_dates.loc[row_index][column_index]
            first_price_barrier_touched = \
                check_ohlc_prices.loc[row_index][column_index]

        #check if lower barrier is touched first
        elif upper_coords[0]>lower_coords[0]:

            row_index, column_index = lower_coords[0], lower_coords[1]

            first_date_barrier_touched = \
                check_ohlc_dates.loc[row_index][column_index]
            first_price_barrier_touched = \
                check_ohlc_prices.loc[row_index][column_index]            

        #check if upper/lower barrier are touched in same row, and get first
        elif upper_coords[0]==lower_coords[0]:
            
            #unique event row is the same for upper/lower case
            unique_event_row = upper_coords[0]
            
            #create a list with columns indices [sorted as O-H-L-C]
            base_list_dates_indices = [upper_coords[1], lower_coords[1]]
            
            #define the dates in which upper/lower case occurs
            upper_date = \
                check_ohlc_dates.loc[unique_event_row][upper_coords[1]]
            lower_date = \
                check_ohlc_dates.loc[unique_event_row][lower_coords[1]]
            
            #create a list of each date for upper and lower case
            list_base_dates = [upper_date, lower_date]

            #get the oldest event as the first event reached
            first_date_barrier_touched = min(list_base_dates)
            
            #get the index of the first barrier date touched
            index_first_barrier_date_touched = list_base_dates.index(
                first_date_barrier_touched
            ) 

            #price from OHLC cols with same index as 1st barrier dates touched
            first_price_barrier_touched = \
                check_ohlc_prices.loc[unique_event_row][
                    base_list_dates_indices[index_first_barrier_date_touched]
                ]
            
        #save price barrier and date barrier for each idx horizon case 
        price_barriers_touched.append(first_price_barrier_touched)
        date_barriers_touched.append(first_date_barrier_touched)
    
    data["barrier_price"] = price_barriers_touched
    data["barrier_date"] = date_barriers_touched
    
    return data        

def barsNameDefinition(bartype, 
                       columnBaseNames = ["open", "high", "low", 
                                          "close", "open_date", 
                                          "high_date", "low_date", 
                                          "close_date", "basic_volatility",
                                          "bar_cum_volume", 
                                          "feat_buyInitTotal",
                                          "feat_sellInitTotal",
                                          "feat_signVolSide",
                                          "feat_accumulativeVolBuyInit",
                                          "feat_accumulativeVolSellInit",
                                          "feat_accumulativeDollarValue",
                                          "feat_hasbrouckSign",                                          
                                          "vwap", 
                                          "fracdiff"]
                       ):
    """
    Function to add a prefix ('bartype') to each information column name.
    """
    return [bartype +  "_" + colName for colName in columnBaseNames]
    
dataset_column_names = ["open_price", 
                        "high_price", 
                        "low_price", 
                        "close_price", 
                        "open_date", 
                        "high_date", 
                        "low_date", 
                        "close_date", 
                        "basic_volatility", 
                        "bar_cum_volume",
                        "feat_buyInitTotal",
                        "feat_sellInitTotal",
                        "feat_signVolSide",
                        "feat_accumulativeVolBuyInit",
                        "feat_accumulativeVolSellInit",
                        "feat_accumulativeDollarValue",
                        "feat_hasbrouckSign",
                        "vwap", 
                        "fracdiff"]

##############################################################################
##############################################################################
################# NEW TRIPLE BARRIER COMPUTATION #############################
##############################################################################
##############################################################################

def ErrorIndexIdxFirstTrue(param):
    """
    Test if there is an IndexError during where-search handling.
    """
    try:
        param[0][0]
    except IndexError:
        return None
    else:
        #only 1st value from tuple of single array
        return param[0][0]





@njit
def barrier_inside_computation(ts_timeframe,prices_timeframe,init_ts,last_ts,upper_bound,lower_bound):
    
    #print("Veamos Inner barrier_inside_computation | LLEVA NUMBA | UTILS.PY (line 2309)")    
    
    selected_indexes  = np.where((ts_timeframe>init_ts)&(ts_timeframe<last_ts))
    segmented_prices = prices_timeframe[selected_indexes[0]]
    segmented_timestamps = ts_timeframe[selected_indexes[0]]
    
    try:
        first_upper_barrier_idx = np.where(segmented_prices > upper_bound)[0][0]
    except:
        first_upper_barrier_idx = None
        
    try:
        first_lower_barrier_idx = np.where(segmented_prices < lower_bound)[0][0]
    except:
        first_lower_barrier_idx = None

    # conditional definition of barrierBoolValues (vertical barrier)
    if segmented_prices.shape[0] == 0:
        return  0, 0, last_ts

    # upper/lower barrier idx values are equal (None)
    elif first_upper_barrier_idx == first_lower_barrier_idx:
        return segmented_prices[-1, 0], 0, last_ts
    
    # case in which one barrier-type is touched
    else: 
        
        #if just upper barrier idx is None (only lower exists)
        if first_upper_barrier_idx is None:
            #set random greater value from lower idx for later comparisson
            first_upper_barrier_idx = first_lower_barrier_idx+1

        #if just lower barrier idx is None (only upper exists)
        if first_lower_barrier_idx is None:
            #set random greater value from upper idx for later comparisson
            first_lower_barrier_idx = first_upper_barrier_idx+1

        #upper barrier happens first (or it's unique) than lower barrier
        if first_upper_barrier_idx < first_lower_barrier_idx:
            #definition of horizontal upper barrier
            
            if segmented_timestamps[first_upper_barrier_idx:,0][0] < 0:
                print("       ::::::: >> Warning! Timestamp selected is negative!! \
                      Upper barrier Case | Check array: ")
                print(segmented_timestamps[first_upper_barrier_idx:,0])

            
            return segmented_prices[first_upper_barrier_idx:,0][0], 1 , segmented_timestamps[first_upper_barrier_idx:,0][0] 
        
        #lower barrier happens first (or it's unique) than upper barrier
        else:
            
            if segmented_timestamps[first_lower_barrier_idx:,0][0] < 0:
                print("       ::::::: >> Warning! Timestamp selected is negative!! \
                      Lower barrier Case | Check array: ")
                print(segmented_timestamps[first_lower_barrier_idx:,0])
            
            #definition of horizontal lower barrier
            return segmented_prices[first_lower_barrier_idx:,0][0],-1, segmented_timestamps[first_lower_barrier_idx:,0][0]

  

def vectorizedTripleBarrier(arr,arr2,path, init, init_ts, last, last_ts, upper_bound, 
                            lower_bound):
    """
    Función Principal para el cómputo de la Triple Barrera.
    
    Función ingestada en 'LabelTripleBarrierComputation'.
    
    Esta a su vez es ingestada en 'new_triple_barrier_computation'
    de tripleBarrier.py a través del multiprocesador 'ray'.
    
    Inputs:
        - path (str): dirección local donde se encuentran el .zarr de una acción det.
        - init (datetime series): pd.Series conteniendo las fechas de c/ bar-event (close_date)
        - init_ts (timestamp series): pd.Series conteniendo las fechas de c/ bar-event en formato timestamp
        - last (datetime series): pd.Series conteniendo las fechas máx. de c/ label (horizon).
        - last_ts (timestamp series): pd.Series conteniendo las fechas máx. de c/ label (horizon) en formato timestamp.
        - upper_bound (float series): pd.Series conteniendo el valor float-price máximo de la barrera horizontal.
        - lower_bound (float series): pd.Series conteniendo el valor float-price mínimo de la barrera horizontal.
        
    Output:
        - Tupla conteniendo tres valores:
            * finalPrice (float)
            * finalLabel (int: -1, 0 ó 1)
            * finalTimestamp (float)
    """
    #days range to search tripleBarrier | timeframe values
    daysList = sel_days(init, last) 

    #open zarr of selected stock
    #zarrds = zarr.open_group(path)
    
    #general zarr dates 
    #zarr_dates = zarrds.date.get_basic_selection()

    #finding dates indices in zarr_dates | timeframe idxs
    #dateIdxs = np.searchsorted(zarr_dates, daysList)    
    dateIdxs = 1
    #check in case there are no dates (missing data values)
    if dateIdxs==0:
        finalPrice, finalLabel, finalTimestamp = 0,0,0
        return finalPrice, finalLabel, finalTimestamp
    
    else: 
        prices_timeframe = ts_timeframe = np.array([])
        for i in daysList:
             prices_timeframe = np.hstack([prices_timeframe,arr[i]])
             ts_timeframe = np.hstack([ts_timeframe,arr2[i]])
        #for i in dateIdxs:
        #    prices_timeframe = np.hstack([prices_timeframe,arr[i+1]])
        #    ts_timeframe = np.hstack([ts_timeframe,arr[i+1]])
        prices_timeframe = prices_timeframe.reshape(-1,1)
        ts_timeframe = ts_timeframe.reshape(-1,1)
        
        finalPrice, finalLabel, finalTimestamp = barrier_inside_computation(
                                                                        ts_timeframe,
                                                                        prices_timeframe,
                                                                        init_ts,last_ts,
                                                                        upper_bound,lower_bound)
    return finalPrice, finalLabel, finalTimestamp 

########### NEW TRIPLE BARRIER COMPUTATION ######################

def LabelTripleBarrierComputation(barDataframe, stock, data_dir):
    """
    Función de ingesta e inicialización de la triple barrera.
    
    Ingesta la función "vectorizedTripleBarrier" para obtener resultados.
    
    Inputs:
        - barDataframe (pd.DataFrame): dataframe central sobre el que computar la triple barrera.
        - stock (str): nombre de la acción.
        - data_dir (str): path donde se encuentra los archivos .zarr con data x tick.
        
    Output:
        - barDataframe reformado inc. las siguientes columnas de la triple barrera:
            * 'barrierPrice'
            * 'barrierLabel'
            * 'barrierTime'
    """
    zarrds = zarr.open_group(data_dir + stock + ".zarr")
    t1 = time()
    
    arr = {}
    arr2 = {}

    for idx,i in enumerate(zarrds.date):
        arr[i] = zarrds.value[idx]
    for idx,i in enumerate(zarrds.timestamp):
        arr2[zarrds.date[idx]] = zarrds.timestamp[idx]

    print("Armado de arrays", stock,time() - t1)
    t2 = time()
    #tripleBarrier vectorized computation version
    tripleBarrierInfo = barDataframe.apply( 
        lambda row: 
            vectorizedTripleBarrier(arr,arr2,
                path = data_dir + stock + ".zarr", 
                init = row["close_date"],
                init_ts = datetime.timestamp(
                    row["close_date"].to_pydatetime()
                    )*1e3,
                last = row["horizon"],
                last_ts = datetime.timestamp(
                    row["horizon"].to_pydatetime()
                    )*1e3,
                upper_bound = row["upper_barrier"],
                lower_bound = row["lower_barrier"])
            , 
            axis=1
            )
    print(tripleBarrierInfo)
    #barrier columns information generation
    barDataframe[['barrierPrice', 'barrierLabel', 'barrierTime']] = \
        pd.DataFrame(
            tripleBarrierInfo.tolist(), 
            index=tripleBarrierInfo.index
            )    
    print(stock, time()-t2)
    return barDataframe


############################ Alternative Method ##############################

def bar_para(dato,num_bar):
    '''
    Función para el tunning de barras convencionales: tick, vol, dollar.
    
    dato    := array con el número de ticks, volume y dollar promedio por día
    num_bar := el número de barras que queremos obtener en promedio
    '''
    bar_para = dato/num_bar
    
    bar_para = bar_para.astype(int)
    
    dic = {
        'tick_t': bar_para [0],
        'volume_t': bar_para[1], 
        'dollar_t':bar_para[2]
        }
    # bar_para:=retorna un arr con el num de ticks, volume y dollar para bars
    return dic 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def saving_basic_bars(path_save, list_datasets, naming,
                      list_stocks, list_bar_names):
    """
    Function to save basic bars.
    """
    
    for idx, dataset in enumerate(list_datasets):
        print("Saving", list_stocks[idx], "...")

        #iteration by bartype dataframes in stock list
        for idx_bartype, dataframe in enumerate(dataset):

            saving_path = path_save + list_stocks[idx] + "_" + \
                list_bar_names[idx_bartype].upper() + "_" + naming + '.csv'

            dataframe.loc[:, 'open_date':'close_date'] = \
                dataframe.loc[:, 'open_date':'close_date'].astype(str)

        
            dataframe.to_csv(
                saving_path, 
                date_format='%Y-%m-%d %H:%M:%S', 
                index=False
                )    
    return "Saving Basic Bars Process Completed"

def saving_unique_entropy_bar(path_save, list_datasets, naming,
                              list_stocks, bar_name):
    """
    Function to save entropy series.
    """
    
    for idx, dataset in enumerate(list_datasets):
        print("Saving", list_stocks[idx], "...")

        saving_path = path_save + list_stocks[idx] + "_" + \
                bar_name.upper() + "_" + naming + '.csv'
        
        #save date as string to preserve mili/microsecond information
        dataset.loc[:, 'open_date':'close_date'] = \
                dataset.loc[:, 'open_date':'close_date'].apply(
                    lambda x: x.dt.strftime("%y-%m-%d %H:%M:%S.%f"),
                    axis=1
                    )              
                
        dataset.to_csv(
                saving_path, 
                date_format='%Y-%m-%d %H:%M:%S', 
                index=False
                )    
    return "Saving Basic Bars with Entropy Process Completed"

def saving_etf_trick_or_sadf(path_save, list_datasets, naming,
                              list_stocks, bar_name):
    """
    Function to save ETF Trick or SADF.
    """
    
    for idx, dataset in enumerate(list_datasets):
        print("Saving...")

        saving_path = path_save + "SERIES_" + \
                bar_name.upper() + "_" + naming + '.csv'
                
        dataset.to_csv(
                saving_path, 
                index=False
                )    
        
    return "Saving Basic Bars with Entropy Process Completed"

def saving_tunning_bars(path_save, list_datasets, naming, 
                       list_stocks):
    """
    Function to save tunning information of each bar by stock.
    """
    
    list_frames = []
    
    for idx, dataset in enumerate(list_datasets):

        dataset["stock"] = list_stocks[idx]
        
        equity_tuning_frame_bar = pd.DataFrame(
            dataset, 
            index=[0]
            )

        list_frames.append(equity_tuning_frame_bar)
    
    general_bar_tunned_frame = pd.concat(list_frames)
    
    saving_path = path_save + naming +'.csv'
    
    general_bar_tunned_frame.to_csv(saving_path, index=False)        
    
    return "Saving Tunning Information Completed"


def get_alpha_calibration_value(path, stock): 
    """
    Get alpha calibration from tunning pandas info.
    """
    
    dataframe_of_tuning_info = pd.read_csv(path)    
    
    return dataframe_of_tuning_info.query("stock==@stock")


def open_bar_files(base_path, stock, bartype):
   
    stock, bartype = stock.upper(), bartype.upper()
    
    pandas_path = base_path+stock+"_"+bartype+"_BAR.csv"

    pandasBar = pd.read_csv(pandas_path, parse_dates=[
        "open_date","high_date","low_date","close_date", "horizon"]
                           )
    #pandasBar = pd.read_csv(pandas_path)

    return pandasBar


def construct_pandas_tunning(list_datasets, list_stocks):
    """
    Función que permite construir el pandas resumen del bars-tunning.
    
    Inputs:
        - list_datasets (lst - df): lista de df conteniendo info tunning de cada acción
        - list_stocks (lst - str): lista de strings con los nombres de las acciones
    
    Outputs:
        - pandas Tunning conteniendo las sig. columnas
            * [volume_t]
            * [dollar_t]
            * [stock]    
    """
    # lista de frames almacenable para concadenar
    list_frames = []
    
    # iteración por idex y dataset
    for idx, dataset in enumerate(list_datasets):
        
        # añade columna stock al dataset principal
        dataset["stock"] = list_stocks[idx]
        
        # convierte al dataset en un pandas 
        equity_tuning_frame_bar = pd.DataFrame(
            dataset, 
            index=[0]
            )
        
        # añade a la lista final
        list_frames.append(equity_tuning_frame_bar)
    
    # retorna tunning bars dataframe
    return pd.concat(list_frames)


##############################################################################

def enigmxSplit(df, labels, pct_average, backtest_comb = 0.5):
    """
    Función que permite el split aleatorio de un dataset según un % de partición.
    
    Es usado principalmente para dividir el stackedDf en tres elementos:
        - stacked para FeatImportance
        - stacked para ModelTunning
        - stacked para Combinatorial backtest.
    
    Inputs:
        - 'df' (pd.DataFrame)  : dataframe a particionar.
        - 'pct_average' (float): valor ]0;1[ como % de primera partición.
        - 'backtest_comb' (float): valor ]0;1[ como % de segunda partición.
    """
    # primera partición: obtener train y pre-test (div. según pct_average)
    msk1 = np.random.rand(len(df)) < pct_average
    
    # obtención de primer train, y test
    train, __test__ = df[msk1], df[~msk1]
     
    #obtención de train y test para los labels
    train2, __test2__ = labels[msk1], labels[~msk1]

    # segunda partición: obtener test y combinatorial data (div. 50% del resto)
    msk2 = np.random.rand(len(__test__)) < backtest_comb 
    
    # obtención de test final y combinatorial dataset
    test, combinatorial = __test__[msk2], __test__[~msk2]

    #obtención de train y test para los labels
    test2, combinatorial2 = __test2__[msk2], __test2__[~msk2]
    
    return train, test, combinatorial, train2, test2, combinatorial2


# FUNCION QUE CONVIERTE UN ARRAY DE 1D 
def decodifierNNKerasPredictions(array_predictions):
    assert len(array_predictions.shape) == 1, 'input should be 1D only'
    return array_predictions - 1

# FUNCION QUE TRANSFORMA UN VECTOR DE ETIQUETAS PARA NN DE KERAS
def transformYLabelsVector(original_labels):
    # cambiamos los labels de pd.Series a encoderVector
    encoder = LabelEncoder()
    encoder.fit(original_labels)
    encoded_Y = encoder.transform(original_labels)
    
    # redefinimos labels como dummy variables (one hot encoded)
    # este es un array 2D -0D: [#POS 0: -1, #POS 1: 0, #POS 2: -1]
    lbl = np_utils.to_categorical(encoded_Y)    
    return lbl

##############################################################################
############################### KENDALL METHOD ###############################
##############################################################################

def kendall_evaluation(importance_series, pca_series, threshold = 0.5):
    
    # calcula el valor de seleccion segun el critico asignado en el imp. vector
    critic_value = importance_series.quantile(0.5)
    
    # determina los feature seleccionados con sus respectivos values segun el critico   
    importance_selection = importance_series[importance_series>critic_value].sort_values(
        ascending=False
        )
    
    importanceSeriesSelection = importance_series[importance_series>critic_value].rank()
    pcaImportanceSelection = pca_series[importanceSeriesSelection.index.values].rank()
    
    print('importance selection quantile')
    print(importanceSeriesSelection)
    print(' ')
    
    
    print('pca selection quantile')
    print(pcaImportanceSelection)
    print(' ')
    
    #pca_rank = pcaSelectionCritic.rank()
    
    #pca_rank_sorted_values = pca_rank.sort_values()
    
    #pca_rank_sorted_features = pca_rank_sorted_values.index.values
    
    #importance_values_vector = importance_series.loc[pca_rank_sorted_features]
    
    #print('PCA RANK SORTED')
    #print(importanceSeriesSelection)
    #print(' ')
    
    #print('IMP VALUES VECTOR')
    #print(pcaImportanceSelection)
    #print(' ')
    
    kendallVal = stats.weightedtau(
                importanceSeriesSelection.values**-1,
                pcaImportanceSelection.values, 
        )[0]
    
    
    # determina los nombres de los features seleccionados segun el critico (np.array)
    importance_selected_features = importance_selection.index.values
    
    # extrae los features del Imp. en el PCA con sus respectivos valores var.
    pca_join_features = pca_series.loc[importance_selected_features]
        
    # rankea los features join entre Imp. y el PCA  
    pca_ranked_join_features = pca_join_features.rank()
    
    # determina el valor del Weighted Kendall's Tau (Corr: 0 to 1)
    #kendallVal = stats.weightedtau(importance_selection.values, pca_ranked_join_features.values**-1)[0]
    
    # error de mensaje en caso no se cumpla la condicion
    errorMsgg = "Kendall Test failed. KendallTau = {} (abs ver.) vs Threshold = {}. \
        Please, check!".format(
        kendallVal, threshold
    )
    
    # condicion: que el kendallVal resultante sea mayor al threshold asignado de Corr
    assert abs(kendallVal) > threshold, errorMsgg
    
    print(f"     >>>>>> Weighted Kendall's Tau Corr: {kendallVal}.")
    
    # une los dataframes con los importance del Imp. y del PCA
    dfMatrix = pd.concat([pca_join_features, importance_selection],axis=1)
    
    return dfMatrix

################## CLUSTERING FEAT IMPORTANCE USEFUL FUNCTIONS ###############
# def __splitStringArray__(string):
#     return string.split("_")[-1] 
# splitStringArray = np.frompyfunc(
#     __splitStringArray__, 1, 1
# ) 


# función genérica que permite resumir el proceso central del featImp
def baseFeatImportance(**kwargs):
    """
    Parámetros permitidos:
        - features_matrix (pd.dataframe)
        - labels_dataframe (pd.dataframe)
        - random_state (int)
        - method (str)
        - model_selected (sklearn package)
        - pct_embargo (0 < float < 1)
        - cv (int)
        - oob (bool)
    """
    # extrae data de train/test desde la matriz ortogonalizada y el vector de etiquetas                
    
    print("      ::::: >>> Running Base FeatImp...")
    
    x_train, x_test, y_train, y_test = train_test_split(
        kwargs['features_matrix'], 
        kwargs['labels_dataframe'],
        random_state = kwargs['random_state']
        )
    
#    # si el método seleccionado es 'MDA'
#    if kwargs['method'] == 'MDA':
#        # verificación de modelo utilizado
#        if type(kwargs['model_selected']).__name__=='RandomForestClassifier':
#            raise ValueError(
#                "{} model is not allowed to implement 'MDA'".format(
#                    'RandomForestClassifier'
#                    )
#                )      
        
    # si el método seleccionado es 'MDI
    if kwargs['method'] == 'MDI': 
        # verificación de modelo utilizado
        if type(kwargs['model_selected']).__name__!='RandomForestClassifier':
            raise ValueError(
                "Only {} is allowed to implement 'MDI'".format(
                    'RandomForestClassifier'
                    )
                )
    
    
    print("REVISANDO INDICES")
    print(x_train.sort_index())
    print(y_train.sort_index())
    print(" ")
    
    # importance values, score con cpkf, y mean val (NaN)
    imp,oos,oob = featImportances(x_train, 
                                  y_train, 
                                  kwargs['model_selected'],
                                  1, #nSample : no useful
                                  method=kwargs['method'],
                                  sample_weight = y_train['w'],
                                  pctEmbargo=kwargs['pct_embargo'],
                                  cv=kwargs['cv'],
                                  oob=kwargs['oob'])
        
    # comp. importance rank: valor alto (importante) | valor bajo (no importante)
    featureImportanceRank = imp['mean'].rank()
        
    # fit del modelo seleccionado para el featImp
    kwargs['model_selected'].fit(x_train,y_train['labels'])
        
    # score sin combinatorial purged kfold (socre del modelo)
    score_sin_cpkf = kwargs['model_selected'].score(x_test, y_test['labels'])
        
    print("FeatImportance Score without PurgedKFold :", score_sin_cpkf)
    print("FeatImportance Score with PurgedKFold    :", oos)
        
    # retorna featuresRank (0), accuracy CPKF, accuracy con CPKF, y el stacked
    return featureImportanceRank, score_sin_cpkf, oos, imp
    
    
##############################################################################
############################### CLICK MESSAGES ###############################
##############################################################################


M1 = "¿Desea continuar el featImp-clusterizado sin residuos? \
    Este genera N combinatorias x cluster evaluadas por el Kendall-Tau Corr."
M2 = "Puede generar MemoryProblems si el dispositivo no cuenta con capacidad. "
M3 = "Asimismo, el featImpClusterizado no está conectado con el proceso post."
M4 = ":::::::: >>>> Responda 'Y' si desea continuar o 'N' si desea detenerlo."
clickMessage1 = M1 + M2 + M3 + M4

##############################################################################
################### FUNCION PARA GUARDAR TABLA SQL ALCHEMY ###################
##############################################################################

def sql_alche_saving(dataframe, backtescode, 
                     server_name, database, uid = '', pwd = ''):
    
    print("------------ SAVING IN 'BACKTEST' sql database -------------")
    print(":::::: >>> SQL Alchemy Initialization for 'backtest' table...\n")
    
    #obtenemos la lista de drivers temporales en uso directamente de pyodbc
    temporalDriver = [item for item in pyodbc.drivers()]
    
    #selecciona el temporal driver idx = 0 | en caso error, usar idx = -1
    temporalDriver = temporalDriver[0]
    
    print(f">>> Temporal Driver Selected is '{temporalDriver}'...")
    print("----> Warning! In case 'InterfaceError':") 
    print("Please, change 'temporalDriver' idx selection in line 2912 utils.py.\n")    
    
    #construimos la sentencia de conexión a través de SQL ALchemy
    mainSQLAlchemySentence = f'DRIVER={temporalDriver};SERVER={server_name};DATABASE={database};UID={uid};PWD={pwd}'
    
    #generamos SQL-AL engine para inserción de nuevas tablas
    params = urllib.parse.quote_plus(mainSQLAlchemySentence)
        
    #inicialización del engine de SQL Alchemy
    engine = sqlalchemy.create_engine(
            "mssql+pyodbc:///?odbc_connect={}".format(params)
            )        
    
    #definimos nombre de la tabla
    tableName = "BACKTEST_TRIAL_{}".format(backtescode)
    
    #escritura del dataframe a SQL usando SQL alchemy
    dataframe.to_sql(tableName, engine, index=False)
    
    print("<<<::::: BACKTEST TABLE COMPUTATION SQL PROCESS FINISHED :::::>>>")
    
 ##############################################################################
################ ENTROPY MATRIX: VARIATION OF INFORMATION ####################
##############################################################################

######################################################################################
# La funcion numBins es una funcion intermedia
# Sirve para determinar el numero optimo de bins para dividir la data
######################################################################################
#-------------------------------------------------------------------------------------
def numBins(nObs,corr=None):
    # Determina el numero optimo de intervalos (bins) en los que se divide los datos.
    # Necesario cuando se trabaja con V.A continuas.

    if corr is None:
        # binning optimo para el "marginal entropy" (Hacine-Gharbi et.al 2012)
        z=(8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1/3.)
        b=round(z/6.+2./(3*z)+1./3)
    else:
        # binning optimo para el "joint entropy" (Hacine-Gharby and Ravier 2018)
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    return int(b)
#-------------------------------------------------------------------------------------
####################################################################################################
# La funcion varInfo es la funcion principal
# Sirve para calcular la metrica "variation of information VI[X,Y]",
# la cual indica el grado de incertidumbre que tengo sobre X si conozco Y
####################################################################################################
def varInfo(x,y,norm=True):
    # variation of information 
    ## 1) Determinar el numero optimo de bins
    bXY=numBins(x.shape[0],corr=np.corrcoef(x,y)[0,1])
    ## 2) Se calcula el mutual information (denotado como iXY = I[X,Y])
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    ## 3) se calcula el marginal entropy de X y Y
    #### PARA MANTENER CONSISTENCIA, SE USA EL MISMO binning
    hX=stats.entropy(np.histogram(x,bXY)[0])
    hY=stats.entropy(np.histogram(y,bXY)[0])
    ## 4) Se calcula el variation of information (denotado como vXY = VI[X,Y])
    vXY=hX+hY-2*iXY # variation of information
    ## 5) Se normaliza el vXY (denotado \tilde{VI}[X,Y] = VI[X,Y]/H[X,Y])
    if norm:
        hXY=hX+hY-iXY # joint entropy
        vXY/=hXY # normalized variation of information
        ### opcional 6) como vXY es una metrica, si es 0 => si te dan Y conoces X (info equivalente)
        ### para que sea comparable con la correlacion, se debería usar 1 - vXY
        #vXY = 1 - vXY
    return vXY
#----------------------------------------------------------------------------------------------------

# La funcion "mutualInfo" es una funcion principal
# Sirve para calcular el "Mutual information I[X,Y]" (no es una metrica),
# la cual indica en que grado se reduce la incertidumbre que tengo sobre X si conozco Y
#---------------------------------------------------------------------------------------------------
def mutualInfo(x,y,norm=True):
    # mutual information
    bXY=numBins(x.shape[0],corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    if norm:
        hX=ss.entropy(np.histogram(x,bXY)[0]) # marginal
        hY=ss.entropy(np.histogram(y,bXY)[0]) # marginal
        iXY/=min(hX,hY) # normalized mutual information
    return iXY
#--------------------------------------------------------------------------------------------------------
# La funcion "prox_matrix" es una funcion que calcula la "Proximity Matrix", insumo del algoritmo ONC
# Para ello, calcula en forma iterativa \tilde{VI}[X_i,X_j], para todo i != j y completa las posiciones
# (i,j) y (j,i) en la "Proximity Matrix" de dimension N, donde N es el numero de features.
def prox_matrix(X):

    # Por definicion, b = \tilde{VI}[X,X] = 0 => 1-b = 1
    # Por este motivo, la diagonal de la matrix tiene puros 1
    NVI = np.identity(X.shape[1])

    I = 1
    L = X.shape[1]

    for i in range(0,L-1):

        for j in range(I,L):
            a = varInfo(X.iloc[:,i],X.iloc[:,j],True)
            # usamos 1-a para que halla similitud con la matrix de correlacion
            NVI[i,j], NVI[j,i] = 1-a, 1-a
        I+= 1

    NVI = pd.DataFrame(NVI,columns=X.columns,index=X.columns)
    return NVI


#Función que separa el dataframe "Stacked" en las bases de entrenamiento de los modelos
#exógeno, endógeno y el backtest final
def backtestSplit(stacked, labels, pct_split,colDates):

    # revisa que el pct_split no sea menor a 0.6
    assert pct_split >= 0.6, "Percentage of 'splits' should be 0.6 (60%) as min."

    # obtención de df aleatorio para modelo exogeno, endogeno y backtest
    df_exo, df_endo, backtest,labels_exo, labels_endo, labels_backtest  = enigmxSplit(
        df = stacked,
        labels = labels,
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
    labels_exo.sort_values(
    by=['close_date']
    )
    labels_endo.sort_values(
    by=['close_date']
    )
    labels_backtest.sort_values(
    by =['close_date']
    )


    # conversión de fecha como string para evitar pérdida de info
    df_exo[colDates] = df_exo[colDates].astype(str)
    df_endo[colDates] = df_endo[colDates].astype(str)
    backtest[colDates] = backtest[colDates].astype(str)

    return backtest, df_endo, df_exo, labels_backtest, labels_endo, labels_exo
