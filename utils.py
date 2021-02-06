"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import os
import zarr
import math
import numpy as np
import pandas as pd
from numba import njit
from numba import float64
from functools import reduce
from numba.typed import List
from datetime import datetime
import pandas_market_calendars as mcal
from fracdiff import StationaryFracdiff

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

#get trading days based on pandas mlcalendar library        
def sel_days(init,last):
    """
    Gets trading days based on NYSE Calendar.
    """
    nyse = mcal.get_calendar('NYSE')
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
def open_zarr_general(zarrDates, range_dates, zarrObject):    
    #resultado de la función mini nueva        
    idxs_ = [np.where(zarrDates == range_dates[0])[0][0],
             np.where(zarrDates == range_dates[-1])[0][0]+1]
    
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
    #print(hyperp_dict)
        
    #init_vals = init_df[symbol]
    
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
    list_stocks = []
    for i,file in enumerate(os.listdir(data_dir)):
        if file.endswith(common_path):
            list_stocks.append(os.path.basename(file)[:-drop_extension])
    return list_stocks

#Data Tunning Preparation
def dataPreparation_forTuning(csv_path,
                              list_features_names, 
                              label_name,
                              set_datetime_as_index = True):
    if not set_datetime_as_index:
        raise ValueError(
            "Only True available for datatime as index until now."
            )
        
    df = pd.read_csv(
        csv_path, index_col='datetime'
        ).dropna()
    
    X = df[list_features_names]
    
    y = df[label_name]
    
    t1 = pd.Series(data=y.index, index=y.index)    
    
    return X, y, t1

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

def dataPreparation(data, 
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
    train['new_index'] = train_indices.copy()
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
        #print('Split Set: ' + str(split))
        
        train = []
        test = []
    
        for piece in range(len(pieces)):
            if np.isin(piece,non_zero):
                #print('Test: ' + str(piece))
                test.append(pieces[piece])
            else:
                #print('Train: ' + str(piece))
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

def __newVolumeBarConstruction__(arrayTime, 
                                 arrayPrice, 
                                 arrayVol, 
                                 alpha_calibration=1e3):

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
     
    return groupTime, groupPrice, groupVol

def __newDollarBarConstruction__(arrayTime, 
                                 arrayPrice, 
                                 arrayVol, 
                                 alpha_calibration=1e3):

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
def OHLC_BAR(list_arrayPrice, list_arraytime, list_arrayVol):
    """
    No vectorized version.
    
    OHLC + VOLATILITY + VOLUME computation.
    
    Includes arrayTime values for OHLC prices.
    """
    barOHLC_plus_time_info = []
    
    #for each array of price (arrays in price same as in Time and Vol)
    for idx in range(len(list_arrayPrice)):
        
        subset_info = list_arrayPrice[idx]
        subset_info_time = list_arraytime[idx]
        
        open_ = subset_info[0]
        high_ = np.max(subset_info)
        high_index = np.where(subset_info==high_)[0][0]
        low_ = np.min(subset_info)
        low_index = np.where(subset_info==low_)[0][0]
        close = subset_info[-1]
        
        #basic volatility as simple standard deviation of bar prices
        basic_volatility = np.std(subset_info)
        
        volume_in_bar = np.cumsum(list_arrayVol[idx])[-1]
        
        barOHLC_plus_time_info.append(
            [
                open_, high_, low_, close,
                subset_info_time[0],
                subset_info_time[high_index], 
                subset_info_time[low_index],
                subset_info_time[-1], 
                basic_volatility, 
                volume_in_bar
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
                     bartype):
    
    """
    Takes the group list of arrays information from 
    'time', 'prices' and 'vols' and, based on the 
    string 'bartype', compute OHLC Bars (inc. 'VOLATILITY' + 'BARVOLUME')
    and VWAP.
    """
    
    if bartype == 'tick':
        #compute vwap from Price and Vol by bar
        vwap = iterativeVwap(grp_prices, grp_vols)     
        
        #List of lists of values:
        #[O, H, L, CL, O-Date, H-Date, L-Date, C-Date, volatility, bar volume] 
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
        OHLCBars = OHLC_BAR(
            grp_prices, grp_time, grp_vols
            ) #ad '_VEC' to func. name for vectorized version
        
        #returns object        
        return OHLCBars, vwap
    
    elif bartype=='dollar':
        #compute vwap from Price and Vol by bar
        vwap = iterativeVwap(grp_prices, grp_vols)     
        
        #List of lists of values:
        #[O, H, L, CL, O-Date, H-Date, L-Date, C-Date, volatility, bar volume] 
        OHLCBars = OHLC_BAR(
            grp_prices, grp_time, grp_vols
            ) #ad '_VEC' to func. name for vectorized version
        
        #returns object        
        return OHLCBars, vwap
    
    else:
        raise ValueError("Not recognized bartype string input")
        
def getTripleBarrierOHLC(df):
    
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
                                          "bar_cum_volume", "vwap", "fracdiff"]
                       ):
    """
    Function to add a prefix ('bartype') to each information column name.
    """
    return [bartype +  "_" + colName for colName in columnBaseNames]
    
dataset_column_names = ["open", "high", "low", "close", "open_date", 
                        "high_date", "low_date", "close_date", 
                        "basic_volatility", "bar_cum_volume", 
                        "vwap", "fracdiff"]

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

def vectorizedTripleBarrier(path, init, init_ts, last, last_ts, upper_bound, 
                            lower_bound):
    
    #days range to search tripleBarrier | timeframe values
    daysList = sel_days(init, last) 
    
    #open zarr of selected stock
    zarrds = zarr.open_group(path)
    
    #general zarr dates 
    zarr_dates = zarrds.date.get_basic_selection()

    #finding dates indices in zarr_dates | timeframe idxs
    dateIdxs = np.searchsorted(zarr_dates, daysList)    
    
    #check in case there are no dates (missing data values)
    if len(dateIdxs)==0:
        finalPrice, finalLabel, finalTimestamp = 0,0,0
        return finalPrice, finalLabel, finalTimestamp
    
    else:

        #getting general prices matrix in timeframe
        prices_timeframe  = zarrds.value.oindex[dateIdxs, :]
      
        #getting general timestamp matrix in timeframe
        ts_timeframe = zarrds.timestamp.oindex[dateIdxs,:]    
        
        #indexes selected by timestamp limits
        selected_indexes = np.where(
            (ts_timeframe>init_ts)&(ts_timeframe<last_ts)
        )    
        
        #prices array segmented by ts over the timeframe | 1D array
        segmented_vector_prices = prices_timeframe[selected_indexes] 
    
        #timestamp array segmented by ts over the timeframe | 1D array
        segmented_vector_timestamps = ts_timeframe[selected_indexes]    
    
        #get the first index for each barrier touched
        first_upper_barrier_idx = ErrorIndexIdxFirstTrue(
            np.where(segmented_vector_prices>upper_bound)
        )
        first_lower_barrier_idx = ErrorIndexIdxFirstTrue(
            np.where(segmented_vector_prices<lower_bound)
        )
        
        ####################barrier computation###############################
        
        #check if there is info available
        if segmented_vector_prices.shape[0] == 0:
            finalPrice, finalLabel, finalTimestamp = (
                0, 0, last_ts
            )
        
        #case upper/lower barrier idx values are equal (None)
        elif first_upper_barrier_idx == first_lower_barrier_idx:
            #definition of vertical barrier
            finalPrice, finalLabel, finalTimestamp = (
                segmented_vector_prices[-1], 0, last_ts
            )
    
        else:
            #if just upper barrier idx is None (only lower exists)
            if first_upper_barrier_idx is None:
                #set random greater value from lower idx for later comparisson
                first_upper_barrier_idx = first_lower_barrier_idx+1
                
            #if just upper barrier idx is None (only lower exists)            
            if first_lower_barrier_idx is None:
                #set random greater value from upper idx for later comparisson
                first_lower_barrier_idx = first_upper_barrier_idx+1
            
            #upper barrier happens first (or it's unique) than lower barrier
            if first_upper_barrier_idx < first_lower_barrier_idx:
                #definition of horizontal upper barrier
                finalPrice, finalLabel, finalTimestamp = (
                    segmented_vector_prices[first_upper_barrier_idx], 
                    1, 
                    segmented_vector_timestamps[first_upper_barrier_idx] 
                ) 
                
            #lower barrier happens first (or it's unique) than upper barrier    
            else:
                #definition of horizontal lower barrier
                finalPrice, finalLabel, finalTimestamp = (
                    segmented_vector_prices[first_lower_barrier_idx],
                    -1,
                    segmented_vector_timestamps[first_lower_barrier_idx]
                )
                
        return finalPrice, finalLabel, finalTimestamp
    
    
########### NEW TRIPLE BARRIER COMPUTATION ######################


def LabelTripleBarrierComputation(barDataframe, stock, data_dir):

    #tripleBarrier vectorized computation version
    tripleBarrierInfo = barDataframe.apply( #para ENTROPY, aplicar lambda también como org. global
        lambda row: 
            vectorizedTripleBarrier(
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
    #barrier columns information generation
    barDataframe[['barrierPrice', 'barrierLabel', 'barrierTimestamp']] = \
        pd.DataFrame(
            tripleBarrierInfo.tolist(), 
            index=tripleBarrierInfo.index
            )    
        
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
        'tick_t': bar_para[0], 
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
