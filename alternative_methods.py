"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
import zarr
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
from enigmx.utils import (
    get_arrs, bar_para, softmax, sel_days, 
    simpleFracdiff, open_bar_files
    )
import scipy.cluster.hierarchy as sch


# El presente file 'alternative_methods.py' posee las func. relevantes
# del primer apartado del workflow (ver ppt "17-01" del 2021).
# Fue dividido para comidad en posibles modificaciones futuras.
        
##############################################################################        
#1. Bar tunning #-------------------------------------------------------------
##############################################################################

@ray.remote
def standard_bar_tunning(url,ticker,num_bar, date_tuple_range):
    '''
    INPUTS:
    * url     : ubicacion donde se encuentra el archivo zarr 
                (ejemplo: 'F:/DESCARGAS_2021/')
    * ticker  : ticker de la acción (ejemplo: MSFT)
    * date_tuple_range : tupla con el rango-dates para computar el tunning.
                         Nomenclatura: (fecha_init, fecha_end)
                         Importante: asegurarse de que la fecha_end 
                                     sea la fecha desde donde se piensa
                                     iniciar la computación de barras 
                                     (evitar look-ahead bias).
    OUTPUT:
    * diccionario con hyperparametros para las tres barras estandar
    ## {'tick_para': a, 'vol_para': b, 'dol_para': c}
    '''

    #obtener el url para aperturar el Zarr
    url = url+ticker+'.zarr'
    
    #apertura remota de zarr
    zarrds = zarr.open_group(url)

    #si la tupla de fechas tiene un valor distinto a None (str fecha x 2)
    #abre todas las fechas
    zarr_dates = zarrds.date.get_basic_selection()

    #encuentra los idxs de la fecha min y fecha max dada en la tupla
    dateIdxs = np.searchsorted(zarr_dates, date_tuple_range)
        
    #guarda número de días del zarr| dif. de indexes
    num_days = dateIdxs[1]-dateIdxs[0]
    
    #guarda número prom diario de ticks, volumen o dollar | init: 0 all
    average_ticks_day, average_volume_day, average_dollar_day = 0, 0, 0

    #iteración por cada index: desde index 0 hasta final del zarr
    for i in range(dateIdxs[0], dateIdxs[1]):
        value = zarrds.value[i]
        value = value[value>0]
        average_ticks_day  += len(value)/num_days
        vol = zarrds.vol[i]
        vol = vol[vol>0]
        average_volume_day += sum(vol)/num_days
        fec = zarrds.date[i]
        arr = np.array(zarrds.date)
        _ = get_arrs(zarrds,arr,fec)
        dol = sum(_[0] * _[1])
        average_dollar_day += dol/num_days    
    
    
    #determina el promedio de los ticks, volumen o dollar de la iteración
    average_ticks_day = int(average_ticks_day)
    average_volume_day = int(average_volume_day)
    average_dollar_day = int(average_dollar_day)
    
    #se construye array resumen
    dato = np.array(
        [average_ticks_day, average_volume_day, average_dollar_day]
        )

    #calibrar los hierparámetros y guardarlos en un diccionario 
    dic = bar_para(dato,num_bar)
    
    return dic

##############################################################################
#2. Entropy Calculation #-----------------------------------------------------
##############################################################################
def getTEventsCumSum(gRaw,h):
    """
    Function for CUMSUM Filter resampling by 'h'.
    
    Higher 'h' means less tEvents values in term of prices diff.
    
    0 < h < 1
    """
    
    tEvents,sPos,sNeg=[],0,0
    #diff = np.diff(g)
    diff=np.diff(gRaw) #differential | eq. returns
    for i in range(1,diff.shape[0]):
        sPos,sNeg=max(0,sPos+diff[i]),min(0,sNeg+diff[i])
        #print(sPos,sNeg)
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return tEvents#pd.DatetimeIndex(tEvents)

#@njit
def matchLength(msg,i,n):
    # Maximum matched length+1, with overlap.
    # i>=n & len(msg)>=i+n
    subS=''
    #if 'n' is not a fixed window, it will be acumulated
    #print("info antes de match length")
    #print(range(n))
    
    #print(len(msg))
    #print(" ")
    for l in range(n):
        msg1=msg[i:i+l+1]
        for j in range(i-n,i):
            msg0=msg[j:j+l+1]
                        
            if msg1==msg0:
                subS=msg1
                break # search for higher l.
    return len(subS)+1 

def konto(msg,window=None):
    """
    Kontoyiannis’ LZ entropy estimate, 2013 version (centered window).
    Inverse of the avg length of the shortest non-redundant substring.
    If non-redundant substrings are short, the text is highly entropic.
    window==None for expanding window, in which case len(msg)%2==0
    If the end of msg is more relevant, try konto(msg[::-1])
    """

    num_, sum_ = 0, 0
        
    if not isinstance(msg,str):msg=''.join(map(str,msg))
    if window is None:
        points=range(1,int(len(msg)/2+1))
        
    else:
        window=int(min(window,len(msg)/2))
        points=range(window,len(msg)-window+1)

    for i in points:
        if window is None:
            
            l=matchLength(msg,i,i)
            
            sum_+=np.log2(i+1)/l # to avoid Doeblin condition
        else:
            l=matchLength(msg,i,window)
            sum_+=np.log2(window+1)/l # to avoid Doeblin condition
            
        num_+=1
    out = 1-(sum_ / num_)/np.log2(len(msg)) # redundancy, 0<=r<=1
    return out

def entropyFeature(row, 
                   path, 
                   entropy_window = None,
                   beta=0.025, 
                   cumsum_sampling = True):
    
    init, last = row[4], row[7]
    init_ts = datetime.timestamp(init.to_pydatetime()) * 1e3
    last_ts = datetime.timestamp(last.to_pydatetime()) * 1e3
    
    #days range to search tripleBarrier | timeframe values
    daysList = sel_days(init, last)
    
    #open zarr of selected stock
    zarrds = zarr.open_group(path)
    
    #general zarr dates 
    zarr_dates = zarrds.date.get_basic_selection()

    #finding dates indices in zarr_dates | timeframe idxs
    dateIdxs = np.searchsorted(zarr_dates, daysList)    

    #getting general timestamp matrix in timeframe
    ts_timeframe = zarrds.timestamp.oindex[dateIdxs,:]    
    
    #indexes selected by timestamp limits
    selected_indexes = np.where(
            (ts_timeframe>init_ts)&(ts_timeframe<last_ts)
        )    
    
    #conditions for indexing | version 1.0 does not have this
    if len(selected_indexes) == 2:
        #take only row dimensions, columns already defined by dateIdxs
        selected_indexes = selected_indexes[1]
          
    if len(dateIdxs) == 0:
        #there is no information from external dimension
        dateIdxs_init, dateIdxs_last = 0, 0
    
    if len(dateIdxs) == 1:
        #there is only one date information from external dimension
        dateIdxs_init, dateIdxs_last = dateIdxs[0], dateIdxs[0] 
    
    if len(dateIdxs) >= 2:
        #there are at least 2 columns information external dimension
        dateIdxs_init, dateIdxs_last = dateIdxs[0], dateIdxs[-1]
        
    #we select only dim [0] cause' this works for bars in the same day
    #if inter-day bars will be used, this selection should be evaluated
    vector_prices = zarrds.value[
        dateIdxs_init:dateIdxs_last+1
    ][:,selected_indexes][0]
    
    #--------------------Entropy calculation--------------------#
    
    #check if there are prices in the interval
    if vector_prices.shape[0] == 0:
        
        #set entropy as zero
        entropy = 0
        
    else: 
        #mostly, when 'entropy_window' = None
        if cumsum_sampling:

            assert 0<beta<1, "Beta should be in the range [0;1["

            #resampling  idxs in terms of returns upper a 'beta' factor
            resampled_idx_vector_prices = getTEventsCumSum(
                vector_prices, 
                beta
            )

            #resampling prices defined by resampling idxs
            vector_prices = vector_prices[resampled_idx_vector_prices]

        #log prices calculation from resampled prices vector
        vector_log_prices = np.log(vector_prices)
        
        #check in case log transformation gets small dataset size
        if vector_log_prices.shape[0] < 10:
            
            #set entropy as zero
            entropy = 0
            
        else:
            #computation of fracdiff from log prices of resampled prices vector
            vector_fracdiff_log_prices = simpleFracdiff(vector_log_prices)
    
            #encoding of fracdiff prices
            binarized_vector_fracdiff_log_prices = np.where(
                    vector_fracdiff_log_prices>np.mean(
                        vector_fracdiff_log_prices
                        ), 
                    1, 0
                )
    
            #redundancy as entropy
            entropy = konto(
                    msg= binarized_vector_fracdiff_log_prices, 
                    window= entropy_window
                )
    return entropy

@ray.remote
def entropyCalculationParallel(zarr_dir, 
                               pandas_bar, 
                               stock,
                               beta = 0.01, 
                               entropy_window = 100, 
                               cumsum_sampling = True):
        
    zarr_path = zarr_dir + stock + ".zarr"
    
    #pandas_bar["entropy"] = 
    pandas_bar["entropy"] = np.apply_along_axis(
        entropyFeature, 1, pandas_bar.values, 
        zarr_path, entropy_window, beta, cumsum_sampling 
        )
    #print(pandas_bar)
    #ga = pandas_bar.apply(lambda row:
    #                       entropyFeature(
    #                           path = zarr_path,
    #                           init = row["open_date"],
    #                           init_ts = datetime.timestamp(
    #                                       row["open_date"].to_pydatetime()
    #                                       ) * 1e3,
    #                           last = row["close_date"],
    #                           last_ts= datetime.timestamp(
    #                                       row["close_date"].to_pydatetime()
    #                                       ) * 1e3, 
    #                           #### Optional Params ####
    #                           beta = beta, 
    #                           entropy_window = entropy_window, 
    #                           cumsum_sampling = cumsum_sampling)
    #                       ,axis=1
    #                      )    
    #print(ga)
    #pandas_bar["entropy"] = ga   
    return pandas_bar

##############################################################################
#3. SADF & ETF Trick #--------------------------------------------------------
##############################################################################
def pcaWeights(cov, 
               riskDist =  None, 
               riskTarget = 1, 
               softmax_method = False,
               preserve_sign = False):
    """
    cov: covariance matrix MXM as dataframe or numpy
    riskDist: list or numpy length = Num Stocks values range(0,1)
    riskTarget: risk for scale resizing
    softmax_method: activate a softmax version for scaling allocation
    """
    
    # Following the riskAlloc distribution, match riskTarget
    eVal,eVec=np.linalg.eigh(cov) # must be Hermitian

    indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
    eVal,eVec=eVal[indices],eVec[:,indices]

    if riskDist is None:
        riskDist=np.zeros(cov.shape[0])
        riskDist[-1]=1.

    loads=riskTarget*(riskDist/eVal)**.5

    wghts=np.dot(eVec,np.reshape(loads,(-1,1)))
    
    #ctr=(loads/riskTarget)**2*eVal # verify riskDist
    if softmax_method:
        result = softmax(wghts)
    else:
        #result = np.abs(wghts) / np.sum(np.abs(wghts))
        result = wghts
    if preserve_sign:
        wghts[wghts < 0], wghts[wghts > 0] = -1, 1
        result = result*wghts 
    
    return result    

def settingTimePandasFormat(dataframe, column_name):
    """
    Set datatime of a pandas from a base dataframe.
    """
    df = pd.DataFrame(dataframe[column_name])
    df.set_index(dataframe['close_date'], inplace=True)
    return df

def optPort(cov,mu=None):
    """
    Weights calculation by inverting cov. matrix.
    """
    inv=np.linalg.inv(cov)
    ones=np.ones(shape=(inv.shape[0],1))
    if mu is None:mu=ones
    w=np.dot(inv,mu)
    w/=np.dot(ones.T,w)
    return w

##############################################################################
######################### THEOTHERICAL RISK ##################################
##############################################################################

class HRP():
    
    def __init__(self, cov,corr):
        self.cov = cov
        self.corr = corr
        self.weights = self.HRP(self.cov, self.corr)
        

    def HRP(self, cov,corr):
        ''' PASO 1 (Tree Clustering) 
            Se agrupan los elementos de la matrix de correlacion en base a su
            distancia
            Plantear utilizar otros procedimientos en la medición 
            de distancias, ejm: scipy.spatial.distance.pdist
        '''

        # distance matrix
        dist=((1-corr)/2.)**.5 
        # linkage matrix object
        link=sch.linkage(dist,'single') ### EVALUAR INCLUIR UN HYPERPARAMETRO
        ''' PASO 2 (Cuasi Diagonalización)
            Se determina el orden de las filas de la matriz de correlación 
            en función de los clusters obtenidos en el paso 1.
        '''
        link=link.astype(int)
        sortIx=pd.Series([link[-1,0],link[-1,1]])
        # número de elementos por grupo (cuarta columna)
        numItems=int(link[-1,3])
        while sortIx.max()>=numItems:
            sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
            df0=sortIx[sortIx>=numItems] # find clusters
            i=df0.index;j=df0.values-numItems
            sortIx[i]=link[j,0] # item 1
            df0=pd.Series(link[j,1],index=i+1)
            sortIx=sortIx.append(df0) # item 2
            sortIx=sortIx.sort_index() # re-sort
            sortIx.index=range(sortIx.shape[0]) # re-index
        sortIx = sortIx.astype(int).tolist()
        ''' PASO 3 (Recursive Bisection)
            Se determinan los pesos asignados a cada acción iterativamente 
            por pares. En consecuencia, el número de acciones
            mínimo debe ser 4. 
        '''
        # generamos un vector para los pesos del portafolio
        w=pd.Series(1,index=sortIx)
        # initialize all items in one cluster
        cItems=[sortIx] 
        while len(cItems)>0:
            cItems=[
                i[j:k] for i in cItems for j,k in (
                    (0,int(len(i)/2)), 
                    (int(len(i)/2),len(i))) if len(i)>1
                ] # bi-section
            for i in range(0,len(cItems),2): # parse in pairs
                cItems0=cItems[i] # cluster 1
                cItems1=cItems[i+1] # cluster 2
                cVar0=self.getClusterVar(cov,cItems0)
                cVar1=self.getClusterVar(cov,cItems1)
                alpha=1-cVar0/(cVar0+cVar1)
                w[cItems0]*=alpha # weight 1
                w[cItems1]*=1-alpha # weight 2
        return w

    #———————————————————————————————————————
    def getIVP(self, cov,**kargs):
    # Compute the inverse-variance portfolio
        ivp=1./np.diag(cov)
        ivp/=ivp.sum()
        return ivp
    #———————————————————————————————————————
    def getClusterVar(self, cov,cItems):
    # Compute variance per cluster
        cov_=cov.iloc[cItems,cItems] # matrix slice
        w_=self.getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar
    #———————————————————————————————————————

##############################################################################
##############################################################################
##############################################################################

def etfTrick(list_bars_stocks, 
             stock_list, 
             k=10000, 
             lower_bound_index = 50, 
             allocation_approach = 'inv',
             output_type = dict):
    """
    Description:
        Function allows to compute ETF Trick for a list of equities.
        Cap 2 - AFML of Marcos Lopez de Prado (Section 2.4.1).
    
    Parameters:
        - 'list_bars_stocks': dataframes list with bar pandas by equity.
        - 'stock_list': lst of strings by stock of 'bar' .csv information.
        - 'bartype': str referring type of bars 'bar' .csv is based on.
        - 'k': initial K (AUM) int to compute ETF Trick.
        - 'lowerb_ound_index': N-int first values taken to 1st covmatrix.
        - 'allocation_approach': str referring weight method computation.
                * 'inv': inverting covariance matrix method.
                * 'hrp': hierarchical risk.
                * otherwise: PCA decomposition.
        - 'output_type': 'dict' statement or anything else referring 
                         to output format.
                * "dict" statement: returns full dictionary of info 
                                    (ETF value series not included).
                * anything else: returns full ETF Trick series.
    Output:
    ------
        - If output_type: 
            * "dict" statement: returns full dictionary of info 
                                (ETF value series not included).
            * anything else: returns full ETF Trick series.        
    """
    #get list stock-bar information dataframe
    #list_stocks = [
    #    open_bar_files(bar_dir, stock, bartype) for stock in stock_list
    #]

    #get idx-time reformed df close dates of each stock-bar dataframe
    list_reformed_df_close_dates = [
        settingTimePandasFormat(stock_frame, "close_date") 
        for stock_frame in list_bars_stocks
    ]
    
    #get idx-time reformed df open prices of each stock-bar dataframe
    list_reformed_df_open_prices = [
        settingTimePandasFormat(stock_frame, "open_price")
        for stock_frame in list_bars_stocks
    ]
    
    #get idx-time reformed df close prices of each stock-bar dataframe
    list_reformed_df_close_prices = [
        settingTimePandasFormat(stock_frame, "close_price")
        for stock_frame in list_bars_stocks   
    ]    
    
    #compute joinned dataframe of dates from all stocks
    #ERROR LINE
    join_dataframe_dates = pd.concat(
        list_reformed_df_close_dates, axis=1
    ).fillna(method='ffill')[1:]
    
    #compute joinned dataframe of open prices from all stocks
    join_dataframe_open_prices = pd.concat(
        list_reformed_df_open_prices, axis=1
    ).fillna(method='ffill')[1:]
    
    #compute joinned dataframe of close prices from all stocks
    join_dataframe_close_prices = pd.concat(
        list_reformed_df_close_prices, axis=1
    ).fillna(method='ffill')[1:]
    
    #transform dataframe as price array - multiple stocks
    join_array_close_prices = join_dataframe_close_prices.values
    
    #transform dataframe as dates array - multiple stocks
    join_array_dates = join_dataframe_dates.values

    #global index available universe for selection
    idx_global_sequential = range(
        lower_bound_index, 
        join_dataframe_close_prices.shape[0]
    )
    
    #if 'covariance inverse matrix' is selected
    if allocation_approach.lower() == 'inv':
        weights_method = optPort
        
    #se aplica el hierarchical risk parity del C. 16 de MLDP(2018)
    elif allocation_approach.lower() == 'hrp':
        weights_method = HRP         
        
    #otherwise, apply simple PCA decomposition | Warning: this doesn't sum 1
    else:
        weights_method = pcaWeights
    
    #-----------------ETF ITERATIVE CONSTRUCTION-----------------# 
    
    #historical empty weights
    info_weights = []
    
    #historical dates information over mkt-2-mkt value change
    info_dates = []
    
    #historical covariance changes
    covs = []
    
    #historical equity holdings ('h')
    _ = []
    
    #idx iteration starts at lower_bound_index
    for idx in idx_global_sequential:
        
        #assign temporal matrix of prices
        temp_matrix_prices = pd.DataFrame(
            join_array_close_prices[0:idx]
        )
        
        #temporal returns matrix
        temp_returns_matrix = temp_matrix_prices.pct_change()
        
        #temporal covariance matrix
        temp_covariance_matrix = temp_matrix_prices.cov()
        
        #save covariance matrix information
        covs.append(temp_covariance_matrix)
        
        if weights_method == HRP:
            #for the HRP method, the correlation matrix is calculated
            #temporal correlation matrix
            temp_correlation_matrix = temp_returns_matrix.corr()
            
            #temporal covariance matrix
            temp_covariance_matrix = temp_returns_matrix.cov()
            
            #weight vector computation (pd.Series)
            weigths = weights_method(
                temp_correlation_matrix, temp_covariance_matrix
                ).weights.sort_index()            
        
        else:
            #weight vector computation
            weigths = weights_method(temp_covariance_matrix).T[0]
        
        #save information of weights
        info_weights.append(weigths)
    
    #final prices segmentation | useful to final computation
    join_array_open_prices = join_dataframe_open_prices.values[
        lower_bound_index:
    ]
    join_array_dates = join_array_dates[lower_bound_index:]
    
    #find when a bar has a 't' index ocurrency | iterate by each 't'
    for stockIdx in range(len(info_weights)-1):

        #if there is no bar 't'
        if pd.isnull(stockIdx): 
            _.append(_[stockIdx-1])

        #if there is bar at 't', we define a new 'h'
        else:

            #find allocation weight vector at 't' by stock index
            allocation_weight_t = info_weights[stockIdx]

            #get open prices by stock idx in the next 't'
            precio_open_tsig = join_array_open_prices[stockIdx+1]

            #find USD value change of 1 point (1%)
            USDValT = precio_open_tsig * 0.01

            #compute delta difference
            delta = \
            temp_matrix_prices.iloc[stockIdx]-join_array_open_prices[stockIdx]

            #if it is the first event
            if stockIdx == 0:
                pass
            
            #otherwise
            else:
                #find the new K at 't'
                k = k + sum(hi_t*USDValT*delta.fillna(0))

            #finally, compute 'h' holding securities at 't'
            hi_t = (
                (allocation_weight_t * k)/(precio_open_tsig*USDValT * sum(
                    abs(allocation_weight_t))
                                          )
            )

            #save 'h' to adress historical holdings
            _.append(hi_t)
            
        info_dates.append(join_array_dates[stockIdx])

    #if output_type is 'dict' return complete info dict report
    if output_type == dict:
        return {
            "info_weights":np.array(info_weights), 
            "info_dates":np.array(info_dates),
            "prices_allocation":np.array(join_array_open_prices)[:-1], 
            "holdings":np.array(_)
        }
    
    else:
        #generate final dataframe with ETF Trick
        final_frame = pd.DataFrame(
            {"value": np.sum(
                np.array(_) * np.array(join_array_open_prices
                                       )[:-1], axis=1)
                }
        )

        #add date-info per stocks to select after the next sampling process
        final_frame[stock_list] = np.array(info_dates)
        
        final_frame[stock_list] = final_frame[stock_list].apply(pd.to_datetime)
        #final_frame.to_csv(
        #    bar_dir + "SERIES_" + bartype + "_ETFTRICK.csv", index=False
        #    )
        return final_frame #"ETFTrick Saved"