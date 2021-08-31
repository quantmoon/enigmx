"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import talib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from collections import ChainMap as cmap
from enigmx.alphas import MainGetAlphasFunction
from tsfresh.feature_extraction import feature_calculators as f

#############################################################################
################## FUNCION - TRANSFORMACION MEDIAS MOVILES ##################
#############################################################################


def movingAveragesSignals(time_frames, close_price_series): 
    
    """
    DEPRECATED**
    
    Lógica general de cómputo:
    
    Las medias móviles serán tomadas como insumos 
    para la elaboración de un signal derivado  en c/u de ellas.
    
    Esta transformación debe cumplir con dos condiciones:
        1. Ser continua y aproximada a una misma media.
        2. Mantener la información de valor de la señal.
        
    En esa medida, y dado que las señales son medias móviles 
    que se encuentran relacionadas a la serie de precios, 
    se propone la siguiente transformación:
    
        signalDerivative = priceSeries / MovingAverageSignal
        
    Así, se tendrá evaluada la relación existente entre la señal
    y la serie de precios con una media constante aproximada igual a 0.
    
    Este proceso se reealizará sobre todos los timeframes 
    que se ingesten para el cómputo de las señales.
    
    De ese modo, por cada timeframe, surgirán N signals,
    siendo N el total de Medias Móviles a utilizar.
    
    Por defecto, la presente función utilizará las sig. medias móviles:
        - EMA
        - KAMA
        - TEMA
        - TRIMA
    
    En caso se agregue una más, recuerde aumentar las variables
    receptoras en la salida de la función al momento de su uso.
    
    Fuente de las medias móviles TALIB:
        - https://mrjbq7.github.io/ta-lib/ 
    """
    
    # revisamos que el input sea una lista
    assert type(time_frames) == list, "Input should be a list of integers."
    
    # definimos los diccionarios vacíos para alm. info
    dictEmas, dictKamas, dictTemas, dictTrimas = {}, {}, {}, {} 
    
    # iteración por cada timeframework definido para las medias
    for timeframe in time_frames:
        
        # cómputo y almacenamiento de señal calculada en función de EMAs
        dictEmas['EMA_'+str(timeframe)+'_signal'] = close_price_series / talib.EMA(
            close_price_series, timeperiod = timeframe
        )
        
        # cómputo y almacenamiento de señal calculada en función de KAMAs
        #dictKamas['KAMA_'+str(timeframe)+'_signal'] = close_price_series / talib.KAMA(
        #    close_price_series, timeperiod = timeframe
        #)
        
        # cómputo y almacenamiento de señal calculada en función de  TEMAs
        #dictTemas['TEMA_'+str(timeframe)+'_signal'] = close_price_series / talib.TEMA(
        #    close_price_series, timeperiod = timeframe
        #)
        
        # cómputo y almacenamiento de señal calculada en función de  TRIMAs
        #dictTrimas['TRIMA_'+str(timeframe)+'_signal'] = close_price_series / talib.TRIMA(
        #    close_price_series, timeperiod = timeframe
        #)
        
    # retorna una lista de dicts sobre la que se mapeará posteriormente
    return [dictEmas, dictKamas, dictTemas, dictTrimas] 

##############################################################################
################## FUNCION - TRANSFORMACION BANDA BOLLINGER ##################
##############################################################################

def bollingerBandsLogic(upperband, middleband, lowerband, price):
    """
    La sig. función traduce los elementos de la BB y la serie de precios
    en una señal de valores discretos pertenecientes al conjunto {1, 0.5, 0, 0.5, -1}.
    
    La lógica empleada es la sig.:
    
    Si close_price > upper band:
     Sign: -100
    Si close_price < lower band:
     Sign: 100
    Else:
        # deprecated: all of this are setting as 0
            Si lower band <= close_price < middleband:
                Sign: -1
            Si upper band >= close_price > middleband:
                Sign: 1
            Else:
                Sign: 0 #useful
                
    Fuente central de lógica:
        https://www.investopedia.com/articles/technical/102201.asp 
    """
    # cambiamos la entrada de serie a arrays para easy-computing
    upperArray, middArray, lowerArray, priceArray = \
        upperband.values, middleband.values, lowerband.values, price.values
    
    # mensaje de error en caso los shapes de los inputs sean distintos
    mssg = '    ::::::>>> Shape Error! ~ B.Bollinger Indicator Error: shapes are different.'
    
    # asset statement | revisando dimensionalidad equivalente
    assert upperArray.shape[0] == middArray.shape[0] == lowerArray.shape[0] == priceArray.shape[0], mssg
    
    # signal modificado - length de la serie
    outputSignal = np.zeros(len(upperArray))
    
    # nan's a todos los valores para preservar su post. eliminacion por defecto
    outputSignal[:] = np.nan
    
    # condicion de signal lateral - inner band | NO SIGNAL (0 value)
    outputSignal[lowerArray <= priceArray] = 0
    outputSignal[priceArray <= upperArray] = 0    
    
    # condicion de signal bajista: -100
    outputSignal[priceArray > upperArray] = -100
    
    # condicion de signal alcista: 100
    outputSignal[priceArray < lowerArray] = 100
    
    # condicion de signal lateral bajista
    #outputSignal[upperArray >= priceArray & priceArray > middArray] = - 1
    
    # condicion de signal eq. a la media movil central | NO SIGNAL (0 value)
    #outputSignal[priceArray == middArray] = 0
    
    # retorna el signal convertido a pandas series
    return pd.Series(outputSignal)

#############################################################################
################## FUNCIONES EXTRAS - FEATURES GENÉRICOS ####################
#############################################################################

# feature z score
def get_z_score(serie):
    return (serie-serie.mean())/serie.std()

# feature z rolling
def get_rolling_z_score(serie, window=200):
    return (
        serie-serie.rolling(
            window=window, min_periods=20
        ).mean())/serie.rolling(window=200, min_periods=20).std()

# feature minmax
def min_max(serie):
    return (serie-min(serie))/(max(serie)-min(serie))

# feeature rolling application
def rolling_application(serie, function, rolling_periods = 200, min_periods=20):
    return serie.rolling(
        rolling_periods,min_periods=min_periods
        ).apply(function)

# feature rolling rank
def rolling_rank(na):
    return rankdata(na)[-1:]

# feature ts rank
def ts_rank(s,window):
    return s.rolling(window).apply(rolling_rank)

#############################################################################
########### FEATURES MICROESTRUCTURALES - FEATURES GENÉRICOS ################
#############################################################################

# feature microstructural | roll 84
def roll84(series,cov_win=10):
    '''
    INSUMOS
    close  : precio close de la barra (panda serie)
    cov_win: tamaño del rolling window utilizado en el calculo
    
    OUTPUT
    spread: estimador del bid ask spread  
    
    Consideraciones: se sigue la modificacion de Lesmond (2005) en este estimador.  
    Es decir, toda covarianza positiva se multiplica por -1
    '''
    close = series['close_price']
    log_close  = np.log(close)
    r_t  = log_close.diff()
    r_t_1= r_t.shift()
    # se calcula la covarianza movil con una ventana de tamaño win
    cov = r_t_1.rolling(window=cov_win).cov(r_t)
    # si la covarianza resulta positiva, se multiplica por -1
    cov[cov>0] = -cov[cov>0]
    
    spread = 2*np.sqrt(-cov)
    return spread

# feature microstructural | corwin schultz
def corwinSchultz(series,mean_win=1):
    high=series['high_price']
    low =series['low_price']
    #get beta
    hl=np.log(high/low)**2
    beta=hl.rolling(window=2).sum()
    beta=beta.rolling(window=mean_win).mean()
    #get gamma
    h2=high.rolling(window=2).max()
    l2=low.rolling(window=2).min()
    gamma=np.log(h2/l2)**2
    #get alpha:
    den = 3 - 2*2**.5
    alpha=(2**.5-1)*(beta**.5)/den - (gamma/den)**.5
    alpha[alpha<0]=0 # set negative alphas to 0 (see p.727 of paper)
    #get Sigma
    k2=(8/np.pi)**.5
    den=3-2*2**.5
    sigma=(2**(-.5)-1)*beta**.5/(k2*den)+(gamma/(k2**2*den))**.5
    sigma[sigma<0]=0
    #get spread
    spread=2*(np.exp(alpha)-1)/(1+np.exp(alpha))
    return spread,sigma

# feature microstructural | kyle 85
def kyle85(series):
    d_p_t  = series['close_price'].diff()
    V_t = series['bar_cum_volume']
    b = d_p_t.copy()
    b[b < 0] = -1
    b[b > 0] =  1
    b[b == 0] =  1
    lambda_k = d_p_t / (V_t * b)
    lambda_k[lambda_k < 0] = 0
    return lambda_k

# feature microstructural | amihud 2002 
def amihud2002(series):
    d_logclose_tau = np.log(series['close_price']).diff()
    sum_p_tV_t = series['feat_accumulativeDollarValue']
    lambda_a = np.abs(d_logclose_tau)/sum_p_tV_t
    return lambda_a

# feature microstructural | hasbrouck
def hasbrouck(series): 
    d_logclose_t  = np.log(series['close_price']).diff()

    lambda_h = d_logclose_t/series['feat_hasbrouckSign']
    lambda_h[lambda_h < 0] = 0
    return lambda_h

# feature microstructural | vpin 2008
def vpin2008(series,sum_win):
    V_b = series['feat_accumulativeVolBuyInit']
    V_s = series['feat_accumulativeVolSellInit']
    V = V_b+V_s
    A = np.abs(V_b-V_s).rolling(window=sum_win).sum()
    B = V.rolling(window=sum_win).sum()
    VPIN =  A / B
    return VPIN

# feature microstructural | countdown
def countdown(o,h,l,c):
    countdown_raw = [0]
    for i in range(1,len(o)):
        up_pres = 0
        down_pres = 0
        if c[i] > o[i]:
            up_pres += 1.0
        if h[i] > h[i-1]:
            up_pres += 1.0
        if c[i] < o[i]:
            down_pres += 1.0
        if l[i] < l[i-1]:
            down_pres += 1.0
        countdown_raw.append(up_pres-down_pres)
    return talib.EMA(np.array(countdown_raw)) 

#############################################################################
#############################################################################
####################### CLASE GENERAL - FEATURE CLASS #######################
#############################################################################
#############################################################################

# MAIN feature class
class FeaturesClass():
    """
    Clase generadora de Features

    Permite generar todas las variables que serán utilizadas 
    en el proceso de entrenamiento y prueba de los modelos de aprendizaje
    automático, como parte del flujo de trabajo de Enigmx.

    Inputs obligatorios:
        - DataFrame de una acción con la info de las barras generadas (OHLCV)

    Inputs opcionales:
        - No precisa

    Métodos:
        - technicals: Variables basadas en indicadores técnicos:
            - De tendencia: trima, ema, kama
            - De oscilación: aroonosc, mfi, rsi, ultosk, willr
            - De volatilidad: atr
        - alphas: Variables basadas en el documento '101 alphas': 
            'https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf'
        - tsfresh: Variables basadas en la librería tsfresh: 
            'https://tsfresh.readthedocs.io/en/latest/text/_generated/tsfresh.feature_extraction.feature_calculators.html'
        - others: estadísticos de cada variable aleatoria:
            - zscore
            - zscore con ventana móvil
            - percentil
            - normalización min/max
        - features: este método aplica los 4 métodos anteriores 
                    para unificar todos los features computables por esta clase
    """    
    
    def __init__(self,df, num_days = 15):
        
        # general input variables
        self.df = df
        
        self.o = self.df['open_price']
        self.h = self.df['high_price']
        self.l = self.df['low_price']
        self.c = self.df['close_price']
        
        self.v = self.df['bar_cum_volume']
        self.vwap = self.df['vwap']          
        
        # numb days to lowback
        self.num_days = num_days
        
        # tsfresh features (non-categorical values)
        self.functions =  [
                f.absolute_sum_of_changes, 
                #f.abs_energy,  # |> eq. al 'abs_energy'
                #f.benford_correlation, # |> retorna un solo val aprox. 
                f.count_above_mean, 
                #f.count_below_mean, # |> eq. al 'count_above_mean'
                f.first_location_of_maximum, 
                f.first_location_of_minimum, 
                f.longest_strike_above_mean, 
                #f.longest_strike_below_mean,   
                f.mean_second_derivative_central, 
                f.percentage_of_reoccurring_datapoints_to_all_datapoints, 
                #f.percentage_of_reoccurring_values_to_all_values, 
                    # |> eq. al 'percentage_of_reoccurring_datapoints_to_all_datapoints' 
                f.sample_entropy, 
                f.sum_of_reoccurring_data_points, 
                #f.sum_of_reoccurring_values, # |> eq. a 'sum_of_reoccurring_data_points'
                #f.sum_values, # |> eq. a 'abs_energy'
                #f.variance_larger_than_standard_deviation, # |> int repetitivos
                f.variation_coefficient
            ]
        
    def technicals(self):
        
        """
        Indicadores técnicos basados princ. en la librería talib.
        """
        
        ###############################################################
        ########### Indicadores Generales (Overlap Studies) ###########
        ###############################################################
        
        # 1) Exponential Moving Average
        ema = talib.EMA(
            self.c,
            timeperiod = 30
            ) # general timeperiod: 30        
        
        
        # 2) Computa media móviles as signals según "NAME_PERIOD_SIGNAL", e.g. "EMA_30_SIGNAL"
        #                       DEPRECATED!!!
        listDictsMA = movingAveragesSignals(
            time_frames = [10, 130], # idealmente, trabjar con: [10, 100, 200]
            close_price_series = self.c
            )
        
            # 2.1. Generamos el diccionario con la info de las MA
        dictMa = dict(cmap(*listDictsMA))
        
        # 5) Parabolic SAR (float)  + Lógica continua | new*
        sar = talib.SAR(
            self.h, 
            self.l, acceleration=0.02, maximum=0.4 # setting parameters
            )
                
            # 5.1. Parabolic SAR Continous Signal (diff. prices)
        sarSignal = self.c - sar
        
        # 7) Lógica de Bandas Bollinger... (categorical)
            # 7.1. Estimación de parámetros de la Banda 

        upperband, middleband, lowerband = talib.BBANDS(
            self.c, 
            timeperiod=15, # general parameter | lowback window
            nbdevup=2,     # general parameter | Std banda up 
            nbdevdn=2,     # general parameter | Std banda down
            matype=0       # general paramter 
            )
        
            # 7.2. Computación de lógica discreta - BB
        bollingerBand = bollingerBandsLogic(
            upperband, 
            middleband, 
            lowerband, 
            self.c
            )
        
            # 7.3. Computación de lógica continua - Comp. de volatilidad BB        
        compresionVolatilidad = upperband - lowerband
        
        # 8) Lógica de MESA: MAMA VS FAMA... (continua)
        mamaLine, famaLine = talib.MAMA(self.c)
        
            # 8.1. Criteria: https://phemex.com/academy/what-is-mesa-adaptive-moving-average 
        mesaSignal = mamaLine - famaLine # signal with continous values 
        
        
        ###############################################################
        ################### Indicadores de Momentum ###################
        ###############################################################  

        # 1) ADX - Average Directional Movement Index
        adx = talib.ADX(
            self.h, 
            self.l,  
            self.c, 
            timeperiod=14 #general timeperiod
            ) 

        # 2) Aroon Oscillator
        aroonosc = talib.AROONOSC(
            self.h, 
            self.l,
            timeperiod=14 # general timeperiod
            ) 
        
        # 3) BOP - Balance Of Power
        bop = talib.BOP(self.o,self.h,self.l,self.c)        
        
        # 4) Commodity Channel Index
        cci = talib.CCI(
            self.h, 
            self.l, 
            self.c, 
            timeperiod=14 # general timeperiod
            )
        
        # 5) Money Flow Index - MFI
        mfi = talib.MFI(
            self.h,
            self.l,
            self.c,
            self.v,
            timeperiod=14 # general timeperiod
            )
        
        # 5) Relative Strength Index - RSI
        rsi = talib.RSI(self.c, timeperiod = 14) # general timeperiod
        
        # 6) Ultimate Oscillator - uOscillator
        ultosk = talib.ULTOSC(
            self.h, 
            self.l, 
            self.c,
            timeperiod1 = 7,  # general timeperiod
            timeperiod2 = 14, # general timeperiod
            timeperiod3 = 28  # general timeperiod
            )
        
        # 7) Williams' %R
        willr = talib.WILLR(
            self.h,
            self.l,
            self.c,
            timeperiod = 14 # general timeperiod        
            ) 
        
        # 8) Lógica de MACD... (categorical) 
        
            # Pendiente...
        
        
        # 9) Lógica de Stochastic... (continua)
        
        #    Based on: https://www.investopedia.com/terms/s/stochasticoscillator.asp
        
            # 9.1. Slow K-D Stochastic Version
        slowk, slowd = talib.STOCH(
            self.h, 
            self.l, 
            self.c, 
            fastk_period=5,  # general timeperiod
            slowk_period=3,  # general timeperiod
            slowk_matype=0,  # general timeperiod
            slowd_period=3,  # general timeperiod
            slowd_matype=0   # general timeperiod
            )
        
        slowStochastic = slowk - slowd #buy > 0; else, sell.
        
            # 9.2. Fast K-D Stochastic Version
        fastk, fastd = talib.STOCHF(
            self.h, 
            self.l, 
            self.c, 
            fastk_period=5,  # general timeperiod
            fastd_period=3,  # general timeperiod
            fastd_matype=0   # general timeperiod
            )
        
        fastStochastic = fastk - fastd
        
        
        ###############################################################
        #################### Indicadores de Volumen ###################
        ###############################################################  

        # 1) Chaikin A/D Line (float) | new*
        ad = talib.AD(self.h, self.l, self.c, self.v)
        
        # 2) On balance volume - OBV (float) | new*
        obv = talib.OBV(self.c, self.v)
        
        ###############################################################
        ################## Indicadores de Volatilidad #################
        ###############################################################         
        
        # 1) Average True Range (ATR)
        atr = talib.ATR(
            self.h, 
            self.l, 
            self.c,
            timeperiod = 14 # general timeperiod  
            )        
        
        ###############################################################
        ########## Indicadores de Transformación de Precios ###########
        ###############################################################          
        
        # Weighted Close Price - WCP (float) | new* 
        #wcp = talib.WCLPRICE(self.h, self.l, self.c)        
        
        ###############################################################
        #################### Indicadores de Ciclos ####################
        ###############################################################         
        
        # Transformada de Hilbert (float de desfase 90° de la señal) | new* 
        #simple_hilbert =  talib.HT_DCPERIOD(self.c)        
        
        ###############################################################
        ########## Indicadores de Reconocimiento de Patrones ########## 
        ###############################################################           
        
        # 1) Patterm feature 1 - Two Crows (int) | new*
        #twocrows = talib.CDL2CROWS(self.o, self.h, self.l, self.c) # DESCARTAR
        
        # 2) Pattern feature 2 - Three Black Crows (int) | new*
        #threecrows = talib.CDL3BLACKCROWS(self.o, self.h, self.l, self.c)
        
        # 3) Pattern feature 3 - Three Inside Up/Down (int) | new* 
        #threeinside = talib.CDL3INSIDE(self.o, self.h, self.l, self.c) 
        
        # 4) Pattern feature 4 - Three-Line Strike (int) | new*
        #threestrike = talib.CDL3LINESTRIKE(self.o, self.h, self.l, self.c) 

        # 5) Pattern feature 5 - Three Stars In The South (int) | new*
        #threestars = talib.CDL3STARSINSOUTH(self.o, self.h, self.l, self.c)        
        
        # 6) Pattern feature 6 - Three Advancing White Soldiers (int) | new*
        #taws = talib.CDL3WHITESOLDIERS(self.o, self.h, self.l, self.c)        
        
        # 7) Pattern feature 7 - Abandoned Baby (int) | new*
        #ababy = talib.CDLABANDONEDBABY(self.o, self.h, self.l, self.c)           
        
        # 8) Pattern feature 8 - Advance Block (int) | new*
        #ablock = talib.CDLADVANCEBLOCK(self.o, self.h, self.l, self.c) 
        
        # 9) Pattern feature 9 - Belt-hold (int) | new*
        belthold = talib.CDLBELTHOLD(self.o, self.h, self.l, self.c)        
        
        # 10) Pattern feature 10 - Breakaway (int) | new*
        #breakaway = talib.CDLBREAKAWAY(self.o, self.h, self.l, self.c) 
        
        # 11) Pattern feature 11 - Closing Marubozu (int) | new*
        #marubozu = talib.CDLCLOSINGMARUBOZU(self.o, self.h, self.l, self.c) 
        
        # 12) Pattern feature 12 - Baby Sawllow (int) | new*
        #baby_swallow = talib.CDLCONCEALBABYSWALL(self.o, self.h, self.l, self.c)                
        
        # 13) Pattern feature 13 - Counterattack (int) | new*
        #counterattack = talib.CDLCOUNTERATTACK(self.o, self.h, self.l, self.c) 
        
        # 14) Pattern feature 14 - Dark Cloud Cover (int) | new*        
        #darkcloud = talib.CDLDARKCLOUDCOVER(self.o, self.h, self.l, self.c) 
        
        # 15) Pattern feature 15 - DOJI (int) | new*
        doji = talib.CDLDOJI(self.o, self.h, self.l, self.c) 
        
        # 16) Pattern feature 16 - Engulfing Pattern (int) | new*
        eng = talib.CDLENGULFING(self.o, self.h, self.l, self.c)
        
        # 17) Pattern feature 17 - Evening Star (int) | new*
        #evening_star = talib.CDLEVENINGSTAR(self.o, self.h, self.l, self.c)        

        # 18) Pattern feature 18 - Up/Down-gap side-by-side white lines (int) | new*
        #up_down_gap = talib.CDLGAPSIDESIDEWHITE(self.o, self.h, self.l, self.c)
        
        # 19) Pattern feature 19 - Hammer (int) | new*
        hammer = talib.CDLHAMMER(self.o, self.h, self.l, self.c)        
        
        # 20) Pattern feature 20 - Hanging Man (int) | new*
        #hanging = talib.CDLHANGINGMAN(self.o, self.h, self.l, self.c)          

        # 21) Pattern feature 21 - Harami Pattern (int) | new*
        harami = talib.CDLHAMMER(self.o, self.h, self.l, self.c)           
        
        # 22) Pattern feature 22 - High-Wave Candle (int) | new*
        hw_candle = talib.CDLHIGHWAVE(self.o, self.h, self.l, self.c) 
        
        # 23) Pattern feature 23 - Hikkake Pattern (int) | new*
        hikkake = talib.CDLHIKKAKE(self.o, self.h, self.l, self.c)         

        # 24) Pattern feature 24 - Homing Pigeon (int) | new*
        #pigeon = talib.CDLHOMINGPIGEON(self.o, self.h, self.l, self.c)          
        
        # 25) Pattern feature 25 - In-Neck Pattern (int) | new*
        #neck = talib.CDLINNECK(self.o, self.h, self.l, self.c)         

        # 26) Pattern feature 26 - Kicking (int) | new*
        #kicking = talib.CDLKICKING(self.o, self.h, self.l, self.c)    

        # 27) Pattern feature 27 - Ladder Bottom (int) | new*
        #ladder = talib.CDLLADDERBOTTOM(self.o, self.h, self.l, self.c)  
        
        # 28) Pattern feature 28 - Marubozu (int) | new*
        marubozu = talib.CDLMARUBOZU(self.o, self.h, self.l, self.c)         
        
        # 29) Pattern feature 29 - Matching Low (int) | new*
        matchinglow = talib.CDLMATCHINGLOW(self.o, self.h, self.l, self.c)    

        # 30) Pattern feature 30 - Mat Hold (int) | new*
        #mathold = talib.CDLMATHOLD(self.o, self.h, self.l, self.c)  
        
        # 31) Pattern feature 31 - Morning Star (int) | new*
        #morning_star = talib.CDLMORNINGSTAR(self.o, self.h, self.l, self.c)           
        
        # 32) Pattern feature 32 - Piercing Pattern (int) | new*
        #piercing = talib.CDLPIERCING(self.o, self.h, self.l, self.c)        
        
        # 33) Pattern feature 32 - Rickshaw Man (int) | new*
        rickshaw = talib.CDLRICKSHAWMAN(self.o, self.h, self.l, self.c)         
        
        # 34) Pattern feature 34 - Shooting Star (int) | new*
        #shooting_star = talib.CDLSHOOTINGSTAR(self.o, self.h, self.l, self.c)

        # 35) Pattern feature 35 - Stick Sandwich (int) | new*
        #stick_sandwich = talib.CDLSTICKSANDWICH(self.o, self.h, self.l, self.c)
        
        # 36) Pattern feature 36 - Tasuki Gap (int) | new*
        #tasuki = talib.CDLTASUKIGAP(self.o, self.h, self.l, self.c)
        
        # 37) Pattern feature 37 - Unique 3 River (int) | new*
        #river = talib.CDLUNIQUE3RIVER(self.o, self.h, self.l, self.c)
        
        ###############################################################
        ################### Indicadores Estadísticos ##################
        ###############################################################         
        
        # 1) Statistical feature 1 - Beta (float ) | new*
        beta_feature = talib.BETA(
            self.h, 
            self.l, 
            timeperiod=5 # general timeperiod: 5
            ) 
        
        # 2) Statistical feature 2 - Time Series Forecast | new*
        #tforecast = talib.TSF(
        #    self.c, timeperiod=14 # genereal timeperiod: 14
        #    )
        
        ###############################################################
        ################### Indicadores No Agrupados ##################
        ###############################################################           
        
        # Disparitiy | insumo EMA 
        disparity = ((self.c/ema)-1)*100

        ###############################################################
        ###############################################################
        ###############################################################
        
        # choppiness feature
        choppiness = []

        # iteración choppines 
        for i in range(0,len(self.c)):
            if i-self.num_days >= 0:
                choppiness.append(
                    100*np.log10(
                        atr[i]/(
                            max(self.c[i-self.num_days:i]) - min(self.c[i-self.num_days:i])
                            )
                        )*(1/np.log10(self.num_days)))
            elif i == 0:
                choppiness.append(
                    100*np.log10(atr[i]/(self.c[i]))*(1/np.log10(self.num_days))
                    )
            else:
                choppiness.append(
                    100*np.log10(
                        atr[i]/(max(self.c[0:i])-min(self.c[0:i]))
                        )*(1/np.log10(self.num_days))
                    )
        
        # fisher feature
        fisher_prev = []
        
        # iteración fisher
        for i in range(0,len(self.c)):
            if i-self.num_days >= 0:
                fisher_prev.append(
                    (
                        self.c[i] - min(self.c[i-self.num_days:i+1])
                        )/(
                            max(self.c[i-self.num_days:i+1]) - min(self.c[i-self.num_days:i+1])
                            )
                    )
            elif i == 0:
                fisher_prev.append(0)
            else:
                fisher_prev.append(
                    (
                        self.c[i] - min(self.c[0:i+1])
                        )/(max(self.c[0:i+1])-min(self.c[0:i+1]))
                    )
        
        # fisher estimation
        fisher_prev = [(2*i - 1) for i in fisher_prev]
        fisher_prev = [i if i != 1.0 else 0.999 for i in fisher_prev]
        fisher_prev = [i if i != -1.0 else -0.999 for i in fisher_prev]
        fisher = [
            (1/2)*(np.log((1+fisher_prev[i])/(1-fisher_prev[i]))) for i in range(len(self.c))
            ]
        
        # countdown feature
        countdown_ind = countdown(self.o, self.h, self.l, self.c)
        
        
        # definimos diccionario general de técnicos (no inc. las medias móviles)
        self.dictTechnicals = (
        	{
			#'ema':ema,
			#'hilbert_trend':trend_hilbert,
			#'kama':kama,
			#'midpoint':midpoint,
			'sar_signal':sarSignal,
			#'trima':trima,
            'bollinger_band_integer': bollingerBand, # lógica técnica discr.
            'bollinger_volatility_compression': compresionVolatilidad, # lógica técnico. cont.
            'mesa_signal': mesaSignal, #lógica técnico de valores continuos
			'adx': adx,
			'aroonosc':aroonosc,
			'bop' : bop,
			'cci': cci,
			'mfi':mfi,
			'rsi':rsi,
			'ultosk':ultosk,
			'willr':willr,
            'slow_stochastic': slowStochastic, # lógica técnico. cont.
            'fast_stochastic': fastStochastic, # lógica técnico. cont.
			'adline':ad,
			'obv':obv,
			'atr':atr,
			#'weighted_close_price':wcp,
			#'hilbert_trans':simple_hilbert,
			#'two_crows_integer':twocrows,
			#'three_crows_integer':threecrows,
			#'three_inside_integer': threeinside,	
			#'three_strike_integer': threestrike,	
			#'three_stars_integer': threestars,
			#'taws_integer_integer':taws,
			#'ababy_integer':ababy,
			#'ablock_integer':ablock,
			'belthold_integer':belthold,
			#'breakaway_integer':breakaway,
			#'baby_swallow_integer':baby_swallow,
			#'counterattack_integer': counterattack,	
			#'darkcloud_integer':darkcloud,
			'doji_integer':doji,
			'eng_integer':eng,
			#'evening_star_integer':evening_star,
			#'up_down_gap_integer':up_down_gap,
			'hammer_integer':hammer,
			#'hanging_integer':hanging,
			'harami_integer':harami,
			'hw_candle_integer':hw_candle,
			'hikkake_integer':hikkake,
			#'pigeon_integer':pigeon,
			#'neck_integer':neck,
			#'kicking_integer':kicking,
			#'ladder_integer':ladder,
			'marubozu_integer':marubozu,
			'matchinglow_integer':matchinglow,
			#'mathold_integer':mathold,
			#'morning_star_integer':morning_star,
			#'piercing_integer':piercing,
			'rickshaw_integer':rickshaw,
        	#'shooting_star_integer':shooting_star,
        	#'stick_sandwich_integer':stick_sandwich,
        	#'tasuki_gap_integer':tasuki,
        	#'river_integer':river,
        	'beta_tech':beta_feature,
        	#'tforecast':tforecast,		
        	'disparity':disparity,
        	'choppiness' : choppiness,
        	'fisher' : fisher,
        	'countdown_ind' : countdown_ind
        		}
        )
        
        # actualizamos el diccionario de técnicos con el de MA's
        self.dictTechnicals = {**dictMa, **self.dictTechnicals}
        
        # creamos DataFrame con todos los features técnicos
        self.dftechnicals  = pd.DataFrame(self.dictTechnicals)
        
        return self.dftechnicals
    
    # alpha features
    def alphas(self):
        
        # get a dataframe 
        return MainGetAlphasFunction(
                o = self.o, 
                h = self.h, 
                l = self.l, 
                c = self.c, 
                v = self.c, 
                vwap = self.vwap
            )
    
    # statistical features - tsfresh
    def tsfresh(self):
        
        # empty dataframe
        self.dftsfresh=pd.DataFrame()
        
        # iteration over tsfresh definided functions
        for i in self.functions:
            self.dftsfresh[str(i.__name__)] = rolling_application(self.c, i)
        return self.dftsfresh
    
    # microstructural features computation
    def microstructural(self):
        
        # micro 1
        roll = roll84(self.df) 
        
        # micro 2
        corSchultz = corwinSchultz(self.df)
        
        # micro 3
        kyle = kyle85(self.df)
        
        # micro 4
        amihud = amihud2002(self.df)
        
        # micro 5
        hasb = hasbrouck(self.df)
        
        # micro 6
        vpin = vpin2008(self.df, sum_win = 5).fillna(0) #definimos la ventana del SumWin
        
        # construye el df con los features de microestructura         
        self.dfmicrostructural = pd.DataFrame(
            {
                'roll' : roll,'corwinschultz' : corSchultz[0],
                'kyle' : kyle, 'amihud' : amihud,
                'hasbrouck' : hasb, 'vpin' : vpin}
            )
        
        # general microstructural dataframe 
        return self.dfmicrostructural

    # other related features
    def others(self):
        
        # z score feature  
        z_score = get_z_score(self.c)
        
        # pct change (returns)
        pct = self.c.rank(pct=True)
        
        # rolling zscore
        rolling_zscore = get_rolling_z_score(self.c)
        
        # minmax feature
        minmax = min_max(self.c)
        
        # general other features dataframe
        self.dfothers = pd.DataFrame(
            {
                'z_score':z_score,
                'pct':pct,
                'rolling_zscore':rolling_zscore,
                'minmax':minmax
                }
            )
        return self.dfothers 
    
    # compiler main features function
    def features(self):
        
        # nombre de variables que no son features
        nonFeatures = list(self.df.columns.values)
        
        print("       >>>> Computing Talib - Technical features... ")
        # features técnicos (df)
        technicals = self.technicals()
            
            #adding general prefix
        technicals = technicals.add_prefix('technical_')
        
        print("       >>>> Computing Tsfresh - Statistical features... ")
        # features estadísticos (stats)
        tsfresh = self.tsfresh()
        
            #adding general prefix
        tsfresh = tsfresh.add_prefix('statistical_')
        
        print("       >>>> Computing MLDP - Microstructural features... ")
        # features microestructurales 
        microstructural = self.microstructural()
        
            #adding general prefix
        microstructural = microstructural.add_prefix('microstructural_')
        
        print("       >>>> Computing Complementary features... ")
        # features others 
        others = self.others()
        
            #adding general prefix
        others = others.add_prefix('related_')        
        
        print("   ||*** Building base dataframe object ***|| ")
        # concadenación global de features
        df = pd.concat(
            [
                self.df,
                technicals,
                tsfresh,
                others,
                microstructural,

            ], axis=1
            )
        
        print("       >>>> Computing Alpha 101 - Signals features... ")
        # features alphas 
        alphas = self.alphas()
        
        print("   ||*** Final Concadenation***|| ")
        # adding alphas to base dataframe
        df[alphas.columns.values] = alphas
        
        # removing NaN's
        df = df.dropna()
        
        # iteración para reasignación de nombre por columna
        for i in df.columns:
        
            # agrega el prefijo "feeature"
            if i not in nonFeatures:
                name = 'feature_'+i
                nonFeatures.append(name)
                
        df.columns = nonFeatures 
        
        return df
