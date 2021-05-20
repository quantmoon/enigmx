"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import pandas as pd
import inspect as ins
import talib
import numpy as np
import scipy as sp
from scipy.stats import rankdata,spearmanr
from tsfresh.feature_extraction import feature_calculators as f
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters

def get_z_score(serie):
    return (serie-serie.mean())/serie.std()

def get_rolling_z_score(serie, window=200):
    return (serie-serie.rolling(window=window, min_periods=20).mean())/serie.rolling(window=200, min_periods=20).std()

def min_max(serie):
    return (serie-min(serie))/(max(serie)-min(serie))

def rolling_application(serie, function, rolling_periods = 200, min_periods=20):
    return serie.rolling(rolling_periods,min_periods=min_periods).apply(function)

def rolling_rank(na):
    return rankdata(na)[-1:]

def ts_rank(s,window):
    return s.rolling(window).apply(rolling_rank)

def alpha2_(v,c,o,window=20):
    alpha2_ = [np.nan]*window
    for i in range(len(v))[window:]:
        alpha2_.append(-1*np.log(v[i-window:i]).diff(2).corr(((c[i-window:i]-o[i-window:i])/o[i-window:i]).rank(),method='spearman'))
    return alpha2_

def alpha3_(v,o,window=20):
    alpha3_ = [np.nan]*window
    for i in range(len(v))[window:]:
        alpha3_.append( -1*o[i-window:i].rank().corr(v[i-window:i].rank(pct=True),method='spearman'))
    return alpha3_

def alpha5_(l): 
    return -1*ts_rank(l.rank(),9)

def alpha9_(c):
    if min(c.diff(1)) > 0:
        return c.diff(1)
    elif max(c.diff(1)) < 0:
        return c.diff(1)
    else: return -1*c.diff(1)

def alpha12_(c,v): 
    return np.sign(v.diff(1))*(-1*c.diff(1))#[-1:])

def alpha14_(c,o,v):
    return(-1*c.pct_change().diff(3).rank()*spearmanr(o,v)[0])

def alpha20_(o,h,c,l):
    return round((-1*(o - h.diff(1)).rank()*(o - c.diff(1)).rank()*(o - l.diff(1)).rank()/1000000))#[-1:])

def alpha23_(h,window=20):
    alpha23_ = [np.nan]*window

    for i in range(len(h))[window:]:
        if sum(h)/len(h) > h.iloc[-1]:
            alpha23_.append(-1*h.diff(2).iloc[-1])
        else:
            alpha23_.append(0)
    return alpha23_

class FeaturesClass():
    """
    Clase generadora de Features

    Permite generar todas las variables que serán utilizadas en el proceso de entrenamiento y prueba de los modelos de aprendizaje
    automático, como parte del flujo de trabajo de Enigmx.

    Inputs obligatorios:
        - DataFrame de una acción con la información de las barras generadas (OHLCV)

    Inputs opcionales:
        - No precisa

    Métodos:
        - technicals: Variables basadas en indicadores técnicos:
            - De tendencia: trima, ema, kama
            - De oscilación: aroonosc, mfi, rsi, ultosk, willr
            - De volatilidad: atr
        - alphas: Variables basadas en el documento '101 alphas': 'https://arxiv.org/ftp/arxiv/papers/1601/1601.00991.pdf'
        - tsfresh: Variables basadas en la librería tsfresh: 'https://tsfresh.readthedocs.io/en/latest/text/_generated/tsfresh.feature_extraction.feature_calculators.html'
        - others: estadísticos de cada variable aleatoria:
            - zscore
            - zscore con ventana móvil
            - percentil
            - normalización min/max
        - features: este método aplica los 4 métodos anteriores para unificar todos los features computables por esta clase
    """    
    
    def __init__(self,df, num_days = 15):
        self.df = df
        self.o = df['open_price']
        self.h = df['high_price']
        self.l = df['low_price']
        self.c = df['close_price']
        self.v = df['bar_cum_volume']
        
        self.functions =  [f.absolute_sum_of_changes, f.abs_energy, f.benford_correlation, f.count_above_mean, f.count_below_mean, 
              f.first_location_of_maximum, f.first_location_of_minimum, f.longest_strike_above_mean, 
              f.longest_strike_below_mean, f.mean_second_derivative_central, 
              f.percentage_of_reoccurring_datapoints_to_all_datapoints, f.percentage_of_reoccurring_values_to_all_values,
              f.sample_entropy, f.sum_of_reoccurring_data_points, f.sum_of_reoccurring_values, f.sum_values,
              f.variance_larger_than_standard_deviation, f.variation_coefficient]
        
    def technicals(self):
        trima = talib.TRIMA(self.c)
        kama = talib.KAMA(self.c)
        ema = talib.EMA(self.c)
        aroonosc = talib.AROONOSC(self.h,self.l)
        mfi = talib.MFI(self.h,self.l,self.c,self.v)
        rsi = talib.RSI(self.c)
        ultosk = talib.ULTOSC(self.h,self.l,self.c)
        willr = talib.WILLR(self.h,self.l,self.c)
        atr = talib.ATR(self.h,self.l,self.c)
        self.dftechnicals = pd.DataFrame({'trima':trima,'kama':kama,'ema':ema,'aroonosc':aroonosc,
                                        'mfi':mfi,'rsi':rsi,'ultosk':ultosk,
                                          'willr':willr,'atr':atr})
        return self.dftechnicals
    
    def alphas(self):
        v = self.v
        o = self.o
        c = self.c
        l = self.l
        h = self.h
        
        alpha2 = alpha2_(v,c,o)
        alpha3 = alpha3_(v,o)
        alpha5 = alpha5_(l)
        alpha9 = alpha9_(c)
        alpha12 = alpha12_(c,v)
        alpha14 = alpha14_(c,o,v)
        alpha20 = alpha20_(o,h,c,l)
        alpha23 = alpha23_(h)
        self.dfalphas = pd.DataFrame({'alpha2':alpha2,'alpha3':alpha3,'alpha5':alpha5,
                                   'alpha9':alpha9,'alpha12':alpha12,'alpha14':alpha14,
                                   'alpha20':alpha20,'alpha23':alpha23})
        #return pd.concat([self.df,self.alphas],axis = 1)
        return self.dfalphas
    
    def tsfresh(self):
        self.dftsfresh=pd.DataFrame()
        for i in self.functions:
            self.dftsfresh[str(i.__name__)] = rolling_application(self.c, i)
        return self.dftsfresh
    
    def others(self):
        z_score = get_z_score(self.c)
        pct = self.c.rank(pct=True)
        rolling_zscore = get_rolling_z_score(self.c)
        #rolling_pct = get_rolling_pct(self.c)
        minmax = min_max(self.c)
        self.dfothers = pd.DataFrame({'z_score':z_score,'pct':pct,'rolling_zscore':rolling_zscore,
                                     'minmax':minmax})
        return self.dfothers 
    
    def features(self):
        
        nonFeatures = list(self.df.columns.values)
        
        alphas = self.alphas()
        technicals = self.technicals()
        tsfresh = self.tsfresh()
        others = self.others()
        df = pd.concat([self.df,technicals,tsfresh,others,alphas],axis=1).dropna()
        
        #names = []
        
        for i in df.columns:
            
            if i not in nonFeatures:
                name = 'feature_'+i
                nonFeatures.append(name)
        
        df.columns = nonFeatures #names
        return df


#dataframe = pd.read_csv("D:/feature_importance/STACKED_EXO_VOLUME_MDA.csv")
#print(dataframe.columns)
#print(FeaturesClass(dataframe).features())