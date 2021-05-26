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

def amihud2002(series):
    d_logclose_tau = np.log(series['close_price']).diff()
    sum_p_tV_t = series['feat_accumulativeDollarValue']
    lambda_a = np.abs(d_logclose_tau)/sum_p_tV_t
    return lambda_a

def hasbrouck(series): 
    d_logclose_t  = np.log(series['close_price']).diff()

    lambda_h = d_logclose_t/series['feat_hasbrouckSign']
    lambda_h[lambda_h < 0] = 0
    return lambda_h

def vpin2008(series,sum_win):
    V_b = series['feat_accumulativeVolBuyInit']
    V_s = series['feat_accumulativeVolSellInit']
    V = V_b+V_s
    A = np.abs(V_b-V_s).rolling(window=sum_win).sum()
    B = V.rolling(window=sum_win).sum()
    VPIN =  A / B
    return VPIN

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
        self.num_days = num_days
        
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
        bop = talib.BOP(self.o,self.h,self.l,self.c)
        disparity = ((self.c/ema)-1)*100
        
        choppiness = []
        for i in range(0,len(self.c)):
            if i-self.num_days >= 0:
                choppiness.append(100*np.log10(atr[i]/(max(self.c[i-self.num_days:i])-min(self.c[i-self.num_days:i])))*(1/np.log10(self.num_days)))
            elif i == 0:
                choppiness.append(100*np.log10(atr[i]/(self.c[i]))*(1/np.log10(self.num_days)))
            else:
                choppiness.append(100*np.log10(atr[i]/(max(self.c[0:i])-min(self.c[0:i])))*(1/np.log10(self.num_days)))
        
        fisher_prev = []
        for i in range(0,len(self.c)):
            if i-self.num_days >= 0:
                fisher_prev.append((self.c[i] - min(self.c[i-self.num_days:i+1]))/(max(self.c[i-self.num_days:i+1]) - min(self.c[i-self.num_days:i+1])))
            elif i == 0:
                fisher_prev.append(0)
            else:
                fisher_prev.append((self.c[i] - min(self.c[0:i+1]))/(max(self.c[0:i+1])-min(self.c[0:i+1])))
        fisher_prev = [(2*i - 1) for i in fisher_prev]
        fisher_prev = [i if i != 1.0 else 0.999 for i in fisher_prev]
        fisher_prev = [i if i != -1.0 else -0.999 for i in fisher_prev]
        fisher = [(1/2)*(np.log((1+fisher_prev[i])/(1-fisher_prev[i]))) for i in range(len(self.c))]
        
        countdown_ind = countdown(self.o, self.h, self.l, self.c)
        
        self.dftechnicals = pd.DataFrame({'trima':trima,'kama':kama,'ema':ema,'aroonosc':aroonosc,
                                        'mfi':mfi,'rsi':rsi,'ultosk':ultosk,
                                          'willr':willr,'atr':atr, 
                                          'bop' : bop, 'disparity' : disparity,
                                          'choppiness' : choppiness,
                                          'fisher' : fisher,
                                          'countdown_ind' : countdown_ind})
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
    
    def microstructural(self):
        roll = roll84(self.df) 
        corSchultz = corwinSchultz(self.df)
        kyle = kyle85(self.df)
        amihud = amihud2002(self.df)
        hasb = hasbrouck(self.df)
        vpin = vpin2008(self.df)
        self.dfmicrostructural = pd.DataFrame({'roll' : roll,'corwinschultz' : corSchultz,
                                             'kyle' : kyle, 'amihud' : amihud,
                                             'hasbrouck' : hasb, 'vpin' : vpin})
        return self.dfmicrostructural
        
    
    
    
    
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
        #microstructural = self.microstructural()
        others = self.others()
        df = pd.concat([self.df,
                        technicals,
                        tsfresh,
                        others,
                        #microstructural,
                        alphas
                        ],axis=1).dropna()
        
        for i in df.columns:
            
            if i not in nonFeatures:
                name = 'feature_'+i
                nonFeatures.append(name)
        
        df.columns = nonFeatures #names
        return df


dataframe = pd.read_csv("C:/Users/ASUS/Desktop/HELI/Quantmoon/enigmx-repo/.STACKED_EXO_VOLUME_MDA.csv")
FeaturesClass(dataframe).features().to_excel('C:/Users/ASUS/Desktop/gas.xlsx')
