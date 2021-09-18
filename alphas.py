import pandas as pd
#import inspect as ins
#import talib
import numpy as np
#import scipy as sp
from scipy.stats import rankdata,spearmanr
#from tsfresh import extract_features

def get_z_score(serie):
    return (serie-serie.mean())/serie.std()
def get_rolling_z_score(serie, window=200):
    return (serie-serie.rolling(window=window, min_periods=20).mean())/serie.rolling(window=200, min_periods=20).std()
def min_max(serie):
    return (serie-min(serie))/(max(serie)-min(serie))
    
def rolling_application(serie, function, rolling_periods = 200, min_periods=20):
    return serie.rolling(rolling_periods,min_periods=min_periods).apply(function)
    
def scale(df, k=1):
    """
    Scaling time serie.
    :param df: a pandas DataFrame.
    :param k: scaling factor.
    :return: a pandas DataFrame rescaled df such that sum(abs(df)) = k
    """
    return df.mul(k).div(np.abs(df).sum())

def rolling_rank(na):
    return rankdata(na)[-1:]

def delay(df, period=1):
    """
    Wrapper function to estimate lag.
    :param df: a pandas DataFrame.
    :param period: the lag grade.
    :return: a pandas DataFrame with lagged time series
    """
    return df.shift(period)

def decay_linear(df, period=10):
    """
    Linear weighted moving average implementation.
    :param df: a pandas DataFrame.
    :param period: the LWMA period
    :return: a pandas DataFrame with the LWMA.
    """
    # Clean data
    if df.isnull().values.any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(value=0, inplace=True)
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :] 
    na_series = df.values
    
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    # Estimate the actual lwma with the actual close.
    # The backtest engine should assure to be snooping bias free.
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    return pd.DataFrame(na_lwma, index=df.index, columns=['CLOSE'])  

def delta(df, period=1):
    """
    Wrapper function to estimate difference.
    :param df: a pandas DataFrame.
    :param period: the difference grade.
    :return: a pandas DataFrame with today’s value minus the value 'period' days ago.
    """
    return df.diff(period)

def covariance(x, y, window=10):
    """
    Wrapper function to estimate rolling covariance.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).cov(y)

def correlation(x, y, window=10):
    """
    Wrapper function to estimate rolling corelations.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return x.rolling(window).corr(y)

def sma(df, window=10):
    """
    Wrapper function to estimate SMA.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).mean()


def stddev(df, window=10):
    """
    Wrapper function to estimate rolling standard deviation.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).std()

def ts_max(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series max over the past 'window' days.
    """
    return df.rolling(window).max()

def ts_min(df, window=10):
    """
    Wrapper function to estimate rolling min.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).min()

def ts_sum(df, window=10):
    """
    Wrapper function to estimate rolling sum.
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: a pandas DataFrame with the time-series min over the past 'window' days.
    """
    return df.rolling(window).sum()

def ts_rank(s,window):
    return s.rolling(window).apply(rolling_rank)

def rank(df):
    """
    Cross sectional rank
    :param df: a pandas DataFrame.
    :return: a pandas DataFrame with rank along columns.
    """
    #return df.rank(axis=1, pct=True)
    return df.rank(pct=True)

def ts_argmax(df, window=10):
    """
    Wrapper function to estimate which day ts_max(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmax) + 1 

def ts_argmin(df, window=10):
    """
    Wrapper function to estimate which day ts_min(df, window) occurred on
    :param df: a pandas DataFrame.
    :param window: the rolling window.
    :return: well.. that :)
    """
    return df.rolling(window).apply(np.argmin) + 1

def alpha1_(c):
    inner = c
    r = inner.pct_change()
    r.iloc[0] = 0
    inner[r < 0] = stddev(r, 20)
    alpha1_ = rank(ts_argmax(inner ** 2, 5))
    return alpha1_

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

def alpha4_(l):
    alpha4_ = -1 * ts_rank(rank(l), 9)
    return alpha4_

def alpha5_(o, c, vwap): 
    alpha5_ = (rank((o - (sum(vwap, 10) / 10))) * (-1 * abs(rank((c - vwap))))) 
    return alpha5_

def alpha6_(o, v):
    df = -1 * correlation(o, v, 10)
    alpha6_ = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return alpha6_
    
def alpha7_(c, v):
    adv20 = sma(v, 20)
    alpha7_ = -1 * ts_rank(abs(delta(c, 7)), 60) * np.sign(delta(c, 7))
    alpha7_[adv20 >= v] = -1
    return alpha7_
        
def alpha8_(o):
    r = o.pct_change()
    r.iloc[0] = 0
    alpha8_ = -1 * (rank(((ts_sum(o, 5) * ts_sum(r, 5)) - delay((ts_sum(o, 5) * ts_sum(r, 5)), 10))))
    return alpha8_
    
def alpha9_(c):
    if min(c.diff(1)) > 0:
        return c.diff(1)
    elif max(c.diff(1)) < 0:
        return c.diff(1)
    else: return -1*c.diff(1)
    
def alpha10_(c):
    delta_close = delta(c, 1)
    cond_1 = ts_min(delta_close, 4) > 0
    cond_2 = ts_max(delta_close, 4) < 0
    alpha10_ = -1 * delta_close
    alpha10_[cond_1 | cond_2] = delta_close
    return np.log(alpha10_) #reformed | eq. to alpha 9
    
def alpha11_(vwap, v, c):
    alpha11_ = ((rank(ts_max((vwap - c), 3)) + rank(ts_min((vwap - c), 3))) *rank(delta(v, 3)))
    return alpha11_
    
def alpha12_(c,v): 
    alpha12_ = np.sign(v.diff(1))*(-1*c.diff(1))#[-1:])
    return (alpha12_/10) ** 2 #reformed | eq. to alpha 9

def alpha13_(c, v):
    alpha13_ = -1 * rank(covariance(rank(c), rank(v), 5))
    return alpha13_

def alpha14_(c,o,v):
    return(-1*c.pct_change().diff(3).rank()*spearmanr(o,v)[0])

def alpha15_(h, v):
    df = correlation(rank(h), rank(v), 3)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha15_ = -1 * ts_sum(rank(df), 3)
    return alpha15_
    
def alpha16_(h, v):
    alpha16_ = -1 * rank(covariance(rank(h), rank(v), 5))
    return alpha16_

def alpha17_(c, v):
    adv20 = sma(v, 20)
    alpha17_ = -1 * (rank(ts_rank(c, 10)) * rank(delta(delta(c, 1), 1)) * rank(ts_rank((v / adv20), 5)))
    return alpha17_

def alpha18_(c, o):
    df = correlation(c, o, 10)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha18_ = -1 * (rank((stddev(abs((c - o)), 5) + (c - o)) + df))
    return alpha18_

def alpha19_(c):
    r = c.pct_change()
    r.iloc[0] = 0
    alpha19_ = ((-1 * np.sign((c - delay(c, 7)) + delta(c, 7))) * (1 + rank(1 + ts_sum(r, 250))))
    return alpha19_

def alpha20_(o,h,c,l):
    return round((-1*(o - h.diff(1)).rank()*(o - c.diff(1)).rank()*(o - l.diff(1)).rank()/1000000))

def alpha21_(c, v):
    cond_1 = sma(c, 8) + stddev(c, 8) < sma(c, 2)
    cond_2 = sma(v, 20) / v < 1
    alpha21_ = pd.DataFrame(np.ones_like(c), index=c.index)
    alpha21_[cond_1 | cond_2] = -1
    return alpha21_.squeeze()

def alpha22_(h, v, c):
    df = correlation(h, v, 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha22_= -1 * delta(df, 5) * rank(stddev(c, 20))
    return alpha22_

def alpha23_(h,window=20):
    alpha23_ = [np.nan]*window
    for i in range(len(h))[window:]:
        if sum(h)/len(h) > h.iloc[-1]:
            alpha23_.append(-1*h.diff(2).iloc[-1])
        else:
            alpha23_.append(0)
    return alpha23_

def alpha24_(c):
    cond = delta(sma(c, 100), 100) / delay(c, 100) <= 0.05
    alpha24_ = -1 * delta(c, 3)
    alpha24_[cond] = -1 * (c - ts_min(c, 100))
    return alpha24_

def alpha25_(c, h, vwap, v):
    r = c.pct_change()
    r.iloc[0] = 0
    adv20 = sma(v, 20)
    alpha25_ = rank(((((-1 * r) * adv20) * vwap) * (h - c)))
    return alpha25_

def alpha26_(h, v):
    df = correlation(ts_rank(v, 5), ts_rank(h, 5), 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha26_ = -1 * ts_max(df, 3)
    return alpha26_

def alpha27_(v, vwap):
    alpha27_ = rank((sma(correlation(rank(v), rank(vwap), 6), 2) / 2.0))
    alpha27_[alpha27_ > 0.5] = -1
    alpha27_[alpha27_ <= 0.5]=1
    return alpha27_

def alpha28_(h, c, l, v):
    adv20 = sma(v, 20)
    df = correlation(adv20, l, 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha28_ = scale(((df + ((h + l) / 2)) - c))
    return alpha28_

def alpha29_(c):
    r = c.pct_change()
    r.iloc[0] = 0
    alpha29_ = (ts_min(rank(rank(scale(np.log(ts_sum(rank(rank(-1 * rank(delta((c - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * r), 6), 5))
    return alpha29_

def alpha30_(c, v):
    delta_close = delta(c, 1)
    inner = np.sign(delta_close) + np.sign(delay(delta_close, 1)) + np.sign(delay(delta_close, 2))
    alpha30_ = ((1.0 - rank(inner)) * ts_sum(v, 5)) / ts_sum(v, 20)
    return alpha30_

def alpha31_(c, l, v):
    adv20 = sma(v, 20)
    df = correlation(adv20, l, 12).replace([-np.inf, np.inf], 0).fillna(value=0)         
    p1=rank(rank(rank(decay_linear((-1 * rank(rank(delta(c, 10)))).to_frame(), 10)))) 
    p2=rank((-1 * delta(c, 3)))
    p3=np.sign(scale(df))
    p1["p2"] = p2.fillna(0)
    p1["p3"] = p3
    p1["alpha31_"] = 0
    p1["alpha31_"] = p1["CLOSE"] + p1["p2"] + p1["p3"]
    alpha31_ = p1["alpha31_"].squeeze()
    return alpha31_

def alpha32_(c, vwap):
    alpha32_ = scale(((sma(c, 7) / 7) - c)) + (20 * scale(correlation(vwap, delay(c, 5),230)))
    return alpha32_

def alpha33_(o, c):
    alpha33_ = rank(-1 + (o / c))
    return alpha33_

def alpha34_(c):
    r = c.pct_change()
    r.iloc[0] = 0
    inner = stddev(r, 2) / stddev(r, 5)
    inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
    alpha34_ = rank(2 - rank(inner) - rank(delta(c, 1)))
    return alpha34_

def alpha35_(c, h, l, v):
    r = c.pct_change()
    r.iloc[0] = 0
    alpha35_ = ((ts_rank(v, 32) * (1 - ts_rank(c + h - l, 16))) * (1 - ts_rank(r, 32)))
    return alpha35_

def alpha36_(c, o, v, vwap):
    r = c.pct_change()
    r.iloc[0] = 0
    adv20 = sma(v, 20)
    alpha36_ = (((((2.21 * rank(correlation((c - o), delay(v, 1), 15))) + (0.7 * rank((o - c)))) + 
              (0.73 * rank(ts_rank(delay((-1 * r), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + 
                (0.6 * rank((((sma(c, 200) / 200) - o) * (c- o)))))
    return alpha36_

def alpha37_(o, c):
    alpha37_ = rank(correlation(delay(o - c, 1), c, 200)) + rank(o - c)
    return alpha37_

def alpha38_(o , c):
    inner = o/c
    inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
    alpha38_ = -1 * rank(ts_rank(o, 10)) * rank(inner)
    return alpha38_

def alpha39_(c, v):
    r = c.pct_change()
    r.iloc[0] = 0
    adv20 = sma(v, 20)
    a = (-1 * rank(delta(c, 7))).fillna(0)
    b = (1 - rank(decay_linear((v / adv20).to_frame(), 9))).fillna(0)
    c = (1 + rank(sma(r, 250))).fillna(0)
    
    b["a"] = 0
    b["a"] = a #b.CLOSE
    b["c"] = 0
    b["c"] = c
    b["alpha39_"] = 0
    b["alpha39_"] = (b["a"] * b["CLOSE"])* b["c"]
    alpha39_ = b["alpha39_"].squeeze()
    return alpha39_

def alpha40_(h, v):
    alpha40_ = -1 * rank(stddev(h, 10)) * correlation(h, v, 10)
    return alpha40_

def alpha41_(h, l, vwap):
    alpha41_ = pow((h * l),0.5) - vwap
    return alpha41_

def alpha42_(c, vwap):
    alpha42_ = rank((vwap - c)) / rank((vwap + c))
    return alpha42_

def alpha43_(c,  v):
    adv20 = sma(v, 20)
    alpha43_ = ts_rank(v / adv20, 20) * ts_rank((-1 * delta(c, 7)), 8)
    return alpha43_

def alpha44_(h, v):
    df = correlation(h, rank(v), 5)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha44_= -1 * df
    return alpha44_

def alpha45_(c, v):
    df = correlation(c, v, 2)
    df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha45_ = -1 * (rank(sma(delay(c, 5), 20)) * df *
                 rank(correlation(ts_sum(c, 5), ts_sum(c, 20), 2)))
    return alpha45_

def alpha46_(c):
    inner = ((delay(c, 20) - delay(c, 10)) / 10) - ((delay(c, 10) - c) / 10)
    alpha46_ = (-1 * delta(c))
    alpha46_[inner < 0] = 1
    alpha46_[inner > 0.25] = -1
    return alpha46_

def alpha47_(c, h, v, vwap):
    adv20 = sma(v, 20)
    alpha47_ = ((((rank((1 / c)) * v) / adv20) * 
                 ((h * rank((h - c))) / (sma(h, 5) /5))) - 
                rank((vwap - delay(vwap, 5))))
    return alpha47_

# Alpha#48	 (indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) *delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))

def alpha49_(c):
    inner = (((delay(c, 20) - delay(c, 10)) / 10) - ((delay(c, 10) - c) / 10))
    alpha49_ = (-1 * delta(c))
    alpha49_[inner < -0.1] = 1
    return alpha49_

def alpha50_(v, vwap):
    alpha50_  = (-1 * ts_max(rank(correlation(rank(v), rank(vwap), 5)), 5))
    return alpha50_


def alpha51_(c):
    inner = (((delay(c, 20) - delay(c, 10)) / 10) - ((delay(c, 10) - c) / 10))
    alpha51_ = (-1 * delta(c))
    alpha51_[inner < -0.05] = 1
    return alpha51_

def alpha52_(c, l, v):
    r = c.pct_change()
    r.iloc[0] = 0
    alpha52_ = (((-1 * delta(ts_min(l, 5), 5)) *
             rank(((ts_sum(r, 240) - 
                    ts_sum(r, 20)) / 220))) * ts_rank(v, 5))
    return alpha52_

def alpha53_(c, h, l):
    inner = (c - l).replace(0, 0.0001)
    alpha53_ = -1 * delta((((c- l) - (h - c)) / inner), 9)
    return alpha53_

def alpha54_(c, h, l, o):
    inner = (l - h).replace(0, -0.0001)
    alpha54_ = -1 * (l -c) * (o ** 5) / (inner * (c ** 5))
    return alpha54_

def alpha55_(c, h, l, v):
    divisor = (ts_max(h, 12) - ts_min(l, 12)).replace(0, 0.0001)
    inner = (c - ts_min(l, 12)) / (divisor)
    df = correlation(rank(inner), rank(v), 6)
    alpha55_ = -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)
    return alpha55_

# Alpha#56	 (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
#本Alpha使用了cap|市值，暂未取到该值
#    def alpha056(self):
#        return (0 - (1 * (rank((sma(self.returns, 10) / sma(sma(self.returns, 2), 3))) * rank((self.returns * self.cap)))))

def alpha57_(c, vwap):
    alpha57_ = (0 - (1 * ((c- vwap) / decay_linear(rank(ts_argmax(c, 30)).to_frame(), 2).CLOSE)))
    return alpha57_

# Alpha#58	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322))
# Alpha#59	 (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap *(1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))

def alpha60_(c, h, l, v):
    divisor = (h - l).replace(0, 0.0001)
    inner = ((c - l) - (h- c)) * v / divisor
    alpha60_ = - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(c, 10))))
    return alpha60_

def alpha61_(v, vwap):
    adv180 = sma(v, 180)
    alpha61_ = (rank((vwap - ts_min(vwap, 16))) < rank(correlation(vwap, adv180, 18)))
    return alpha61_

def alpha62_(o, h, l, v, vwap):
    adv20 = sma(v, 20)
    alpha62_ = ((rank(correlation(vwap, sma(adv20, 22), 10)) < rank(((rank(o) +
                                                                           rank(o)) < (rank(((h + l ) / 2)) + 
                                                                                               rank(h))))) * -1)
    return alpha62_

# Alpha#63	 ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180,37.2467), 13.557), 12.2883))) * -1)

def alpha64_(o, h, l, v, vwap):
    adv120 = sma(v, 120)
    alpha64_ = ((rank(correlation(sma(((o * 0.178404) + (l * (1 - 0.178404))), 13),sma(adv120, 13), 17)) < rank(delta(((((h + l) / 2) * 0.178404) + 
                                                                                                                                      (vwap * (1 -0.178404))), 3.69741))) * -1)
    return alpha64_

def alpha65_(o, v, vwap):
    adv60 = sma(v, 60)
    alpha65_ = ((rank(correlation(((o * 0.00817205) + (vwap * (1 - 0.00817205))), sma(adv60,9), 6)) < rank((o - ts_min(o, 14)))) * -1)
    return alpha65_

def alpha66_(o, h, l, vwap):
    alpha66_ = ((rank(decay_linear(delta(vwap, 4).to_frame(), 7).CLOSE) + 
                 ts_rank(decay_linear(((((l* 0.96633) + 
                                         (l * (1 - 0.96633))) - vwap) /(o -((h + l) / 2))).to_frame(), 11).CLOSE, 7)) * -1)
    return alpha66_

# Alpha#67	 ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)

def alpha68_(c, h, l, v):
    adv15 = sma(v, 15)
    alpha68_ = ((ts_rank(correlation(rank(h), rank(adv15), 9), 14) <rank(delta(((c * 0.518371) + 
                                                                                (l * (1 - 0.518371))), 1.06157))) * -1)
    return alpha68_

# Alpha#69	 ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416),9.0615)) * -1)
# Alpha#70	 ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1)

def alpha71_(o, c, l, vwap, v):
    adv180 = sma(v, 180)
    p1=ts_rank(decay_linear(correlation(ts_rank(c, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16)
    p2=ts_rank(decay_linear((rank(((l + o) - (vwap + vwap))).pow(2)).to_frame(), 16).CLOSE, 4)
    df=pd.DataFrame({'p1':p1,'p2':p2})
    df.at[df['p1']>=df['p2'],'max']=df['p1']
    df.at[df['p2']>=df['p1'],'max']=df['p2']
    return df['max'].squeeze()
    #return max(ts_rank(decay_linear(correlation(ts_rank(self.close, 3), ts_rank(adv180,12), 18).to_frame(), 4).CLOSE, 16), ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap +self.vwap))).pow(2)).to_frame(), 16).CLOSE, 4))

def alpha72_(h, l, v, vwap):
    adv40 = sma(v, 40)
    alpha72_= (rank(decay_linear(correlation(((h + l) / 2), adv40, 9).to_frame(), 10).CLOSE) /
               rank(decay_linear(correlation(ts_rank(vwap, 4), ts_rank(v, 19), 7).to_frame(),3).CLOSE))
    return alpha72_


def alpha73_(o, l, vwap):
    p1=rank(decay_linear(delta(vwap, 5).to_frame(), 3).CLOSE)
    p2=ts_rank(decay_linear(((delta(((o * 0.147155) + (l * (1 - 0.147155))), 2) / 
                              ((o *0.147155) + (l * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)
    df=pd.DataFrame({'p1':p1,'p2':p2})
    df.at[df['p1']>=df['p2'],'max']=df['p1']
    df.at[df['p2']>=df['p1'],'max']=df['p2']
    alpha73_ = (-1*df['max']).squeeze()
    return alpha73_
    #return (max(rank(decay_linear(delta(self.vwap, 5).to_frame(), 3).CLOSE),ts_rank(decay_linear(((delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open *0.147155) + (self.low * (1 - 0.147155)))) * -1).to_frame(), 3).CLOSE, 17)) * -1)

def alpha74_(c, h, v, vwap):
    adv30 = sma(v, 30)
    alpha74_ = ((rank(correlation(c, sma(adv30, 37), 15)) <
                 rank(correlation(rank(((h * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(v), 11)))* -1)
    return alpha74_

def alpha75_(l, v, vwap):
    adv50 = sma(v, 50)
    alpha75_ = (rank(correlation(vwap, v, 4)) < rank(correlation(rank(l), rank(adv50),12)))
    return alpha75_

# Alpha#76	 (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1)

def alpha77_(h, l, v, vwap):
    adv40 = sma(v, 40)
    p1=rank(decay_linear(((((h + l) / 2) + h) - (vwap + h)).to_frame(), 20).CLOSE)
    p2=rank(decay_linear(correlation(((h + l) / 2), adv40, 3).to_frame(), 6).CLOSE)
    df=pd.DataFrame({'p1':p1,'p2':p2})
    df.at[df['p1']>=df['p2'],'min']=df['p2']
    df.at[df['p2']>=df['p1'],'min']=df['p1']
    return df['min'].squeeze()
    #return min(rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)).to_frame(), 20).CLOSE),rank(decay_linear(correlation(((self.high + self.low) / 2), adv40, 3).to_frame(), 6).CLOSE))

def alpha78_(v, vwap, l):
    adv40 = sma(v, 40)
    alpha78_ = (rank(correlation(ts_sum(((l * 0.352233) + 
                              (vwap * (1 - 0.352233))), 20),ts_sum(adv40,20), 7)).pow(rank(correlation(rank(vwap), rank(v), 6))))
    return alpha78_

# Alpha#79	 (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))),IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150,9.18637), 14.6644)))
# Alpha#80	 ((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))),IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)

#def alpha81_(v, vwap):
#    adv10 = sma(v, 10)
#    alpha81_ = ((rank(log(product(rank((rank(correlation(vwap, ts_sum(adv10, 50),8)).pow(4))), 15))) < 
#             rank(correlation(rank(vwap), rank(v), 5))) * -1)
#    return alpha81_

# Alpha#82	 (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)

def alpha83_(c, h, l, v, vwap):
    alpha83_ = ((rank(delay(((h - l) / (ts_sum(c, 5) / 5)), 2)) * rank(rank(v))) / 
                (((h - l) / (ts_sum(c, 5) / 5)) / (vwap - c)))
    return alpha83_

def alpha84_(c, vwap):
    alpha84_ = pow(ts_rank((vwap - ts_max(vwap, 15)), 21), delta(c,5))
    return alpha84_

def alpha85_(c, h, l, v):
    adv30 = sma(v, 30)
    alpha85_ = (rank(correlation(((h * 0.876703) + (c * (1 - 0.876703))), adv30,10)).pow(rank(correlation(ts_rank(((h + l) / 2), 4), ts_rank(v, 10),7))))
    return alpha85_

def alpha86_(o, c, v, vwap):
    adv20 = sma(v, 20)
    alpha86_ = ((ts_rank(correlation(c, sma(adv20, 15), 6), 20) < 
                 rank(((o + c) - (vwap + o)))) * -1)
    return alpha86_

# Alpha#87	 (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)

def alpha88_(o, c, h, l, v):
    adv60 = sma(v, 60)
    p1=rank(decay_linear(((rank(o) + rank(l)) - (rank(h) + rank(c))).to_frame(),8).CLOSE)
    p2=ts_rank(decay_linear(correlation(ts_rank(c, 8), ts_rank(adv60,21), 8).to_frame(), 7).CLOSE, 3)
    df=pd.DataFrame({'p1':p1,'p2':p2})
    df.at[df['p1']>=df['p2'],'min']=df['p2']
    df.at[df['p2']>=df['p1'],'min']=df['p1']
    return df['min'].squeeze()
    #return min(rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))).to_frame(),8).CLOSE), ts_rank(decay_linear(correlation(ts_rank(self.close, 8), ts_rank(adv60,20.6966), 8).to_frame(), 7).CLOSE, 3))

# Alpha#89	 (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012))
# Alpha#90	 ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
# Alpha#91	 ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)

def alpha92_(o, h, c, l, v):
    adv30 = sma(v, 30)
    p1=ts_rank(decay_linear(((((h + l) / 2) + c) < (l + o)).to_frame(), 15).CLOSE,19)
    p2=ts_rank(decay_linear(correlation(rank(l), rank(adv30), 8).to_frame(), 7).CLOSE,7)
    df=pd.DataFrame({'p1':p1,'p2':p2})
    df.at[df['p1']>=df['p2'],'min']=df['p2']
    df.at[df['p2']>=df['p1'],'min']=df['p1']
    return df['min'].squeeze()
    #return  min(ts_rank(decay_linear(((((self.high + self.low) / 2) + self.close) < (self.low + self.open)).to_frame(), 15).CLOSE,19), ts_rank(decay_linear(correlation(rank(self.low), rank(adv30), 8).to_frame(), 7).CLOSE,7))

# Alpha#93	 (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664)))

def alpha94_(v, vwap):
    adv60 = sma(v, 60)
    alpha94_ = ((rank((vwap - ts_min(vwap, 12))).pow(ts_rank(correlation(ts_rank(vwap,20), ts_rank(adv60, 4), 18), 3)) * -1))
    return alpha94_

def alpha95_(o, h, l, v):
    adv40 = sma(v, 40)
    alpha95_ = (rank((o - ts_min(o, 12))) < 
                ts_rank((rank(correlation(sma(((h + l)/ 2), 19), sma(adv40, 19), 13)).pow(5)), 12))
    return alpha95_

def alpha96_(c, v, vwap):
    adv60 = sma(v, 60)
    p1=ts_rank(decay_linear(correlation(rank(vwap), rank(v).to_frame(), 4),4).CLOSE, 8)
    p2=ts_rank(decay_linear(ts_argmax(correlation(ts_rank(c, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)
    df=pd.DataFrame({'p1':p1,'p2':p2})
    df.at[df['p1']>=df['p2'],'max']=df['p1']
    df.at[df['p2']>=df['p1'],'max']=df['p2']
    return (-1*df['max']).squeeze()
    #return (max(ts_rank(decay_linear(correlation(rank(self.vwap), rank(self.volume).to_frame(), 4),4).CLOSE, 8), ts_rank(decay_linear(ts_argmax(correlation(ts_rank(self.close, 7),ts_rank(adv60, 4), 4), 13).to_frame(), 14).CLOSE, 13)) * -1)

# Alpha#97	 ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)

def alpha98_(o, v, vwap):
    adv5 = sma(v, 5)
    adv15 = sma(v, 15)
    alpha98_ = (rank(decay_linear(correlation(vwap, sma(adv5, 26), 5).to_frame(), 7).CLOSE) -
                rank(decay_linear(ts_rank(ts_argmin(correlation(rank(o), rank(adv15), 21), 9),7).to_frame(), 8).CLOSE))
    return alpha98_

def alpha99_(h, l, v):
    adv60 = sma(v, 60)
    alpha99_ = ((rank(correlation(ts_sum(((h + l) / 2), 20), ts_sum(adv60, 20), 9)) <
                 rank(correlation(l, v, 6))) * -1)
    return alpha99_

# Alpha#100	 (0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high -close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) -scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))),IndClass.subindustry))) * (volume / adv20))))

def alpha101_(o, h, c, l):
    alpha101_ = (c- o) /((h - l) + 0.001)
    return alpha101_



def MainGetAlphasFunction(o, h, l, c, v, vwap):
    alpha1 = alpha1_(c)
    alpha2 = alpha2_(v,c,o)
    alpha3 = alpha3_(v,o)
    alpha4 = alpha4_(l)
    alpha5 = alpha5_(o, c, vwap)
    alpha6 = alpha6_(o, v)
    alpha7 = alpha7_(c, v)
    alpha8 = alpha8_(o)
    alpha9 = alpha9_(c)
    alpha10 = alpha10_(c)
    alpha11 = alpha11_(vwap, v, c)
    alpha12 = alpha12_(c,v)
    alpha13 = alpha13_(c, v)
    #alpha14 = alpha14_(c,o,v)
    alpha15 = alpha15_(h,v)
    alpha16 = alpha16_(h,v)
    alpha17 = alpha17_(c,v)
    alpha18 = alpha18_(c,o)
    alpha19 = alpha19_(c)
    alpha20 = alpha20_(o,h,c,l)
    alpha21 = alpha21_(c, v)
    alpha22 = alpha22_(h,v, c)
    alpha23 = alpha23_(h, window=20)
    alpha24 = alpha24_(c)
    alpha25 = alpha25_(c, h, vwap, v)
    alpha26 = alpha26_(h, v)
    #alpha27 = alpha27_(v, vwap)
    alpha28 = alpha28_(h, c, l, v)
    alpha29 = alpha29_(c)
    alpha30 = alpha30_(c, v)
    alpha31 = alpha31_(c, l, v)
    alpha32 = alpha32_(c, vwap)
    alpha33 = alpha33_(o, c)
    alpha34 = alpha34_(c)
    alpha35 = alpha35_(c, h, l, v)
    alpha36 = alpha36_(c, o, v, vwap)
    alpha37 = alpha37_(o, c)
    alpha38 = alpha38_(o , c)
    alpha39 = alpha39_(c, v)
    alpha40 = alpha40_(h, v)
    alpha41 = alpha41_(h, l, vwap)
    alpha42 = alpha42_(c, vwap)
    alpha43 = alpha43_(c,  v)
    alpha44 = alpha44_(h, v)
    alpha45 = alpha45_(c, v)
    alpha46 = alpha46_(c)
    alpha47 = alpha47_(c, h, v, vwap)
    
    alpha49 = alpha49_(c)
    alpha50 = alpha50_(v, vwap)
    alpha51 = alpha51_(c)
    alpha52 = alpha52_(c, l, v)
    alpha53 = alpha53_(c, h, l)
    alpha54 = alpha54_(c, h, l, o)
    alpha55 = alpha55_(c, h, l, v)
    
    alpha57 = alpha57_(c, vwap)
    
    alpha60 = alpha60_(c, h, l, v)
    alpha61 = alpha61_(v, vwap)
    alpha62 = alpha62_(o, h, l, v, vwap)
    
    alpha64 = alpha64_(o, h, l, v, vwap)
    alpha65 = alpha65_(o, v, vwap)
    alpha66 = alpha66_(o, h, l, vwap)
    
    #alpha68 = alpha68_(c, h, l, v)
    
    
    alpha71 = alpha71_(o, c, l, vwap, v)
    alpha72 = alpha72_(h, l, v, vwap)
    alpha73 = alpha73_(o, l, vwap)
    
    
    alpha77 = alpha77_(h, l, v, vwap)
    
    
    alpha88 = alpha88_(o, c, h, l, v)
    
    
    alpha92 = alpha92_(o, h, c, l, v)
    
    alpha94 = alpha94_(v, vwap)
    alpha95 = alpha95_(o, h, l, v)
    alpha96 = alpha96_(c, v, vwap)
    
    alpha98 = alpha98_(o, v, vwap)
    alpha99 = alpha99_(h, l, v)
    
    alpha101 = alpha101_(o, h, c, l)

    dfalphas = pd.DataFrame(
        {
            'alpha1':alpha1,'alpha2':alpha2,
            'alpha3':alpha3,'alpha4':alpha4,
            'alpha5':alpha5,'alpha6':alpha6,
            'alpha7':alpha7,'alpha8':alpha8,
            'alpha9':alpha9,
#            'alpha10':alpha10,
            'alpha11':alpha11,'alpha12':alpha12,
            'alpha13':alpha13, 
            #'alpha14':alpha14,    # returns only nan values
            'alpha15':alpha15, 'alpha16':alpha16,
            'alpha17':alpha17,'alpha18':alpha18,
#            'alpha19':alpha19,
            'alpha20':alpha20,
            'alpha21_integer':alpha21,'alpha22':alpha22,
            #'alpha23':alpha23,    # get the same number
            'alpha24':alpha24,
            'alpha25':alpha25, 'alpha26':alpha26,
            'alpha28':alpha28,'alpha29':alpha29,
            'alpha30':alpha30,'alpha31':alpha31,
#            'alpha32':alpha32,
            'alpha33':alpha33,
            'alpha34':alpha34,'alpha35':alpha35,
#            'alpha36':alpha36,
#            'alpha37':alpha37,
            'alpha38':alpha38,
            'alpha39':alpha39,
            'alpha40':alpha40,'alpha41':alpha41,
            'alpha42':alpha42,'alpha43':alpha43,
            'alpha44':alpha44,'alpha45':alpha45,
            'alpha46':alpha46,'alpha47':alpha47,
            'alpha49':alpha49,'alpha50':alpha50,
            'alpha51':alpha51,
#            'alpha52':alpha52,
            'alpha53':alpha53,'alpha54':alpha54,
            'alpha55':alpha55,'alpha57':alpha57,
            'alpha60':alpha60,
            'alpha61_integer':alpha61,
            'alpha62_integer':alpha62,
            'alpha64_integer':alpha64,
            'alpha65_integer':alpha65,
#            'alpha66':alpha66,
            'alpha71':alpha71,
            'alpha72':alpha72,
            'alpha73':alpha73,
            'alpha77':alpha77, 
            'alpha88':alpha88, 
            'alpha92':alpha92,
#            'alpha94':alpha94,
            #'alpha95':alpha95, # get the same number
            'alpha96':alpha96,
            'alpha98':alpha98,
            'alpha99_integer':alpha99,
            'alpha101':alpha101
            }
        )
        
    return dfalphas
    
    
    
# class ExtraFeaturesClass():
    
#     def __init__(self, open_prices, high_prices, low_prices, close_prices, volume, vwap):
#         #self.df = df
        
#         self.open = open_prices #df['open_price']
#         self.high = high_prices #df['high_price']
#         self.low = low_prices # df['low_price']
#         self.close = close_prices # df['close_price']
#         self.volume = volume #df['bar_cum_volume']
#         self.vwap_series = vwap #df["vwap"]
    
#     def get_alphas(self):
#         v = self.volume
#         o = self.open
#         c = self.close
#         l = self.low
#         h = self.high
#         vwap = self.vwap_series
        
#         alpha1 = alpha1_(c)
#         alpha2 = alpha2_(v,c,o)
#         alpha3 = alpha3_(v,o)
#         alpha4 = alpha4_(l)
#         alpha5 = alpha5_(l)
#         alpha6 = alpha6_(o, v)
#         alpha7 = alpha7_(c, v)
#         alpha8 = alpha8_(o)
#         alpha9 = alpha9_(c)
#         alpha10 = alpha10_(c)
#         alpha11 = alpha11_(vwap, v, c)
#         alpha12 = alpha12_(c,v)
#         alpha13 = alpha13_(c, v)
#         alpha14 = alpha14_(c,o,v)
#         alpha15 = alpha15_(h,v)
#         alpha16 = alpha16_(h,v)
#         alpha17 = alpha17_(c,v)
#         alpha18 = alpha18_(c,o)
#         alpha19 = alpha19_(c)
#         alpha20 = alpha20_(o,h,c,l)
#         alpha21 = alpha21_(c, v)
#         alpha22 = alpha22_(h,v, c)
#         alpha23 = alpha23_(h, window=20)
#         alpha24 = alpha24_(c)
#         alpha25 = alpha25_(c, h, vwap, v)
#         alpha26 = alpha26_(h, v)
#         #alpha27 = alpha27_(v, vwap)
#         alpha28 = alpha28_(h, c, l, v)
#         alpha29 = alpha29_(c)
#         alpha30 = alpha30_(c, v)
#         alpha31 = alpha31_(c, l, v)
#         alpha32 = alpha32_(c, vwap)
#         alpha33 = alpha33_(o, c)
#         alpha34 = alpha34_(c)
#         alpha35 = alpha35_(c, h, l, v)
#         alpha36 = alpha36_(c, o, v, vwap)
#         alpha37 = alpha37_(o, c)
#         alpha38 = alpha38_(o , c)
#         alpha39 = alpha39_(c, v)
#         alpha40 = alpha40_(h, v)
#         alpha41 = alpha41_(h, l, vwap)
#         alpha42 = alpha42_(c, vwap)
#         alpha43 = alpha43_(c,  v)
#         alpha44 = alpha44_(h, v)
#         alpha45 = alpha45_(c, v)
#         alpha46 = alpha46_(c)
#         alpha47 = alpha47_(c, h, v, vwap)
        
#         alpha49 = alpha49_(c)
#         alpha50 = alpha50_(v, vwap)
#         alpha51 = alpha51_(c)
#         alpha52 = alpha52_(c, l, v)
#         alpha53 = alpha53_(c, h, l)
#         alpha54 = alpha54_(c, h, l, o)
#         alpha55 = alpha55_(c, h, l, v)
        
#         alpha57 = alpha57_(c, vwap)
        
#         alpha60 = alpha60_(c, h, l, v)
#         alpha61 = alpha61_(v, vwap)
#         alpha62 = alpha62_(o, h, l, v, vwap)
        
#         alpha64 = alpha64_(o, h, l, v, vwap)
#         alpha65 = alpha65_(o, v, vwap)
#         alpha66 = alpha66_(o, h, l, vwap)
        
#         #alpha68 = alpha68_(c, h, l, v)
        
        
#         alpha71 = alpha71_(o, c, l, vwap, v)
#         alpha72 = alpha72_(h, l, v, vwap)
#         alpha73 = alpha73_(o, l, vwap)
        
        
#         alpha77 = alpha77_(h, l, v, vwap)
        
        
#         alpha88 = alpha88_(o, c, h, l, v)
        
        
#         alpha92 = alpha92_(o, h, c, l, v)
        
#         alpha94 = alpha94_(v, vwap)
#         alpha95 = alpha95_(o, h, l, v)
#         alpha96 = alpha96_(c, v, vwap)
        
#         alpha98 = alpha98_(o, v, vwap)
#         alpha99 = alpha99_(h, l, v)
        
#         alpha101 = alpha101_(o, h, c, l)
    
#         self.dfalphas = pd.DataFrame({'alpha1':alpha1,'alpha2':alpha2,'alpha3':alpha3,'alpha4':alpha4,'alpha5':alpha5,
#                                       'alpha6':alpha6,'alpha7':alpha7,'alpha8':alpha8,'alpha9':alpha9,'alpha10':alpha10,
#                                       'alpha11':alpha11,'alpha12':alpha12,'alpha13':alpha13,'alpha14':alpha14,'alpha15':alpha15,
#                                       'alpha16':alpha16,'alpha17':alpha17,'alpha18':alpha18,'alpha19':alpha19,'alpha20':alpha20,
#                                       'alpha21':alpha21,'alpha22':alpha22,'alpha23':alpha23,'alpha24':alpha24,'alpha25':alpha25,
#                                       'alpha26':alpha26,'alpha28':alpha28,'alpha29':alpha29,'alpha30':alpha30,
#                                       'alpha31':alpha31,'alpha32':alpha32,'alpha33':alpha33,'alpha34':alpha34,'alpha35':alpha35,
#                                       'alpha36':alpha36,'alpha37':alpha37,'alpha38':alpha38,'alpha39':alpha39,'alpha40':alpha40,
#                                       'alpha41':alpha41,'alpha42':alpha42,'alpha43':alpha43,'alpha44':alpha44,'alpha45':alpha45,
#                                       'alpha46':alpha46,'alpha47':alpha47,'alpha49':alpha49,'alpha50':alpha50,
#                                       'alpha51':alpha51,'alpha52':alpha52,'alpha53':alpha53,'alpha54':alpha54,'alpha55':alpha55,
#                                       'alpha57':alpha57,'alpha60':alpha60,
#                                       'alpha61':alpha61,'alpha62':alpha62,'alpha64':alpha64,'alpha65':alpha65,
#                                       'alpha66':alpha66,
#                                       'alpha71':alpha71,'alpha72':alpha72,'alpha73':alpha73,
#                                       'alpha77':alpha77,
#                                       'alpha88':alpha88,
#                                       'alpha92':alpha92,'alpha94':alpha94,'alpha95':alpha95,
#                                       'alpha96':alpha96,'alpha98':alpha98,'alpha99':alpha99,
#                                       'alpha101':alpha101})
        
#         #return pd.concat([self.df,self.alphas],axis = 1)
#         return self.dfalphas
    
    # def tsfresh(self):
    #     self.dftsfresh=pd.DataFrame()
    #     for i in self.functions:
    #         self.dftsfresh[str(i.__name__)] = rolling_application(self.c, i)
    #     return self.dftsfresh
    
    # def others(self):
    #     z_score = get_z_score(self.c)
    #     pct = self.c.rank(pct=True)
    #     rolling_zscore = get_rolling_z_score(self.c)
    #     #rolling_pct = get_rolling_pct(self.c)
    #     minmax = min_max(self.c)
    #     self.dfothers = pd.DataFrame({'z_score':z_score,'pct':pct,'rolling_zscore':rolling_zscore,
    #                                  'minmax':minmax})
    #     return self.dfothers 
    
    # def features(self):
    #     alphas = self.alphas()
    #     technicals = self.technicals()
    #     tsfresh = self.tsfresh()
    #     others = self.others()
    #     df = pd.concat([self.df,technicals,tsfresh,others,alphas],axis=1).dropna()
    #     names = []
    #     for i in df.columns:
    #         name = 'feature_'+i
    #         names.append(name)
    #     df.columns = names
    #     return df
