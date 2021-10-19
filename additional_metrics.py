"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
from math import exp

#### Funcion que estima el bid/ask spread con las OHLC Vol Bars
def bidAskOperator(current_high, lagged_high, current_low, lagged_low):
  # det. max value del HighPrice y LowPrice actual y laggeado
  maxHigh, maxLow = max(current_high, lagged_high),max(current_low, lagged_low)
  # gamma como el log del ratio entre maxHigh y maxLow al cuadrado
  gamma = np.log(maxHigh / maxLow) ** 2
  # beta como el log de la sumatoria de los ratios high::low 
  beta = np.log(current_high/current_low)**2 + \
        np.log(lagged_high/lagged_low) ** 2
  # estimación de Alpha = ((3-2√2) x √beta) - √(gamma ÷ (√2-1))
  alpha = 2.414213562373093 * (beta ** 0.5) - (gamma / 0.17157287525381) ** 0.5
  # estimación del exponencial de e
  alphaExp = exp(alpha)
  # estimación del spread final
  spread = 2 * (alphaExp-1)/(alphaExp+1) 
  return spread

#### funcion integradora del bid/ask spread
def bidAskSpread(dfTable):
  """
  Reference in Php: https://bit.ly/3lJHmNO (StackExchange)
  """
  # agregamos columna de los precios High laggeados 1 obs. atrás
  dfTable["lagged_high"] = dfTable.high_price.shift(1)
  # agregamos columna de los precios Low laggeados 1 obs. atrás
  dfTable["lagged_low"] = dfTable.low_price.shift(1)
  # borramos fila conteniendo nan
  dfTable.dropna(inplace=True)
  # reindex de la tabla
  dfTable = dfTable.reset_index(drop=True)
  # det. el bid-ask spread y la agrega como nueva columna
  dfTable["bidask_spread"] = dfTable.apply(
      lambda row: bidAskOperator(
          row.high_price, row.lagged_high, row.low_price, row.lagged_low), 
          axis = 1
        ) 
  return dfTable
