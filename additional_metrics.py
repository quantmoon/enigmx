"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import sys
import numpy as np
import pandas as pd 
from math import exp

##############################################################################
################### CAL. BID/ASK SPREAD (databundle step) ####################
##############################################################################

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

##############################################################################
########## Func. Genérica para la clase de CashReturn Metrics ################
##############################################################################

# funcion base de retornos
def return_func(initial_value, final_value, categoricalPred):
  """
  Func. Base de cálculo de los retornos.

  Estima la variación porcentual entre el precio close y el precio de la barra.

  A esta variación porcentual, la ajusta según la categoría de la predicción.

  Si predCat = 1:
      var % * 1.
  Si predCat = -1:
      var % * -1.
  Si predCat = 0:
      var % * 0

  >>>> Mayor detalle en texto ISO 00010
  """
  # estimacion de la tasa de cambio (var%)
  changeRate = (final_value - initial_value)/initial_value

  # inversion del signo de la var% para la pred. -1 
  if categoricalPred == -1:
    changeRate = -1 * changeRate
  
  # inversion del signo de la var% para la pred. 0 (no existe retorno a computar)
  if categoricalPred == 0:
    changeRate = changeRate * 0
  
  return changeRate

# funcion para estimación del precio inicial de inversión
def investedPrice(close_price, leverage): 
  """
  Estima el precio de inversión (inicial) ajustado al apalancamiento.

  >>>> Mayor detalle en texto ISO 0011
  """  
  # ajuste del valor de apalancamiento 
  if leverage < 0:
    leverage = leverage * -1

  # definición del factor de apalancamiento
  fLev = 1 + leverage

  # estimación de precio ajustado por apalancamiento
  adjustedPrice = close_price / fLev

  return adjustedPrice

# funcion para estimación del precio final de inversión
def finalPrice(investedPrice, returns, leverage):
  """
  Estima el precio de salida de inversión (final) ajustado al retorno.
  
  >>>> Mayor detalle en texto ISO 0011
  """
  # ajuste del valor de apalancamiento  
  if leverage < 0:
    leverage = leverage * -1

  # estimación del precio final ajustado por retornos apalancados 
  finalPrice = investedPrice * (1 + (returns * leverage)) 
  
  return finalPrice

# funcion para estimación del cash "net" que identifica ganancias y/o pérdidas
def netCashEstimation(df, fixed_comission):

  # IMPORTANTE...
  # agregar de entrada total de acciones a comprar como parametro
  # multiplicar con ello el final price y el initial price para cal. del 'net'
  # también add el parametro rep. del costo FIJO (AJUSTE bidask va out of func)

  """
  La presente función estructura la secuencia temporal
  para las salidas y entradas de efectivo durante
  las operaciones por un trial y un modelo en concreto.

  La selección indv. de trial y modelo es exógena a la presente func. 

  >>>> Mayor detalle en texto ISO 0011
  """

  # define una col. con los idx consec. del trial según ocurrencia de close
  df["close_price_index"] = df.index

  # ordenamiento según el eje temporal del barrierTime (fin de evento)
  df = df.sort_values(by="barrierTime")

  # redefine el index original del df org. según el total de eventos
  df.index = range(df.shape[0])

  # define una col. con los idx consec. del trial según ocurrencia de barrier
  df["barrier_price_index"] = df.index

  # reordenamiento según eje temporal del closePrice (inicio de evento)
  df = df.sort_values(by="close_price_index")

  # redefinición del index org. del df org. según total de eventos
  df.index = range(df.shape[0])  


  ### CONDICIONAL TEMPORAL!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  # final price = final price * holds
  # initial price = initial price * holds
  # estimación del net: variación entre precio final y precio inicial
  df["net"] = ((df["final_price"]  - df["invest_price"]) * 10) - fixed_comission 

  return df

def bidAskAdjustment(close_price, barrier_price, bidask_spread, pred_cat):
  """
  Función que determina el precio de entrada y salida de los trades
  ajustados al valor del el bid/ask spread.

  Inputs
  ------
    - close_price: (float) precio de cierre de la barra (entrada teórica).
    - barrier_price: (float) precio de finalización del evento (salida teórica).
    - bidask_spread: (float) valor del bidask en base %/100.
    - pred_cat: (int) categoría de predicción del evento

  Outputs
  ------
    - entryPrice: (float) precio de entrada del trade.
    - exitPrice: (float) precio de salida del trade.

    >>>> Más detalles, revisar ISO 00012
  """
  # bid/ask spread ratio 
  spreadRatio = bidask_spread#/100

  # si la predicción es 1 (LONG)
  if pred_cat == 1:
    # penaliza :: entrada == close_price | compra (al ask) a un precio mayor
    entryPrice = close_price * (1 + spreadRatio)
    # penaliza :: salida == barrier_price | vende (al bid) a un precio menor
    exitPrice = barrier_price * (1 - spreadRatio)

  # si la predicción es -1 (SHORT)
  if pred_cat == -1:
    # penaliza :: entrada == close_price | vende (al bid) a un precio menor
    entryPrice = close_price * (1 - spreadRatio)
    # penaliza :: salida == barrier_price | ccompra (al ask) a un precio mayor
    exitPrice = barrier_price * (1 + spreadRatio)
  
  # si la predicción es 0 (NO OPERATION)
  if pred_cat == 0:
    # no existe penalización (los precios se mantienen)
    entryPrice, exitPrice = close_price, barrier_price

  return entryPrice, exitPrice

# funcion para estimación de la secuencia/flujo de efectivo
def cashFlowSequential(df): 
  """
  Genera la tabla de valores secuenciales de cashFlow.

  Toma como entrada el df actualizado con los valores de índices 
  tanto del closePrice como del barrierPrice y, además, la columna 'net'.

  Define una columna denominada "is_close".

    - if "is_close" == 0: salida de cash.
    - if "is_close" == 1: entrada de cash.

  El valor de entrada/salida de cash está definido por el 'net'.
  
  >>>> Más detalles, revisar ISO 00011
  """

  # STEP 1 | genera df tomando el "final_price" (final) con el barrierTime idx
  final_prices = df.sort_values(by="barrierTime")[
                ["barrierTime", "final_price", "barrier_price_index","net"]
              ]

  # setea a todos los eventos de cierre un valor categórico igual a 0
  final_prices["is_close"] = 0

  # genera una copia para la indexación
  final_prices_ = final_prices.copy()

  # ajuste de finalprices en funcion al indice de ocurrencia de evento (idx)
  final_prices_.barrier_price_index = final_prices_.index

  final_prices_.columns = [
              "close_date", "invest_price", 
              "close_price_index", "net", "is_close"
              ]

  # STEP 2 |  genera df tomando el "invest_price" (initial) con barrierTime idx
  initial_prices = df.sort_values(by="barrierTime")[
                ["close_date", "invest_price", "close_price_index", "net"]
              ]
  
  # setea a todos los eventos de cierre un valor categórico igual a 1
  initial_prices["is_close"] = 1

  # une la copia del df original con la nueva
  total_prices = initial_prices.append(final_prices_, ignore_index=True)

  # ordenamiento final de dataframe
  total_prices = total_prices.sort_values(by="close_date")
  total_prices.index = range(len(total_prices))

  return total_prices

# funcion (NO UTIL POR AHORA) para estimacion de costo de comision % de IBK 
def ibkFixedCostAdivsorAccount(holdings, cost_per_share =0.0005, ibk_val = 0.35):
  """
  Comisiones Tiered de IBK Reatil Advisors/Brokers para Equities

  Nomenclatura: https://www.interactivebrokers.com/en/index.php?f=1590&p=stocks2 

  NOTA: esta función SOLO sirve para la suposición de que se
        utilice la cuenta tipo Advisors (Friends and Family).
        Para una cuenta retail individual, el costo estandar es:
            * 0.0035 por share para ≤ 300,000 Shares
                            ó
            * 0.35 por trade como costo mínimo.
            
            Siendo la comisión como máximo:
                - 1.0% of trade value 

        Para mayor detalle, revisar el mismo link.
  """
  return min((holdings * cost_per_share), ibk_val)

# funcion para construcción de tabla binaria de entradas (1) y salidas (0)
def moneyEVA(binaryDF, initial_cap = 5000, cashF = 1):
  """
  Func. que utiliza la tabla binaria de entradas (0) y salidas (1) de cash
  para obtener la evolución del capital (CAP), el flujo de efectivo (CASH) y 
  el Asset Under Management (AUM).

  Inputs:
  -------
    - binaryDF : pd.DataFrame que contiene la info de las transacciones, 
                 y los valores categóricos representativos de entradas (0)
                 y salidas (1) de efectivo. Las columnas deben ser:
                    *   close_date (datetime de la obs.) 	
                    *   invest_price (float) ---> adj. por leverage y costos
                    *   close_price_index (int único) 	
                    *   net (float) ---> dif. neto de profit  	
                    *   is_close ---> categórico de entrada o salida de cash
                  
                  Nota: el 'binaryDF' es computado para un solo trial y un 
                        solo modelo a la vez.  

    - initial_capital : int/float representativo del capital initial.
    - cashF : int/float ∈ [0;1] representativo de la porción de capital útil 
              a tomarse en cuenta.    

  Output:
  ------
    - assetum: pd.Dataframe con la siguiente info x columna.
                  * close_date (datetime de la obs.)
                  * aum (float) valor $ de las tenencias hasta esa fecha. 
                  * cash (float) dinero efectivo disponible del cap. total
                                 (capital invertible y no invertible)
                  * cap (float) evolución del capital invertible. 
  """  

  # condicional de valor max y min para el cashF
  if cashF > 1 or cashF <= 0:
    sys.exit("Warning! 'cashF' value should belong to the set [0;1] only...")

  # def. de cash como capital inicial entre factor de cash (%)
  cash = [initial_cap/cashF]
  # AUM como lst con 0 inicial (ningún activo previo en hold para el trial)
  aum = [0]
  # capital inicial bruto general 
  cap = [initial_cap]
  # lista vacía de fechas almacenables para la construcción del df. final
  dates = [0]
  # temporal iterativo generla
  temp = 0
  # suma de cash acumulable (temporal)
  sum_cash = 0
  # suma de capital acumulable (temporal)
  sum_cap = 0
  # valor restante del AUM (temporal)
  minus_aum = 0
  # Eventos no tomados en cuenta dado el limitante del capital 
  no_van = []

  # iteración general por el total de eventos 
  for i in range(binaryDF.shape[0]):
    # si el evento representa LA ÚLTIMA salida de cash (1)...
    if i == binaryDF.query("is_close == 1").index[-1]:
      # evalúa si el evento no se encuentra en la lista de eventos que no van
      last_rows=binaryDF.loc[~binaryDF.close_price_index.isin(no_van)].loc[i+1:]
      # utiliza el neto de beneficio/pérdida para inicializar cap, cash y aum
      sum_cap = sum_cap + last_rows.net.sum()
      minus_aum = minus_aum + last_rows.invest_price.sum()
      sum_cash = sum_cash + last_rows.invest_price.sum()
    
    ##########################################################################
    ################### CONDICIONALES DE CASH, CAPITAL Y AUM #################
    ##########################################################################

    #Para todo condicional existen dos casos principales, si el is_close es 0
    #O si el is_close es 1
    #is_close=0: es decir que estamos en la liquidación de un trade, la salida
    #is_close=1: es decir que estamos entrando al trade, apertura de operación

    #Flujo de Efectivo (CASH)
    
    #Si nos encontramos en una liquidación de la operación
    if binaryDF["is_close"].iloc[i] == 0:
      #Y esta operación si fue aperturada (no está en no_van)
      if binaryDF["close_price_index"].iloc[i] not in no_van:
        #suma las liquidaciones para añadirlas en el próximo is_close=1
        sum_cash = sum_cash + binaryDF["invest_price"].iloc[i]
      else:
        #si la operación no fue aperturada, no sumar esa operación
        sum_cash = sum_cash
    #Si, en cambio, se encuentra en la apertura del trade
    elif binaryDF["is_close"].iloc[i] == 1:
      dates.append(binaryDF["close_date"].iloc[i])
      if sum_cash == 0:
        if (cash[temp] - binaryDF["invest_price"].iloc[i]) >= 0:
            cash.append(cash[temp] - binaryDF["invest_price"].iloc[i])
            temp = temp + 1
        else:
          cash.append(cash[temp])
          no_van.append(binaryDF["close_price_index"].iloc[i])
          temp = temp + 1
      if sum_cash !=0 :
        if (cash[temp] + sum_cash - binaryDF["invest_price"].iloc[i]) >= 0:
          cash.append(cash[temp] + sum_cash - binaryDF["invest_price"].iloc[i])
          temp = temp + 1
        else:
          cash.append(cash[temp] + sum_cash)
          no_van.append(binaryDF["close_price_index"].iloc[i])
          temp = temp + 1
        sum_cash = 0

    #Asset Under Management (AUM)
    if binaryDF["is_close"].iloc[i] == 0:
      if binaryDF["close_price_index"].iloc[i] not in no_van:
        minus_aum = minus_aum + binaryDF["invest_price"].iloc[i]
      else:
        minus_aum = minus_aum
    elif binaryDF["is_close"].iloc[i] == 1:
      if minus_aum == 0:
        if binaryDF["close_price_index"].iloc[i] not in no_van:
          aum.append(aum[temp-1] + binaryDF["invest_price"].iloc[i])
        else:
          aum.append(aum[temp-1])
      elif minus_aum != 0:
        if binaryDF["close_price_index"].iloc[i] not in no_van:
          if (aum[temp-1] + binaryDF["invest_price"].iloc[i] - minus_aum) <= 0:
            aum.append(0)
          else:
            aum.append(
                aum[temp-1] + binaryDF["invest_price"].iloc[i] - minus_aum
                )
        else:
          if (aum[temp-1] - minus_aum) <= 0 :
            aum.append(0)
          else:
            aum.append(aum[temp-1] - minus_aum)
        minus_aum = 0

    #Evolución de la Capitalización (CAP)
    if binaryDF["is_close"].iloc[i] == 0:
      if binaryDF["close_price_index"].iloc[i] not in no_van:
        sum_cap += binaryDF["net"].iloc[i]
      else:
        sum_cap = sum_cap
    elif binaryDF["is_close"].iloc[i] == 1:
      if sum_cap != 0:
        cap.append(cap[temp - 1] + sum_cap)
        sum_cap = 0
      elif sum_cap == 0:
        cap.append(cap[temp-1])
    else:
      continue

    # si se topa con la última salida de cash...
    if i == binaryDF.query("is_close == 1").index[-1]:
      # rompe el loop
      break

  # pandas vacío para almacenamiento de info final
  assetum = pd.DataFrame()
  # añadidura de fecha de evento 
  assetum["close_date"] = dates[1:] # 'close_date' como nombre para post. merge
  # añadidura de AUM
  assetum["aum"] = aum[1:]
  # añadidura de CASH
  assetum["cash"] = cash[1:]
  # añadidura de CAP
  assetum["cap"] = cap[1:]

  return assetum

