"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, norm

from enigmx.additional_metrics import (
        bidAskAdjustment, 
        return_func, 
        investedPrice, 
        finalPrice, 
        netCashEstimation, 
        cashFlowSequential, 
        moneyEVA
    )

##############################################################################
###################### SUBCLASE #1 CASHR-ETURN METRICS #######################
##############################################################################

class CashReturnMetrics(object):
  
  """
  Class que engloba las metricas de performance de efectivo (cash) y retornos.

  Inputs generales:
    - trials_frame: pd.Dataframe con la informacion resultante de c/trial
                    del combinatorial backtest x modelo.
    - initial_cap: int/float con el valor del capital inicial seleccionado.  
    - capital_factor: int/float entre 0 y 1 con el % del capital disponible
                      para los trades. 
  """

  def __init__(self, 
               trials_frame, 
               initial_cap = 5000, 
               capital_factor = 1,
               fixed_comission = 0.70):

    # definición de valores básicos para la clase
    self.trials_frame = trials_frame
    self.initial_cap = initial_cap 
    self.capital_factor = capital_factor
    self.fixed_comission = fixed_comission 


  # actualizacion de precios ajustados x bid/ask spread según cat. de entrada
  def __entryExitPrices__(self):
      
    # pd.Series con tupla de entry y exit price
    seriesTemp = self.trials_frame.apply(
        lambda row: bidAskAdjustment(
            row.close_price, row.barrierPrice, row.bidask_spread, row.predCat
            ), axis=1
          ) 
    # creación de cols entryPrice y exitPrice en df org.
    self.trials_frame[['entryPrice', 'exitPrice']] = pd.DataFrame(
          seriesTemp.tolist(), 
          index=seriesTemp.index
        )

  # calculo del retorno por evento  
  def __returnsUpdating__(self):
    self.trials_frame["returns"] = self.trials_frame.apply(
        lambda row: return_func(row.entryPrice, row.exitPrice, row.predCat),
        axis=1
      )
  
  # actualizacion de los precios (inicial y final) segun apalancamiento
  def __priceUpdating__(self):
    # actualizacion de initial price
    self.trials_frame["invest_price"] = self.trials_frame.apply(
        lambda row: investedPrice(row.entryPrice, row.leverage),
        axis =1
      )
    # actualizacion de final price
    self.trials_frame["final_price"] = self.trials_frame.apply(
        lambda row: finalPrice(row.invest_price, row.returns, row.leverage),
        axis=1
      ) 
  
  # estima cash neto como diferencial entre precio ajustado inicial y final
  def __netCashEstimation__(self):
    return netCashEstimation(self.trials_frame, self.fixed_comission)

  # generacion de la tabla binaria secuencial de cash (binaryCashFlowTable)
  def __binaryCashFlowTable__(self, dfNetCash):
    return cashFlowSequential(dfNetCash)

  # generacion de la tabla con la evolucion del CASH, AUM y CAP 
  def __moneyAndCapitalAsessment__(self, cashFlowTable):
    return moneyEVA(
        binaryDF = cashFlowTable, 
        initial_cap = self.initial_cap, 
        cashF = self.capital_factor
        )

  def get_info(self):
    # actualizacion de precios segun bid/ask spread
    self.__entryExitPrices__() 
    # actualizacion de computo de retornos
    self.__returnsUpdating__()
    # actualizacion de precios iniciales y finles x apalancamiento
    self.__priceUpdating__()
    # calculo de valor netcash
    dfNetCash = self.__netCashEstimation__()
    # generacion de tabla binarizada de estructura de cash flow 
    cashFlowTable = self.__binaryCashFlowTable__(dfNetCash)
    # tabla de evolución del cashflow, aum y capitalizacion
    assetumTable = self.__moneyAndCapitalAsessment__(cashFlowTable)
    # agregamos columna con el costo fijo
    assetumTable["fixedCost"] = self.fixed_comission
    return assetumTable  


##############################################################################
###################### SUBCLASE #2 COMPLEMENTARY METRICS #####################
##############################################################################


class ComplementaryMetrics(object):
  """
  Clase de Métricas complementarias
  --------------------------------

  Input:
    - pd.Dataframe con las columnas de cap, cash y AUM *.

    (*) pd.Dataframe para solo un trial y un modelo.

  Output:
    - lst() con la información de las métricas complementarias.

  Listado de Métricas x Método:

    - Profit and Loss:       __Pnl__() ... 
    - Retornos anualizados:  __Aror__()...
    - Ratio Hit:             __HitRatio__()...
    - Retorno medio x hit:   __AvgReturnsHits__()...
    - Herfindahl-Hirschman:  __HHI__(signo)...
            * signo :: 'p' >> concentración de los retornos positivos
            * signo :: 'n' >> concentración de los retornos negativos
    - Ratio Sharpe:          __sharpreRatio__()...
    - Probabilistic Sharpe:  __probSharpeRatio__(benchmark)...:
            * benchmark: valor de Sharpe Ratio estimado
    - Return over costs:     __returnOverCosts__()
    - Drawndown & TimeUnd:   __ddtu__(dollar)...:
            * dollar : boleano para estimar el DD y TU en formato $$$.
    - Retorno medio x rotación de capital : __dollarPerformanceTurnover__()

  Método principal de llamado:
    - compute(psp_benchmark)...

  """

  # clase de inicialización
  def __init__(self, df):
    # df segmentado por trial y modelo inc. métricas de cash computadas 
    self.df = df
  
  # cómputo de profit and loss = cap final - cap inicial
  def __Pnl__(self):
    return self.df.cap.iloc[-1] - self.df.cap.iloc[0] 

  # cómputo de Annualized Rate of Returns (AROR)
  def __Aror__(self):
    # length en días del trial
    days = (
        self.df.close_date.iloc[-1] - self.df.close_date.iloc[0]
        ).days
    # tasa anualizada de los retornos
    return (((self.__Pnl__() / self.df.cap.iloc[0]) + 1)**(360/days) ) - 1 

  # hit ratio :: total de bets positivos (otrogaron ganancias)
  def __HitRatio__(self):
    # levantamos el df segmentado de returns > 0 objeto a la clase general
    self.positiveReturnsDf = self.df.query("returns > 0")
    # estimación del hit ratio como conteo total de positiveness 
    return self.positiveReturnsDf.shape[0]

  # retorno promedio por hit
  def __AvgReturnsHits__(self):
    return self.positiveReturnsDf.returns.sum() / self.__HitRatio__()
  
  # Herfindahl-Hirschman :: medida de concentración de los retornos
  def __HHI__(self, sign):
    # si es un HHI para retornos positivos
    if sign == 'p':
      ret_info = self.positiveReturnsDf.returns
    # si es un HHI para retornos negativos
    if sign == 'n':
      ret_info = self.df.query("returns <= 0").returns
    # si no se cuenta con suficiente data
    if ret_info.shape[0]<=2:return np.nan
    # estimacion de weights
    wght = ret_info / ret_info.sum()
    # estimacion del HHI
    hhi = (wght**2).sum()
    return (hhi-ret_info.shape[0]**-1)/(1.-ret_info.shape[0]**-1)
  
  # sharpe ratio
  def __sharpreRatio__(self):
    # mean of returns (mu value)
    mu = self.df.returns.mean()
    # std of returns (std val)
    std = self.df.returns.std()
    # compute and return sharpe ratio
    return mu/std

  # probabilistic sharpe ratio
  def __probSharpeRatio__(self, benchmark):
    # estimación del sharpe ratio
    sharpeRatio = self.__sharpreRatio__()
    # total events
    t = self.df.returns.shape[0]
    # skew (asimetría) de los retornos 
    skewReturns = skew(self.df.returns)
    # kurtosis (shape dist) de los retornos
    kurtosisReturns = kurtosis(self.df.returns)
    # prob. Sharpe Ratio
    prob_sharpe_ratio = norm.cdf(
        (sharpeRatio - benchmark) * ( (t-1) **.5 ) / 
        ( ( 1 - skewReturns * sharpeRatio + (
            (kurtosisReturns-1)/4
            ) * (sharpeRatio ** 2)) **.5 )
        )
    return prob_sharpe_ratio

  # returns over execution costs
  def __returnOverCosts__(self):
    # dollar performance y levantamiento como param general
    self.dollar_performance = (self.df.cap.iloc[-1] / self.df.cap.iloc[0]) - 1 
    # media de los costos de ejecución
    execution_costs = self.df.fixedCost.mean()
    # retornamos retorno sobre costos
    return self.dollar_performance / execution_costs

  # Drawndown & TimeUnderwater
  def __ddtu__(self, dollar = False):
    # se añade columna con máximos en el dataframe general, a medida que aparece
    # un nuevo máximo, el .expanding() lo modifica en la serie
    self.df['maximos'] = self.df.cap.expanding().max()

    # se hace un groupby por máximos, pero con los valores mínimos de cada máximo
    minimos = self.df.groupby('maximos').min().reset_index()[["maximos", "cap"]]
    minimos.columns=['maximos','min']
    minimos.index=self.df['maximos'].drop_duplicates(keep='first').index 

    # muestra los índices en minimos donde hubo drawdown
    # un máximo seguido de un mínimo
    minimos=minimos[minimos['maximos']>minimos['min']]

    #Si está en dólares, el drawdown es la resta
    if dollar:dd=minimos['maximos']-minimos['min']
    #Si no es porcentaje
    else:dd=(minimos['maximos']-minimos['min'])/minimos['maximos']
    
    #Obtienes los índices y el tiempo entre índices del drawdown / No corre
    tuw=(minimos.index[1:]-minimos.index[:-1])
    #tuw=(minimos.index[1:]-minimos.index[:-1]).astype(int).values
    #Conviertes a serie los tiempos tuw
    tuw=pd.Series(tuw,index=minimos.index[:-1])
    return dd.max(), tuw.max()/60
  
  # dollar performance per turnover 
  def __dollarPerformanceTurnover__(self):
    # retorno promedio por cada rotación
    self.turnover = len(self.df.query("aum == 0"))-1
    return self.dollar_performance/self.turnover

  # callable method || input extra: benchmark para el probabilistic sharpe ratio
  def compute(self, psp_benchmark):
    # IMPORTANT! :::: psp_benchmark ::: float between 0 - 1 (if not, psrp  = 0)
    
    #######################################################
    ##### Estimación de métricas complementarias ##########
    #######################################################
    # PnL
    pnl = self.__Pnl__()
    # Annualized Rate of Returns (AROR)
    aror = self.__Aror__()
    # Hit Ratio
    hitRatio = self.__HitRatio__()
    # Avg. Returns of Hits
    avghits = self.__AvgReturnsHits__()
    # Herfindahl-Hirschman :: concentración de retornos positivos
    hh_positive = self.__HHI__('p')
    # Herfindahl-Hirschman :: concentración de retornos negativos
    hh_negative = self.__HHI__('n')
    # Sharpe Ratio
    sharpe_ratio = self.__sharpreRatio__()
    # probabilistic sharpe ratio
    probabilistic_sharpe = self.__probSharpeRatio__(psp_benchmark)
    # retornos sobre costos
    return_over_costs = self.__returnOverCosts__()
    # drawndown and time underwater...
    drawndown = self.__ddtu__()[0]
    time_underwater = self.__ddtu__()[1]
    # dolllar performance per turnover...
    dollar_performance_per_turnover = self.__dollarPerformanceTurnover__()

    # lista de resultados finales
    resultsList = [
          pnl, 
          aror, 
          hitRatio, 
          avghits, 
          hh_positive,
          hh_negative,
          sharpe_ratio,
          probabilistic_sharpe,
          return_over_costs,
          drawndown,
          time_underwater,
          dollar_performance_per_turnover
        ]
    return resultsList

##############################################################################
############## CLASE PRINCIPAL | CÓMPUTO DE MÉTRICAS COMPLETAS ###############
##############################################################################

class EnigmxMetrics(object):
    
    """
    Clase que integra las subclases:
        1. Subclase de Cálculo de métricas de cash (subclase #1)
        2. Subclase de Cálculo de métricas financieras y ML (subclase #2)
    
    La presente clase funciona solo con el insumo de 'df_trials' 
    para un trial y un modelo en particular.  
    
    Debe ser ingestada dentro de una función paralelizadora (Ray).

    Método único:
        - generate():
                * Permite el cómputo de los VALORES de métricas.
                  Estos son computados en una lista que posee 
                  el siguiente ordenamiento de valores:
                      ** pnl :: en $$ 
                      ** aror :: % retorno anualizado
                      ** hitRatio :: total de bets positivos (profits)
                      ** avghits  :: retorno promedio por hit %
                      ** hh_positive :: concentración de los retornos + en %
                      ** hh_negative :: concentración de los retornos - en %
                      ** sharpe_ratio :: ratio sharpe
                      ** probabilistic_sharpe :: ratio sharpe probabilístico
                      ** return_over_costs :: retorno sobre costos en %
                      ** drawndown :: máximo drawdown en %
                      ** time_underwater :: tiempo en negativo (minutos)
                      ** dollar_performance_per_turnover :: performance x c/ rotación de cap. en %
                    
    """
    
    def __init__(self, 
                 df_trials, 
                 initial_cap, 
                 capital_factor, 
                 fixed_comission = 0.70, 
                 psp_benchmark = 0.15):
        
        # definición de parámetros generales
        
        # df con info del trial N# | DEBE TENER UN SOLO TRIAL DE UN SOLO MODELO
        self.df_trials = df_trials 
        # int de capital inicial
        self.initial_cap = initial_cap
        # factor de capital: de 0 a 1 en % sobre capital útil
        self.capital_factor = capital_factor
        # monto de comision fija x trade
        self.fixed_comission = fixed_comission
        # benchmark de 0 a 1 para el probabilistic sharpe ratio
        self.psp_benchmark = psp_benchmark 

        print("Computando las métricas de un trial",flush = True)
        
    def generate(self):
        
        # computo de la primera subclase #1 | cómputo de métricas de cash
        cashDf = CashReturnMetrics(
            trials_frame = self.df_trials,
            initial_cap = self.initial_cap,
            capital_factor = self.capital_factor,
            fixed_comission = self.fixed_comission
            ).get_info()
        
        # actualizacion del df de trials
        df_trials_updated = pd.merge(
            self.df_trials, cashDf, "left", on = "close_date"
            )
        
        # computo de la segunda subclase #2 | cómputo de métricas financieras
        metricsDf = ComplementaryMetrics(df_trials_updated).compute(
            psp_benchmark = self.fixed_comission
            )
        
        return metricsDf

@ray.remote
def metricsParalelization(
        dataframe_segmented, 
        initial_cap, 
        capital_factor, 
        fixed_comission, 
        psp_benchmark
        ):
    
    instance = EnigmxMetrics(
        dataframe_segmented,
        initial_cap, 
        capital_factor, 
        fixed_comission, 
        psp_benchmark
        )
    
    return instance.generate()
