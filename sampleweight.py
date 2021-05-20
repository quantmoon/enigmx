"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import itertools
import numpy as np
import pandas as pd


def detectingOverlapEvents(barrierTimeArrayDate, closeArrayDate):
  """
  Detecta los overlap en el dataframe principal.
  Devuelve los índices de ocurrencia de los mismos. 

  Sección del libro de AFML: Pág.60.
  """
  # list para guardar los indices con overlap
  overlap_indices = []

  # iteración por indice del total de información
  for idx in range(0, barrierTimeArrayDate.shape[0]):

    # selección de índice con overlap
    # Ocurrencia: desde 'close_date' hasta 'barrier_time'  
    indices_selected = np.where(
        (barrierTimeArrayDate[idx] > closeArrayDate) & 
        (closeArrayDate[idx] < closeArrayDate)
        )[0]
    
    # almacenamiento de indices con overlap
    overlap_indices.append(indices_selected)

  #retornar los indices de overlap
  return overlap_indices

def mpSampleW(df):
  """"
  Return Attribution for Weights Main Function
  ----------------------------------------------

  Calcula el sample-weight por "return attribution"
  Derive sample weight by return attribution  
  """
  # define al precio de cierre como index del dataframe
  df = df.set_index("close_date")
  
  # calcula el log-return de los precios close 
  ret = np.log(df.close_price).diff() # log-returns, so that they are additive
  
  # define un pd.Series vacío con index 'close_date' y valores float64
  wght = pd.Series(index=ret.index, dtype='float64') 

  # iteración para llenar el pd.Series con los sample-weights corrsp.
  for tIn, tOut in df["barrierTime"].iteritems():
    # tIn: fecha de término de la barra
    # tOut: fecha de materialización del evento objetivo (barrierTime)  
    
    # segmentamos los valores de los retornos segun tIn y tOut
    locElements1 = ret[(ret.index >= tIn) & (ret.index <= tOut)]
    
    # segmentamos la existencia de overlaps en del df princ. segun tIn y tOut
    locElements2 = df[(df.index >= tIn) & (df.index <= tOut)].overlap
    
    # asignación del peso correspondiente según la fecha "tIn" 
    wght.loc[tIn] = (locElements1 / locElements2).sum()

  # devolvemos el 'absolute' return con reemplazo de 'inf' por 0
  return wght.abs().replace([np.inf, -np.inf], 0)


def getTimeDecay(tW, clfLastW=1.):
  """
  TimeDecay Function
  -------------------

  Función de Marcos López de Prado para ajustar el weights con un 
  factor de decaimiento temporal (clfLastW, o factor 'c'). 

  
  - c = 1 : significa que no hay decadencia en el tiempo.
  - 0 < c <1 : significa que las ponderaciones decaen linealmente con el tiempo, 
            pero c/ observación aún recibe una ponderación positiva, 
            independientemente de la antigüedad.
  - c = 0 : los pesos convergen linealmente a 0 a medida que envejecen.
  - c <0 : la parte más antigua de las observaciones recibe un peso cero 
            (es decir, se borran de la memoria).  

  Nota: La función está tal cual fue desarrollada por MLDP.
  """

  # apply piecewise-linear decay to observed uniqueness (tW)
  # newest observation gets weight=1, oldest observation gets weight=clfLastW

  # sumatoria de indices
  clfW=tW.sort_index().cumsum()

  # si el valor de 'c' es mayor igual a cero
  if clfLastW>=0:
    
    # calcula la pendiente de decaimiento 
    slope=(1.-clfLastW)/clfW.iloc[-1]
  
  else:

    # calcula la pendiente de decaimiento
    slope=1./((clfLastW+1)*clfW.iloc[-1])
  
  # calcula el fector de decaimiento constante
  const=1.-slope*clfW.iloc[-1]
  
  # calcula los nuevos pesos
  clfW=const+slope*clfW
 
  #redefine en caso el peso sea cero
  clfW[clfW<0]=0
  
  #print(const,slope)
  
  return clfW  


##############################################################################
##############################################################################
####################### MAIN WEIGHTS CLASS GENERATION ######################## 
##############################################################################
##############################################################################

class WeightsGeneration(object):
  """
  Clase principal para generar los pesos para cada sample.

  C/ peso para cada sample está asociado al "uniqueness" de tal evento (label),
  sobre los overlap entre etiquetas y ajustado temporalmente (TimeDecay).
  """

  # ingestamos el único parámetro principal: dataframe de SQL x stock y bartype
  def __init__(self, df):
    # datafrane completo
    self.df = df
  
  # método que permite encontrar los idx con overlap
  def __findingOverlappingEvents__(self):
    
    # definimos arrays temporales: ocurrencia de barrera y fin de barra 
    barrierTimeArrayDate, closeArrayDate = (
        self.df["barrierTime"].values, 
        self.df["close_date"].values
        )
    
    # lista de arrays con overlap-indices con repitencias (legnth > df)
    generalOverlapIndices = detectingOverlapEvents(
        barrierTimeArrayDate, 
        closeArrayDate
    )

    # array 1D con overlap-indices sin repitencias (legnth < df)
    overlapEvents = np.unique(list(itertools.chain(*generalOverlapIndices)))
    
    #devuelve array 1D con indices donde ocurre overlap
    return overlapEvents

  # método que permite la definición de weights (sin TimeDecay)
  def __baseSampleWeightDefinition__(self):

    # extrae los indices donde ocurre overlap sin repitencia
    overlapEvents = self.__findingOverlappingEvents__()
    
    # define una nueva columna con valores: 0 (sin overlap) y 1 (hay overlap)
    self.df["overlap"] = np.where(self.df.index.isin(overlapEvents), 1, 0)

    # calcula el weight individual para cada evento
    self.df["weight"] = mpSampleW(self.df).reset_index(drop=True)

    # divide cada weight por la suma total para completar el cálculo
    self.df["weight"] = self.df["weight"]/self.df["weight"].sum()
    
    # devuelve dataframe original actualizado
    return self.df

  # método principal que permite la definición de weights (SIN/CON TimeDecay)
  def getWeights(self, decay_factor = 0.5):
    """
    Método principal para actualizar el df princ. y obtener los sample weights.
    Añade columnas: 
      - "overlap" (int [0,1], donde 0: no hay overlap, y 1: hay overlap)
      - "weight" (float con los weights por cada sample SIN TimeDecay)
      - "weightTime" (float con los weights por cada sample CON TimeDecay)

    Recordar sobre 'decay_factor' o 'c':

      - c = 1 : no hay decadencia en el tiempo.
      - 0 < c <1 : las ponderaciones decaen linealmente con el tiempo, 
                pero c/ observación aún recibe una ponderación positiva, 
                independientemente de la antigüedad.
      - c = 0 : los pesos convergen linealmente a 0 a medida que envejecen.
      - c <0 : la parte más antigua de las observaciones recibe un peso cero 
                (es decir, se borran de la memoria).  
    """
    
    # calculamos los weight SIN timedecay
    updated_df = self.__baseSampleWeightDefinition__()
    
    # añadimos columna con los weight CON timedecay
    updated_df["weightTime"] = getTimeDecay(
          updated_df.weight, clfLastW=.5
          )
    
    return updated_df