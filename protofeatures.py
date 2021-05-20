"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np

def fill_zeros_with_last(arr):
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

def tickRuleVector(vector):
    # encuentra las diferencias del vector 1d
    vectorResult = np.diff(vector)
    
    # primer elemento (idx 0) cat. 1
    vectorResult = np.insert(vectorResult, 0, 1)
    #vectorResult [0] = 1
    
    # filled vector (ffill)
    vectorResult = fill_zeros_with_last(vectorResult)
    
    # reemplaza los valores negativos o ig. a 0 por -1
    vectorResult[vectorResult < 0] = -1 
    
    # reemplaza los valores positivos por 1
    vectorResult[vectorResult > 0] = 1
    
    return vectorResult
class protoFeatures:
    """
    Elementos computables para features (columnas)

    1. BuyInit Total Count
    2. SellInit Total Count
    3. Sign Volume: sum((BuyInit * TickVol - SellInit * TickVol) x Tick)) x barra
    4. Acumulative Dolar Value | Inside Bar: sum(tick price * TickVol) x barra
    5. Sign Hasbrouck Volume: 
       sum((BuyInit * sqrt(tick price * TickVol) - SellInit * sqrt(tick price * TickVol)) x barra
    6. Signed Volume: Tick Rule Sign * Diferencial del volumen.
       sum(Vols positivos) acumulada x barra & sum(Vols negativos) acumulada x barra

    BuyInit & SellInit: es un valor 1 o -1 en el punto 't'.
    """
    def __init__(self, price_vector, volume_vector, tick_rule_vector): #groupTickRule 1D
        
        # define el vector de precios
        self.price_vector = price_vector
        
        # define el vector de volumen
        self.volume_vector = volume_vector
        
        # define el vector del tick rule ya computado previamente
        self.tick_rule_vector = tick_rule_vector
        
        # definimos mensaje de error de shape
        errorMssg = "Error! Differences among 'volume', 'price' & 'tickRule' vector shape."
        
        assert self.volume_vector.shape[0] == self.price_vector.shape[0], errorMssg
        assert self.price_vector.shape[0] == self.tick_rule_vector.shape[0], errorMssg
        
    def __mainSupplies__(self):
        
        # calcula diff categorizada de precios del vector de precios | Tick Rule
        self.priceDiff = self.tick_rule_vector[1:] #findingDifferences(self.price_vector)
        
        # define la condicion del buyInit * sellInit
        conditionBuyInit, conditionSellInit = self.priceDiff == 1, self.priceDiff == -1
        
        # indices donde hay aggressorSide alcista
        self.indiciesBuyInit = np.where(conditionBuyInit)[0]
        
        # indices donde hay aggressorSide bajista
        self.indiciesSellInit = np.where(conditionSellInit)[0]
        
    def __findBuySellAggressorSide__(self):
        """
        Conteno de cuantos buyInit existen (cat.: 1)
        
        Conteo de cuantos sellInit existen (cat.: -1)
        
        Retorna:
            - self.buy_aggressor_count
            - self.sell_aggressor_count
        """
        # aggresor side alcista como un count de dif. positivas entre ticks
        self.buy_aggressor_count = np.sum(self.priceDiff > 0, axis=0)
        
        # aggresor side bajista como un count de dif. negativas entre ticks (inc. 0)
        self.sell_aggresor_count = self.priceDiff.shape[0] - self.buy_aggressor_count
        
    def __signVolumeSum__(self): 
        
        # calculamos el signo del volumen para la barra segun sus ticks
        self.signVolumeSum = \
        np.sum(self.volume_vector[1:][self.indiciesBuyInit]) \
        - np.sum(self.volume_vector[1:][self.indiciesSellInit])
        
    def __accumulativeTypeVolume__(self):
        
        # calculamos el tipo de volumen buySide y sellSide 
        volume_buyInit = self.volume_vector[1:][self.indiciesBuyInit]
        volume_sellInit = self.volume_vector[1:][self.indiciesSellInit]
        
        # calculamos el acumulado segun tipo de volumen buySide y sellSide
        self.volumeBuyInitTotal = np.sum(volume_buyInit)
        self.volumeSellInitTotal = np.sum(volume_sellInit)
        
    def __accumulativeDollarValue__(self):
        
        # calculamos el Dollar Value acumulado de la barra segun sus ticks
        self.accumulativeDollarValue = np.sum(self.price_vector * self.volume_vector)
        
    def __signHasbrouckVolume__(self):
        """
        Hasbrouck Volume Sign.
        
        Formulacion:
        
        sum((BuyInit:sqrt(tick price * TickVol) - SellInit:sqrt(tick price * TickVol))
        
        Retorna:
            - self.hasbrouckSignVol: float positivo o negativo
            
            Si es negativo, mas volume bajistas; si es positivo, mas volume alcista.
        """
        
        # calculamos los precios asociados al aggressor side alcista y bajista
        prices_buyInit = self.price_vector[1:][self.indiciesBuyInit]
        prices_sellInit = self.price_vector[1:][self.indiciesSellInit]
        
        # calculamos los volumenes asociados al aggresor side alcista y bajista
        volume_buyInit = self.volume_vector[1:][self.indiciesBuyInit]
        volume_sellInit = self.volume_vector[1:][self.indiciesSellInit]
        
        # raiz de los buyInitSide price & vol, y del sellInitSide price * vol
        sqrtBuyPriceVol = np.sqrt(prices_buyInit * volume_buyInit)
        sqrtSellPriceVol = np.sqrt(prices_sellInit * volume_sellInit)
        
        # signo del hasbbrouckVol con la resta de la sumatoria de las raices
        self.hasbrouckSignVol = np.sum(sqrtBuyPriceVol) - np.sum(sqrtSellPriceVol)
        
    def get_proto_features(self):
        
        self.__mainSupplies__()
        self.__findBuySellAggressorSide__()
        self.__signVolumeSum__()
        self.__accumulativeTypeVolume__()
        self.__accumulativeDollarValue__()
        self.__signHasbrouckVolume__()
        
        final_elements= (
            # buyInit Total Count
            float(self.buy_aggressor_count),
            # sellInit Total Count
            float(self.sell_aggresor_count),
            # sign Volume Sum by Side
            float(self.signVolumeSum),
            # accumulative volume by BuySide
            float(self.volumeBuyInitTotal),
            # accumulative volume by SellSide
            float(self.volumeSellInitTotal),
            # accumulative dollar value 
            float(self.accumulativeDollarValue),
            # hasbrouck sign volume
            float(self.hasbrouckSignVol)
        )
        return final_elements