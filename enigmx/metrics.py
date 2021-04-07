"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import kurtosis, skew, norm

# diccionario con elementos base para computación de las métricas de backtest
baseCriticsDicforBacktestMetrics = {
    'Pnl': [0.1], 
    'Annualized Rate of Returns': [0.45], 
    'Hit Ratio': [0.33], 
    'Average Return of Hits': [0.01],
    'Max Drawdown': [0.01], 
    'Max Time Under Water': [0.1], 
    'HHI positive': [-0.1], 
    'HHI negative': [-0.1], 
    'Sharpe Ratio': [0.88], 
    'Probabilistic Sharpe Ratio': [0.9], 
    'Dollar Performance': [0.1], 
    'Return Over Execution Costs': [0.5]
    }

class Metrics():
    
    """
        Construye un DataFrame que inicializa columnas aleatorias para el costo y deriva el precio de estrategia por la
        triple barrera y los retornos netos de costos del DataFrame que recibe, luego cálcula las estadísticas de
        backtesting en cada uno de sus métodos y en el método final calcula el desempeño de estas en base a los criterios
        esperados.
        
        Inputs Obligatorios:
        
        -df: DataFrame del que se derivan columnas de costos aleatorias, precio y retornos
        
        Inputs Accesitarios:
        -crits: Diccionario con criterios discrecionales sobre el cual analizar desempeño
        -benchm: Benchmark de Sharpe Ratio utilizado en el método .psr(self, benchm), incluye el sharpe ratio objetivo
        
        Diccionario de datos genéricos por método (crits)
        
        Crits para uso en .performance(crits):
        
        crits = {'Pnl': [0.1], 
                'Annualized Rate of Returns': [0.45], 
                'Hit Ratio': [0.33], 
                'Average Return of Hits': [0.01],
                'Max Drawdown': [0.01], 
                'Max Time Under Water': [0.1], 
                'HHI positive': [-0.1], 
                'HHI negative': [-0.1], 
                'Sharpe Ratio': [0.88], 
                'Probabilistic Sharpe Ratio': [0.9], 
                'Dollar Performance': [0.1], 
                'Return Over Execution Costs': [0.5]}
        
        Output: 
        Clase: DataFrame con columnas de costos, precio y retornos
        Métodos: Resultado de cada estadística de backtesting por método
        Performance: Resultado de estadísticas totales y score heurístico de performance
        
    """
    def __init__(self,df):
        self.df2 = df
        y_price = []
        
        """
                                       Y_PRICE
            Se obtienen las columnas de precios en base a la barrera
            Si la barrera tiene de etiqueta 0, se asigna el precio close
            Si la barrera tiene de etiqueta 1, se asigna el tope arriba de la barrera
            Si la barrera tiene de etiqueta -1, se asigna el tope abajo de la barrera
        """
        for i in range(len(self.df2)):
            if self.df2['barrierLabel'].iloc[i] == 0:
                y_price.append(self.df2['close_price'].iloc[i])
            elif self.df2['barrierLabel'].iloc[i] == 1:
                y_price.append(self.df2['upper_barrier'].iloc[i])
            else:
                y_price.append(self.df2['lower_barrier'].iloc[i])
                
             
        self.df = self.df2[['close_price','close_date','barrierPrice','barrierLabel']] #Se seleccionan solo unas columnas

        #Se completa barrier price con precio close cuando el barrier price es 0
        self.df['barrierPrice'] = self.df.apply(lambda x: x.barrierPrice if x.barrierPrice!=0 else x.close_price, axis=1)
        
        self.df["costs"] = np.random.random(size=len(self.df2))*y_price*0.01
        
        """
            Se obtienen las columnas de retornos en base a la barrera
            Si la barrera tiene de etiqueta 0, se asigna el retorno 0
            Si la barrera tiene de etiqueta 1, se asigna el retorno positivo menos costos
            Si la barrera tiene de etiqueta -1, se asigna el retorno negativo menos costos
        """
        self.returns = []
        for i in range(len(self.df2)):
            if self.df2['barrierLabel'].iloc[i] == 0:
                self.returns.append(0)
            elif self.df2['barrierLabel'].iloc[i] == 1:
                self.returns.append((self.df['barrierPrice'].iloc[i]-self.df['close_price'].iloc[i]-self.df["costs"].iloc[i])/self.df['close_price'].iloc[i])
            else:
                self.returns.append((self.df['close_price'].iloc[i]-self.df['barrierPrice'].iloc[i]-self.df["costs"].iloc[i])/self.df['close_price'].iloc[i])
         
        """
            Pnl: El total de dólares obtenidos en el período, close * retornos (retornos ya inlcuye costos)
        """

        self.pnl = self.df['close_price']*self.returns
        self.df['returns'] = self.returns
        self.df['pnl'] = self.pnl
        self.initial_cap = 5000
        
        #El realized cap va sumando retornos al capital inicial (sin costos)
        self.df['realized_cap'] = self.initial_cap
        for i in range(len(self.df['realized_cap'])):
            if i == 0:
                self.df['realized_cap'].iloc[i] = self.initial_cap + self.df['pnl'].iloc[i]
            else:
                self.df['realized_cap'].iloc[i] = self.df['realized_cap'].iloc[i-1]+self.df['pnl'].iloc[i]

        self.ret = pd.Series(self.returns)
        self.ret.index = self.df['close_date']
        
        
        self.cap = self.df['realized_cap']
        self.cap.index = pd.DatetimeIndex(self.df.close_date)
        self.costs = self.df.costs
        
        
    """
        El Pnl total resta las columnas del realized cap (ya con costos) para el total del período
    """
    def get_pnl(self):
        return self.cap[-1]-self.cap[0]

    """
        Annualized rate of returns, el retorno neto de costos anualizado
    """
    def aror(self):#annualized rate of returns
        date_format = "%Y-%m-%d" #"%m/%d/%Y"
        a = datetime.strptime(self.df['close_date'].max().split(" ")[0], date_format)
        b = datetime.strptime(self.df['close_date'].min().split(" ")[0], date_format)
        days = (a-b).days
        return ((self.cap[-1]-self.cap[0])/self.cap[0] +1)**(360/days) - 1 

    """
        Número de bets positivos
    """
    def hitratio(self):
        hit = 0
        for i in self.ret: 
            if i > 0: hit+=1
        return len(self.ret[self.ret>0])

    
    """
        Retorno promedio por hit
    """
    def avgrethits(self): #Average return from hits
        posret = self.ret[self.ret>0]
        arfh = sum(posret) / len(posret)
        return arfh

    """
        Drawdown y TimeUnderWater, computa los drawdown dentro del dataframe y el tiempo que pasa hasta recuperar el 
        máximo previo
    """
    def computeDD_TuW(self,dollars=False):
    # compute series of drawdowns and the time under water associated with them
        #columna con serie de capital normal:
        df0=self.cap.to_frame('pnl')
        
        #columna con máximos:
        df0['hwm']=self.cap.expanding().max()
        
        
        #agrupamos por máximos mostrando el valor mínimo en ese máximo:
        df1=df0.groupby('hwm').min().reset_index()
        
        #renombras columnas:
        df1.columns=['hwm','min']
        
        #Asignas los indices donde empieza el primer máximo (columna hwm)
        df1.index=df0['hwm'].drop_duplicates(keep='first').index # time of hwm
        
        #muestra los índices en df1 donde hubo drawdown
        df1=df1[df1['hwm']>df1['min']] # hwm followed by a drawdown
        
        #Si está en dólares, el drawdown es la resta 
        if dollars:dd=df1['hwm']-df1['min']
        
        #Si no es porcentaje
        else:dd=1-df1['min']/df1['hwm']
        
        #Obtienes los índices y el tiempo entre índices del drawdown / No corre
        tuw=(df1.index[1:]-df1.index[:-1])/np.timedelta64(1,'Y')
        #tuw=(df1.index[1:]-df1.index[:-1]).astype(int).values
        
        #Conviertes a serie los tiempos tuw
        tuw=pd.Series(tuw,index=df1.index[:-1])
        return dd,tuw
    
    
    """
        Herfindahl-Hirschman, medida de concentración, mide que tan concentrados están los retornos
        Si el índice se acerca a 1, los retornos se concentran mucho en un solo bet de suerte
        Si el índice se acerca a 0, hay menor concentración
    """
    def getHHI(self,sign):
        if sign == 'p':
            ret = self.ret[self.ret>=0]
        if sign == 'n':
            ret = self.ret[self.ret<=0]
        if ret.shape[0]<=2:return np.nan
        wght=ret/ret.sum()
        hhi=(wght**2).sum()
        hhi=(hhi-ret.shape[0]**-1)/(1.-ret.shape[0]**-1)
        return hhi
    
    """
        Sharpe Ratio
    """
    def sr(self):
        mu = self.ret.mean()
        sd = self.ret.std()
        return mu/sd
    
    """
        Probabilistic Sharpe Ratio
    """
    def psr(self,benchm):
        sratio = self.sr()
        t = len(self.ret)
        y3 = skew(self.ret)
        y4 = kurtosis(self.ret)
        psratio = norm.cdf((sratio-benchm)*((t-1)**.5)/((1-y3*sratio+((y4-1)/4)*(sratio**2))**.5))
        return psratio
    
    """
        Mínimo tamaño de la muestra para obtener un probabilistic sharpe ratio válido
    """
    def min_n(self):
        sratio = self.sr()
        y3 = skew(self.ret)
        y4 = kurtosis(self.ret)
        MinTRL_n = 1+(1-y3*sratio+(y4-1)/4.*sratio**2)*(norm.ppf(0.05)/(sratio-self.bechmMin))**2
        return MinTRL_n, len(self.ret)
        
    """
        Dollar Performance per Turnover, retorno promedio por cada rotación de dinero invertido sobre AUM
    """
    def dppt(self): #Dollar performance per turnover: 
        dollar_perfomance = (self.cap[-1]/self.cap[0])-1
        init_cap = self.cap[0]
        turnover = init_cap/self.cap.mean() # dinero invertido/aum
        return dollar_perfomance/turnover
    
    """
        Retorno sobre costos
    """
    def roec(self): #Return on execution costs
        dollar_perfomance = (self.cap[-1]/self.cap[0])-1
        execution_costs = self.costs.mean()
        return dollar_perfomance/execution_costs
    
    """
        Función que genera la métrica de perfomance de todos los statistics en base 100
    """
    def performance(self, crits):
        
        pt_1 = self.get_pnl()/self.cap[0] # el pnl se divide sobre el cap inicial para obtener un retorno pnl porcentual
        pt_2 = self.aror()
        pt_3 = self.hitratio()/len(self.ret) #el número de hit se divide al número total para obtener un porcentual
        pt_4 = self.avgrethits()
        pt_5 = self.computeDD_TuW(dollars=False)[0].max() #solo se toma el maximo drawdown
        pt_6 = self.computeDD_TuW(dollars=False)[1].max() #solo se toma el máximo time under water
        pt_7 = self.getHHI("p")
        pt_8 = self.getHHI("n")
        pt_9 = self.sr()
        pt_10 = self.psr(crits['Sharpe Ratio'][0]) #el sharpe ratio toma de benchmark el sharpe que se pide
        pt_11 = self.dppt()
        pt_12 = self.roec()
        
        frame = pd.DataFrame({"Pnl": [pt_1], "Annualized Rate of Returns": [pt_2], "Hit Ratio": [pt_3],
                             "Average Return of Hits": [pt_4], "Max Drawdown": [pt_5], "Max Time Under Water": [pt_6],
                             "HHI positive": [pt_7], "HHI negative": [pt_8], "Sharpe Ratio": [pt_9], 
                             "Probabilistic Sharpe Ratio": [pt_10], "Dollar Performance": [pt_11],
                             "Return Over Execution Costs": [pt_12]})
        frame = frame.append(pd.DataFrame(crits), ignore_index=True)
        
        #Todas las métricas reales se dividen sobre las exigidas y se promedian a base 100.
        #Las métricas de HHI se miden en negativo ya que menos concentrado es mejor siempre.
        score = (frame.iloc[0]/frame.iloc[1]).mean()*100
        frame.index = ["Real Values", "Benchmarks"]
        return frame, score
        
        
"""-------------------------------------------------------------------------------------------------------------------------
    Función que aplica la clase Metrics de estadísticas de backtest para dividir el dataframe en paths
    sobre los que se aplica la clase y se retorna un DataFrame compuesto por el puntaje de desempeño
    de cada path
    
    Inputs:
    df: DataFrame sobre el cual dividir y aplicar la clase
    crits: Criterios de desempeño esperados de las estrategias
    
    Outputs:
    df_2: DataFrame con las métricas reales por cada path dentro del frame inicial y el puntaje de desempeño
    
    Diccionario de datos genéricos
    
    Crits para uso en .performance(crits):
        
        crits = {'Pnl': [0.1], 
                'Annualized Rate of Returns': [0.45], 
                'Hit Ratio': [0.33], 
                'Average Return of Hits': [0.01],
                'Max Drawdown': [0.01], 
                'Max Time Under Water': [0.1], 
                'HHI positive': [-0.1], 
                'HHI negative': [-0.1], 
                'Sharpe Ratio': [0.88], 
                'Probabilistic Sharpe Ratio': [0.9], 
                'Dollar Performance': [0.1], 
                'Return Over Execution Costs': [0.5]}
    
"""

def metricsByPath(df, crits):
    df_2 = pd.DataFrame()
    for x in df.trial.unique():
        df_ = df[df.trial == x]
        clase = Metrics(df_)
        clase_df, clase_score = clase.performance(crits)
        clase_df["Score"] = clase_score
        clase_df["Trial"] = x
        df_2 = df_2.append(clase_df.loc["Real Values"])
        df_2.index = range(len(df_2))
    return df_2