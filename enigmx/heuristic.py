"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from enigmx.PSO import PSO_particles, PSO

class heuristic():
    """
        Recibe un dataframe con estadísticas de backtesting 
        y devuelve la mejor combinación de estos en base a una
        heurística de evaluación obtenida por dos métodos: PSO o MinMax.

        Inputs:
        
            - df: dataframe con estadísticas
            - method: ["pso", "minmax"]
                * "pso": método por heurística PSO (Particle Swarm Optimization), 
                         minimiza los valores hacia un vector de mejor combinación, 
                         sobre este calcula la distancia euclideana de cada fila y 
                         retorna la más cercana a la solución pso.
                * "minmax": obtiene los mínimos y máximos objetivos dentro del dataframe de forma directa, 
                            sobre estos calcula la distancia euclideana de cada fila y 
                            retorna la más cercana a la solución minmax.
            - to_maximize: lista con los nombres de las columnas a maximizar.
            - to_minimize: lista con los nombres de las columnas a minimizar. 
            - fitness_pso_fuction: funcion fitness a usarse en el PSO
        
        Outputs:
            .df_ : retorna el dataframe con las distancias euclideanas de cada fila,
            .target :  retorna los valores objetivo a aproximar, ya sea por pso o por minmax.
            .best_solution: retorna la mejor solución del pso, 
                            la cual luego se aproximará con distancia euclideana.
            .optimum: devuelve el óptimo de estadísticas según el método utilizado.
    """    
    def __init__(
        self, df, 
        method = "pso", 
        to_maximize = ['Annualized Rate of Returns',
                       'Average Return of Hits', 
                       'Dollar Performance', 'Hit Ratio', 
                       'Pnl', 'Probabilistic Sharpe Ratio',
                       'Return Over Execution Costs', 'Sharpe Ratio'], 
        to_minimize=['HHI positive', 
                     'HHI negative', 
                     'Max Drawdown', 
                     'Max Time Under Water'],
        fitness_function = None):
        
        print("      ▶▶▶▶▶▶▶ Loading Heuristic Assessment Process ▶▶▶▶▶▶▶      ")
        
        # rellenamos nans con 0
        self.initial_df = df.fillna(0)
        
        # def objeto de escalamiento 
        s = StandardScaler()
        
        # ejecutamos el fit de escalamiento solo en cols para min/max
        s.fit(self.initial_df[to_maximize + to_minimize])
        
        # ejecutamos transformacion con el objeto de escalamiento para min/max
        self.df_ = s.transform(self.initial_df[to_maximize + to_minimize])
        
        # redefinimos el dataframe con la matriz de valores transformada
        self.df_ = pd.DataFrame(self.df_, columns=self.initial_df[to_maximize + to_minimize].columns)
        
        # seleccionamos las columnas/vectores de maximizacion y minimizacion
        self.df_ = self.df_[to_maximize + to_minimize]
        
        # asignamos la funcion fitness
        self.fitness_function = fitness_function 
        
        # PSO method - Activation
        if method == "pso":
            
            print("    :::: >> Particles Swarm Optimization ----> initialized")
            assert self.fitness_function != None , "Fitness param 'fitness_function' is not defined. Please, check!"
            
            # extraccion de minimos y maximos
            Min = self.df_.min()
            Max = self.df_.max()
            
            # definicion de los minimos y negativizacion del maximos minimizables 
            self.min_ = Min
            self.min_[to_maximize] = -Max[to_maximize]
            
            # definicion de los maximos y negativizacion del minimos maximizables 
            self.max_ = Max
            self.max_[to_maximize] = -Min[to_maximize]
            
            # conversion de minimos y maximos en listas
            self.min_ = list(self.min_)
            self.max_ = list(self.max_)
        
            # generacion de las particulas del PSO 
            self.initial = PSO_particles(5, self.min_, self.max_)
            self.pso_optimizer_ackley = PSO(self.fitness_function, self.initial, 0.01, 1, 1, 100) 
            self.best_solution, best_fitness, \
            history_bestfitness, history_bestsolution = self.pso_optimizer_ackley.optimize()       

            # negativizacion de vectores de maximizacion
            self.best_solution[:len(to_maximize)] = -self.best_solution[:len(to_maximize)]
            
            # iteracion por vector y calculo de distancias          
            self.distances = []
            for x in range(len(self.df_)):
                self.distances.append(euclidean_distances(self.best_solution.reshape(1, -1), np.array(self.df_.iloc[x]).reshape(1, -1))[0][0])
            self.distances = pd.Series(self.distances, index = self.df_.index)
            self.df_["euclidean_distances"] = self.distances
            self.target = self.min_            
            
            # iteracion por distancias y seleccion del valor de maximizacion (distancia mas alejada para PSO)
            for h in range(len(to_maximize)):
                self.target[h] = -self.target[h]
            self.target = pd.Series(self.target, index = to_maximize + to_minimize)
            self.heuristic = self.initial_df.iloc[self.distances.argmax()]
            
            # definimos el dataframe con los indicadores de distancia y valor que encontramos 
            self.data_frame = pd.concat(
                [
                    self.target, 
                    pd.Series
                    (self.best_solution, index = to_maximize + to_minimize),
                    self.heuristic[to_maximize + to_minimize]
                ], 
                axis=1
            )
            self.data_frame.columns = [
                "Objetivo min_max", 
                "Solución PSO", 
                "Solución clase"
            ]
            
            print("    :::: >> Particles Swarm Optimization ----> finished")
        
        # MinMax Method - Activation
        elif method == "minmax":
            
            print("    :::: >> MinMax Method ----> initialized")
            
            # lista vacia para almacenamiento de minimos y maximos
            self.min_max = []
            
            # iteracion para recoleccion de valores sujetos a maximizacion  
            for m in to_maximize:
                    self.min_max.append(max(self.df_[m]))
                    
            # iteracion para recoleccion de valores sujetos a minimizacion
            for n in to_minimize:
                    self.min_max.append(min(self.df_[n]))
            
            # iteracion por vector y calculo de distancias 
            self.distances = []
            for x in range(len(self.df_)):
                self.distances.append(euclidean_distances(np.array(self.min_max).reshape(1, -1), np.array(self.df_.iloc[x]).reshape(1, -1))[0][0])
            self.distances = pd.Series(self.distances, index = self.df_.index)
            self.df_["euclidean_distances"] = self.distances
            self.target = pd.Series(self.min_max, index = to_maximize + to_minimize)
            self.heuristic = self.initial_df.iloc[self.distances.argmin()]
            self.data_frame = pd.concat([self.target, self.heuristic[to_maximize + to_minimize]], axis=1)
            self.data_frame.columns = ["Objetivo min_max", "Solución clase"]
            
            print("    :::: >> MinMax Method ----> finished")
        
       
