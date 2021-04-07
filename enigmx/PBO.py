#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from itertools import combinations 

import matplotlib.pyplot as plt

def Sharpe_Ratio(array):
    
    mu_A,std_A = np.average(array, axis=0),np.std(array, axis=0)
    SR = mu_A/std_A
    return SR


# In[5]:


class Overfit_Stats():
    
    '''
    Descripción: Esta clase es el complemento del procedimiento CSCV o 
    Combinatorially Symmetric Cross-Validation detallado en Bailey et.al 2014.
    
    Tiene como objetivo generar Estadísticos de Overfitting: 
    1) Probability of Backtest Overfit (PBO).   
    2) Probability of Loss (ProbLoss).
    3) Stochastic Dominance (StochDom)
    
    Esta clase retorna cada uno de estos estadísticos utilizando los métodos correspondientes
    
    INPUTS: 
    
    Lambda_c    -> array de logits para cada "c \in C" obtenido utilizando la clase CSCV.
    
    R_triple    -> array de tripletes [R_n*,R_bar_n*,np.mean(R_bar)] para cada "c \in C" obtenido utilizando la clase CSCV.
                       
    METODOS:
    
    PBO      := Calcula la probabilidad de overfitting
    ProbLoss := Calcula la probabilidad de que el performance del modelo sea negativo
    StochDom := 
    '''
        
    def __init__(self,lambda_c,R_triple):
        self.lambda_c = lambda_c
        self.R_triple = R_triple
        self.PBO      = self.PBO(self.lambda_c)
        self.ProbLoss = self.ProbLoss(self.R_triple)
#        self.StochDom = self.StochDom(self.R_triple)
    
    def PBO(self,lambda_c):
        
        PBO = len(lambda_c[lambda_c < 0])/len(lambda_c)
        
        return PBO
    
    def ProbLoss(self,R_triple):
        
        R_train_n_star = R_triple[:,0]
        R_test_n_star  = R_triple[:,1]
        
        ProbLoss_IS  = len(R_train_n_star[R_train_n_star<0])/len(R_train_n_star)
        ProbLoss_OOS = len(R_test_n_star[R_test_n_star<0])/len(R_test_n_star)
        
        return ProbLoss_IS,ProbLoss_OOS

#     def StochDom(self,R_triple):
        
#         R_test_n_star = R_triple[:,1]
#         R_test_mean  = R_triple[:,2]
        
#         StochDom
        
#         ProbLoss_IS,ProbLoss_OOS


# In[6]:


import pandas as pd
import numpy as np
from itertools import combinations 
import matplotlib.pyplot as plt

class CSCV():
    
    '''
    Descripción: Esta clase es la implementación del algoritmo denominado CSCV o 
    Combinatorially Symmetric Cross-Validation detallado en Bailey et.al 2014.
    
    La idea del procedimiento CSCV es: 
    1) Obtener un gran número de "Train-Test samples" al combinar de diferentes maneras S particiones.   
    2) Para cada combinación "c \in C" 
    2.a) Se calcula una metrica de performance IS (R) y OOS (R_bar).
    2.b) Se rankea R y R_bar en los vectores r y r_bar y se selecciona el modelo con mejor performance IS (n*)
    2.c) Se calcula el ranking relativo OOS para el modelo n* (omega_c)
    2.d) Se calcula el valor logistico (logit) de omega_c (lambda_c)
    3) Se computan los siguientes vectores: 
    3.a) La distribución de lambda_c para cada "c \in C" (Lambda_c)
    3.b) La distribución del vector [R_n*,R_bar_n*,np.mean(R_bar)] para cada "c \in C" (R_triple)
    
    Esta clase retorna Lambda_c y R_triple usando el método CSCV(*).Lambda
    
    INPUTS: 
    
    returns    -> array de tamaño T con los retornos de nuestro target.
    
    models     -> matriz de tamñano TxN donde cada columna es un modelo distinto.
    
    tipoPL     -> Determina el tipo de matriz M (o matriz de P&L):
                   a) tipoPL = False => M es una matriz con los retornos de cada modelo.
                   b) tipoPL = True  => M es una matriz con el valor acumulado por cada modelo.
                   
    S          -> Número de particiones a nuestra matriz M, S debe ser un número par. 
    
    metric     -> Una función que calcula las metricas de performance de los modelos para cada "c \in C".
    
    fix_n_star -> Dos opciones:
                   a) fix_n_star = None => n* sera el indice (de columna) del mejor modelo IS en cada "c \in C"
                   b) fix_n_star = n    => n* sera el indice de la columna n
                   
    METODOS:
    
    Lambda        := método principal, representa el cuarto y quinto paso en el procedimiento CSCV, 
                     tiene como dependencia el método "combinaciones" y el método "ranking".
                     Utiliza todos los Inputs.
    combinaciones := representa el tercer paso en el procedimiento CSCV, 
                     tiene como dependencia el método "M_s" y el paquete itertools.combinations.
                     Utiliza los siguiente Inputs:
                     a) returns,
                     b) models,
                     c) tipoPL,
                     d) S
    ranking       := dependencia del método Lambda, determina el ranking (de menor a mayor) en un array
                     Se utiliza para rankear el array de performance de los N modelos para cada "c \in C"
    M_s           := representa el segundo paso en el procedimiento CSCV, 
                     tiene como dependencia el método "PandL"
                     Utiliza los siguiente Inputs:
                     a) returns,
                     b) models,
                     c) tipoPL,
                     d) S
    PandL         := representa el primer paso en el procedimiento CSCV, 
                     Utiliza los siguiente Inputs:
                     a) returns,
                     b) models,
                     c) tipoPL
    '''
    
    def __init__(self,returns,models,tipoPL,S,metric,fix_n_star):
        self.returns = returns
        self.models = models
        self.tipoPL = tipoPL
        self.S = S
        self.metric = metric
        self.fix_n_star = fix_n_star
        self.Lambda = self.Lambda(self.returns,self.models,self.tipoPL,self.S,self.metric,self.fix_n_star)
        #self.combinaciones = self.combinaciones(self.returns,self.models,self.tipoPL,self.S)
        #self.Ms = self.M_s(self.returns,self.models,self.tipoPL,self.S)
        #self.PandL = self.PandL(self.returns,self.models,self.tipoPL)     
        
    def Lambda(self,returns,models,tipoPL,S,metric,fix_n_star):
        '''
        Cuarto y Quinto paso del algoritmo CSCV (4/5 y 5/5)
        * Genera un array (Lambda_c) que contiene un logit (lambda_c) para cada combinacion c en C_s  
        * Genera un array (R_triple) que contiene un array con tres elementos (R_n*,R_n*_bar y np.mean(R_bar)) 
          para cada c en C_s
        '''
        
        Cs = self.combinaciones(returns,models,tipoPL,S)
        
        Cs_2 = Cs.copy()
        Cs_2.reverse()

        Lambda = np.zeros(len(Cs))
        R_triple = np.zeros([len(Cs),3])
        for i in range(0,len(Cs)):

            A = np.vstack(list(Cs[i])) # TRAIN SAMPLE
            E  = np.vstack(list(Cs_2[i])) # TEST SAMPLE

            R_train,R_test = self.metric(A),self.metric(E)
            r_train,r_test = self.ranking(R_train),self.ranking(R_test)

            if fix_n_star == None:
                n_star = np.argmax(r_train, axis=0)
            else:
                n_star = fix_n_star
            #print(n_star)
            
            triple = np.array([R_train[n_star],R_test[n_star],np.mean(R_test)])
            
            N = len(r_train)

            w_c = r_test[n_star]/(N+1)

            lambda_c = np.log(w_c/(1-w_c))

            Lambda[i] = lambda_c
            R_triple[i,:] = triple 
        return Lambda,R_triple

    def combinaciones(self,returns,models,tipoPL,S):
        '''
        Tercer paso del algoritmo CSCV (3/5)
        * Genera una lista con todas las posibles combinaciones de tamaño S/2 de los elementos en la lista Ms, 
          donde cada elemento es una matriz M_s con shape (T/S,N)
        '''
        
        Ms = self.M_s(returns,models,tipoPL,S)
        
        S = len(Ms)
        S_2 = int(S/2)
        Cs  = list(combinations(Ms,S_2))
        return Cs    
    
    def M_s(self,returns,models,tipoPL,S):
        '''
        Segundo paso del algoritmo CSCV (2/5)
        * Particiona la matriz M con shape (T,N) de Profit & Losses en 
          S matrices M_s con shape (T/S,N), las cuales guarda en una lista
        '''
        
        M = self.PandL(returns,models,tipoPL)
        
        if S/2 - round(S/2) == 0.0:
            Ms = np.array_split(M, S)
            return Ms
        else:
            print("S debe ser par") 
        
    def PandL(self,returns,models,tipoPL= False):
        '''
        Primer paso del algoritmo CSCV (1/5)
        * Transforma una serie de retornos y N series con predicciones de modelos en 
          N series de Profit and Loss (P&L)
        '''
        T,N = models.shape
        r   = np.repeat(returns,N).reshape(T,N)
        models_ret_fact = 1+models*r

        if tipoPL == True:
            M = np.cumprod(models_ret_fact, axis=0)
        else:
            M = models_ret_fact-1
        return M

    def ranking(self,array):
        '''
        Dependencia del Cuarto paso del algoritmo CSCV
        * Genera un array "ranks" con el ranking de los elementos del array insumo "array"
          del más bajo (1) al más alto (N = len(array))
        '''
        temp = array.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(array))+1
        return ranks



# # EJEMPLO

# In[43]:


r = np.random.rand(1000)

models = np.rint(2*np.random.rand(1000,100)-1)

print(r[0:10])

print(models[0:10,0:5])


# In[54]:


lambda_c,R_triple = CSCV(r,models,tipoPL= False,S = 16 , metric = Sharpe_Ratio,fix_n_star = None).Lambda

# print("valores de lambda_c")
# print(np.unique(lambda_c))

OvStats = Overfit_Stats(lambda_c,R_triple)
PBO = OvStats.PBO
ProbLoss_IS, ProbLoss_OOS = OvStats.ProbLoss

print("PBO =",PBO," |  Prob of Loss OOS = ",ProbLoss_OOS)


# ## Stochastic Dominance 

# In[55]:


num_comb = len(R_triple[:,0])
y = np.round(np.arange(0,num_comb)/(num_comb-1),4)
plt.plot(np.sort(R_triple[:,1]),y, color='blue',label = "R_bar_n*")
plt.plot(np.sort(R_triple[:,2]),y, color='red' ,label = "E[R_bar]")
plt.legend()
plt.show()

