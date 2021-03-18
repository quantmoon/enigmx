"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.svm import NuSVC
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix 
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

#BetSize Forgein Functions
def discreteSignal(signal0,stepSize):
    # discretize signal
    signal1=(signal0/stepSize).round()*stepSize # discretize
    signal1[signal1>1]=1 # cap
    signal1[signal1<-1]=-1 # floor
    return signal1

def getSignal_ASC(stepSize, prob, pred, numClasses):
    # get signals from predictions
    #1) generate signals from multinomial classification (one-vs-rest, OvR)
    z = (prob-1./numClasses)/(prob*(1.-prob))**.5 # t-value of OvR
    m = 2*norm.cdf(z)-1
    signal0=pred*m # signal=side*size
    signal1=discreteSignal(signal0,stepSize=stepSize) 
    return signal1

#BetSize Main Class
class BetSize(object):
  """
  Clase BetSize Central [clase objeto].

  Objetivo: desarrollar el entrenamiento de un 'Metalabeling-Model' 
            útil para la predicción del tamaño destino a inversión ('size').
  
  Métodos:
  --------

  __init__ (base): método base que recibe las entradas.

                    * array_features: matriz de características ZxN conteniendo 
                                      los features con los que se entrenó el 
                                      modelo base.

                    * array_predictions: vector Zx1 con las predicciones
                                         realizadas por el modelo base 
                                         a partir del 'array_features'.

                    * array_labels: vector Zx1 con los verdaderos labels 
                                    comparativos al 'array_predictions'.

                    * endogenous_model: str conteniendo las siglas del modelo
                                        sklearn seleccionado para fungir de 
                                        'Metalabeling-model'. Solo 'rbf' o 'svm'

                    * test_size: tamaño de partición de Train/Test para los 
                                 array de características y etiquetas. 
                    
                    * rebalance: booleano que permite el rebalanceo de la data
                                 en caso esta se presente desbalanceada 
                                 (evitar el 'lack-of-training')

                    * balance_method: str con el name del método de rebalanceo.
                                      Disponible 'SMOTE' y 'MLDP'(no impl.). 

  __warningStatements__: método que verifica el correcto ingreso de los inputs.

  __dataManagement__: método para concadenar features con las predicciones org.,
                      binarizar el vector de predicciones originales, y devolver
                      el Train/Test split con base al 'test_size'.

  __dataRebalanced__: método que ejecuta el rebalanceo de data de solicitarse.

  __randomGridVariablesRF__: método que devuelve el randomGridDict del RF. 

  __randomGridVariablesSVM__: método que devuelve el randomGridDict de la SVM.

  __endogenousModel__: método que ejecuta el randomGridSearch según el modelo.
                       Devuelve el modelo entrenado con los parámetros tuneados.

                       Este ya constituye un output en sí mismo según el método
                       de acceso 'get_betsize'.

  
  __predictionEndogenousModel__: ejecuta la predicción del modelo y construye
                                 el betSize para el CONJUNTO DE DATOS CONOCIDO
                                 compuesto por el array_features a partir de 
                                 'getSignal_ASC' (función externa).

  __dataExplorer__: calcula el % de pesos con base al BetSize Signal para el
                    CONJUNTO DE DATOS CONOCIDO compuesto por el array_features.
                    Devuelve 'allocations'.

  get_betsize: método de llamado de la clase. No necesita ningún parámetro ad. 
               Devuelve el método entrenado del Betsize ('Metalabeling-Model') 
               listo para usarse en un CONJUNTO DE DATOS NO CONOCIDO.

            Parámetros optativos:
                *  data_explorer: booleano para acceder al dataFrame generado
                                  para el CONJUNTO DE DATOS CONOCIDO compuesto
                                  por el array_features otorgado como input.

                                  Duelve dicho pd.DataFrame., ya no el modelo.

                *  confusion_matrix: booleano para plotear la matriz de conf.

                *  dollar_capital: int/float que represente la cant. capital
                                   disponible para asignar entre los activos
                                   del CONJUNTO DE DATOS CONOCIDO (solo si
                                   'data_explorer = True'). 
  """
  
  def __init__(self, 
               array_features, 
               array_predictions, 
               array_labels,
               endogenous_model = 'rf',
               test_size = 0.25,
               rebalance = True, 
               balance_method = 'smote'):
    
    self.array_features = array_features
    self.array_predictions = array_predictions 
    self.array_labels = array_labels
    self.endogenous_model = endogenous_model.lower() 
    self.test_size = test_size
    self.rebalance = rebalance
    self.balance_method = balance_method.lower()
  
  def __warningStatements__(self):

    if self.test_size >= 1:
      raise ValueError(
          "BROKEN CODE: 'test_size' should be less than 1."
      )
    if len(self.array_features.shape) != 2:
      raise ValueError(
          "BROKEN CODE: 'array_features' should be 2D."
      )
    if len(self.array_predictions.shape) != 1:
      raise ValueError(
          "BROKEN CODE: 'array_predictions' should be 1D."
      )    
    if len(self.array_labels.shape) > 2:
      raise ValueError(
          "BROKEN CODE: 'array_labels' should be 2D or less."
      )
    if np.unique(self.array_labels).shape[0] > 3:
      raise ValueError(
          "BROKEN CODE: 'array_labels' only for 3 labes or less."
      )    
    if self.balance_method not in ['smote', 'mldp']:
      raise ValueError(
          "BROKEN CODE: 'balance_method' only 'smote' or 'mldp'."
      )
    if self.endogenous_model not in ['rf', 'svm']:
      raise ValueError(
          "BROKEN CODE: 'endogenous_model' only 'rf' or 'svm'."
      )
    
    
    #if len(self.array_labels.shape) == 1:
    #  self.array_labels = self.array_labels.reshape(
    #      self.array_labels.reshape(
    #          self.array_labels.shape[0], 1
    #          )
    #      )
    
  def __dataManagement__(self):
    self.__warningStatements__()

    #features matrix inc. [-1,0,1] prediction as new feature
    new_array_features = np.hstack(
        (
            self.array_features, 
            np.vstack(
                self.array_predictions
                )
            )
        )
    
    #transform labels [-1,0,1] as a binary set!
    new_array_labels = (self.array_labels!=0)*1

    #return x_train, x_test, y_train, y_test
    return train_test_split(new_array_features, 
                            new_array_labels, 
                            test_size=self.test_size, 
                            random_state=0
                            )
    
  def __dataRebalanced__(self):
      
    #Splitting process over original data (binarized labels)
    (
        new_x_train, new_x_test, 
        new_y_train, new_y_test
     ) = self.__dataManagement__()
    
    #SMOTE Rebalancing Process
    if self.balance_method == 'smote':
      sm = SMOTE(random_state=2) 
      new_x_train_res, new_y_train_res = sm.fit_sample(
           new_x_train, 
           new_y_train.ravel()
           )
      
    #Marquitos Rebalancing Process
    if self.balance_method == 'mldp':
      print("No balance method 'mldp' defined yet.")

    return new_x_train_res, new_x_test, new_y_train_res, new_y_test    

  def __randomGridVariablesRF__(self): #AGREGAR AQUI MÁS PARÁMETROS!!!

    #parameters for RandomForest RandomGridSearch
      # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(
        start = 100, 
        stop = 1000, 
        num = 1) #10
    ]
    
      # Number of features to consider at every split
    max_features = ['auto', 'sqrt']

      # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(
        start = 10, 
        stop = 100, 
        num = 10)
    ]
    max_depth.append(None)

      # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
      # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
      # Method of selecting samples for training each tree
    bootstrap = [True, False]

      #return random grid dictionary
    return {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}

  def __randomGridVariablesSVM__(self):
      
      #parameters for SVM RandomGridSearch
      list_nus = np.linspace(0.01,0.99,2)
      list_kernels = ["rbf", "sigmoid"]  
      list_coef0 = [0.0,.1,.5,.9]      

      return {'nu': list_nus,
            'kernel': list_kernels,
            'coef0': list_coef0}

  def __endogenousModel__(self):
    
      #SMOTE rebalanced data, if it's required 
    if self.rebalance:
      (
          new_x_train_res, new_x_test, 
          new_y_train_res, new_y_test
       ) = self.__dataRebalanced__()
      
      #original no-rebalanced data, otherwise
    else: 
      (
          new_x_train_res, new_x_test, 
          new_y_train_res, new_y_test
       ) = self.__dataManagement__()
    
    #uploading to __init__ variables
    self.new_x_test = new_x_test
    self.new_y_test = new_y_test
    
    #if endogenous model is random forest
    if self.endogenous_model == 'rf':
        
      #RF Dict of parameters  
      random_grid_dictionary = self.__randomGridVariablesRF__()

      #Random Forest Classifier 
      rf = RandomForestClassifier()
      
      #RandomGridSearch over RF
      rf_random = RandomizedSearchCV(
          estimator = rf, 
          param_distributions = random_grid_dictionary, 
          n_iter = 50, 
          cv = 3, 
          verbose=2, 
          random_state=42, 
          n_jobs = -1
      )

      rf_random.fit(new_x_train_res, new_y_train_res)

      model_selected = rf_random.best_estimator_
    
    #otherwise, the endogenous model is a SVM
    else:
      
      #SVM Dict of parameters  
      random_grid_dictionary = self.__randomGridVariablesSVM__()
      
      #Nu Support Vector Machine
      svm = NuSVC(probability = True)
      
      #RandomGridSearch over NuSVC
      svm_random = RandomizedSearchCV(
          estimator = svm, 
          param_distributions = random_grid_dictionary, 
          n_iter = 25, 
          cv = 3, 
          verbose=2, 
          random_state=42, 
          n_jobs = -1
      )      

      svm_random.fit(new_x_train_res, new_y_train_res)

      model_selected = svm_random.best_estimator_

    return model_selected
  
  def __predictionEndogenousModel__(self, 
                                    plot_confusion_matrix = False):

    #get endogenous trained model 
    model_selection = self.__endogenousModel__()
    
    #prediction of label size - category [1, 0]
    prediction_label_size = model_selection.predict(self.new_x_test) 

    #prediction of label size - probability
    probabilities_label_size = model_selection.predict_proba(self.new_x_test)

    #final dataframe with general information
    final_set = pd.DataFrame(
          self.new_x_test, 
          columns=['volume', 'volatility', 'fracdiff', 'bet_side']
          )

    #bet size prediction inclusion from new bet side    
    final_set['bet_side_new'] = final_set['bet_side'] * prediction_label_size
    final_set['bet_size'] = probabilities_label_size.max(axis=1)

    #if user want a confusion matrix plot
    if plot_confusion_matrix:
        size_bet_conf_matrix = confusion_matrix(
            final_set['bet_side'], 
            final_set['bet_side_new']
            )
        f1_score_ = f1_score(
            final_set['bet_side'], 
            final_set['bet_side_new'], 
            average='macro'
            )
        plt.figure() 
        plot_confusion_matrix(
            size_bet_conf_matrix, classes=[1, 0, -1], 
            title='Confusion matrix Bet Size'
            )
        plt.show()
        
        print("F1-Score: {}".format(f1_score_))

    final_set['signal'] = getSignal_ASC(
            stepSize=0.2, 
            prob=final_set['bet_size'], 
            pred=final_set['bet_side_new'], 
            numClasses=3
        )
    
    #return final set | no model
    return final_set

  def __dataExplorer__(self, 
                      dollar_capital, 
                      plot_confusion_matrix=False):
      
        #get final set with endogenous model prediction  
        final_set = self.__predictionEndogenousModel__(
              plot_confusion_matrix
              )
        
        #assign new column as pct_allocation
        final_set['pct_allocation'] = (
            final_set.signal / abs(final_set.signal).sum()
            )
        
        #assign new column as cash_allocation
        final_set['cash_allocation'] = (
              final_set.pct_allocation * dollar_capital
              )
        
        #get only rows where pct_allocation != 0
        allocations = final_set[final_set.pct_allocation!=0]
        return allocations

  def get_betsize(self, 
                  data_explorer = False, 
                  confusion_matrix = False, 
                  dollar_capital = None):
      
      #if user want only to get a data exploration sample
      if data_explorer:
          #no 'dollar_capital' int/float defined raise Error
        if dollar_capital == None:
          raise ValueError(
              "You should assign a dollar_capital value [int/float]."
          )
          
        #else, return the pd.DataFrame as data exploration sample
        return self.__dataExplorer__(
            dollar_capital, 
            plot_confusion_matrix=confusion_matrix
            )
    
      #otherwise, the method returns the betsize model trained
      else:
        return self.__endogenousModel__() 
    
    
def betsize_livetrading(features_matrix, prediction_vector, 
                        betsize_model, scaling = False, 
                        stepSize=0.2, weighting = False):
  """
  Posibilita la aplicación del betsize para livetrading.
  Utiliza para ello el modelo 'meta-labeling' entrenado en la clase 'BetSize'.

  Inputs:
  ------
    - features_matrix: np.ndarray de MxN dim con las características.
    - prediction_vector: Mx1 dim np.ndarray con las predicciones 
                         dadas por el modelo exógeno (base).
    - betsize_model: 'metalabeling' model generado por la clase 'BetSize'.
    - scaling: booleano para escalar la matriz de características.
    - stepSize: variable para la construcción final de la señal del betsizing.
                Esta se encuentra entre ]0;0.5]
    - weighting: booleano para habilitar el cálculo de pesos porcentual. 
                 Caso contrario, retorna el betsize Signal original.                 
  
  Output:
  -------
    np.ndarray de tamaño M con el total de pesos asignados para 
    aquellas posiciones validadas por el betsize.

  Importante:
    'M': cantidad total de acciones
    'N': cantidad total de features 
    
    
    AGREGAR APALANCAMIENTO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
  """

  #revisar data type de algunos inputs
  if not all(
      isinstance(
          element , np.ndarray
          ) for element in [features_matrix, 
                            prediction_vector]
             ):
    raise ValueError(
        "Only np.ndarray for 'features_matrix' & 'precition_vector'."
    )

  #verificar si se escala o no
  if scaling:
    features_matrix = StandardScaler().fit_transform(
        features_matrix
        )
    
  #obtener data para metalabelling
  data_for_metalabelling = np.append(
      features_matrix, 
      prediction_vector, 
      axis=1
      )

  #obtener predicciones (binarizadas: 1 o 0) y sus probabilidades
  (
      prediction_label_size, probabilities_label_size
   ) = (
       betsize_model.predict(
           data_for_metalabelling
           ), 
        betsize_model.predict_proba(
            data_for_metalabelling
            )
        )

  #obtener señal BetSize   
  bet_size_signal = getSignal_ASC(
            stepSize=stepSize, 
            prob = probabilities_label_size.max(axis=1), 
            pred = data_for_metalabelling[:,-1] * prediction_label_size, 
            numClasses = 3
        ) 
  
  
  #obtener pesos % para cada input-equity (fila de la matriz)
  if weighting:
      if np.unique(bet_size_signal).shape[0]==1:
          return bet_size_signal
      else:
          bet_size_signal = (
            bet_size_signal/abs(
                bet_size_signal[bet_size_signal!=0]
                ).sum()
                )

  return bet_size_signal