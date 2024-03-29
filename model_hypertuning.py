"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import numpy as np
from sklearn.metrics import *
from scipy.stats import rv_continuous
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier
from enigmx.purgedkfold_features import PurgedKFold 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from keras.models import Sequential
from enigmx.utils import transformYLabelsVector
from keras.wrappers.scikit_learn import KerasClassifier

#---------------------MODEL HYPERTUNING BASE FUNCTIONS-----------------------#
class MyPipeline(Pipeline):
    """
    Main Class Pipeline Function for Model Hypertunning
    """
    def fit(self, X, y, sample_weight=None, **fit_params):
        if sample_weight is not None:
            fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight
        return super(MyPipeline, self).fit(X, y, **fit_params)
    
def clfHyperFit(feat, lbl, t1, param_grid, 
                pipe_clf = None, cv=3, bagging=[0, 0, 1.0],
                rndSearchIter=0, n_jobs=None, pctEmbargo=0, 
                scoring=None, **fit_params):
    """
    Parámetros:
        -feat: Dataframe de X's 
        -lbl: Dataframe labels (y's)
        -t1: pd.Series de tiempo correspondiente al dataframe de labels
        -param_grid: Diccionario con los parámetros a probar y una lista
                     con los posibles valores. Va a depender del modelo a 
                     tunear
        -pipe_clf: modelo a tunear!!!
        -cv: folds para el KFold
        -rndSearchIter: en caso haya una lista muy grande de valores para 
                        tunear, estos se escogerán aleatoriamente
    """
    
    # determinar las categorias de label que hay: si solo son 2, usa 'f1Score'
    if set(lbl.values) == {0, 1}:
        scoring = make_scorer(
            f1_score, 
            average='macro'
            )
    
    # si hay mas de 2 label a predecir, define un scoring diferente: 'log loss'
    else:
        scoring = make_scorer(
            log_loss, 
            greater_is_better=False, 
            needs_proba=True, 
            labels=lbl
            )
        
    # revisamos si el modelo ingresado es una NN de Keras
    if type(pipe_clf) in (KerasClassifier, Sequential):
        
        # cambiamos los labels de pd.Series a encoderVector
        # este es un array 2D -0D: [#POS 0: -1, #POS 1: 0, #POS 2: -1]
        lbl = transformYLabelsVector(lbl)
        
        
    # 1) hyperparameter searching, on train data
    inner_cv = PurgedKFold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    
    # tipo de gridesearch | gridSearch tradicional
    if rndSearchIter == 0:
        gs = GridSearchCV(estimator=pipe_clf, #ingestamos modelo
                          param_grid=param_grid, #ingestamos params for tunning
                          scoring=scoring, #metrica de scoring para seleccion
                          cv=inner_cv, #cant. de folds para el cross validation
                          n_jobs=n_jobs) #nJbos para paralelizacion

    # tipo de gridesearch | gridSearch random        
    else:
        gs = RandomizedSearchCV(estimator= pipe_clf, #ingestamos modelo
                                param_distributions= param_grid, #params for tunning
                                scoring= scoring, #metrica de scoring para seleccion
                                cv= inner_cv, # cant. de folds para el cross val
                                n_jobs= n_jobs, #nJbos para paralelizacion
                                iid= False, #definir si la data es iiD
                                n_iter= rndSearchIter) # num Iteraciones randomSearch
    
    # fit de la busqueda grid con los parametros X e Y
    gs = gs.fit(feat, lbl, **fit_params).best_estimator_
    
    
    # 2) fit validated model on the entirety of the data
    if bagging[1] > 0:
        gs = BaggingClassifier(
            base_estimator=MyPipeline(gs.steps), 
            n_estimators=int(bagging[0]), 
            max_samples=float(bagging[1]),
            max_features=float(bagging[2]), 
            n_jobs=n_jobs
            )
        
        gs = gs.fit(
            feat, lbl,
            sample_weight = fit_params[
                gs.base_estimator.steps[-1][0] + '__sample_weight'
                ]
            )
        
        gs = Pipeline([('bag', gs)])
    return gs


class logUniform_gen(rv_continuous):
    # Method to get random numbers log-uniformly distributed between 1 and e
    def _cdf(self, x):
        return np.log(x / self.a) / np.log(self.b / self.a)
    
def logUniform(a=1, b=np.exp(1)):
    #Method to construct LogUniform values
    return logUniform_gen(a=a, b=b, name='logUniform')