"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples


#BASE SNIPPET | CLUSTERING KMEANS
#---------------------------------------------------
def clusterKMeansBase(corr0,maxNumClusters=None,n_init=10,val_abs=False):
    """
    Functions computes base clustering process.
    
    Paramteres:
        - corr0: correlation matrix among features (pandas series/df)
        - maxNumClusters: number of clusters desired (int)
        - n_init: iteration search steps (int)
        
    Output:
        - corr1: sorted correlation among clusters 
        - clstrs: clusters of features (dict of lists of strings)
        - silh: pandas series representing each silhouette score per feature
        
    Remember:
        The 'maxNumClusters' cannot be higher than the half of total features.
        E.g.: 
            Being N number of features, maxNumClusters should be N/2 at most.
            
        This is cause' the algoritm start searching by one cluster based on
        two features at least.
    """
    
    if val_abs == False:
        dist,silh=((1-corr0.fillna(0))/2.)**.5,pd.Series()# distance matrix
    else:
        dist,silh=(1-np.abs(corr0.fillna(0)))**.5,pd.Series()# distance matrix
        
    if maxNumClusters is None: 
        maxNumClusters = int(corr0.shape[0]/2)

    for init in range(n_init):
        for i in range(2,maxNumClusters+1): # find optimal num clusters
            kmeans_=KMeans(n_clusters=i,n_jobs=1,n_init=1)
            kmeans_=kmeans_.fit(dist)
            silh_=silhouette_samples(dist,kmeans_.labels_)
            stat=(silh_.mean()/silh_.std(),silh.mean()/silh.std())
            if np.isnan(stat[1]) or stat[0]>stat[1]:
                silh,kmeans=silh_,kmeans_ #silhouette score

    newIdx=np.argsort(kmeans.labels_)
    corr1=corr0.iloc[newIdx] # reorder rows
    corr1=corr1.iloc[:,newIdx] # reorder columns
    clstrs={i:corr0.columns[np.where(
        kmeans.labels_==i)[0]].tolist() for i in np.unique(kmeans.labels_)
        } # cluster members
    silh=pd.Series(silh,index=dist.index)
    return corr1,clstrs,silh


#SNIPPET 6.4 CLUSTERED MDI
#---------------------------------------------------
def groupMeanStd(df0,clstrs):
    out=pd.DataFrame(columns=['mean','std'])
    for i,j in clstrs.items():
        df1=df0[j].sum(axis=1)
        out.loc['C_'+str(i),'mean']=df1.mean()
        out.loc['C_'+str(i),'std']=df1.std()*df1.shape[0]**-.5
    return out

def featImpMDI_Clustered(clf,X,y,featNames,clstrs):
    fit=clf.fit(X,y)
    df0={i:tree.feature_importances_ for i,tree in enumerate(fit.estimators_)}
    df0=pd.DataFrame.from_dict(df0,orient='index')
    df0.columns=featNames
    df0=df0.replace(0,np.nan) # because max_features=1
    imp=groupMeanStd(df0,clstrs)
    imp/=imp['mean'].sum()
    return imp

#SNIPPET 6.5 CLUSTERED MDA
#---------------------------------------------------
def featImpMDA_Clustered(clf,X,y,clstrs,n_splits=10):
    from sklearn.metrics import log_loss
    from sklearn.model_selection._split import KFold
    cvGen=KFold(n_splits=n_splits)
    scr0,scr1=pd.Series(),pd.DataFrame(columns=clstrs.keys())
    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0=X.iloc[train,:],y.iloc[train]
        X1,y1=X.iloc[test,:],y.iloc[test]
        fit=clf.fit(X=X0,y=y0)
        prob=fit.predict_proba(X1)
        scr0.loc[i]=-log_loss(y1,prob,labels=clf.classes_)
        for j in scr1.columns:
            X1_=X1.copy(deep=True)
            for k in clstrs[j]:
                np.random.shuffle(X1_[k].values) # shuffle cluster
            prob=fit.predict_proba(X1_)
            scr1.loc[i,j]=-log_loss(y1,prob,labels=clf.classes_)
    imp=(-1*scr1).add(scr0,axis=0)
    imp=imp/(-1*scr1)
    imp=pd.concat({'mean':imp.mean(),'std':imp.std()*imp.shape[0]**-.5},
                  axis=1)
    imp.index=['C_'+str(i) for i in imp.index]
    return imp

##############################################################################
################## Clustering Feature Importance Class #######################
##############################################################################

class ClusteredFeatureImportance(object):
    
    def __init__(self, 
                 feature_matrix, 
                 model, 
                 method,
                 max_number_clusters = None,
                 number_initial_iterations = 10):
        
        # ingestando variables base
        self.feature_matrix=feature_matrix
        self.model=model
        self.method=method
        
        # se define al max clusters como la mitad - 1 del total de features
        if max_number_clusters==None: 
            self.max_number_clusters= int(self.feature_matrix.shape[1]/2 - 1)
        # si está definido, se toma el valor directamente
        else:
            self.max_number_clusters=max_number_clusters 
        
        self.number_initial_iterations=int(number_initial_iterations)
        
    def __baseClusterization__(self):
        
        # verificando que el input de features matrix sea array o pandas
        assert type(self.feature_matrix) == pd.core.frame.DataFrame, \
                "'feature_matrix' format is not correct."
        
        # estima correlacion de tipica features no clusterizada
        corr0 = self.feature_matrix.corr()
        
        # clusterización base: corr clusterizada, clusters y silhouette score
        corr1, clstrs, silh = clusterKMeansBase(
            corr0 = corr0, 
            maxNumClusters= self.max_number_clusters, 
            n_init= self.number_initial_iterations
            )
        
        # levanta la información a la clase general
        self.corr1, self.clstrs, self.silh = corr1, clstrs, silh
        
    def get_clustering_feature_importance(self, labels):
        
        print("    ::::>> Initializing Clustering Feat Importance Process...")
        
        # verificación de congruencia del modelo para el FeatImp | caso: 'MDI'
        if self.method == 'MDI': 
            # verificación de modelo utilizado
            if type(self.model).__name__!='RandomForestClassifier':
                raise ValueError(
                    "Only {} is allowed to implement 'MDI'".format(
                        'RandomForestClassifier'
                        )
                    )
            # elije el método de feature importance
            methodFunction = featImpMDI_Clustered
                
        # verificación de congruencia del modelo para el FeatImp | caso: 'MDA'
        if self.method == 'MDA':
            # verificación de modelo utilizado
            if type(self.model).__name__=='RandomForestClassifier':
                raise ValueError(
                    "{} model is not allowed to implement 'MDA'".format(
                        'RandomForestClassifier'
                        )
                    )
            # elije el método de feature importance
            methodFunction = featImpMDA_Clustered
        
        # computando clusterización base de los features con base a su corr.
        self.__baseClusterization__()
        
        # computando feature importance en los clusters 
        featureImportance = methodFunction(
                    self.model, 
                    self.feature_matrix, 
                    labels, 
                    self.feature_matrix.columns, 
                    self.clstrs
                )
        
        # ordenando los features por 'importance'
        sotredFeatImportance = featureImportance['mean'].sort_values(
                            ascending=False
                        )
        
        # retorna el sorted featImportance por cluster, y los clusters
        return sotredFeatImportance, self.clstrs

        
        
        
        
        
        
