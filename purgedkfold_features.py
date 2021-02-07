"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss,accuracy_score
from sklearn.model_selection._split import _BaseKFold

#Feature Importance MDI base function
def featImpMDI(model,featNames):
    """
    Get Mean Decrease Impurity
    fit = Classifier Model (tree-based)
    featNames = Names of feats
    x_train = values of the features (pd.DataFrame)
    y_train = values of labels (array of values)
    """

    df = {
        i:tree.feature_importances_ for i,tree in enumerate(model.estimators_)
        }
    df = pd.DataFrame.from_dict(df,orient="index")
    df.columns = featNames
    imp = pd.concat(
        {'mean':df.mean(),"std":df.std()*df.shape[0]**-.5}, axis=1
        )
    imp /= imp['mean'].sum()
    return imp

#Main Purged KFold Class
class PurgedKFold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples.
    """
    def __init__(self,n_splits=3,t1=None,pctEmbargo=0.):
        if not isinstance(t1, pd.Series):
            raise ValueError('Label Through Dates must be a pd.Series')
        super(PurgedKFold,self).__init__(
            n_splits,shuffle=False,random_state=None
            )
        self.t1=t1
        self.pctEmbargo=pctEmbargo

    def split(self,X,y=None,groups=None):
        if (X.index==self.t1.index).sum()!=len(self.t1):
            raise ValueError('X and ThruDateValues must have the same index')

        indices=np.arange(X.shape[0])
        mbrg=int(X.shape[0]*self.pctEmbargo)
        test_starts=[
            (i[0],i[-1]+1) for i in np.array_split(np.arange(X.shape[0]),
                                                   self.n_splits)
        ]
        
        for i,j in test_starts:
            
            t0=self.t1.index[i] # start of test set
            test_indices=indices[i:j]
            maxT1Idx=self.t1.index.searchsorted(
                self.t1[test_indices].max()
                )
            train_indices=self.t1.index.searchsorted(
                self.t1[self.t1<=t0].index
                )
            train_indices = train_indices-1 
            
            if maxT1Idx<X.shape[0]: # right train ( with embargo)
                train_indices=np.concatenate(
                    (train_indices, indices[maxT1Idx+mbrg:])
                    )
            yield train_indices,test_indices
            
#Cross-Validation Score Calculation
def cvScore(clf,X,y,sample_weight=None,scoring='neg_log_loss',
            t1=None,cv=None,cvGen=None,pctEmbargo=None):
    if scoring not in ['neg_log_loss','accuracy']:
        raise Exception('wrong scoring method.')
    
    idx = pd.IndexSlice
    if cvGen is None:
        #Purging Process
        cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) 
    score=[]
    
    for train,test in cvGen.split(X=X):

        fit=clf.fit(X.iloc[idx[train],:],y.iloc[idx[train]],
                    sample_weight=sample_weight.iloc[idx[train]].values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X.iloc[idx[test],:])
            score_=-log_loss(y.iloc[idx[test]], prob,
                             labels=clf.classes_)
        else:
            pred=fit.predict(X.iloc[idx[test],:])
            score_=accuracy_score(y.iloc[idx[test]],pred)

        score.append(score_)
    return np.array(score)

#Feature Importance MDA
def featImpMDA(clf,X,y,cv,sample_weight,t1,pctEmbargo,scoring='neg_log_loss'):
    # feat imporant based on OOS score reduction
    if scoring not in ['neg_log_loss','accuracy']:
        raise ValueError('wrong scoring method.')

    #Purged Cross-Validation
    cvGen=PurgedKFold(n_splits=cv,t1=t1,pctEmbargo=pctEmbargo) # purged cv
    scr0,scr1=pd.Series(), pd.DataFrame(columns=X.columns)

    for i,(train,test) in enumerate(cvGen.split(X=X)):
        X0,y0,w0=X.iloc[train,:],y.iloc[train],sample_weight.iloc[train]
        X1,y1,w1=X.iloc[test,:],y.iloc[test],sample_weight.iloc[test]
        fit=clf.fit(X=X0,y=y0,sample_weight=w0.values)
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1)
            scr0.loc[i]=-log_loss(y1,prob,sample_weight=w1.values,
                                      labels=clf.classes_)
        else:
            pred=fit.predict(X1)
            scr0.loc[i]=accuracy_score(y1,pred,sample_weight=w1.values)

    for j in X.columns:
        X1_=X1.copy(deep=True)
        np.random.shuffle(X1_[j].values) # permutation of a single column
        if scoring=='neg_log_loss':
            prob=fit.predict_proba(X1_)
            scr1.loc[i,j]=-log_loss(y1,prob,sample_weight=w1.values,
                                    labels=clf.classes_)
        else:
            pred=fit.predict(X1_)
            scr1.loc[i,j]=accuracy_score(y1,pred,sample_weight=w1.values)
    imp=(-scr1).add(scr0,axis=0)
    if scoring=='neg_log_loss':imp=imp/-scr1
    else: imp=imp/(1.-scr1)
    imp=(pd.concat({'mean':imp.mean(),
                    'std':imp.std()*imp.shape[0]**-0.5},
                   axis=1))
    return imp,scr0.mean()


#General Feature Importance Function    
def featImportances(trnsX,cont,model,n_estimators=1000,cv=10,
                    max_samples=1.,numThreads=11,pctEmbargo=0,
                    scoring='accuracy',method='SFI',minWLeaf=0.,oob=False,
                    sample_weight = None,**kargs):
    
    #Feature Importance calculation based on MDI & MDA process

    fit=model.fit(X=trnsX,y=cont['labels'],sample_weight=sample_weight)
    if oob == True:
        oob=fit.oob_score_
    else:
        oob = None
    if method=='MDI':
        imp=featImpMDI(fit,featNames=trnsX.columns)
        oos=cvScore(model,X=trnsX,y=cont['labels'],
                    cv=cv,sample_weight=sample_weight,
                    t1=cont['t1'],pctEmbargo=pctEmbargo,
                    scoring=scoring).mean()
    elif method=='MDA':
        imp,oos=featImpMDA(model,X=trnsX,y=cont['labels'],cv=cv,
                           sample_weight=sample_weight,t1=cont['t1'],
                           pctEmbargo=pctEmbargo,scoring=scoring)
    return imp,oos,oob

#Plot Feature Importance
def plotFeatImportance(pathOut,imp,oob,oos,method,
                       tag=0,simNum=0,model=None,**kargs):
    # plot mean imp bars with std
    plt.figure(figsize=(10,imp.shape[0]))
    imp=imp.sort_values('mean',ascending=True)
    ax=imp['mean'].plot(kind='barh',color='b',alpha=0.25,xerr=imp['std'],
                        error_kw={'ecolor':'r'})
    if method=='MDI':
        plt.xlim([0,imp.sum(axis=1).max()])
        plt.axvline(1./imp.shape[0],lw=1.,color='r',ls='dotted')
    ax.get_yaxis().set_visible(False)
    
    for i,j in zip(ax.patches,imp.index):
        ax.text(i.get_width()/2, i.get_y()+i.get_height()/2,
                j,ha='center',va='center',color='k')
    if method == "MDA":
        plt.title('tag='+tag+' | simNUm='+str(simNum)+
                  ' | oob='+str(round(oob,4))+' | oos='+str(round(oos,4))+
                  ' | model='+model)
    else:    
        plt.title('tag='+tag+' | simNUm='+str(simNum)+
                  ' | oob='+str(round(oob,4))+' | oos='+str(round(oos,4)))
    
    plt.savefig(pathOut+'featImportance_'+str(simNum)+'.png',dpi=100)
    plt.clf()
    plt.close()
