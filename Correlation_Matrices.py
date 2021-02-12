import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neighbors.kde import KernelDensity

import numpy as np
import scipy.cluster.hierarchy as sch
import pandas as pd

import scipy.spatial.distance as ssd

class denoise_cov():
    
    """
        Inputs
        
        q: valor T/N, debe ser mayor a la división real (T/N), nunca menor, 
            si fuera menor, subestima las cov entre las variables
            si fuera mayor, genera un trade off entre las covarianzas
        sugerencia: fijar q entre dos a cuatro veces T/N
        bWidth: 0.01, ancho entre distribuciones, determina que tan ruidosa es la distribución
        cov: matriz de covarianzas
    """
    
    def __init__(self, cov,q,bWidth):
        self.cov = cov
        self.q = q
        self.bWidth = bWidth
        self.denoise_cov=self.deNoiseCov(self.cov, self.q, self.bWidth)[0]
        self.denoise_corr = self.deNoiseCov(self.cov, self.q, self.bWidth)[1]
    
        ''' Usa la función KDE (Kernel Density Estimate) para adecuar la distribución 
        empírica a la distribución Marcenko-Pastur, la cual siguen los eigenvalues
        de una matriz, está también explica por que la matriz de covarianza está
        cargada de ruido
        Arroja un pandas dataframe con la distribución

        Inputs
        obs: data empirica
        bWidth: bandwidth of the kernel
        kernel: distribución
        x : the array of values on which the fit KDE will be evaluated
        '''
    def fitKDE(self, obs,bWidth=.25,kernel='gaussian',x=None):
        # Fit kernel to a series of obs, and derive the prob of obs
        # x is the array of values on which the fit KDE will be evaluated
        if len(obs.shape)==1:obs=obs.reshape(-1,1)
        kde=KernelDensity(kernel=kernel,bandwidth=bWidth).fit(obs)
        if x is None:x=np.unique(obs).reshape(-1,1)
        if len(x.shape)==1:x=x.reshape(-1,1)
        logProb=kde.score_samples(x) # log(density)
        pdf=pd.Series(np.exp(logProb),index=x.flatten())
        return pdf
    #--------------------------------------------------
    ''' Simula la distribución Marcenko-Pastur
        Inputs:
        var: matriz de varianzas-covarianzas
        q: valor T/N
        pts: Número de simulaciones
    '''
    def mpPDF(self, var,q,pts):
     # Marcenko-Pastur pdf
        # q=T/N
        eMin,eMax=var*(1-(1./q)**.5)**2,var*(1+(1./q)**.5)**2

        eVal=np.linspace(eMin,eMax,pts)

        eVal = eVal.reshape(-1,)
        pdf=q/(2*np.pi*var*eVal)*((eMax-eVal)*(eVal-eMin))**.5
        pdf=pd.Series(pdf,index=eVal)
        return pdf
    #------------------------------------------------------------------------------
    ''' Retorna la suma de errores cuadrado entre la distribución teórica y 
        la empírica
        Inputs:
        var: matriz de varianzas-covarianzas
        eVal: datos empíricos
        q: valor T/N
        bWidth: bandwidth of the kernel
        pts: Observaciones de la matriz teórica
    '''
    def errPDFs(self, var,eVal,q,bWidth,pts=1000):
        # Fit error
        #print(var)

        pdf0=self.mpPDF(var,q,pts) # theoretical pdf
        #print(pdf0)
        pdf1=self.fitKDE(eVal,bWidth,x=pdf0.index.values) # empirical pdf
        sse=np.sum((pdf1-pdf0)**2)
        return sse
    #------------------------------------------------------------------------------
    ''' Convierte la matriz de correlaciones a covarianzas
        Inputs:
        corr: matriz de correlaciones
        std: matriz de desviaciones estándar
    '''
    def corr2cov(self, corr,std):
        cov=corr*np.outer(std,std)
        return cov
    #------------------------------------------------------------------------------
    ''' Convierte la matriz de covarianzas a correlaciones
        Inputs:
        cov: matriz de varianzas-covarianzas
    '''
    def cov2corr(self, cov):
        # Derive the correlation matrix from a covariance matrix
        std=np.sqrt(np.diag(cov))
        corr=cov/np.outer(std,std)
        corr[corr<-1],corr[corr>1]=-1,1 # numerical error
        return corr
    #------------------------------------------------------------------------------
    '''
        Encuentra el máximo autovalor de la matriz de varianzas covarianzas para la data empírica. Este paso
        es necesario para encontrar el cluster de máxima señal dentro de la matriz y "aislarlo" para diluir 
        la señal
        Inputs:
        eVal: data empírica
        q: valor T/N
        bWidth: bandwidth of the kernel

        Outputs:
        Var: varianza aleatoria mas cercana a la data empírica
        eMax: Máximo autovalor de la distribución Marcenko-Pastur usando varianza que minimiza error

        Descripción
        Utiliza la función minimize para minimizar los errores al cuadrado entre la distribución
        empírica y la distribución teórica, inicia la función en 0.5 y encuentra la matriz
        de varianzas teórica que se parece más a la data empírica
    '''
    def findMaxEval(self, eVal,q,bWidth):
        # Find max random eVal by fitting Marcenko's dist to the empirical one

        out=minimize(lambda *x:self.errPDFs(*x),.5,args=(eVal,q,bWidth),bounds=((1E-5,1-1E-5),))
        if out['success']:var=out['x'][0]
        else:var=1
        eMax=var*(1+(1./q)**.5)**2
        return eMax,var
    #------------------------------------------------------------------------------
    '''
        A partir de una matriz Hermitiana calcula los eigenvalues y eigenvectors, 
        los cuales son ordenados de mayor a menor y la matriz de eigenvalues
        es retornada como matriz diagonal

        inputs
        matrix: matriz hermitiana de entrada

        outputs
        eVal: matriz diagonal de eigenvalues ordenada descendente y en diagonal
        eVec: matriz de eigenvectors ordenada de forma descendente
    '''
    def getPCA(self, matrix):
        # Get eVal,eVec from a Hermitian matrix
        eVal,eVec=np.linalg.eigh(matrix)
        indices=eVal.argsort()[::-1] # arguments for sorting eVal desc
        eVal,eVec=eVal[indices],eVec[:,indices]
        eVal=np.diagflat(eVal)
        return eVal,eVec
    #------------------------------------------------------------------------------
    ''' Redistribuye los autovalores aleatorios que no son señal en igualmente promediados
        Solo mantiene los autovalores que se deben a clusters de correlación y no se ajustan a la distribución
        Luego reconstruye la matriz sobre esos autovalores, los autovectores no son alterados


        Inputs
        eVal: matriz diagonal de eigenvalues ordenada descendente 
        eVec: matriz de eigenvectors ordenada de forma descendente
        nFacts: la cantidad de primeros autovalores que son mayores al maximo autovalor 
        que se ajusta a la distribución teórica, da el punto de corte de arriba a abajo
        del máximo autovalor aleatorio que explica la varianza ajustada, los valores restantes
        son señales por ser clusters de alta correlación

        Output
        corr1: Matriz de correlaciones sin ruido
    '''
    def denoisedCorr(self, eVal,eVec,nFacts):
     # Remove noise from corr by fixing random eigenvalues
        eVal_=np.diag(eVal).copy()
        #Convierte eVal_[a partir del valor ajustado:] en vector de valores iguales
        eVal_[nFacts:]=eVal_[nFacts:].sum()/float(eVal_.shape[0]-nFacts)
        eVal_=np.diag(eVal_)
        corr1=np.dot(eVec,eVal_).dot(eVec.T)
        corr1=self.cov2corr(corr1)
        return corr1
    #------------------------------------------------------------------------------
    '''
        Funcion principal: Retira el ruido de una matriz de covarianzas
        Paso 1: Obtiene sus auto valores y autovectores ordenados de mayor a menor diagonalmente (PCA)
        Paso 2: Encuentra el maximo autovalor mediante el ajuste de la distribución teórica y la data empírica
        Paso 3: Modificar los autovalores de la matriz manteniendo la señal de clusters de correlación y retirando el ruido

        Inputs:
        cov0: Mtriz de covarianzas
        q: valor N/T
        bWidth: bandwidth of the kernel

        Output:
        Matriz de covarianzas sin ruido

    '''
    def deNoiseCov(self, cov0,q,bWidth):
        corr0=self.cov2corr(cov0)
        eVal0,eVec0=self.getPCA(corr0)
        eMax0,var0=self.findMaxEval(np.diag(eVal0),q,bWidth)
        nFacts0=eVal0.shape[0]-np.diag(eVal0)[::-1].searchsorted(eMax0)
        corr1=self.denoisedCorr(eVal0,eVec0,nFacts0)
        cov1=self.corr2cov(corr1,np.diag(cov0)**.5)
        return cov1, corr1
    
class HRP():
    
    def __init__(self, cov,corr):
        self.cov = cov
        self.corr = corr
        self.weights = self.h_risk_par(self.cov, self.corr)
        

    def h_risk_par(self, cov,corr):
        ''' PASO 1 (Tree Clustering) 
            Se agrupan los elementos de la matrix de correlacion en base a su distancia

            Plantear utilizar otros procedimientos en la medición de distancias, ejm: scipy.spatial.distance.pdist
        '''

        # distance matrix
        dist=((1-corr)/2.)**.5 
        # linkage matrix object
        link=sch.linkage(dist,'single') ### EVALUAR INCLUIR UN HYPERPARAMETRO
        ''' PASO 2 (Cuasi Diagonalización)

            Se determina el orden de las filas de la matriz de correlación en función de los clusters obtenidos en el paso 1.
        '''
        link=link.astype(int)
        sortIx=pd.Series([link[-1,0],link[-1,1]])
        # número de elementos por grupo (cuarta columna)
        numItems=int(link[-1,3])
        while sortIx.max()>=numItems:
            sortIx.index=range(0,sortIx.shape[0]*2,2) # make space
            df0=sortIx[sortIx>=numItems] # find clusters
            i=df0.index;j=df0.values-numItems
            sortIx[i]=link[j,0] # item 1
            df0=pd.Series(link[j,1],index=i+1)
            sortIx=sortIx.append(df0) # item 2
            sortIx=sortIx.sort_index() # re-sort
            sortIx.index=range(sortIx.shape[0]) # re-index
        sortIx = sortIx.astype(int).tolist()
        ''' PASO 3 (Recursive Bisection)

            Se determinan los pesos asignados a cada acción iterativamente por pares. En consecuencia, el número de acciones
            mínimo debe ser 4. 
        '''
        # generamos un vector para los pesos del portafolio
        w=pd.Series(1,index=sortIx)
        # initialize all items in one cluster
        cItems=[sortIx] 
        while len(cItems)>0:
            cItems=[i[j:k] for i in cItems for j,k in ((0,int(len(i)/2)),(int(len(i)/2),len(i))) if len(i)>1] # bi-section
            for i in range(0,len(cItems),2): # parse in pairs
                cItems0=cItems[i] # cluster 1
                cItems1=cItems[i+1] # cluster 2
                cVar0=self.getClusterVar(cov,cItems0)
                cVar1=self.getClusterVar(cov,cItems1)
                alpha=1-cVar0/(cVar0+cVar1)
                w[cItems0]*=alpha # weight 1
                w[cItems1]*=1-alpha # weight 2
        return w


    #———————————————————————————————————————
    def getIVP(self, cov,**kargs):
    # Compute the inverse-variance portfolio
        ivp=1./np.diag(cov)
        ivp/=ivp.sum()
        return ivp
    #———————————————————————————————————————
    def getClusterVar(self, cov,cItems):
    # Compute variance per cluster
        cov_=cov.iloc[cItems,cItems] # matrix slice
        w_=self.getIVP(cov_).reshape(-1,1)
        cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
        return cVar
        
    
class Noisy_TIC(object):
    """
    Inputs
    tree: Arbol de relaciones teóricas
    corr: Matriz de correlaciones
    sd: Vector de desviaciones estándar del dataframe
    """
    
    def __init__(self, tree, corr, sd):
        self.tree = tree
        self.corr = corr
        self.sd = sd
        self.linkage = self.getLinkage_corr(self.tree, self.corr)
        self.corr_matrix=self.link2corr(self.linkage[1], self.corr.index)
        self.cov_matrix = self.corr2cov(self.corr_matrix, self.sd)
    
    #Realiza el cluster por proximidad, utilizando linkClusters, que transforma el cluster resultante de sch.linkage a uno global
    #Añade un nivel superior al cluster a formarse, asegurando tu relación teórica previa al algoritmo de clustering
    #Divide los niveles del arbol general en subarboles de dos niveles sobre los que agrupa hacia el mayor nivel
    #Una vez agregados sobre el mayor nivel de los dos subniveles, calcula el cluster parcial con respeto a la teoría
    #Lo modifica de uno parcial a uno global con link_clusters, actualiza las distancias con update_dist y arroja el resultado
    def getLinkage_corr(self, tree,corr):
        tree = tree.copy()
        if len(np.unique(tree.iloc[:,-1]))>1:tree['All']=0 # add top level  #si hay mas de sector, agrega la columna all=0
        lnk0=np.empty(shape=(0,4))
        lvls=[[tree.columns[i-1],tree.columns[i]] for i in range(1,tree.shape[1])] #itera de dos en dos sobre las columnas del tree
        dist0=((1-corr)/2.)**.5 # distance matrix
        items0=dist0.index.tolist() # map lnk0 to dist0 #lista del index de corr. Las acciones en lista
        for cols in lvls:
            grps=tree[cols].drop_duplicates(cols[0]).set_index(cols[0]).groupby(cols[1]) #AGRUPAS por el nivel mas alto del subarbol
            for cat,items1 in grps: #categoria e items agrupados en el nivel mas alto del subarbol
                items1=items1.index.tolist() #lista de las acciones dentro del nivel mas alto del subarbol
                if len(items1)==1: # single item: rename #si solo tiene un item el nivel
                    items0[items0.index(items1[0])]=cat #en items0 añadimos el subnivel mas alto, dentro de la lista
                    #crea la matriz de correlaciones de los activos del maximo nivel, sus indices son los números
                    dist0=dist0.rename({items1[0]:cat},axis=0) 
                    dist0=dist0.rename({items1[0]:cat},axis=1) 
                    continue
                dist1=dist0.loc[items1,items1] #dist0 donde los items son números y solo son los del maximo nivel de relacion teo.
                lnk1=sch.linkage(ssd.squareform(dist1,force='tovector',
                    checks=(not np.allclose(dist1,dist1.T))),
                    optimal_ordering=True) # cluster that cat #cluster sobre los items del maximo nivel seleccionados en dist1
                #link1 arroja el cluster
                lnk_=self.linkClusters(lnk0,lnk1,items0,items1) #modifica el cluster formado según la función del snippet 2
                lnk0=np.append(lnk0,lnk_,axis=0) #almacena el cluster como fila
                items0+=range(len(items0),len(items0)+len(lnk_)) #?
                dist0=self.updateDist(dist0,lnk0,lnk_,items0) 
                items0[-1]=cat
                dist0.columns=dist0.columns[:-1].tolist()+[cat]
                dist0.index=dist0.columns
        df = pd.DataFrame(lnk0,columns = ["i0","i1","dist","num"])
        #lnk0=np.array(map(tuple,lnk0),dtype=[('i0',int),('i1',int), ('dist',float),('num',int)])
        #lnk0=np.array(map(tuple,lnk0),dtype=[[('i0',int),('i1',int), ('dist',float),('num',int)]]()
        return lnk0,df


    #modifica el cluster
    #items 0, todas las acciones, items1:sectores del cluster, #lnk0: vacio, #lnk1 cluster ya creado
    #Modifica los componentes del cluster siguiendo condicionales, para volverlo global y no un cluster parcial
    #Agrega acciones faltantes del sector y la distancia correspondiente al cluster, manteniendo la estructura
    def linkClusters(self, lnk0,lnk1,items0,items1): 
        # transform partial link1 (based on dist1) into global link0 (based on dist0)
        nAtoms=len(items0)-lnk0.shape[0] #universo acciones-vector de 0,4 resultado (n) acciones totales
        lnk_=lnk1.copy()
        for i in range(lnk_.shape[0]): #bucle sobre el numero de filas del cluster
            i3=0 #se hará la distancia
            for j in range(2): #bucle de 0 y 1
                if lnk_[i,j]<len(items1): 
                    lnk_[i,j]=items0.index(items1[int(lnk_[i,j])]) #si i0 es menor que los sectores, reemplaza i0 por 
                    #reemplaza i0 por las acciones de su sector
                else:
                    lnk_[i,j]+=-len(items1)+len(items0)#si i0 no es menor, sumale len(items0-items1) #el resto de acciones delsector
                # update number of items
                if lnk_[i,j]<nAtoms:i3+=1 #si i0 es menor que nAtoms, i3+=1, faltan items en dist, sumale 1
                else:
                    if lnk_[i,j]-nAtoms<lnk0.shape[0]:  #si es menor que 1, que filas del vector
                        i3+=lnk0[int(lnk_[i,j])-nAtoms,3]
                    else:
                        i3+=lnk_[int(lnk_[i,j])-len(items0),3]

            lnk_[i,3]=i3
        return lnk_
    
    # expand dist0 to incorporate newly created clusters
    def updateDist(self, dist0,lnk0,lnk_,items0,criterion=None):
        nAtoms=len(items0)-lnk0.shape[0]
        newItems=items0[-lnk_.shape[0]:]
        for i in range(lnk_.shape[0]):
            i0,i1=items0[int(lnk_[i,0])],items0[int(lnk_[i,1])]
            if criterion is None:
                if lnk_[i,0]<nAtoms:w0=1.
                else:w0=lnk0[int(lnk_[i,0])-nAtoms,3]
                if lnk_[i,1]<nAtoms:w1=1.
                else:w1=lnk0[int(lnk_[i,1])-nAtoms,3]
                dist1=(dist0[i0]*w0+dist0[i1]*w1)/(w0+w1)
            else:dist1=criterion(dist0[[i0,i1]],axis=1) # linkage criterion
            dist0[newItems[i]]=dist1 # add column
            dist0.loc[newItems[i]]=dist1 # add row
            dist0.loc[newItems[i],newItems[i]]=0. # main diagonal
            dist0=dist0.drop([i0,i1],axis=0)
            dist0=dist0.drop([i0,i1],axis=1)
        return dist0
    
    # get all atoms included in an item
    def getAtoms(self, lnk,item):
        anc=[item]
        while True:
            item_=max(anc)
            if item_<=lnk.shape[0]:break
            else:
                anc.remove(item_)
                anc.append(lnk['i0'][item_-lnk.shape[0]-1])
                anc.append(lnk['i1'][item_-lnk.shape[0]-1])
        return anc
    
    def link2corr(self, lnk,lbls):
        corr=pd.DataFrame(np.eye(lnk.shape[0]+1),index=lbls,columns=lbls,dtype=float)
        for i in range(lnk.shape[0]):
            x=self.getAtoms(lnk,lnk['i0'][i])
            y=self.getAtoms(lnk,lnk['i1'][i])
            corr.loc[lbls[x[0]],lbls[y[0]]]=1-2*lnk['dist'][i]**2 # off-diagonal values
            corr.loc[lbls[y[0]],lbls[x[0]]]=1-2*lnk['dist'][i]**2 # symmetry
        return corr
    
        #------------------------------------------------------------------------------
    ''' Convierte la matriz de correlaciones a covarianzas
        Inputs:
        corr: matriz de correlaciones
        std: matriz de desviaciones estándar
    '''
    def corr2cov(self, corr,std):
        cov=corr*np.outer(std,std)
        return cov
    #------------------------------------------------------------------------------
    ''' Convierte la matriz de covarianzas a correlaciones
        Inputs:
        cov: matriz de varianzas-covarianzas
    '''
    def cov2corr(self, cov):
        # Derive the correlation matrix from a covariance matrix
        std=np.sqrt(np.diag(cov))
        corr=cov/np.outer(std,std)
        corr[corr<-1],corr[corr>1]=-1,1 # numerical error
        return corr
    #------------------------------------------------------------------------------