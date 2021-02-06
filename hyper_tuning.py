"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import pandas as pd
from enigmx.utils import (
                        load_data,
                        compute_Ts_DIB,
                        init_values, 
                        )

#Hypertunning Variables per stock for Imbalance Bars Construction
def comparative_df(
        list_stocks, 
        dates, 
        alpha_1_interval, 
        alpha_2_interval, 
        bars_interval,
        path, hyperp_path, 
        tipo = 'DIB',
        drop_dup = False, plot = False):
    
    """
    Main Function to define Imbalance Bars parameters.
    
    It saves paratmers for each stock like:
                    	    A	         AA      ...
        alpha_1	     9.00E-05	   9.00E-05      ...
        alpha_2	     1.00E-05	   5.00E-05      ...	
        num_bars	       12	         14      ...	
        ET_init	  1215.147727	1578.525974      ...	
        Eb_init	 -0.013919436  -0.010053447      ...	
        Ebv_init -221.2989616  -88.02652877      ...	
        
    This function returns a .csv in certain path to use during Imbalance 
    construction.
    """
    
    df = pd.DataFrame(
        columns=[
            'symbol', 'num_bars','alpha1',
            'alpha2','len_Ts','max_threshold',
            'min_threshold'
            ]
        )
    
    hyperp = {}
    
    for symbol in list_stocks:
        print(symbol)
        ts_ref = 1
        val=[0.0001,0.0001,5,1,1,1]
        df_init = load_data(
            symbol, path, dates, drop_dup = True
            )
        
        for num_bars in bars_interval:
            X, ET_init, Eb_init, Ebv_init = init_values(
                df_init, 
                tipo=tipo, 
                num_bars=num_bars
                )
            X = np.array(X.bv,dtype=np.float64)
            
            for alpha_1 in alpha_1_interval:
                
                for alpha_2 in alpha_2_interval:
                    Ts, thres = compute_Ts_DIB(
                            X, 
                            ET_init, 
                            Ebv_init, 
                            alpha_1, 
                            alpha_2, 
                            50, 
                            len(dates)
                        )
                    
                    if len(Ts)>ts_ref:
                        
                        if thres[-1]<thres[-2]*10:
                            val[0] = alpha_1
                            val[1] = alpha_2
                            val[2] = num_bars
                            val[3] = ET_init
                            val[4] = Eb_init
                            val[5] = Ebv_init
                            ts_ref = len(Ts)
                            
        hyperp[symbol] = val
        
    df = pd.DataFrame(
        hyperp,index = [
            'alpha_1','alpha_2',
            'num_bars',
            'ET_init',
            'Eb_init','Ebv_init'
            ]
        )
    df.to_csv(hyperp_path+"hyperp.csv")
                
    return "Imbalance Hyperparameters Tunned"
