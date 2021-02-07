"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import numpy as np
import pandas as pd

def getTEventsCumSum(gRaw,h):
    """
    Function for CUMSUM Filter resampling by 'h'.
    
    Higher 'h' means less tEvents values in term of prices diff.
    
    0 < h < 1
    """
    
    tEvents,sPos,sNeg=[],0,0
    #diff = np.diff(g)
    diff=np.diff(gRaw) #differential | eq. returns
    for i in range(1,diff.shape[0]):
        sPos,sNeg=max(0,sPos+diff[i]),min(0,sNeg+diff[i])
        #print(sPos,sNeg)
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return tEvents#pd.DatetimeIndex(tEvents)

def samplingFilter(base_dataframe, main_column_name, h, selection = True):
    if main_column_name.upper() == 'SADF':
        series_array = base_dataframe.query("SADF>=0").SADF.values
    elif main_column_name.lower() == 'entropy':
        series_array = base_dataframe['entropy'].values
    else:
        raise ValueError(
            "Not recognized 'main_column_name'."
        )
    
    event_indices = getTEvents(series_array, h)
    
    if selection:
        return base_dataframe.loc[event_indices]
    else:
        return event_indices
    
def getSamplingFeatures(
        path_entropy_or_sadf_allocated, 
        main_column_name,
        h_value,
        bartype,
        select_events = True):
    
    if main_column_name.upper() == 'SADF':
        direction = (
            path_entropy_or_sadf_allocated+"SERIES_"+
            bartype.upper()+"_SADF.csv"
            )
    elif main_column_name.lower() == 'entropy':
        direction = (
            path_entropy_or_sadf_allocated+"SERIES_"+
            bartype.upper()+"_ENTROPY.csv"
            )
    else:
        raise ValueError(
            "Not recognized 'main_column_name' paramter."
            )
    
    base_df = pd.read_csv(direction)

    final_selected_object = samplingFilter(
                                        base_df, 
                                        main_column_name, 
                                        h_value, 
                                        selection = select_events
                                        )    
    
    if select_events:

        final_selected_object.to_csv(
            path_entropy_or_sadf_allocated+"FEATURES_"+
            bartype.upper() +"_"+ main_column_name.upper() +"_SAMPLING.csv", 
            date_format="%y-%m-%d %H:%M:%S.%f", 
            index=False
            )    
    else: 
        return final_selected_object
                        
