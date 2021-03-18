"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pandas as pd

def getTEvents(gRaw,h):
    tEvents,sPos,sNeg=[],0,0
    diff = gRaw #np.diff(gRaw)
    for i in range(1,diff.shape[0]):
        sPos,sNeg=max(0,sPos+diff[i]),min(0,sNeg+diff[i])
        
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
        base_df,
        main_column_name,
        h_value,
        select_events = True):
    
    
    final_selected_object = samplingFilter(
                                        base_df, 
                                        main_column_name, 
                                        h_value, 
                                        selection = select_events
                                        )    
    
    return final_selected_object


# Function for sampled selection based on structural breaks/entropy | base bars 
def crossSectionalDataSelection(sampled_dataframe, 
                                list_stocks_bars, list_stocks):
    

    #select sample of information in base bar df using sampled values in SADF
    selection_samples = [
        list_stocks_bars[idx].loc[
            list_stocks_bars[idx]["close_date"].isin(
                sampled_dataframe[stock]
            )
        ] for idx, stock in enumerate(list_stocks)
    ]
    
    #return a list of dataframes
    return selection_samples