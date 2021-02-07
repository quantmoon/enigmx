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
            index=False
            )    
        return "Data Events Sampled."
    else: 
        return final_selected_object


# Function for sampled selection based on structural breaks/entropy | base bars 
def crossSectionalDataSelection(path_bars, list_stocks, bartype, save = True):
    
    base_sample_type = 'SADF'
    
    #open selected dataframe of sampled values in SADF
    sampling_events_from_etf_and_method = pd.read_csv(
        path_bars + "FEATURES_" + bartype + "_" + 
        base_sample_type + "_SAMPLING.csv"
    )
    
    #open base bar dataframe by stock 
    list_stocks_bars = [
        pd.read_csv(
            path_bars + stock + "_" + bartype + "_BAR.csv"
        ) for stock in list_stocks
    ]
    
    #select sample of information in base bar df using sampled values in SADF
    selection_samples = [
        list_stocks_bars[idx].loc[
            list_stocks_bars[idx]["close_date"].isin(
                sampling_events_from_etf_and_method[stock]
            )
        ] for idx, stock in enumerate(list_stocks)
    ]
    
    #save in same path
    if save:
        for idx, frame in enumerate(selection_samples):
            frame = frame.reset_index(drop=True)
            frame.to_csv(
                path_bars + list_stocks[idx] + "_" 
                + bartype + "_" + base_sample_type + "_SAMPLED.csv",
                index=False
            )
            
        return ("Sampled dataframes selected")
    else: 
        #return list of pandas sampled
        return selection_samples