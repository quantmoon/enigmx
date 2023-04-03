"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

#import pandas as pd

def getTEvents(gRaw,h):
    
    """
    Método de sampleo por eventos.
    
    Toma una serie 'graw' y un valor de filtro límite o 'h'.
    
    Samplea con base al nivel de desviación con 'h' como trehshold.
    
    Retorna el vector de eventos sampleados.
    """
    
    tEvents,sPos,sNeg=[],0,0
    diff = gRaw #np.diff(gRaw)
    for i in range(1,diff.shape[0]):
        sPos,sNeg=max(0,sPos+diff[i]),min(0,sNeg+diff[i])
        
        if sNeg<-h:
            sNeg=0;tEvents.append(i)
        elif sPos>h:
            sPos=0;tEvents.append(i)
    return tEvents #pd.DatetimeIndex(tEvents)

def samplingFilter(base_dataframe, main_column_name, h, selection = True):
    
    """
    Función central del proceso de sampling filter.
    
    Es ingestada por la función 'getSamplingFeatures'.
    
    Inputs:
        - base_dataframe: dataframe a samplear.
        - main_column_name: str con nombre de columna de ref para aplicar el sampling.
            Solo puede ser 'SADF' o 'entropy'
        - h: valor float 'h' del método de sampleo por evento.
        - selection: bool para activar o desactivar selección de eventos sig.
        
    Output:
        Si 'selection' = True:
            - dataframe con los eventos sampledos.
        En caso contrario:
            - lista con los indices de los eventos elegidos.        
    """
    
    # si se elecciona como main column a SADF
    if main_column_name.upper() == 'SADF':
        series_array = base_dataframe.query("SADF>=0").SADF.values
    
    # si se selecciona como main column a 'entropy'
    elif main_column_name.lower() == 'entropy':
        series_array = base_dataframe['entropy'].values
        
    # caso contrario, deten el proceso por error
    else:
        raise ValueError(
            "Not recognized 'main_column_name'."
        )
    
    # obten los eventos sampleados
    event_indices = getTEvents(series_array, h)
    
    # si se activa selección
    if selection:
        # retorna el datframe sampleado
        return base_dataframe.loc[event_indices]
    # si no se activa
    else:
        # retorna solo los eventos seleccionados en el sampling
        return event_indices
    
def getSamplingFeatures(
        base_df,
        main_column_name,
        h_value,
        select_events = True,
        stock = None):
    
    """
    Función resumen que ingesta el proceso de sampling filter.
    
    Inputs:
        - base_df: dataframe sobre el que aplicar el sampling.
        - main_column_name: str con nombre de columna de ref para aplicar el sampling.
            Solo puede ser 'SADF' o 'entropy'
        - h_value: valor float 'h' del método de sampleo por evento.
        - select_events: bool para activar o desactivar selección de eventos sig.
        
    Output:
        Si 'select_events' = True:
            - dataframe con los eventos sampledos.
        En caso contrario:
            - lista con los indices de los eventos elegidos.
    """
    
    # objeto final seleccionado en el sampling
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
    
    """
    Selección de eventos con structural break según SADF seleccionado.
    
    Inputs:
        - sampled_dataframe: dataframe ya sampleado.
        - list_stocks_bars: lista de bars dataframes x cada acción
        - list_stocks: lista de strings con nombres x cada acción
        
    Output:
        - lista de dataframes debidamente sampleados x cada acción
    """
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
