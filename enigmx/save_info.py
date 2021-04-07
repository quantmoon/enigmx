"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import ray
import numpy as np
import pandas as pd
from datetime import datetime
from enigmx.databundle import DataRespositoryInitialization

from enigmx.utils import (
    getDailyVolatility, 
    reformedVolatility, 
    doubleReformedVolatility,
    simpleFracdiff, 
    dataset_column_names,
    saving_basic_bars,
    saving_tunning_bars,
    saving_unique_entropy_bar,
    saving_etf_trick_or_sadf
    )


@ray.remote
def generate_datasets(stock, 
                      bartypesList,  
        		      data_dir, 
                      range_dates,                   
                      imbalance_dict,
                      bar_grp_freq= 1,
                      bar_grp_horizon='d',  
                      alpha_calibration = 3692637,
                      volatility_version = 'ver2',
                      window_application_fracdiff=2,                     
                      window_application_horizon_barrier= 1,
                      limit_date_of_data_for_horizon = None,                       
                      data_application_volatility_fracdiff='close'): 
    
    """    
    Nombre: 'generate_datasets'
    Consideración: Ray Remote Function
    Funcionalidad: Función principal generadora de información base.
    
    Descripción:
    La siguiente función incializa el proceso de generación de barras.
    Utiliza la información de los archivos '.zarr' para generarlas. 
    Es la base principal de la construcción de información para EnigmX.
    
    Inputs (Obligatorios):
    -----------------------    
        1. 'stock': str con el nemónico del asset deseado.
        2. 'bartypesList': str-list de los tipos de barras deseados x asset.
        3. 'data_dir': lugar str-path donde se alojan los archivos '.zarr'.
        4. 'imbalance_dict': dict con los parámetros de tunning del imbalance.
    
    Inputs (Predefinidos - Optativos):
        1. 'bar_grp_freq': frecuencia de agrupamiento de barras (int)
        2. 'bar_grp_horizon': plazo temporal de agrupamiento de barras (str)
        3. 'alpha_calibration': int TICKS|VOL|DOLLAR máx. x barra (deprecated),
                                ó dataframe para seleccionar x stock (yes).
        4. 'volatility_version': versión de cálculo de volatiliad diaria (str). 
            4.1. 'ver1': call 'getDailyVolatility' (MLDP original version).
            4.2. 'ver2': call 'reformedVolatility' (intraday returns / set).
            4.3. 'ver3': call 'doubleReformedVolatility' (rolling daily std).
        5. 'window_aplication_fracdiff': window to calculate fracdiff (int>1).
        6. 'window_application_horizon_barrier': 
            Useful window to define horizontal barrier limit (int).
        7. 'limit_date_of_data_for_horizon': 
            Date to avoid non-info error if there is no horizon-data (str).
            If 'None', there is no evaluation required.
        8. 'data_application_volatility_fracdiff': 
            Info name to compute volatility and fracdiff base series (str).
            Only 'close' (close_price) and 'open' (open_price) is available.
                                                
    Output:
    -------        
        List per asset containing bar-dataframe.
        
        Each list elemenet bar-dataframe has the following columns:
            
            * Bar Price Information:
                "open"	"high"	"low"  "close"	
            * Bar Date Information:
                "open_date"	"high_date"	"low_date"  "close_date"	
            * Bar Extra Information:
                "basic_volatility"	"bar_cum_volume"	"vwap"	
            * Bar Base General Information:                
                "fracdiff"	"volatility"	
            * Triple Barrrier Information from Bars:
                "horizon"  "upper_barrier"	"lower_barrier"
    """
    
    #volatility selection
    if volatility_version.lower() == "ver1":
        volatility_function = getDailyVolatility
    elif volatility_version.lower() == "ver2":
        volatility_function = reformedVolatility
    elif volatility_version.lower() == "ver3": 
        volatility_function = doubleReformedVolatility    
    else:
        raise ValueError(
            "Not 'volatility_version' recognized. Check args. params."
            )
    
    if data_application_volatility_fracdiff=='close':
        data_index_for_fracdiff_and_volatility = 3
        datatime_index_for_volatility = 7
    elif data_application_volatility_fracdiff== 'open':
        data_index_for_fracdiff_and_volatility = 0
        datatime_index_for_volatility = 4
    else:
        raise ValueError(
            "Only 'close'&'open' for fracdiff estimation. Check args. params."
            )
    
    list_of_list = []
            
    
    #iteración simple día x día para cálculo de barras
    for date in range_dates:            

        #inicialización de la clase repositorio de datos
        QMREPOSITORY = DataRespositoryInitialization(
                                        data_dir = data_dir, 
                                        start_date = date, 
                                        end_date = None,
                                        stock = stock
                                )

        #iteración por tipo de barra
        result_value = [
            QMREPOSITORY.geneticIterativeFunction(
                freq = bar_grp_freq,
                time = bar_grp_horizon,
                bartype = bartype,
                imbalance_list = imbalance_dict[stock], 
                daily_time_bars_organization=\
                    alpha_calibration[bartype+'_t'].item()
                )
            for bartype in bartypesList]

        list_of_list.append(result_value)
    
    
    datasets_to_save = []
    #iteration by arrayBarInfo | len(list_bar_arrays) == len(bartypesList)
    for idx_bar in range(len(bartypesList)):
        
        #construction of barInformation 
        arrayBarInfo = np.vstack(list(zip(*list_of_list))[idx_bar]) 
        
        #closeprices[idx: 3]/closedates[idx: 7] to get Daily Volatility Series
        daily_volatility = volatility_function(
            pd.Series(
                arrayBarInfo[...,data_index_for_fracdiff_and_volatility], 
                arrayBarInfo[...,datatime_index_for_volatility]
                )
            ) 
        
        #take selected info (close|open) and compute Fractional Diff | *****
        bar_price_fracdiff = simpleFracdiff(
            arrayBarInfo[...,data_index_for_fracdiff_and_volatility], 
            window_application_fracdiff
            )
        
        #include fracdiff price series as one last dim in arrayBarInfo
        arrayBarInfoUpdated = np.column_stack(
            [arrayBarInfo, bar_price_fracdiff]
            )
        
        #dataframe construction: full bar info, daily volatility and fracdiff
        barDataframe = pd.DataFrame(
            data = arrayBarInfoUpdated, 
            columns = dataset_column_names
            )
        
        #definition of daily_volatility index pd.Series dates as only date()
        volatility_dates_list = daily_volatility.index.date
        
        #inc. volatility value to main df if it meets selected bar date
        barDataframe["volatility"] = \
            barDataframe.iloc[:,datatime_index_for_volatility].apply(
                lambda date_bar: daily_volatility.loc[str(date_bar.date())] 
                if date_bar.date() in volatility_dates_list else 0
                )
        
        #definition of timestamp horizon for vertical barriers
        barDataframe["horizon"] = \
            barDataframe.iloc[:,datatime_index_for_volatility] + pd.Timedelta(
                days=window_application_horizon_barrier
                )
            
        #upper barrier definition    
        barDataframe["upper_barrier"] =barDataframe.close * (
            1 + barDataframe.volatility
            )
        
        #lower barrier definition
        barDataframe["lower_barrier"] =barDataframe.close * (
            1 - barDataframe.volatility
            )
        
        #drop when if volatility is zero, and reset the index 
        barDataframe = barDataframe.query("volatility!=0").reset_index(
            drop=True
            )            
        
        if limit_date_of_data_for_horizon != None:
            #limit data-day to avoid indexing error with non-existent dates
            limit_data_date = datetime.strptime(
                limit_date_of_data_for_horizon, 
                '%Y-%m-%d'
                )
            
            #segmentation over horizon based on limit day (last rows/events)
            barDataframe = barDataframe.query("horizon < @limit_data_date")
               
        datasets_to_save.append(barDataframe)
    
    return datasets_to_save

def getting_ray_and_saving(ray_object_list, 
                           path_save, 
                           list_stocks,
                           naming,
                           saving_type,
                           list_element_types = None):
    """
    Nombre: 'getting_ray_and_saving'
    Consideración: Python Function.
    Funcionalidad: Función que permite el almacenaje de información.
    
    Descripción:
    'getting_ray_and_saving' toma el objeto remoto ray de la función
    "generate_datasets" y almacena la información de c/tipo de barra x acción
    en una locación seleccionada.
    
    Inputs (obligatorios): 
        1. 'ray_object_list': objeto ray remoto de la func."generate_datasets".
        2. 'path_save': dirección donde se alojará la info extraída (str).
        3. 'list_stocks': lista de stocks (str) involucradas en el proceso.
                        Debe ser la misma que en "generate_datasets"
        4. 'naming': nomenclatura para identificación de c/ file (str).  

    Inputs (optativos):            
        5. 'list_element_types': lista con los nombres (str) de las barras 
                           involucradas en el proceso. Deben ser las mismas
                           que en "generate_datasets".
        
    Output:
        En consola: mensaje de proceso de finalización de guardado de info.
        En dispositivo: archivos '.csv' almacenados en la dirección otorgada.
    """
    #list of bar types dataframes stock list 
    list_datasets =  ray.get(ray_object_list)
    #list_datasets = ray_object_list
    
    #save tuning elements for each bar by stock
    if saving_type.lower() == 'bar_tunning':
        saving_tunning_bars(path_save, list_datasets, 
                            naming, list_stocks)
    
    #save base bar frame by stock
    elif saving_type.lower() == 'bar_frame':
        saving_basic_bars(path_save, list_datasets, naming,
                              list_stocks, list_bar_names = list_element_types)
        
    elif saving_type.lower() == 'entropy':
        saving_unique_entropy_bar(path_save, list_datasets, naming,
                              list_stocks, bar_name = list_element_types)
        
    elif saving_type.lower() == 'etftrick' or saving_type.lower() == 'sadf':
        saving_etf_trick_or_sadf(path_save, list_datasets, naming, list_stocks, 
                         bar_name = list_element_types)
    
    else:
        raise ValueError(
            "Paramter 'saving_type' should be defined as:\
                'bar_tunning', 'bar_frame' only. Other values not allowed."
            )
        
    print("Saving Porcess Ended")
    return None

    
