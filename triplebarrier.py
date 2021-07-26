"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
        
import ray
import numpy as np
import pandas as pd
from datetime import datetime
from enigmx.utils import (
    get_horizons, getBarrierCoords, LabelTripleBarrierComputation
    )

#Multilabel Computation | Sigma value as frontier | deprecated
def multilabel_computation(priceAchieved, upper, lower, sigma):
    difference = upper - lower
    
    if priceAchieved > upper:
        return 1
    elif upper - difference*sigma < priceAchieved <= upper:
        return 0.75
    elif lower <= priceAchieved < lower + difference*sigma:
        return 0.25
    elif (priceAchieved < lower and priceAchieved > 0):
        return 0
    else:
        return 0.5
    
#Segmented triple barrier computation | deprecated    
def trilabel_computation(priceAchieved, upper, lower):

    if priceAchieved > upper:
        return 1
    if priceAchieved < lower:
        if priceAchieved != 0:
            return -1
        else:
            return 0

#TripleBarrier Generation    
#@ray.remote
def generateTripleBarrier(data_dir,
                          stock,
                          bartype,
                          alpha = 2.5,
                          window_volatility=1, 
                          window_horizon=1,
                          sigma = 0.3):
    
    """
    Función general que permite la computación de la triple barrera.
    
    Utiliza aún csv para computación de triple barrera.
    
    Depreciado.
    """
    
    # lectura de dataframe
    df_ = pd.read_csv(
        data_dir + stock + "_" + bartype.upper() + '_BAR.csv',
        parse_dates= [
            "open_date",
            "high_date",
            "low_date",
            "close_date",
            "horizon"]
        )
    
    # asignación de columna datetime
    df_['datetime'] = pd.to_datetime(
                                df_['datetime']
                                )
    
    # redefinición de dataframe con index datetime
    df_ = df_.set_index('datetime')
    
    # eliminación del primer evento
    df_ = df_[1:]
    
    # extracción de variables add
    special_time = df_[['special_time']]
    volatilities = df_[['volatility']]
    
    #compute horizons 
    #usa 'special_time' (daily open) como inicio de horizonte
    #ahora deberá usar el inicio/finalización de cada barra
    horizons_ = get_horizons(
            special_time, window=window_horizon
            )
    
    #compute base special_time dataframe
    special_time = pd.concat([special_time,volatilities], axis=1)
    
    #defining upper and lower horizontal barriers            
    special_time = special_time.assign(
            upper= lambda x: (
                x.special_time * (1 + x.volatility*alpha)
            ), 
            lower= lambda x: (
                x.special_time * (1 - x.volatility*alpha)
            )
        )
    
    #join base special_time dataframe with horiontal barriers
    final_special_time = pd.concat(
            [
                special_time, horizons_.to_frame("horizon")
            ],
            axis=1
        ).dropna()
    
    #redefining time types
    df_.index = np.datetime_as_string(
                            df_.index, unit='D'
                            )    
    
    final_special_time.index = np.datetime_as_string(
                            final_special_time.index, unit='D'
                            )
    
    final_special_time.horizon = np.datetime_as_string(
                            final_special_time.horizon, unit='D'
                            )

    #computing tripple barrier: price and time values
    tripleBarrier = [
         getBarrierCoords(
             'D:/data_repository/'+ stock +".zarr",
             initTime, horizonTime, upperValue, lowerValue, 
             sigma
             ) for (
                 initTime, 
                 horizonTime, 
                 upperValue, 
                 lowerValue
                 ) 
                 in zip(
                     final_special_time.index.values,
                     final_special_time.horizon.values,
                     final_special_time.upper.values,
                     final_special_time.lower.values
                     )
                 ] 
        
    #setting new information columns 
    final_special_time[
        ["priceAchieved", "timeAchieved"]
        ] = list(tripleBarrier)
    
    final_special_time["tripleBarrier"] = final_special_time.apply(
        lambda x: 
            trilabel_computation(x.priceAchieved, x.upper, x.lower), axis=1
        )
    
    result_dataset = pd.concat(
        [final_special_time.iloc[:,2:], df_], axis=1
        )
    
    result_dataset["stock"] = stock
    
    return result_dataset

#Generate triple barrier using RAY
def getting_ray_triple_barrier(ray_object_list, data_dir_last, list_stocks):
    
    """
    Función para computar la triple barrera (formato antiguo).
    
    Utiliza aún csv para computación de triple barrera.
    
    Depreciado
    """
    
    # obtener lista de datasets
    
    list_datasets =  ray.get(ray_object_list)
    
    # iteración para guardado
    for idx, dataset in enumerate(list_datasets):
        
        print("Saving", list_stocks[idx], "...")
        
        dataset.to_csv(
            data_dir_last + list_stocks[idx] + '_COMPLETE'+'.csv', 
            date_format='%Y-%m-%d %H:%M:%S', 
            index=True, index_label='datetime')
        
    print("Saving Porcess Ended")
    return None

##############################################################################
###################### USEFUL TRIPLE BARRIER COMPUTATION #####################
##############################################################################

#@ray.remote
def new_triple_barrier_computation(sampled_df, stock, zarr_path):
    """
    Función ingesta la computación de la triple barrera usando 'Ray'.
    
    Inputs:
        - sampled_df (pd.DataFrame): dataframe utilizado para computar la triple barrera.
        - stock (str): nombre de la acción sobre la que se computara triple barrera.
        - zarr_path (str): dirección path donde se encuentra alojado los archivos .zarr.
    
    Output:
        - dataset (pd.DataFrame): actualizado con las columnas 
            * barrierLabel
            * barrierTime
            * barrierPrice
    """
    # computación triplebarrera
    print(stock)
    dataset = LabelTripleBarrierComputation(sampled_df, stock, zarr_path)
    
    # transformación de serie numérica timestamp a datetimeObj
    dataset["barrierTime"] = dataset.barrierTime.transform(
        lambda x: datetime.fromtimestamp(x/1e3)
        )

    # retorno final del dataset
    return dataset
