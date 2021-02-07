"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

        
from enigmx.sadf import gettingSADF

from enigmx.save_info import (
                    generate_datasets, 
                    getting_ray_and_saving
                    )
from enigmx.triplebarrier import (
                    generateTripleBarrier, 
                    getting_ray_triple_barrier
                    )

from enigmx.utils import (sel_days, 
                        float_range,
                        global_list_stocks,
                        get_imbalance_parameters,
                        get_alpha_calibration_value
                        )

from enigmx.sampling_features import getSamplingFeatures

from enigmx.alternative_methods import (standard_bar_tunning, 
                                        entropyCalculationParallel, 
                                        etfTrick)


#from enigmx.alternative_methods import standard_bar_tuning


#basic params for tuning bars
#data_dir = 
#ticker = 'MSFT'
#num_desired_bars = 10 
#max_date = None


#basic params for generate basic dataset
data_dir = 'D:/data_repository/'
list_stocks = ['MSFT'] #iterar stock x stock
start_date = "2020-11-02" 
end_date = "2020-12-14" 
#list_stocks = global_list_stocks(data_dir)[275:310]
list_of_bartypes = ['volume']
alpha_calibration = 3692637
basic_stacked_path = 'D:/data_split_stacked/' 



print("Paso 0")    

#tunning | puede ser sobre varios tipos de barras y varias acciones
getting_ray_and_saving(
    ray_object_list = [standard_bar_tunning.remote(
                        url = data_dir, 
                        ticker = stock, 
                        num_bar = 7, 
                        #### Optional Params ####
                        date_tuple_range = ("2020-10-02", start_date)
                        ) for stock in list_stocks],
    path_save = basic_stacked_path,
    list_stocks = list_stocks,
    naming = 'TICK_VOL_DOL_TUNNED',
    saving_type = 'bar_tunning',
    #### Optional Params ####
    list_element_types = None
    )

print("Paso 1")

#basic bar | puede ser sobre varios tipos de barras y varias acciones
getting_ray_and_saving(
        ray_object_list = [generate_datasets.remote(
                        stock = stock, 
                        bartypesList = list_of_bartypes, 
                        data_dir = data_dir, 
                        range_dates = sel_days(start_date, end_date), 
                        imbalance_dict = dict.fromkeys(list_stocks, 1),
                        #### Optional Params ####
                        bar_grp_freq = 1,
                        bar_grp_horizon = 'd',                        
                        alpha_calibration = get_alpha_calibration_value(
                            basic_stacked_path +'TICK_VOL_DOL_TUNNED'+'.csv',
                            stock),
                        volatility_version = 'ver2',
                        window_application_fracdiff = 2,
                        window_application_horizon_barrier = 1,
                        limit_date_of_data_for_horizon = None,
                        data_application_volatility_fracdiff = 'close',
                        ) for stock in list_stocks], 
        path_save = basic_stacked_path, 
        list_stocks = list_stocks, 
        naming = "BAR",
        saving_type = 'bar_frame',
        #### Optional Params ####
        list_element_types = list_of_bartypes, #aqui iteracion
        )

print("Paso 2")

#entropy | puede ser sobre varias acciones, pero solo sobre un tipo de barra 
getting_ray_and_saving(
        ray_object_list = [entropyCalculationParallel.remote(
                        zarr_dir = data_dir,
                        pandas_dir = basic_stacked_path,
                        stock = stock, 
                        bartype = 'VOLUME',
                        #### Optional Params ####
                        beta = 0.01, 
                        entropy_window = 100, 
                        cumsum_sampling = True                        
                        ) for stock in list_stocks],
        path_save = basic_stacked_path, 
        list_stocks = list_stocks, 
        naming = "ENTROPY",
        saving_type = 'entropy',
        #### Optional Params ####
        list_element_types = 'VOLUME', #aqui iteracion    
    )

print("Paso 3")

#ETF Trick Computation | toma toda la lista de acciones
getting_ray_and_saving(
    ray_object_list = [etfTrick.remote(
                    bar_dir = basic_stacked_path, 
                    stock_list = list_stocks, #takes all equities in a list
                    bartype = 'VOLUME', 
                    #### Optional Params ####
                    k=10000, 
                    lower_bound_index = 50, 
                    allocation_approach = 'inv',
                    output_type = None)],
        path_save = basic_stacked_path, 
        list_stocks = list_stocks, 
        naming = "ETFTRICK",
        saving_type = 'etftrick',
        #### Optional Params ####
        list_element_types = 'VOLUME', #aqui iteracion        
        )


print("Paso 4")

#SADF | utiliza el dataframe de ETF TRICK
getting_ray_and_saving(
    ray_object_list = [gettingSADF.remote(
                    path_etf_frame = basic_stacked_path, 
                    bartype = 'VOLUME', #takes ETF Trick | No iteration 
                    #### Optional Params ####
                    lags = None, 
                    main_value_name = 'value'
                    )],
        path_save = basic_stacked_path, 
        list_stocks = list_stocks, 
        naming = "SADF",
        saving_type = 'sadf',
        #### Optional Params ####
        list_element_types = 'VOLUME', #aqui iteracion        
        )


print("Paso 5")

#SAMPLING FEATURES | utiliza path donde esté el dataframe de sadf o entropy
getSamplingFeatures(
    path_entropy_or_sadf_allocated = basic_stacked_path, 
    main_column_name = 'sadf', #'entropy' or 'sadf'
    h_value = .5,
    bartype = 'VOLUME',
    #### Optional Params ####
    select_events=True
    )


print("Paso 6") 


####triple barrier


print("paso 7")


#features computation

print("Paso 8")

#feature importance

print()

