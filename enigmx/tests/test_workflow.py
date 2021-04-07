"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

<<<<<<< HEAD
        
from enigmx.sadf import gettingSADF

from enigmx.save_info import (
                    generate_datasets, 
                    getting_ray_and_saving
                    )
from enigmx.triplebarrier import (
                    generateTripleBarrier, 
                    getting_ray_triple_barrier,
                    new_triple_barrier_computation
                    )

from enigmx.utils import (sel_days, 
                        float_range,
                        global_list_stocks,
                        get_imbalance_parameters,
                        get_alpha_calibration_value
                        )

from enigmx.sampling_features import getSamplingFeatures, crossSectionalDataSelection

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
data_dir = 'C:/Users/sony/Downloads/QUANTMOON/Chapter 2/' 
list_stocks = ['MSFT', 'NFLX', 'A', 'ADBE', 'AMZN'] #iterar stock x stock
start_date = "2020-09-02" 
end_date = "2020-12-14" 
#list_stocks = global_list_stocks(data_dir)[275:310]
list_of_bartypes = ['volume']
alpha_calibration = 3692637
basic_stacked_path = 'C:/Users/sony/Downloads/QUANTMOON/Chapter 2/' 


print("Paso 0")    
#tunning | puede ser sobre varios tipos de barras y varias acciones
getting_ray_and_saving(
    ray_object_list = [standard_bar_tunning.remote(
                        url = data_dir, 
                        ticker = stock, 
                        num_bar = 7, 
                        #### Optional Params ####
                        date_tuple_range = ("2020-08-02", start_date)
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
#getting_ray_and_saving(
#        ray_object_list = [entropyCalculationParallel.remote(
#                        zarr_dir = data_dir,
#                        pandas_dir = basic_stacked_path,
#                        stock = stock, 
#                        bartype = 'VOLUME',
#                        #### Optional Params ####
#                        beta = 0.02, 
#                        entropy_window = 100, 
#                        cumsum_sampling = True                        
#                        ) for stock in list_stocks],
#        path_save = basic_stacked_path, 
#        list_stocks = list_stocks, 
#        naming = "ENTROPY",
#        saving_type = 'entropy',
#        #### Optional Params ####
#        list_element_types = 'VOLUME', #aqui iteracion    
#    )

print("Paso 3")

#ETF Trick Computation | toma toda la lista de acciones
etfTrick(
            bar_dir = basic_stacked_path, 
            stock_list = list_stocks, #takes all equities in a list
            bartype = 'VOLUME', 
            #### Optional Params ####
            k=10000, 
            lower_bound_index = 50, 
            allocation_approach = 'inv',
            output_type = None
            )


print("Paso 4")

#SADF | utiliza el dataframe de ETF TRICK
gettingSADF(
    path_etf_frame = basic_stacked_path, 
    bartype = 'VOLUME', #takes ETF Trick | No iteration 
    #### Optional Params ####
    lags = None, 
    main_value_name = 'value'
    )


print("Paso 5")

#SAMPLING FEATURES | utiliza path donde esté el dataframe de sadf o entropy
# if 'main_column_name' is 'entropy', 'paso 6' is not necessary
getSamplingFeatures(
    path_entropy_or_sadf_allocated = basic_stacked_path, 
    main_column_name = 'sadf', #'entropy' or 'sadf'
    h_value = 2.5,
    bartype = 'VOLUME',
    #### Optional Params ####
    select_events=True
    )

print("Paso 6")
#CROSS-SECTIONAL DATA SELECTION BY SAMPLING INFO | ONLY FOR SADF 
crossSectionalDataSelection(
    path_bars = basic_stacked_path, 
=======
from enigmx.utils import EquitiesEnigmxUniverse
from enigmx.interface import EnigmXinterface

server_name = "DESKTOP-N8JUB39"
pathzarr = 'D:/data_repository/'
list_stocks = EquitiesEnigmxUniverse[10:20] #global_list_stocks(pathzarr)[10:20]
#list_stocks = ["MSFT","NFLX","A","ADBE","AMZN"]
start_date = "2020-08-09" 
end_date = "2020-09-09" 
desired_bars = 10
bartype = 'volume'

print("inicializando clase")
enigmx = EnigmXinterface(
    server = server_name, 
    pathzarr = pathzarr, 
>>>>>>> 6270c9a74f482960316bbe7d09e73a104907ceeb
    list_stocks = list_stocks, 
    bartype = bartype, 
    start_date = start_date, 
    end_date = end_date, 
    desired_bars = desired_bars)

print("creando tablas")
enigmx.create_table_database(
    bars_tunning = True, 
    bars_basic = True, 
    bars_entropy = False, 
    etfs_trick = True, 
    bars_sampled = True, 
    bars_barrier = True,
    bars_weights = True,
    creation_database = True)

print("subiendo info")
enigmx.compute_info_to_sql(
    bars_tunning_process = True, 
    bar_construction_process = True, 
    entropy_construction_process = False, 
    etftrick_construction_process = True, 
    sampling_features_process = True, 
    triple_barrier_computation_process = True,
    sample_weight_computation_process = True #agregar computación features
    )

