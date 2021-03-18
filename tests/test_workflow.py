"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

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

