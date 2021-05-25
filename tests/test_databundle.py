"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
ray.init(include_dashboard=(False),ignore_reinit_error=(True))

#from enigmx.utils import EquitiesEnigmxUniverse
from enigmx.databundle_interface import SQLEnigmXinterface

server_name = "WINDOWS-NI805M6" 
referential_base_database = 'TSQL'
pathzarr = 'C:/data/'
list_stocks = ['ACIW', 'KLXE', 'AZN', 'TH']
start_date = "2021-01-25" 
end_date = "2021-03-25" 
desired_bars = 10
bartype = 'volume'
driver = "{SQL Server}"
uid = ""
pwd = ""

print("inicializando clase")
enigmxsql = SQLEnigmXinterface(
   driver = driver,
    uid = uid,
    pwd = pwd, 
    server = server_name, 
    pathzarr = pathzarr, 
    list_stocks = list_stocks, 
    bartype = bartype, 
    start_date = start_date, 
    end_date = end_date, 
    desired_bars = desired_bars,
    referential_base_database = referential_base_database)

print("creando tablas")
enigmxsql.create_table_database(
    bars_tunning = False, 
    bars_basic = False, 
    bars_entropy = False, 
    etfs_trick = False, 
    bars_sampled = False, 
    bars_barrier = False,
    bars_weights = False,
    bars_features = True,
    creation_database = False)

print("subiendo info")
enigmxsql.compute_info_to_sql(
    bars_tunning_process = False, 
    bar_construction_process = False, 
    entropy_construction_process = False, 
    etftrick_construction_process = False, 
    sampling_features_process = False, 
    triple_barrier_computation_process = False,
    sample_weight_computation_process = False,
    features_bar_computation_process = True,
    tunning_interval = "10D",
    )


