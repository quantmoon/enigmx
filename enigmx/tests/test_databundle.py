"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
ray.init(include_dashboard=(False),ignore_reinit_error=(True))

#from enigmx.utils import EquitiesEnigmxUniverse
from enigmx.databundle_interface import SQLEnigmXinterface

server_name = "35.223.72.148" 
referential_base_database = 'TSQL'
pathzarr = '/home/dataquantmoon/.local/lib/python3.7/site-packages/enigmx/transform/'
list_stocks = ['AFL', 'AGCO', 'AGI', 'AGIO']
start_date = "2020-08-09" 
end_date = "2020-09-09" 
desired_bars = 10
bartype = 'volume'
driver = "{ODBC DRIVER 17 for SQL Server}"
uid = "sqlserver"
pwd = "quantmoon2019"

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
    bars_tunning = True, 
    bars_basic = True, 
    bars_entropy = False, 
    etfs_trick = True, 
    bars_sampled = True, 
    bars_barrier =True,
    bars_weights = True,
    bars_features =True,
    creation_database = True)

print("subiendo info")
enigmxsql.compute_info_to_sql(
    bars_tunning_process = True, 
    bar_construction_process = True, 
    entropy_construction_process = False, 
    etftrick_construction_process = True, 
    sampling_features_process = True, 
    triple_barrier_computation_process = True,
    sample_weight_computation_process = True,
    features_bar_computation_process = True,
    tunning_interval = "15D",
    )


 
