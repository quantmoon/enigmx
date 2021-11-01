"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import ray
ray.init(include_dashboard=(False),ignore_reinit_error=(True),num_cpus=2)

#from enigmx.utils import EquitiesEnigmxUniverse
from enigmx.databundle_interface import SQLEnigmXinterface
from enigmx.tests.telegram import send_message
from enigmx.tests.stocks import stocks



server_name = "35.192.156.82"
referential_base_database = 'TSQL'
pathzarr = '/var/data/data/'
list_stocks = ['VTOL', 'ZNGA']
start_date = "2020-12-01" 
end_date = "2021-03-25" 
desired_bars = 10
bartype = 'volume'
driver = ("{ODBC DRIVER 17 for SQL Server}"),
uid = "sqlserver"
pwd = "quantmoon2021"


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
    bars_features = False,
    backtest_database = False,
    bars_stacked = True,
    creation_database = True)

print("subiendo info")
enigmxsql.compute_info_to_sql(
            bars_tunning_process = False, 
            bar_construction_process = False, 
            entropy_construction_process = False, 
            etftrick_construction_process = False, 
            sampling_features_process = False, 
            triple_barrier_computation_process = False, 
            sample_weight_computation_process = False,
            features_bar_computation_process = False,
            features_stacking = True,
	    )



