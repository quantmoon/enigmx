"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
#importing SQL Manager 
from enigmx.databundle import QuantmoonSQLManager as QSQL

#Define tables of information based on type of data
table_params_fracdiff = {
    'datetime':'DATETIME', 'fracdiff':'float'
    } 
table_params_tick_VWAP = {
    'grpID':'float', 'datetime':'DATETIME', 'price':'float'
    }
table_params_tick_NOVWAP = {
    'datetime':'DATETIME', 'price':'float'
    }
table_params_time_VWAP = {
    'datetime':'DATETIME', 'price':'float'
    }
table_params_time_NOVWAP = {
    'datetime':'DATETIME', 'price':'float', 'vol':'float'
    }

table_params_time_NOVWAP_SIMPLE = {
    'datetime':'float', 'price':'float', 'vol':'int'
    }
#############################################################################
table_params_bars_tunning = {
    'tick_t': 'int', 'volume_t':'int', 'dollar_t':'int', 'stock': 'VARCHAR(10)'
    }

table_standard_bar_structure = {
    'open_price':'float', 'high_price':'float', 'low_price':'float',
    'close_price':'float', 'open_date':'DATETIME',
    'high_date':'DATETIME', 'low_date':'DATETIME',	
    'close_date':'DATETIME', 'basic_volatility':'float',
    'bar_cum_volume':'int', 'vwap':'float',	'fracdiff':'float',	
    'volatility':'float','horizon':'DATETIME', 'upper_barrier':'float',
    'lower_barrier':'float'
    }

table_entropy_bar_structure = {
    'open_price':'float', 'high_price':'float', 'low_price':'float',
    'close_price':'float', 'open_date':'DATETIME',
    'high_date':'DATETIME', 'low_date':'DATETIME',	
    'close_date':'DATETIME', 'basic_volatility':'float',
    'bar_cum_volume':'int', 'vwap':'float',	'fracdiff':'float',	
    'volatility':'float','horizon':'DATETIME', 'upper_barrier':'float',
    'lower_barrier':'float', 'entropy':'float'
    }

table_barrier_bar_structure = {
    'open_price':'float', 'high_price':'float', 'low_price':'float',
    'close_price':'float', 'open_date':'DATETIME',
    'high_date':'DATETIME', 'low_date':'DATETIME',	
    'close_date':'DATETIME', 'basic_volatility':'float',
    'bar_cum_volume':'int', 'vwap':'float',	'fracdiff':'float',	
    'volatility':'float','horizon':'DATETIME', 'upper_barrier':'float',
    'lower_barrier':'float', 'barrierPrice': 'float', 'barrierLabel': 'int',
    'barrierTime': 'DATETIME'
    }

table_weighted_bar_structure = {
    'open_price':'float', 'high_price':'float', 'low_price':'float',
    'close_price':'float', 'open_date':'DATETIME',
    'high_date':'DATETIME', 'low_date':'DATETIME',	
    'close_date':'DATETIME', 'basic_volatility':'float',
    'bar_cum_volume':'int', 'vwap':'float',	'fracdiff':'float',	
    'volatility':'float','horizon':'DATETIME', 'upper_barrier':'float',
    'lower_barrier':'float', 'barrierPrice': 'float', 'barrierLabel': 'int',
    'barrierTime': 'DATETIME', 'overlap': 'int', 'weight': 'float', 
    'weightTime':'float'
    }

#############################################################################
#Definition of Main DataBundle Instance Function
def databundle_instance(server, 
                        bartype_database, 
                        driver, 
                        uid,
                        pwd,
                        create_database,
                        global_range, 
                        referential_base_database):
    """
    Función canalizadora de la interfaz de SQL. 
    
    Inputs:
        - server (str): nombre del servidor SQL
        - bartype_database (str): nombre de la base de datos
        - create_database (bool): booleano para ver si se crea una base de datos.
        - global_range (bool): booleano para catalogar a los datos como globales o parciales.
        
    Output:
        - Tupla: instancia de SQl, conexión a base de datos y puntero
    """
    
    #Define general parameters to initialize SQL
    SQLFRAME = QSQL(server=server, 
                    driver = driver, 
                    uid = uid, 
                    pwd = pwd,
                    database_name=bartype_database, 
                    globalRange=global_range, 
                    base_database=referential_base_database) 
    
    # OPTIONAL: create database if it wasn't created
    if create_database:
        SQLFRAME.create_new_database()
        
    # Returns tuple: (SQLFRAME, dbcon, cursor)    
    dbconn, cursor = SQLFRAME.select_database() 
    
    return SQLFRAME, dbconn, cursor
