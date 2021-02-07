"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import time
import numpy as np
import pandas as pd
from enigmx.databundle import DataRespositoryInitialization
from dbgenerator import (table_params_time_NOVWAP_SIMPLE,
                         databundle_instance)

list_stocks = ['MSFT']

# general paramters
data_dir = 'D:/data_repository'
bartype = 'time'
start_date = '2020-09-28'
end_date = '2020-10-02'
vwap = False
global_range=True
server = "DESKTOP-N8JUB39"
time_vwap = 'Min'
freq_vwap = 1
creation_database = False
table_format = table_params_time_NOVWAP_SIMPLE

#######################QUANTMOON SQL TEST EXAMPLE##############################

# 0) Inicializar SQL y crear la base de datos (OP - no iterative)
SQLFRAME, dbconn, cursor = databundle_instance(
                                        server, 
                                        bartype,  
                                        create_database = creation_database, 
                                        vwap=vwap, 
                                        global_range=global_range
                                    )

# 1) Crear la nueva tabla con 'stock_GLOBAL' como tabla (OP - no iterateive)
for stock in list_stocks:
    SQLFRAME.create_new_table(
                    table_name = stock, 
                    dict_info = table_format, 
                    dbconn_ = dbconn, 
                    cursor_= cursor
                    )
    print("stock table created:", stock)

# 2) Crea el statement para escribir los nuevos datos
for stock in list_stocks:
    statement = SQLFRAME.get_write_statement(
                                    database_name = bartype, 
                                    table_name = stock, 
                                    cursor_ = cursor
                                    )
    
    print("Initialization starting")
    start_ = time.time()
    # 3) Inicializa el repositorio de datos 
    QMREPOSITORY = DataRespositoryInitialization(
                                    data_dir= data_dir, 
                                    start_date= start_date, 
                                    end_date= end_date,
                                    stock= stock
                                    )
    print(time.time()-start_)
    
    print("Information Tuple Collection")
    start_ = time.time()
    info = QMREPOSITORY.infoTimePriceVol
    print(time.time()-start_)
    
    #results = pd.DataFrame(
    #    {
    #        'datetime':info[0], 
    #        'price':info[1], 
    #        'vol':info[2]
    #        }
    #    )
    
    #4) Escribe los datos en la tabla
    #SQLFRAME.write_table_info(
    #                statement_= statement, 
    #                dbconn_= dbconn, 
    #                cursor_= cursor, 
    #                data_= results, 
    #                bartype_= bartype,
    #                vwap_=vwap
    #                )
    
    print("Insert Statement Process")
    start_ = time.time()
    SQLFRAME.globalSimpleInsert(statement, info, dbconn, cursor)
    print(time.time()-start_)
    
    print("stock saved:", stock)
    

#result = SQLFRAME.read_table_info(
#    "SELECT * FROM [TIME_NOVWAP].[dbo].MSFT_GLOBAL", 
#    dbconn, 
#    cursor, 
#    dataframe=True
#    )

#print(result)