# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 15:21:55 2021

@author: FRANK
"""
import ray
import pandas as pd
from datetime import datetime 
from enigmx.sadf import gettingSADF
from enigmx.sampleweight import WeightsGeneration
from enigmx.save_info import generate_datasets
from enigmx.triplebarrier import new_triple_barrier_computation

from enigmx.utils import (
    sel_days, 
    construct_pandas_tunning
    )

from enigmx.dbgenerator import (
    databundle_instance, 
    table_params_bars_tunning, 
    table_standard_bar_structure,
    table_entropy_bar_structure,
    table_barrier_bar_structure,
    table_weighted_bar_structure
    )


from enigmx.sampling_features import (
    getSamplingFeatures, 
    crossSectionalDataSelection
    )


from enigmx.alternative_methods import (
    standard_bar_tunning, 
    entropyCalculationParallel, 
    etfTrick
    )

class EnigmXinterface(object):
    
    def __init__(self, server, 
                 pathzarr, list_stocks, 
                 bartype, start_date, 
                 end_date, desired_bars, 
                 global_range=True):
        
        self.server_name = server
        self.pathzarr = pathzarr
        self.list_stocks = list_stocks
        self.bartype = bartype
        self.start_date = start_date
        self.end_date = end_date
        self.desired_bars = desired_bars
        self.global_range = global_range
        
    def __barTunningProcess__(self, SQLFRAME, dbconn, cursor, tunning_interval):
                
        #fecha inicial a partir de la fecha final (parametro "start_date")
        data_tuple_range_for_tunning = (
            (
                datetime.strptime(self.start_date, '%Y-%m-%d').date() - 
                pd.Timedelta(tunning_interval)
                ).strftime('%Y-%m-%d'), 
            self.start_date
            )
        
        #obtien obj. ray con los parametros tuneados por tipo de barra (dict)
        ray_object_list = [
            standard_bar_tunning.remote(
                                url = self.pathzarr, 
                                ticker = stock, 
                                num_bar = self.desired_bars, 
                                #### Optional Params ####
                                date_tuple_range = data_tuple_range_for_tunning
                                ) for stock in self.list_stocks
            ]
        
        
        #obten la lista de información con los diccionarios
        list_datasets =  ray.get(ray_object_list)
        
        #transforma los diccionarios en un pandas con los params de tunning x bar
        tunning_pandas = construct_pandas_tunning(list_datasets, self.list_stocks)
        
        print("::::> RUNNING: Writting TUNNING PARAMS into SQL Table for Bars")
        
        #llena la tabla única "BARS_TUNNING" con los params seleccionados x barra
        SQLFRAME.write_table_info(
                statement_= "INSERT INTO TUNNING.dbo.BARS_PARAMS_GLOBAL \
                    (tick_t,volume_t,dollar_t,stock) VALUES (?,?,?,?)", 
                            dbconn_= dbconn, 
                            cursor_= cursor, 
                            data_= tunning_pandas, 
                            bartype_= 'TUNNING',
                            )        
            
        
        print("<<<::::: BAR TUNNING SQL PROCESS FINISHED :::::>>>")
    
    def __barConstructionProcess__(self, SQLFRAME, dbconn, cursor, 
                                   volVersion, 
                                   fracdiffWindow,
                                   barrierWindow,
                                   barrierPrice,
                                   bar_group_factor,
                                   bar_grp_horizon):
        
        #lee la tabla de bars tunning | insumo de creación de barras
        tunning_sql_frame = SQLFRAME.read_table_info(
            statement = "SELECT * FROM [TUNNING].[dbo].BARS_PARAMS_GLOBAL", 
            dbconn_= dbconn, 
            cursor_= cursor, 
            dataframe=True
            )    
        
        #iteración por acción
        ray_object_list = [
            generate_datasets.remote(
                            stock = stock, 
                            bartypesList = [self.bartype], 
                            data_dir = self.pathzarr, 
                            range_dates = sel_days(
                                    self.start_date, self.end_date
                                    ), 
                            imbalance_dict = dict.fromkeys(
                                    self.list_stocks, 1
                                    ),
                            #### Optional Params ####
                            bar_grp_freq = bar_group_factor,
                            bar_grp_horizon = bar_grp_horizon,                        
                            alpha_calibration = tunning_sql_frame.query(
                                    "stock==@stock"),
                            volatility_version = volVersion,
                            window_application_fracdiff = fracdiffWindow,
                            window_application_horizon_barrier = barrierWindow,
                            limit_date_of_data_for_horizon = None,
                            data_application_volatility_fracdiff = barrierPrice,
                            ) 
            for stock in self.list_stocks
            ]
        
        #extraemos los objetos ray con los dataframe para la escritura en SQL
        list_datasets =  ray.get(ray_object_list)
    
        #primer bloque iterativo: loop sobre tipo de barra (si lo hubiere)
        for idx, dataset in enumerate(list_datasets):
            
            print("::::> RUNNING: Writting {} into SQL Table for BaseBars".format(
                    self.list_stocks[idx]
                    )
                )        
            
            #segundo bloque iterativo: loop por activo (si lo hubiere)
            for idx_bartype, dataframe in enumerate(dataset):    
                                
                statement = "INSERT INTO BARS.dbo.{}_{}_GLOBAL \
                    {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(
                    self.list_stocks[idx], self.bartype.upper(),
                    "(open_price,high_price,low_price,close_price,\
                        open_date,high_date,low_date,close_date,basic_volatility,\
                            bar_cum_volume,vwap,fracdiff,volatility,\
                                horizon,upper_barrier,lower_barrier)"
                    )    
                    
                #writting info into its SQL table
                SQLFRAME.write_table_info(
                        statement_= statement, 
                                dbconn_= dbconn, 
                                cursor_= cursor, 
                                data_= dataframe, 
                                bartype_= 'BARS',
                            )        
        
        print("<<<::::: BASIC BAR CONSTRUCTION SQL PROCESS FINISHED :::::>>>")
    
    def __barEntropyProcess__(self, SQLFRAME, dbconn, cursor,
                              beta, entropyWindow,
                              cumsum_sampling):
        
        #lee las tablas de barras por acción | insumo para añadir entropy column
        pandas_basic_bars = {stock:
            SQLFRAME.read_table_info(
                    statement= "SELECT * FROM [BARS].[dbo].{}_{}_GLOBAL".format(
                        stock, self.bartype.upper() 
                        ), 
                    dbconn_= dbconn, 
                    cursor_= cursor, 
                    dataframe=True
                ) 
            for stock in self.list_stocks}     
        
        
        #iteración por acción| siki admite un bartype único (no lista)
        ray_object_list = [
            entropyCalculationParallel.remote(
                                zarr_dir = self.pathzarr,
                                pandas_bar = pandas_basic_bars[stock],
                                stock = stock, 
                                #### Optional Params ####
                                beta = beta, 
                                entropy_window = entropyWindow, 
                                cumsum_sampling = cumsum_sampling
                                ) 
            for stock in self.list_stocks
            ]
        
        #extraemos los objetos ray con los dataframe para la escritura en SQL
        list_datasets =  ray.get(ray_object_list)
    
        #bloque iterativo único (no existe bloque de =! tipos de 'bartype')
        for idx, dataset in enumerate(list_datasets):
            
            print("::::> RUNNING: Writting {} into SQL Table for entropy".format(
                    self.list_stocks[idx]
                    )
                ) 
            
            #sentencia SQL (statement) para gestionar el guardado de la info
            statement = "INSERT INTO ENTROPY.dbo.{}_{}_GLOBAL \
                {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(
                self.list_stocks[idx], self.bartype.upper(),
                "(open_price,high_price,low_price,close_price,\
                    open_date,high_date,low_date,close_date,basic_volatility,\
                        bar_cum_volume,vwap,fracdiff,volatility,\
                            horizon,upper_barrier,lower_barrier, entropy)"
                )     
                
            #writting info into its SQL table
            SQLFRAME.write_table_info(
                        statement_= statement, 
                                dbconn_= dbconn, 
                                cursor_= cursor, 
                                data_= dataset, 
                                bartype_= 'ENTROPY',
                            )        
        
        print("<<<::::: BAR ENTROPTY CONSTRUCTION SQL PROCESS FINISHED :::::>>>")
    
    def __etfTrickProcess__(self, SQLFRAME, dbconn, cursor, 
                            kInit, lowerBoundIndex, 
                            allocationApproach):
        
        #lee las tablas de barras por acción | insumo para construir ETF TRICK
        pandas_basic_bars = [
            SQLFRAME.read_table_info(
                    statement="SELECT * FROM [BARS].[dbo].{}_{}_GLOBAL".format(
                        stock, self.bartype.upper()
                        ), 
                    dbconn_= dbconn, 
                    cursor_= cursor, 
                    dataframe=True
                ) 
            for stock in self.list_stocks]
        
        print("::::> RUNNING: Writting ETF TRICK 'NO SAMPLED' into SQL Table")
        
        #cómputo del ETF Trick (sin iteración)
        etf_pandas = etfTrick(
                    list_bars_stocks = pandas_basic_bars, 
                    stock_list = self.list_stocks, #takes all equities in a list
                    #### Optional Params ####
                    k= kInit, 
                    lower_bound_index = lowerBoundIndex, 
                    allocation_approach = allocationApproach,
                    output_type = None
                    )
        
        #recopila stocks-name desde el etf_pandas | omite 1era columna 'value'
        baseStockItems = etf_pandas.columns.values[1:]
        
        #convierte lista de stocks en un string único con las acciones
        stock_list = ",".join(str(e) for e in baseStockItems)
        
        #construye un solo string por cada stock para rellenado de sql
        signQuestionFillSQL = ",".join("?" for e in baseStockItems)
        
        #construye el statement con la información referida
        statement = "INSERT INTO ETFTRICK.dbo.ETF_TRICK_VOLUME_GLOBAL \
            (value,{}) VALUES (?,{})".format(stock_list, signQuestionFillSQL)

        cursor.executemany(
           statement,
           list(etf_pandas.itertuples(index=False, name=None))
        )
        cursor.commit()    
        
        print("<<<::::: ETF TRICK CONSTRUCTION SQL PROCESS FINISHED :::::>>>")
    
    def __samplingFeaturesProcess__(self, SQLFRAME, dbconn, cursor, 
                                    lagsDef, hBound):
        
        #extrae el pandas del ETF TRICK de la tabla SQL
        etf_pandas = SQLFRAME.read_table_info(
            statement="SELECT * FROM [ETFTRICK].[dbo].ETF_TRICK_{}_GLOBAL".format(
               self. bartype.upper()
                ), 
            dbconn_= dbconn, 
            cursor_= cursor, 
            dataframe=True
            )
        
        #utiliza df del ETF TRICK y añade col 'SADF' | entropy no habilitado
        etf_sadf_pandas = gettingSADF(
            etf_df = etf_pandas,
            #### Optional Params ####
            lags = lagsDef, 
            main_value_name = 'value'
            )
    
        #samplea el ETF Trick en los valores relevantes
        etf_sampled = getSamplingFeatures(
            base_df = etf_sadf_pandas, 
            main_column_name = 'sadf', #'entropy' or 'sadf'
            h_value = hBound,
            #### Optional Params ####
            select_events=True
            )
        
        print("::::> RUNNING: Writting ETF TRICK 'SADF' into ETFTRICK SQL Table")
        
        #recopila stocks-name desde el etf_pandas | omite 1era columna 'value'
        baseStockItems = etf_sampled.columns.values[1:]
        
        #convierte lista de stocks en un string único con las acciones
        stock_list = ",".join(str(e) for e in baseStockItems)
        
        #construye un solo string por cada stock para rellenado de sql
        signQuestionFillSQL = ",".join("?" for e in baseStockItems)        
        
        #statement para almacenamiento 
        statement = "INSERT INTO ETFTRICK.dbo.ETF_TRICK_SAMPLED_{}_GLOBAL \
               (value, {}) VALUES (?,{})".format(
               self.bartype.upper(), stock_list, signQuestionFillSQL  
               )
        
        #guardar la información del ETF SAMPLED TRICK en la tabla SQL correspd.
        cursor.executemany(
            statement, 
            list(etf_sampled.itertuples(index=False, name=None))
        )
        
        print("WARNING: don't interrupt | Writting Process continues...")     
        
        #abrimos las tablas en formato dataframe por acción desde SQL
        pandas_basic_bars = [
            SQLFRAME.read_table_info(
                    statement="SELECT * FROM [BARS].[dbo].{}_{}_GLOBAL".format(
                        stock, self.bartype.upper()
                        ), 
                    dbconn_= dbconn, 
                    cursor_= cursor, 
                    dataframe=True
                ) 
            for stock in self.list_stocks]
    
        #selección de eventos con "structural break" según SADF de c/ df org.
        pandas_sampled_bars = crossSectionalDataSelection(
            sampled_dataframe = etf_sampled, 
            list_stocks_bars = pandas_basic_bars,
            list_stocks = self.list_stocks
            )
        
        #bloque iterativo único escritura en tabla (no existe bloque 'bartype')
        for idx, dataset in enumerate(pandas_sampled_bars):
            
            print("::::> RUNNING: Writting {} into SQL Sampled Bars Table".format(
                self.list_stocks[idx])
                ) 
            
            statement = "INSERT INTO BARS_SAMPLED.dbo.{}_SAMPLED_{}_GLOBAL \
                {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(
                self.list_stocks[idx], self.bartype.upper(),
                "(open_price,high_price,low_price,close_price,\
                    open_date,high_date,low_date,close_date,basic_volatility,\
                        bar_cum_volume,vwap,fracdiff,volatility,\
                            horizon,upper_barrier,lower_barrier)"
                )
    
            #writting info into its SQL table
            SQLFRAME.write_table_info(
                statement_= statement, 
                dbconn_= dbconn, 
                cursor_= cursor, 
                data_= dataset, 
                bartype_= "BARS_SAMPLED",
                        )        
        print("<<<::::: SAMPLING SELECTION SQL PROCESS FINISHED :::::>>>")
    
    def __tripleBarrierProcess__(self, SQLFRAME, dbconn, cursor):
        
        #tabla de barras sampledas x acción | barrier 'price', 'label' y 'time'.
        pandas_sampled_bars_from_sql = {
            stock:
                SQLFRAME.read_table_info(
                    statement= "SELECT * FROM \
                        [BARS_SAMPLED].[dbo].{}_SAMPLED_{}_GLOBAL".format(
                        stock, self.bartype.upper()
                        ), 
                    dbconn_= dbconn, 
                    cursor_= cursor, 
                    dataframe= True
                ) 
            for stock in self.list_stocks}         
        
    
        #generación de la triple barrera a través de paralelización ray
        ray_object_list = [
            new_triple_barrier_computation.remote(
                sampled_df = pandas_sampled_bars_from_sql[stock],
                stock = stock,                        
                zarr_path = self.pathzarr,
                                  ) 
            for stock in self.list_stocks
            ]
        
        #extraemos los objetos ray con los dataframe para la escritura en SQL
        list_datasets =  ray.get(ray_object_list)
        
        #primer bloque iterativo: loop sobre tipo de barra (si lo hubiere)
        for idx, dataset in enumerate(list_datasets):
            
            print("::::> RUNNING: Writting {} with Barriers into SQL Table".format(
                    self.list_stocks[idx]
                    )
                )        
                    
            #SQL statement for write information
            statement = "INSERT INTO BARS_COMPLETED.dbo.{}_{}_GLOBAL \
                {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(
                self.list_stocks[idx], self.bartype.upper(),
                "(open_price,high_price,low_price,close_price,\
                    open_date,high_date,low_date,close_date,basic_volatility,\
                        bar_cum_volume,vwap,fracdiff,volatility,\
                            horizon,upper_barrier,lower_barrier, \
                                barrierPrice, barrierLabel, barrierTime)"
                )        
    
            #writting info into its SQL table
            SQLFRAME.write_table_info(
                        statement_= statement, 
                                dbconn_= dbconn, 
                                cursor_= cursor, 
                                data_= dataset, 
                                bartype_= 'BARS_COMPLETED',
                            )    
                    
        print("<<<::::: TRIPLE BARRIER COMPUTATION SQL PROCESS FINISHED :::::>>>")
        
        
    def __sampleWeights__(self, SQLFRAME, dbconn, cursor, decayfactor):
        
        #tabla de barras x acción con info. completa
        pandas_sampled_bars_from_sql = {
            stock:
                SQLFRAME.read_table_info(
                    statement= "SELECT * FROM \
                        [BARS_COMPLETED].[dbo].{}_{}_GLOBAL".format(
                        stock, self.bartype.upper()
                        ), 
                    dbconn_= dbconn, 
                    cursor_= cursor, 
                    dataframe= True
                ) 
            for stock in self.list_stocks}                
                
                
        for stock, dataframe in pandas_sampled_bars_from_sql.items():
            
            updatedDf = WeightsGeneration(dataframe).getWeights(
                decay_factor= decayfactor
                )
            
            print("::::> RUNNING: Writting {} Weighted Ver. into SQL Table".format(
                    stock
                    )
                )      
            
            #SQL statement for write information
            statement = "INSERT INTO BARS_WEIGHTED.dbo.{}_{}_GLOBAL \
                {} VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)".format(
                stock, self.bartype.upper(),
                "(open_price,high_price,low_price,close_price,\
                    open_date,high_date,low_date,close_date,basic_volatility,\
                        bar_cum_volume,vwap,fracdiff,volatility,\
                            horizon,upper_barrier,lower_barrier, \
                                barrierPrice, barrierLabel, barrierTime,\
                                    overlap, weight, weightTime)"
                )        
    
            #writting info into its SQL table
            SQLFRAME.write_table_info(
                        statement_= statement, 
                                dbconn_= dbconn, 
                                cursor_= cursor, 
                                data_= updatedDf, 
                                bartype_= 'BARS_WEIGHTED',
                            )                
            
        print("<<<::::: BAR WEIGHTs COMPUTATION SQL PROCESS FINISHED :::::>>>")
   
    def create_table_database(self, bars_tunning, bars_basic,
                              bars_entropy, etfs_trick, bars_sampled, 
                              bars_barrier, bars_weights, 
                              creation_database = True):
        
        
        #si la base de datos principal es "TUNNING", solo puede crear una tabla.
        if bars_tunning:
            
            #nombre de la base de datos
            bartype_database = 'TUNNING'
                
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear la tabla: si la tabla está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )            
            
            #redefinición del nombre de la tabla a crear
            stock_tablename = "BARS_PARAMS"
            
            #ejecución de método de creación de nueva tabla 
            SQLFRAME.create_new_table(
                    table_name = stock_tablename, 
                    dict_info = table_params_bars_tunning, 
                    dbconn_ = dbconn, 
                    cursor_= cursor
                    )
            
            dbconn.commit()
            dbconn.close()
    
        #si se ingestó como parámetro el proceso de 'construcción de barra'
        if bars_basic:    
                
            #nombre de la base de datos
            bartype_database = 'BARS'
            
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear la tabla: si la tabla está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )            
                
            #por cada nombre de acción en la lista de acciones 
            for stock_tablename in self.list_stocks: 
        
                #crea una tabla según el elemento para cada STOCK
                SQLFRAME.create_new_table(
                            table_name = stock_tablename + "_" + self.bartype.upper(), 
                            dict_info = table_standard_bar_structure, 
                            dbconn_ = dbconn, 
                            cursor_= cursor
                            )
            dbconn.commit()
            dbconn.close()
                    
        #si se ingresó como parámetro el proceso de 'cálculo de entropy'                
        if bars_entropy:
            
            #nombre de la base de datos
            bartype_database = 'ENTROPY'
                
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear la tabla: si la tabla está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )              
                
            #por cada nombre de acción en la lista de acciones
            for stock_tablename in self.list_stocks: 
        
                #crea una tabla según el elemento para cada STOCK
                SQLFRAME.create_new_table(
                            table_name = stock_tablename + "_" + self.bartype.upper(), 
                            dict_info = table_entropy_bar_structure, 
                            dbconn_ = dbconn, 
                            cursor_= cursor
                            )            
            dbconn.commit()
            dbconn.close()        
        
                
        #si se ingresó como parámetro el proceso de 'cálculo de ETF TRICK'         
        if etfs_trick:
            
            #nombre de la base de datos
            bartype_database = 'ETFTRICK'
                
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear tabla: si está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )      
                
            #lista de listas con [STOCK, TYPE] para escritura en tabla
            info_for_table_params = [
                    [stock, 'DATETIME'] for stock in self.list_stocks
                    ]
                
            #inserta columna universal ["value", FLOAT] para tabla
            info_for_table_params.insert(0, ['value', 'float'])
                
            stock_tablename = "ETF_TRICK" + "_" + self.bartype.upper()
                
            #ejecución de método de creación de nueva tabla 
            SQLFRAME.create_new_table(
                                table_name = stock_tablename,
                                dict_info = dict(info_for_table_params), 
                                dbconn_ = dbconn, 
                                cursor_= cursor
                                )         
                
            #lista de listas con [STOCK, TYPE] para escritura en tabla
            info_for_table_params = [
                        [stock, 'DATETIME'] for stock in self.list_stocks
                        ]
                    
            #inserta columna universal ["value", FLOAT] para tabla
            info_for_table_params.insert(0, ['value', 'float'])
                    
            #diccionario de información para escritura de tabla 
            dict_info_table = dict(info_for_table_params)
                    
            #adding 'SADF' column
            dict_info_table['SADF'] = 'float'
                    
                    
            stock_tablename = "ETF_TRICK_SAMPLED" + "_" + self.bartype.upper()
                    
            #ejecución de método de creación de nueva tabla 
            SQLFRAME.create_new_table(
                                    table_name = stock_tablename, 
                                    dict_info = dict_info_table, 
                                    dbconn_ = dbconn, 
                                    cursor_= cursor
                                    )                      
    
            dbconn.commit()
            dbconn.close()            
                
        #si se ingresó la solicitud de creación de tabla sampled x acción
        if bars_sampled:
            
            #nombre de la base de datos
            bartype_database = 'BARS_SAMPLED'
                
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear la tabla: si la tabla está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )      
                        
                
            #por cada nombre de acción en la lista de acciones
            for stock_tablename in self.list_stocks: 
    
                #crea una tabla según el elemento para cada STOCK
                SQLFRAME.create_new_table(
                            table_name = stock_tablename + "_SAMPLED" +
                            "_" + self.bartype.upper(), 
                            dict_info = table_standard_bar_structure, 
                            dbconn_ = dbconn, 
                            cursor_= cursor
                            )          
    
            dbconn.commit()
            dbconn.close()            
    
                    
        #si se ingresó solicitud de creación de tabla bars inc. TripleBarrier
        if bars_barrier:
            
            #nombre de la base de datos
            bartype_database = 'BARS_COMPLETED'
                
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear la tabla: si la tabla está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )      
                                
            #por cada nombre de acción en la lista de acciones 
            for stock_tablename in self.list_stocks: 
        
                #crea una tabla según el elemento para cada STOCK
                SQLFRAME.create_new_table(
                            table_name = stock_tablename + "_" + self.bartype.upper(), 
                            dict_info = table_barrier_bar_structure, 
                            dbconn_ = dbconn, 
                            cursor_= cursor
                            )            
                
            dbconn.commit()
            dbconn.close()                          
            
            
        if bars_weights:
            
            #nombre de la base de datos
            bartype_database = 'BARS_WEIGHTED'
                
            SQLFRAME, dbconn, cursor = databundle_instance(
                    #nombre del servidor SQL Local
                    server = self.server_name, 
                    #nombre que se le asignará a la base de datos matriz
                    bartype_database = bartype_database,
                    #boleano para crear la tabla: si la tabla está creada, debe ser False
                    create_database = creation_database, 
                    #nombre global para cada tabla | "GLOBAL" x defecto
                    global_range = self.global_range
                    )      
                                
            #por cada nombre de acción en la lista de acciones 
            for stock_tablename in self.list_stocks: 
        
                #crea una tabla según el elemento para cada STOCK
                SQLFRAME.create_new_table(
                            table_name = stock_tablename + "_" + self.bartype.upper(), 
                            dict_info = table_weighted_bar_structure, 
                            dbconn_ = dbconn, 
                            cursor_= cursor
                            )            
                
            dbconn.commit()
            dbconn.close()                          
                        
            
        return "SQL updating process finished." 

    def compute_info_to_sql(self, 
                            bars_tunning_process, 
                            bar_construction_process, 
                            entropy_construction_process,
                            etftrick_construction_process,
                            sampling_features_process,
                            triple_barrier_computation_process, 
                            sample_weight_computation_process,
                            #GENERAL NAME FOR DATABASE | SOLO REF.
                            bartype_database = 'BARS',
                            #TUNNING PROCESS PARAMS
                            tunning_interval = '21D', 
                            #BASIC BAR CONSTRUCTION PARAMS
                            volVersion = 'ver2', 
                            fracdiffWindow = 2,
                            barrierWindow = 1,
                            barrierPrice = 'close',
                            bar_group_factor = 1,
                            bar_grp_horizon = 'd',
                            #ENTROPY CONSTRUCTION PARAMS
                            beta = 0.02, 
                            entropyWindow = 50,
                            cumsum_sampling = True,
                            #ETF TRICK CONSTRUCTION PRAMS
                            kInit=10000, 
                            lowerBoundIndex = 50, 
                            allocationApproach = 'inv',
                            #SAMPLING FEATURES PROCESS PARAMS
                            lagsDef = None, 
                            hBound = 2.5,
                            #SAMPLE WEIGHT PROCESS PARAMS
                            decayfactor = 0.5):
        
        #valores de ejecución: instancia SQL, conexión a la base de datos, cursor
        SQLFRAME, dbconn, cursor = databundle_instance(
            #nombre del servidor SQL Local
            server = self.server_name, 
            #nombre que se le asignará a la base de datos matriz
            bartype_database = bartype_database ,
            #boleano para crear la tabla: en la inicialización, siempre será False
            create_database = False, 
            #nombre global para cada tabla | "GLOBAL" x defecto
            global_range = self.global_range
            )        
        
        #proceso de tunning de barras
        if bars_tunning_process:
            self.__barTunningProcess__(SQLFRAME, dbconn, cursor,
                                       tunning_interval)
        
        #proceso de construcción de barras básicas
        if bar_construction_process:
            self.__barConstructionProcess__(SQLFRAME, dbconn, cursor,
                                            volVersion, 
                                            fracdiffWindow,
                                            barrierWindow,
                                            barrierPrice,
                                            bar_group_factor,
                                            bar_grp_horizon)
            
        #proceso de construcción de variable/columnba entropy
        if entropy_construction_process:
            self.__barEntropyProcess__(SQLFRAME, dbconn, cursor, 
                                       beta, 
                                       entropyWindow,
                                       cumsum_sampling)
        
        #proceso de construcción de ETF Trick | sin sampling
        if etftrick_construction_process:
            self.__etfTrickProcess__(SQLFRAME, dbconn, cursor,
                                     kInit, 
                                     lowerBoundIndex, 
                                     allocationApproach)
        
        #proceso de construcción del sampling series | app. en ETF Trick y Bars
        if sampling_features_process:
            self.__samplingFeaturesProcess__(SQLFRAME, dbconn, cursor, 
                                             lagsDef,
                                             hBound)
        
        #proceso de construcción de la triple barrera
        if triple_barrier_computation_process:
            self.__tripleBarrierProcess__(SQLFRAME, dbconn, cursor)
            
        #proceso de construcción de los "sample weights"    
        if sample_weight_computation_process:
            self.__sampleWeights__(SQLFRAME, dbconn, cursor, 
                                   decayfactor = decayfactor)
        
        print("::::> UPDATED SQL INFORMATION FINISHED <::::")
        

