"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import zarr
import pyodbc
import numpy as np

from enigmx.protofeatures import tickRuleVector

import pandas as pd
from itertools import chain
from enigmx.utils import (
    open_zarr_general, 
    check_zarr_data,
    sel_days,
    __volatility__,
    get_horizons,
    barrierCoords,
    __newTickBarConstruction__,
    __newVolumeBarConstruction__,
    __newDollarBarConstruction__,
    infoBarGenerator,
    )
from enigmx.tests.telegram import send_message

#1. Main Quantmoon SQL Manager
class QuantmoonSQLManager(object):
    """
    Clase SQL Manager. 
    
    Instancia general del Data Bundle.
    
    Métodos principales:
        - 'create_new_database'
        - 'select_database'
        - 'create_new_table'
        - 'read_table_info'
        - 'get_write_statement'
        - 'get_write_statement'
        - 'write_table_info'
        - 'globalSimpleInsert'
        
    Métodos interiores:
        - '__checkDatabase__'
        - '__getTables__'
        - '__checkTable__'
        - '__configurateTableStatement__'
        - '__configurateWrittingInfo__'
        
    """
    def __init__(self, server, database_name, driver,uid,pwd,
                 base_database='TSQL', globalRange=False,
                 loggin="",
                 access=""):
        
        self.server = server
        self.base_database = base_database

        self.globalRange = globalRange

        self.driver = driver
        self.uid = uid
        self.pwd = pwd
        
        self.database_name = database_name.upper()
        
        print(' ')
        print('Accessing SQL Info --->>>')
        

        # check if input is string (running local device) or tuple (running google cloud)
        if type(self.uid) == str and type(self.pwd) == str and type(self.driver) == str:
            print("|----- Local Process Detected -----|")
            self.loggin =  ('Driver='+self.driver+
                            ';Server='+ self.server +
                            ';Database='+self.base_database+
                            ';Uid='+self.uid+
                            ';Pwd='+self.pwd+';')
            
        else:
            print("|----- Cloud Process Detected -----|")
            self.loggin = ('Driver='+self.driver[0][0]+
                           ';Server='+ self.server +
                           ';Database='+self.base_database+
                           ';Uid='+self.uid[0]+
                           ';Pwd='+self.pwd[0]+';')

        # setting global Range sufix element
        if self.globalRange:
            self.global_ = '_GLOBAL'
        else:
            self.global_ = '_NOGLOBAL'
        
        # check if input is string (running local device) or tuple (running google cloud)
        if type(self.uid) == str and type(self.pwd) == str and type(self.driver) == str:
            self.access = ('Driver='+self.driver+
                           ';Server='+ self.server +
                           ';Database='+self.database_name+
                           ';Uid='+self.uid+
                           ';Pwd='+self.pwd+';')
        else:
            self.access = ('Driver='+self.driver[0][0]+
                           ';Server='+ self.server +
                           ';Database='+self.database_name+
                           ';Uid='+self.uid[0]+
                           ';Pwd='+self.pwd[0]+';')
        
    def __checkDatabase__(self):
        # revisar si la base de datos ya existe con ese nombre
        if self.database_name.lower() == self.base_database.lower():
            raise ValueError(
                "Database name's equivalent already exist. Change it."
                )
            
    def __getTables__(self, cursor): 
        # obtener las tablas seleccionadas
        statement = 'SELECT name FROM {}.sys.Tables'.format(
                                            self.database_name
                                            )        
        try:
            cursor.execute(statement)
        except:
            raise ValueError (
                "Check 'statement' or SQLDataBase Initialization"
                )
        else:
            return [db[0] for db in cursor] 
        
    def __checkTable__(self, cursor, table_name):
        # revisar si ya existen las tablas 
        self.tableName = table_name + self.global_
        return self.tableName in self.__getTables__(cursor)
    
    def __configurateTableStatement__(self, 
                                      table_name, 
                                      dictColVal):
        
        # configurar la tabla
        __base_statement__ = ('''CREATE TABLE {} ({})''')
        
        # generar sentencia para la config. de la tabla
        sentence_ = ('{} {},' * len(dictColVal))[:-1]
        table_elements = tuple(chain(*dictColVal.items()))
        tablestructure_ = sentence_.format(*table_elements)
        
        # construcción de nombre de la tabla
        self.tableName = table_name + self.global_
        
        # genera el statement para escritura de la tabla
        return __base_statement__.format(self.tableName, tablestructure_)
    
    def __configurateWrittingInfo__(self, database_name, 
                                    table_name, cursor_):
        
        # configurar nombre de la tabla para conf. escritura
        self.tableName = table_name + self.global_
        
        # realiza búsqueda de columnas según la tabla
        searchColumnsName = (
            "SELECT COLUMN_NAME FROM {}.INFORMATION_SCHEMA.COLUMNS " 
            "WHERE TABLE_NAME = N'{}'").format(
                            database_name, self.tableName
                            )
                
        # ejecuta el cursor con la búsqueda de columnas
        cursor_.execute(searchColumnsName)
        
        # extrae los nombres de las columnas
        SQLCOLNAMES = [element[0] for element in cursor_]
        
        # verifica la cant de columnas
        SQLCOLNUM = len(SQLCOLNAMES)
        
        # añade un '?' para agregado de elementos x columna
        base_values = ("?," * (SQLCOLNUM))[:-1]
        
        # define INSERT statement
        insert_ = "INSERT INTO {}.dbo.{}".format(
                                self.database_name, 
                                self.tableName
                                )
        
        # construye predicado de inserción
        _predicative_ = ("{}," * SQLCOLNUM)[:-1]
        predicate_ = _predicative_.format(*SQLCOLNAMES)
        
        # añade los valores a insertar
        values_ = "VALUES ({})".format(base_values)
        
        # define el statement general para usarse en SQL
        final_statement = "{} ({}) {}".format(
                                insert_, 
                                predicate_, 
                                values_
                                )
        return final_statement 
            
    def create_new_database(self):
        
        # método para crear una nueva base de datos
        
        # chequear que no exista previamente dicha base de datos
        self.__checkDatabase__()
        
        # loggearse a la base de datos
        dbconn = pyodbc.connect(self.loggin)
        
        # definición y ejecución del cursor
        cursor = dbconn.cursor()
        cursor.execute("SELECT name FROM master.dbo.sysdatabases")
        existence_ = [db[0].lower() for db in cursor]
        
        # extracción del nombre de la base de datos
        data_base_name = self.database_name
        
        # si la base de datos no existe, créala
        if data_base_name not in existence_:
            dbconn.autocommit = True
            cursor.execute("CREATE DATABASE [{}]".format(
                                                self.database_name
                                                )
                            )
            print("Database '%s' successfully created" % 
                                          self.database_name
                                          )
            # cierra el cursor y la conexión a la base de datos
            cursor.close()
            dbconn.close()
            
        # en caso exista la base de datos, interrumpe el proceso
        else:
            
            # cierra el cursor y la conexión a SQL
            cursor.close()
            dbconn.close()
            raise ValueError(
                "Database name's equivalent already exist. Change it."
                )
            
    def select_database(self):
        # método para seleccionar una base de datos
        print(":::::::::>> SQL Connection Path...")
        print(self.access)
        print(' ')

        try: 
            # conexión a la base de datos y creación de cursor
            dbconn = pyodbc.connect(self.access)
            cursor = dbconn.cursor()
            
        # en caso surga un error:
        except pyodbc.Error as ex:
            # imprime el error de conexión fallida
            print("Error statement appears:{}".format(ex))
            return "Circuit Breaker: Non SQL Connection initialized"
        
        # si todo marcha bien, retorna conexión base de datos y cursor
        return dbconn, cursor 
        
    def create_new_table(self, 
                         table_name, 
                         dict_info, 
                         dbconn_, 
                         cursor_):
        
        # método para crear una nueva tabla
        
        # chequea primero si la tabla no existe
        if self.__checkTable__(cursor_, table_name):
            # si ya existe con ese nombre, retorna error
            raise ValueError(
                    "Table's name already exists. "
                    "Change it or use 'write_table_info()' "
                    "method instead."
                )
        # caso contrario, genera el statement para crear la tabla
        statement = self.__configurateTableStatement__(
                                        table_name, dict_info
                                        )
        # intenta la creación de tabla
        try:
            cursor_.execute(statement)
            dbconn_.commit()
            
        # en caso error, muéstralo
        except pyodbc.Error as ex:
            print("Error statement computation: {}".format(ex))
            return "Breaks: Creating Table Process Failed"
        print("Table '{}' has been successfully created.".format(
                                                        self.tableName
                                                        )
                )
        
    def read_table_info(self, statement, dbconn_, cursor_, dataframe=False):
        # método para lectura de información de tabla
        
        # si no se quiere un dataframe como resultado
        if dataframe==False:
            # ejecuta la lectura del statement
            cursor_.execute(statement)     
            
            # retorna el cursor
            return cursor_
        
        #si se quiere un dataframe
        else:
            # lectura de la tabla
            sql_query = pd.read_sql_query(statement, dbconn_)
            
            # retorna la tabla
            return sql_query 
    
    def get_write_statement(self, database_name, 
                            table_name, cursor_):
        """
        Genera el statement (str) de escritura para tabla SQL.
        """
        return self.__configurateWrittingInfo__(
                                        database_name, 
                                        table_name, 
                                        cursor_
                                        )
        
    def write_table_info(self, statement_, 
                         dbconn_, cursor_, 
                         data_, bartype_):
        # método para escribir información en las tablas
        
        # dado un df, itera por cada fila y llénalo en la tabla selecccionada
        for idx, row in data_.iterrows():
            
            # si el tipo de tabla es TUNNING 
            if bartype_.upper() == 'TUNNING':
                cursor_.execute( 
                    statement_, 
                    row.tick_t, 
                    row.volume_t, 
                    row.dollar_t, 
                    row.stock
                    )
                
            # si el tipo de tabla es para las barras o barras sampleadas ### POSIBLE ERROR AQUI | MISMA COMPOSICION QUE BARS SAMPLED | NO AGREGA COLUMNA 'SADF'
            elif bartype_.upper()=='BARS' or  bartype_.upper()=='BARS_SAMPLED':
                cursor_.execute(
                    statement_, 
                    row.open_price, 
                    row.high_price, 
                    row.low_price, 
                    row.close_price, 
                    row.open_date, 
                    row.high_date, 
                    row.low_date, 
                    row.close_date, 
                    row.basic_volatility, 
                    row.bar_cum_volume, 
                    row.feat_buyInitTotal,
                    row.feat_sellInitTotal,
                    row.feat_signVolSide,
                    row.feat_accumulativeVolBuyInit,
                    row.feat_accumulativeVolSellInit,
                    row.feat_accumulativeDollarValue,
                    row.feat_hasbrouckSign,
                    row.vwap, 
                    row.fracdiff,
                    row.volatility, 
                    row.horizon,
                    row.upper_barrier,
                    row.lower_barrier,
                    row.bidask_spread,                    
                    )
            
            # si el tipo de tabla es para el cómputo de la entropía
            elif bartype_.upper() == 'ENTROPY':
                cursor_.execute(
                    statement_, 
                    row.open_price, 
                    row.high_price, 
                    row.low_price, 
                    row.close_price, 
                    row.open_date, 
                    row.high_date, 
                    row.low_date, 
                    row.close_date, 
                    row.basic_volatility, 
                    row.bar_cum_volume, 
                    row.feat_buyInitTotal,
                    row.feat_sellInitTotal,
                    row.feat_signVolSide,
                    row.feat_accumulativeVolBuyInit,
                    row.feat_accumulativeVolSellInit,
                    row.feat_accumulativeDollarValue,
                    row.feat_hasbrouckSign,                    
                    row.vwap, 
                    row.fracdiff,
                    row.volatility, 
                    row.horizon,
                    row.upper_barrier,
                    row.lower_barrier,                  
                    row.entropy,
                    row.bidask_spread,                      
                    )            
                
            # si el tipo de tabla es para el cómputo del etf trick
            elif bartype_.upper() == 'ETFTRICK':
                cursor_.execute(
                    statement_, 
                    row.value, 
                    row.high_price, 
                    row.low_price, 
                    row.close_price, 
                    row.open_date, 
                    row.high_date, 
                    row.low_date, 
                    row.close_date, 
                    row.basic_volatility, 
                    row.bar_cum_volume, 
                    row.feat_buyInitTotal,
                    row.feat_sellInitTotal,
                    row.feat_signVolSide,
                    row.feat_accumulativeVolBuyInit,
                    row.feat_accumulativeVolSellInit,
                    row.feat_accumulativeDollarValue,
                    row.feat_hasbrouckSign,                    
                    row.vwap, 
                    row.fracdiff,
                    row.volatility, 
                    row.horizon,
                    row.upper_barrier,
                    row.lower_barrier,          
                    row.entropy,
                    row.bidask_spread,                              
                    )               
                
            # si el tipo de tabla es para el cómputo de las barras completas (no finales)
            elif bartype_.upper() == 'BARS_COMPLETED':
                cursor_.execute(
                    statement_, 
                    row.open_price, 
                    row.high_price, 
                    row.low_price, 
                    row.close_price, 
                    row.open_date, 
                    row.high_date, 
                    row.low_date, 
                    row.close_date, 
                    row.basic_volatility, 
                    row.bar_cum_volume, 
                    row.feat_buyInitTotal,
                    row.feat_sellInitTotal,
                    row.feat_signVolSide,
                    row.feat_accumulativeVolBuyInit,
                    row.feat_accumulativeVolSellInit,
                    row.feat_accumulativeDollarValue,
                    row.feat_hasbrouckSign,                    
                    row.vwap, 
                    row.fracdiff,
                    row.volatility, 
                    row.horizon,
                    row.upper_barrier,
                    row.lower_barrier,                    
                    row.barrierPrice,
                    row.barrierLabel,
                    row.barrierTime,
                    row.bidask_spread,                              
                    )            
                
            # si el tipo de tabla es para el cómputo de las barras con weight y label (final)
            elif bartype_.upper() == 'BARS_WEIGHTED':
                cursor_.execute(
                    statement_, 
                    row.open_price, 
                    row.high_price, 
                    row.low_price, 
                    row.close_price, 
                    row.open_date, 
                    row.high_date, 
                    row.low_date, 
                    row.close_date, 
                    row.basic_volatility, 
                    row.bar_cum_volume, 
                    row.feat_buyInitTotal,
                    row.feat_sellInitTotal,
                    row.feat_signVolSide,
                    row.feat_accumulativeVolBuyInit,
                    row.feat_accumulativeVolSellInit,
                    row.feat_accumulativeDollarValue,
                    row.feat_hasbrouckSign,                    
                    row.vwap, 
                    row.fracdiff,
                    row.volatility, 
                    row.horizon,
                    row.upper_barrier,
                    row.lower_barrier,
                    row.barrierPrice,
                    row.barrierLabel,
                    row.barrierTime,
                    row.overlap,
                    row.weight,
                    row.weightTime,
                    row.bidask_spread,                              
                    )                         
            
                
            else: 
                raise ValueError(
                    "Not recognized database name. \
                        Normally bartype's name or 'TUNNING'. Please check."
                    )
                    
        dbconn_.commit()

        print("---Information was written---")
        
    def globalSimpleInsert(self, statement_, infoTuple, dbconn_, cursor_):
        
        # método simplificado para inserción de elementos genéricos en tabla
        for idx in range(0, infoTuple[0].shape[0]):
            cursor_.execute(
                statement_, 
                infoTuple[0][idx], 
                infoTuple[1][idx], 
                int(infoTuple[2][idx])
                )
        
        dbconn_.commit()
        print("Information '{}' was wrriten".format(
                                        self.database
                                        )
                                    )
        
##############################################################################

#2. Data Repository Initialization | General Class for Feature Construction                
class DataRespositoryInitialization(object):
    """


    Clase para el manejo del repositorio de datos Zarr a SQL. 
    
    Clase DataRepositoryInitialization:
        
    
    Métodos principales:
        - 'bar_novect_construction': permite construcción no vectorizada de las barras
        - 'generateTripleBarrier': DEPRECATED 
        - 'geneticIterativeFunction': DEPRECATED
        - 'general_matrix': DEPRECATED

    Métodos interiores:
        - '__makeTupleContent__'

    """
    def __init__(self, 
                 data_dir, 
                 stock, 
                 start_date = None, 
                 end_date = None,
                 check_days = True):
        
        self.data_dir = data_dir
        self.stock = stock
        self.start_date = start_date
        self.end_date = end_date
        self.check_days = check_days
        

        #conditional path dir format
        if self.data_dir[-1] == '/':
            self.pathFullZarr = self.data_dir + self.stock + ".zarr"
        else:
            self.pathFullZarr = self.data_dir + '/' + self.stock + ".zarr"
        
        #Get Zarr Base Object
        self.zarrObject = zarr.open(self.pathFullZarr)

        #get dates from object
        self.zarrDates = np.array(
            self.zarrObject.date)
        
        #Get information from the base DataRepository Instance
        
        #check if days are well-defined
        if check_days: 
                   
            if type(self.start_date) != str:
                raise ValueError(
                        'Only string type for start_date and end_date'
                    )
            else: 
                if self.end_date==None:
                    self.range_dates = [self.start_date]
                else:
                    self.range_dates = sel_days(
                                    self.start_date, 
                                    self.end_date
                                    )
        else:
            self.range_dates = [self.start_date]

        #Get ticks information of Time, Prices and Vol from Zarr Objects    
        self.infoTimePriceVol = open_zarr_general(
            self.zarrDates, self.range_dates, self.zarrObject, self.stock
            ) 

        #Check ticks information for removing/fixed missed values
        self.__stateResults__ = check_zarr_data(
            self.infoTimePriceVol
            )

    
    def __makeTupleContent__(self, freq, time):
        """
        Get relevant info to construct Marco's Bars.
        
        Inputs:
        ------
            freq (int>0): defines cuantitative horizon for get information.
            time (str = 'd'): defines the timeframework horizon for get 
                                information (only days is allowed).
            
            Lecture: given freq = 1 and time = 'd', we will construct
                     each BAR using one day information.
                     
        outputs (tuple of 6 elements):
        -----------------------------
            [0]groupDataTime: list of arrays containing datetime object for 
                              each datapoint per bar.
            [1]groupDataPrice: list of arrays containing prices values for 
                               each datapoint per bar.
            [2]groupDataVolume: list of arrays containing volume values for 
                                each datapoint per bar.
            [3]num_time_bars: int representing total bars over freq/time
                              horizon.
            [4]priceVol: 2D [PRICE, VOL] array by tick over freq/time horizon.
            
            [5]total_ticks: int given the total ticks over freq/time horizon.             
            
        Important:
            'priceVol' and 'total_ticks' are not useful by default to 
            calculate main bars. However, they might be useful for other
            purposes (like TripleBarrier computation).
        """
        
        #check time_vwap daily upper/lower case
        if time == 'd':
            time = time.upper()
        
        #get dates information from ticks
        datesOnly = self.__stateResults__[0].astype(
            "M8[{}{}]".format(
                        freq, time
                    )
            )
        
        #get tick rule over general 1d price vector
        main_tick_rule_vector = tickRuleVector(self.__stateResults__[1])
        
        #get general IdX info from dates
        generalIdxInfo = np.unique(
                            datesOnly, 
                            return_counts=True
                    )[1].cumsum()
        
        #OUTPUT 1: 'group_data' resample outputs by Time
        groupDataTime = np.split(
            self.__stateResults__[0], generalIdxInfo
            )[:-1]
        
        #OUTPUT 2: 'group_data' resample outputs by Price
        groupDataPrice = np.split(
            self.__stateResults__[1], generalIdxInfo
            )[:-1]
        
        #OUTPUT 3: 'group_data' resample outputs by Volume
        groupDataVolume = np.split(
            self.__stateResults__[2], generalIdxInfo
            )[:-1]
        
        #OUTPUT 4: 'group_data' resample outputs by TickRule
        groupTickRule = np.split(
            main_tick_rule_vector, generalIdxInfo
            )[:-1]
        
        #OUTPUT 4: num time bars
        num_time_bars = len(groupDataTime)
        
        #OUTPUT 5: 2D [[PRICE, VOL],...[PRICE, VOL]] array
        priceVol = np.vstack(
            (
                self.__stateResults__[1], 
                self.__stateResults__[2]
                )
            ).T 
        
        #OUTPUT 6: total ticks
        total_ticks = self.__stateResults__[1].shape[0]
        
        return ( #agregar el groupTickRule
            groupDataTime, 
            groupDataPrice, 
            groupDataVolume, 
            num_time_bars, 
            priceVol, 
            total_ticks,
            groupTickRule
            )
        

    def bar_novect_construction(self, 
                                 bartype, 
                                 info_tuple,
                                 imbalance_list,
                                 window_fracdiff = 2,
                                 daily_time_bars = 10):
        """
        'bartype' string with bartype name.

        'info_tuple' information by idx '__makeTupleContent__':
            [0]: groupDataTime (list of arrays), 
            [1]: groupDataPrice (list of arrays), 
            [2]: groupDataVolume (list of arrays), 
            [3]: num_time_bars (int), 
            [4]: priceVol (2D [PRICE, VOL] array by tick), 
            [5]: total_ticks (int)        
            
        'num_time_bars': define num time bars each global freq. 
        
        'window_fracdiff': define fracdiff horizon    
        
        As long as 'info_tuple' is list ordered by info,
        we will select:
            
            info_tuple[0][0] --> el único elemento
            
            Si:
                freq: 1
                time: 'd'
            
        """
        
        if bartype=='tick':
            
            #get list of column names for dataframe construction 
            
            #results_tuple info (last arg. is 'alpha_calibration')   
            result_info = __newTickBarConstruction__(
                info_tuple[0][0], 
                info_tuple[1][0], 
                info_tuple[2][0], 
                alpha_calibration=daily_time_bars
                )
            
            #elementos: [0] OHLC info (prices + dtimes + volatility), [1] vwap
            resultInfo = infoBarGenerator(
                result_info[0], result_info[1], result_info[2], bartype
                )
            
            
            #array OHLC list of list transformation & VWAP dim concadenation
            result_dataset = np.column_stack(
                [
                    np.array(resultInfo[0]), resultInfo[1]
                    ]
                )
            
                  
        elif bartype=='volume':
            
            #get list of column names for dataframe construction 
            
            #results_ tuple info (last arg. is 'alpha_calibration')   
            try:
                a=info_tuple[0][0]
            except:
                print(self.stock,"Gaaaaaaaaaaaa")

            result_info = __newVolumeBarConstruction__(
    	        info_tuple[0][0],
		info_tuple[1][0], 
                info_tuple[2][0],
                info_tuple[6][0],
                alpha_calibration=daily_time_bars)
                
            
            #elementos: [0] OHLC info (prices + dtimes + volatility), [1] vwap
            #computa tambien los proto-features: 7 en total
            resultInfo = infoBarGenerator(
                result_info[0], result_info[1], 
                result_info[2], result_info[3], 
                bartype #groupTickRule agrupado 
                )

            #array OHLC list of list transformation & VWAP dim concadenation
            result_dataset = np.column_stack(
                [
                    np.array(resultInfo[0]), resultInfo[1]
                    ]
                )
            
            
        elif bartype=='dollar':
            #get list of column names for dataframe construction 
            
            #results_ tuple info (last arg. is 'alpha_calibration')   
            result_info = __newDollarBarConstruction__(
                info_tuple[0][0], 
                info_tuple[1][0], 
                info_tuple[2][0], 
                alpha_calibration=daily_time_bars
                )
            
            #elementos: [0] OHLC info (prices + dtimes + volatility), [1] vwap
            resultInfo = infoBarGenerator(
                result_info[0], result_info[1], result_info[2], bartype
                )
            
            #array OHLC list of list transformation & VWAP dim concadenation
            result_dataset = np.column_stack(
                [
                    np.array(resultInfo[0]), resultInfo[1]
                    ]
                )

      
        # ---------------- pendiente ------------------------
        elif bartype == 'imbalance':
            raise ValueError(
                "Bartype 'imbalance' not currently available.\
                Check 'bar_novect_construction-databundle.py'."
                )
            
        return result_dataset 

    def generateTripleBarrier(self, 
                              dictInfo, 
                              window_volatility=1, 
                              window_horizon=1, 
                              tabular=False):
        """
        Deprecated Method to compute Triple Barrier.
        """
        
        # método para el cálculo de la triple barrera 
        
        #openPrice at each datapoint | price bars
        df_, concated_volbar = (
            dictInfo['special_time'], 
            dictInfo["volume"]
            )
        
        volatilities = __volatility__(
            concated_volbar, window=window_volatility
            ).shift(1).dropna()
        
        horizons_ = get_horizons(
            df_, window=window_horizon
            )
        
        horizons_.index = horizons_.index.date
        
        horizons_ = horizons_.apply(
            lambda x: x.date()
            ) 
    
        df_.index = df_.index.date
        df_ = pd.concat([df_,volatilities],axis=1)
            
        df_ = df_.assign(
            upper= lambda x: (
                x.price * (1 + x.volatility*2)
            ), 
            lower= lambda x: (
                x.price * (1 - x.volatility*2)
            )
        )
        
        df_ = pd.concat(
            [
                df_, horizons_.to_frame("horizon")
            ],
            axis=1
        ).dropna()
        
        tripleBarrier = barrierCoords(
            path = self.data_dir+"/"+ self.stock +".zarr", 
            initHorizon = df_.index.values,
            endHorizon = df_.horizon.values,
            upperBound = df_.upper.values,
            lowerBound = df_.lower.values
            )
        
        if tabular!=True:
            
            uppers = df_.upper.values
            lowers = df_.lower.values
            
            barrierPrices, barrierTs = list(zip(*tripleBarrier))
            barrierPrices = np.array(barrierPrices)
            
            uppers_ = uppers[barrierPrices!=0]
            lowers_ = lowers[barrierPrices!=0]
            barrierPrices_ = barrierPrices[barrierPrices!=0]
            
            aboveUpper = (barrierPrices_ > uppers_).astype('int64')
            belowLower = (barrierPrices_<lowers_).astype('int64')*-1
            
            tripleBarrier = aboveUpper + belowLower
            
            barrierPrices[barrierPrices!=0] = tripleBarrier
            
            return barrierPrices
    
        else:
            df_[["dataPrice", "dataTime"]] = list(tripleBarrier)
            
            df_["tripleBarrier"] = df_.apply(
                lambda x: 
                    1 if x.dataPrice > x.upper else (
                        -1 if (
                            x.dataPrice < x.lower and x.dataPrice != 0.0
                            ) else 0
                    ), axis=1
                )
            
            return df_
        
        
    def geneticIterativeFunction(self, 
                                 freq, 
                                 time,
                                 bartype,
                                 imbalance_list, 
                                 window_fracdiff = 2,
                                 daily_time_bars_organization = None):
        
        """
        Deprecated method to organize an iterative bar-no-vect-construction.
        """
        
        # método para computar las barras | computa el contenido esencial
        info_tuple = self.__makeTupleContent__(freq, time)
        
        # define la frecuencia de las barras a computarse
        daily_time_bars = info_tuple[3]
        
        # revisa si se define una seleccion de barras concreta por dia para redf.
        if daily_time_bars_organization != None:
            daily_time_bars = daily_time_bars_organization
                    
        # inicializa la generacion de las barras (metodo no vectorizado)
        barConstructed = self.bar_novect_construction(
                bartype, 
                info_tuple,
                imbalance_list,
                window_fracdiff,
                daily_time_bars
                )
        
        return barConstructed    
    
    def general_matrix(self, 
                       time, 
                       freq,
                       dataframes_dict,
                       window_volatility=1,
                       window_horizon=1,
                       window_fracdiff=2,
                       tabular_output=False):
        """
        
        DEPRECATED METHOD FOR GENERAL MATRIX CONSTRUCTION
        
        Método para construir una matriz general de eventos x día.
        
        dictInfo results:
            - Fracdiff: daily Fracdiff based on VWAP each day
            - tick: total tick bars in a specific day
            - volume: total volume bars in a specific day
            - time: price VWAP datapoint per day (to wavelet)
            - special_time: N freq Min/Sec series for TripleBarrier 
            
            The "special_time" should be changed to a Matrix Search:
                La idea central es realizar la búsqueda en cada 
                horizonte temporal de la triple barrera y detener
                dicha búsqueda apenas se cumpla una condición dada:
                    1.- Se alcance la barrera horizontal superior
                    2.- Se alcance la barrera horizontal inferior
                    
                Recuerde que 'special_time' = open_price
            
                Primero implementar tal cual está en el notebook
                y luego modificar esta parte.
        
        """
        
        # método depreciado para construcción de eventos x día inc. barras
        dictInfo = dataframes_dict
        
        resultBarrier = self.generateTripleBarrier(
            window_volatility=window_volatility, 
            window_horizon=window_horizon,
            dictInfo=dictInfo, 
            tabular=tabular_output
            )
        
        dictInfo["volume"]['datetime']=dictInfo["volume"]['datetime'].dt.date
        dictInfo["volume"] = dictInfo["volume"].groupby(["datetime"]).last(
            )[["grpId"]]
        
        dictInfo["volume"].columns,dictInfo["tick"].columns = (
            ["grpIdVol"], ["grpIdTick"]
            )
        
        return pd.concat(
            [
                resultBarrier, 
                dictInfo["tick"], 
                dictInfo["volume"], 
                dictInfo["fracdiff"]
                ], axis=1
            ).dropna()
    
            
                    
                    
            
        
            
            
    
    
