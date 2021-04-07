"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import zarr
import pyodbc
import numpy as np
import pandas as pd
from itertools import chain
from fracdiff import StationaryFracdiff
from enigmx.utils import (
    open_zarr_general, 
    check_zarr_data,
    sel_days,
    forwardFillOneDimension,
    vectorizedSimpleVwap,
    volumeBarConstruction,
    tickBarConstruction,
    __volatility__,
    get_horizons,
    __vectorizedVwap__,
    __tickBarConstruction__,
    __volumeBarConstruction__,
    simpleBarVolume,
    simpleBarTick,
    imb_feat,
    barrierCoords,
    iterativeVwap,
    __newTickBarConstruction__,
    __newVolumeBarConstruction__,
    __newDollarBarConstruction__,
    infoBarGenerator,
    barsNameDefinition
    )

#1. Main Quantmoon SQL Manager
class QuantmoonSQLManager(object):
    
    def __init__(self, server, database_name, 
                 base_database='TSQL', 
                 vwap=False, globalRange=False, 
                 remote=False, loggin='', access=''):
        
        self.server = server
        self.base_database = base_database
        self.vwap = vwap
        self.globalRange = globalRange
        self.remote = remote
        
        self.loggin = ('Driver={SQL Server};Server='+server+
                       ';Trusted_Connection=yes;')
        
        if self.vwap:
            self.database_name = database_name.upper() + '_VWAP'
        else:
            self.database_name = database_name.upper() + '_NOVWAP'
            
        if self.globalRange:
            self.global_ = '_GLOBAL'
        else:
            self.global_ = '_NOGLOBAL'
            
            
        self.access = ('Driver={SQL Server};Server='+ server +
                       ';Database='+ self.database_name +
                       ';Trusted_Connection=yes;')
        
    def __checkDatabase__(self):
        if self.database_name.lower() == self.base_database.lower():
            raise ValueError(
                "Database name's equivalent already exist. Change it."
                )
            
    def __getTables__(self, cursor): 
        
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

        self.tableName = table_name + self.global_
        return self.tableName in self.__getTables__(cursor)
    
    def __configurateTableStatement__(self, 
                                      table_name, 
                                      dictColVal):
        
        __base_statement__ = ('''CREATE TABLE {} ({})''')
        
        sentence_ = ('{} {},' * len(dictColVal))[:-1]
        table_elements = tuple(chain(*dictColVal.items()))
        tablestructure_ = sentence_.format(*table_elements)
        
        self.tableName = table_name + self.global_
        
        return __base_statement__.format(self.tableName, tablestructure_)
    
    def __configurateWrittingInfo__(self, database_name, 
                                    table_name, cursor_):
        
        self.tableName = table_name + self.global_
        
        searchColumnsName = (
            "SELECT COLUMN_NAME FROM {}.INFORMATION_SCHEMA.COLUMNS " 
            "WHERE TABLE_NAME = N'{}'").format(
                            self.database_name, self.tableName
                            )

                
        cursor_.execute(searchColumnsName)
        SQLCOLNAMES = [element[0] for element in cursor_]
        SQLCOLNUM = len(SQLCOLNAMES)
        
        base_values = ("?," * (SQLCOLNUM))[:-1]
        
        insert_ = "INSERT INTO {}.dbo.{}".format(
                                self.database_name, 
                                self.tableName
                                )
        
        _predicative_ = ("{}," * SQLCOLNUM)[:-1]
        predicate_ = _predicative_.format(*SQLCOLNAMES)
        
        values_ = "VALUES ({})".format(base_values)
        
        final_statement = "{} ({}) {}".format(
                                insert_, 
                                predicate_, 
                                values_
                                )
        return final_statement 
            
    def create_new_database(self):
        
        self.__checkDatabase__()
        
        dbconn = pyodbc.connect(self.loggin)
        cursor = dbconn.cursor()
        cursor.execute("SELECT name FROM master.dbo.sysdatabases")
        existence_ = [db[0].lower() for db in cursor]
        
        data_base_name = self.database_name
        
        if data_base_name not in existence_:
            dbconn.autocommit = True
            cursor.execute("CREATE DATABASE [{}]".format(
                                                self.database_name
                                                )
                            )
            print("Database '%s' successfully created" % 
                                          self.database_name
                                          )
            cursor.close()
            dbconn.close()
        else:
            cursor.close()
            dbconn.close()
            raise ValueError(
                "Database name's equivalent already exist. Change it."
                )
            
    def select_database(self):
        
        try: 
            dbconn = pyodbc.connect(self.access)
            cursor = dbconn.cursor()
        except pyodbc.Error as ex:
            print("Error statement appears:{}".format(ex))
            return "Circuit Breaker: Non SQL Connection initialized"
        return dbconn, cursor 
        
    def create_new_table(self, 
                         table_name, 
                         dict_info, 
                         dbconn_, 
                         cursor_):

        if self.__checkTable__(cursor_, table_name):
            raise ValueError(
                    "Table's name already exists. "
                    "Change it or use 'write_table_info()' "
                    "method instead."
                )
        
        statement = self.__configurateTableStatement__(
                                        table_name, dict_info
                                        )
        
        try:
            cursor_.execute(statement)
            dbconn_.commit()
        except pyodbc.Error as ex:
            print("Error statement computation: {}".format(ex))
            return "Breaks: Creating Table Process Failed"
        print("Table '{}' has been successfully created.".format(
                                                        self.tableName
                                                        )
                )
        
    def read_table_info(self, statement, dbconn_, cursor_, dataframe=False):
        
        if dataframe==False:
            cursor_.execute(statement)     
            return cursor_
        else:
            sql_query = pd.read_sql_query(statement, dbconn_)
            return sql_query 
    
    def get_write_statement(self, database_name, 
                            table_name, cursor_):
        
        return self.__configurateWrittingInfo__(
                                        database_name, 
                                        table_name, 
                                        cursor_
                                        )
        
    def write_table_info(self, statement_, 
                         dbconn_, cursor_, 
                         data_, bartype_,
                         vwap_):
        
        for idx, row in data_.iterrows():
            
            if bartype_=='time' and vwap_:
                cursor_.execute(
                    statement_, row.datetime, row.price
                    )
            elif bartype_=='time' and vwap_==False:
                cursor_.execute(
                    statement_, row.datetime, row.price, row.vol
                    )     
            elif bartype_=='tick' and vwap_:
                cursor_.execute(
                    statement_, row.grpId, row.datetime, row.price
                    )
            elif bartype_=='tick' and vwap_==False:
                cursor_.execute(
                    statement_, row.datetime, row.price
                    )                
            elif bartype_=='fracdiff':
                cursor_.execute(
                    statement_, row.datetime, row.fracdiff
                    )
        dbconn_.commit()
        #cursor_.close()
        print("Information '{}' was written".format(
                                        self.database_name
                                        )
                                    )
        
    def globalSimpleInsert(self, statement_, infoTuple, dbconn_, cursor_):
        
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
        self.zarrObject = zarr.open(
            self.data_dir+"/"+ self.stock +".zarr"
            )
        
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
            self.zarrDates, self.range_dates, self.zarrObject
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
        
        return (
            groupDataTime, 
            groupDataPrice, 
            groupDataVolume, 
            num_time_bars, 
            priceVol, 
            total_ticks
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
            #colNames = barsNameDefinition(bartype)
            
            #results_ tuple info (last arg. is 'alpha_calibration')   
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
            
            
            
            #print(result_dataset, type(result_dataset))            
            #print(caca)
            
            #dataframe computation | first only OHLC info
            #result_dataset = pd.DataFrame.from_records(
            #    resultInfo[0], 
            #    columns=colNames
            #    )
            

            
            #adding vwap bar information 
            #result_dataset[bartype + "_vwap"] = resultInfo[1]
                  
        elif bartype=='volume':
            
            #get list of column names for dataframe construction 
            #colNames = barsNameDefinition(bartype)
            
            #results_ tuple info (last arg. is 'alpha_calibration')   
            result_info = __newVolumeBarConstruction__(
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
                                    
            #dataframe computation | first only OHLC info
            #result_dataset = pd.DataFrame.from_records(
            #    resultInfo[0], 
            #    columns=colNames
            #    )
            
            #adding vwap bar information  
            #result_dataset[bartype + "_vwap"] = resultInfo[1]
            
            
        elif bartype=='dollar':
            #get list of column names for dataframe construction 
            #colNames = barsNameDefinition(bartype)
            
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
                        
            
            #dataframe computation | first only OHLC info
            #result_dataset = pd.DataFrame.from_records(
            #    resultInfo[0], 
            #    columns=colNames
            #    )
            
            #adding vwap bar information  
            #result_dataset[bartype + "_vwap"] = resultInfo[1]            
      
        #PENDIENTE!!!!!!!    
        elif bartype == 'imbalance':
            raise ValueError(
                "Bartype 'imbalance' not currently available.\
                Check 'bar_novect_construction-databundle.py'."
                )
            
            #result_dataset = imb_feat(
            #                    info_tuple[1][0], 
            #                    info_tuple[2][0], 
            #                    imbalance_list
            #                )
            
        return result_dataset 

    def generateTripleBarrier(self, 
                              dictInfo, 
                              window_volatility=1, 
                              window_horizon=1, 
                              tabular=False):
    
        
        #openPrice at each datapoint | price bars
        #precio open de cada día | "result_dataset = info_tuple[1][0][0]" 
        #no útil porque se tiene el precio open/close de cada barra
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
        
        #ya ejecuta la apertura del zarr
        
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
        
        info_tuple = self.__makeTupleContent__(freq, time)
        
        daily_time_bars = info_tuple[3]
        
        if daily_time_bars_organization != None:
            daily_time_bars = daily_time_bars_organization
                    
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
        #important: 
        #'parse_dates = [COLUMN OF DATES]' 
        #in 'read_csv' allows to transform str dates to datetime        
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
    
            
                    
                    
            
        
            
            
    
    