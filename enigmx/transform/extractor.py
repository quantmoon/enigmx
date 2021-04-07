"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import sys
import csv
import re
import os
import time
import warnings
import threading
import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
from urllib.parse import urlencode
from dateutil import parser
from pytz import timezone
import pandas_market_calendars as mcal
import finnhub
warnings.filterwarnings("ignore")



#Base Class for Data Extraction
class BaseExtractor(object):
    """"
    Clase principal para la extracción de data desde Finnhub.
    """
    #Define Trading Calendar from CSV | Deprecated
    def trading_calendar():
        """
        Gets trading days based on NYSE Calendar.
        """
        nyse = mcal.get_calendar('NYSE')
        early = nyse.schedule(start_date='2015-01-01', end_date='2021-04-28')
        dts = list(early.index.date)
        
        #transform as datetime.date() each string date
        return dts
    
    #Extract date available markets
    def extract_date_available_market(self, 
                                      start_, 
                                      end_, 
                                      trd_cal_= trading_calendar()):
        """
        Match input days with NYSE trading calendar days.
        """

        startDate=dt.datetime.strptime(start_,'%Y-%m-%d')
        endDate=dt.datetime.strptime(end_,'%Y-%m-%d')

        if startDate == endDate:

            list_pre = [startDate.date()]
            date = min(
                trd_cal_, key= lambda x: abs(x - list_pre[0])
            )

            if date == list_pre[0]:
                idx = [trd_cal_.index(date)]
                return [trd_cal_[idx[0]].strftime('%Y-%m-%d')]         
            else:
                print("No trading days at {}".format(
                    startDate.date())
                     )
                sys.exit()

        else:

            date = min(
                trd_cal_, key=lambda x: abs(x - startDate.date())
            )
            idx_1 = trd_cal_.index(date)

            date = min(
                trd_cal_, key=lambda x: abs(x - endDate.date())
            )
            idx_2 = trd_cal_.index(date)

            resulted_dates_range = trd_cal_[idx_1:idx_2+1]

            if len(resulted_dates_range)<1:
                print("No trading days in {} to {}".format(
                    startDate.date(), 
                    endDate.date())
                     )
                sys.exit()
            else:
                return [result_date_.strftime('%Y-%m-%d') 
                        for result_date_ in resulted_dates_range]  
    
    #Check if date available markets are correct
    def _check_dates(self, path, symbol,date_str):
        
        try:
            values = xr.open_zarr(path+"/"+symbol+'.zarr').date.values
        except:
            answer = 'continue'
        else:        
            for value in values:
                ts = pd.to_datetime(str(value)).strftime('%Y-%m-%d')
                if date_str == ts:
                    print('For:',symbol,',',date_str,'is already in storage')
                    answer='break'
                    break
                else: 
                    answer='continue'
        return answer

#Main Class for Data Extraction    
class Extractor(BaseExtractor):
    """
    This class summarize all the extraction process over zarr format file.
    """

    def __init__(self,list_stocks, start_date, end_date, 
                 path, limit = 25000, feature = 'tick',
                 api_key = "c04f66748v6u76cjieag",
                 length = 1000000,
                 chunk = 900000, 
                 stocks_file_info = "company_info_zarr.csv",
                 threads = 1, 
                 tupleTimeZone = (4,5) #for Cloud | (9,10) for local
                 ):
        
        self.finnhub_client = finnhub.Client(api_key=api_key)

        self.list_stocks = list_stocks #define list of stocks
        self.start_date = start_date #define start date as string
        self.end_date = end_date #define end date as string
        self.path = path #define path to storage as string
        self.feature = feature #define feature of data as string
        self.api_key = api_key #define Finnhub apikey as string
        self.url="https://finnhub.io/api/v1/stock/{}?".format(feature)
        self.final_dates = self._check_for_final_dates(self.start_date,
                                                       self.end_date)
        self.tupleTimeZone = tupleTimeZone #hrs range to fit TimeZone
        self.length = length
        self.threads = threads
        self.limit = limit
        self._parallelize_extraction(self.list_stocks)
        #self._extraction()
        
    #check if the start and end date are well-defined
    def _check_for_final_dates(self,start_date,end_date):
        
        final_dates = self.extract_date_available_market(
            start_date, 
            end_date)
        # Check if there is range of dates
        if final_dates == ["No Markets Days"]:
            raise "No Markets Days. Please input a different day"
        return final_dates
    
    #define initialization & finalization TIMESTAMPS
    def init_end_Mar_day(self, date_,init_, last_):
        _initialization = dt.datetime.timestamp(
                parser.parse(
                        date_ + " " + init_
                        )
                ) * (10**3)
        _finalization = dt.datetime.timestamp(
                parser.parse(
                        date_ + " " + last_
                        )
                ) * (10**3)                      
        return int(_initialization), int(_finalization)

    #compute data extraction based on dates
    def _data_extraction(self,symbol,date,api_key):
        #Define 3 listas que van a ser las variables del futuro Dataset
        timestamp=[]
        value=[]
        vol = []
        
        formatt = "%Y-%m-%d %H:%M:%S"
        
        
        init = '09:30:00'
        last = '16:00:00'  
        init = date + " " + init
        last = date + " " + last
        init = dt.datetime.strptime(init,formatt)
        last = dt.datetime.strptime(last,formatt)
        ts_init = time.mktime(init.timetuple())*1000
        ts_last = time.mktime(last.timetuple())*1000
        
        init2 = '08:30:00'
        last2 = '15:00:00' 
        init2 = date + " " + init2
        last2 = date + " " + last2
        init2 = dt.datetime.strptime(init2,formatt)
        last2 = dt.datetime.strptime(last2,formatt)
        ts_init2 = time.mktime(init2.timetuple())*1000
        ts_last2 = time.mktime(last2.timetuple())*1000
            
        #Define general params for extraction | URL construction
        for skip in range(0,self.length,25000):
            params=urlencode([       
                        ("symbol",symbol),
                        ("date",date),
                        ("token",api_key),
                        ("limit",25000),
                        ("skip",skip),
                        ("format","csv")])
            print("    >>>> download... {}".format(symbol))  
            print("       > skip {} rows".format(skip))            
            
            res = self.finnhub_client.stock_tick(symbol,date,self.limit,skip)
            df_ = pd.DataFrame(res)

            
            print("    >>>> download finished...{}".format(symbol))
            #Verifica si hay data en la última descarga (ya que la descarga
            #se hace cada 25 000 ticks).

            if df_['t'].shape[0] != 0:
            #Si hay data, los valores de timestamp, precio y volumen, se 
            #añaden a las listas que van a ser las variables del dataset
                
                if init.astimezone(timezone('US/Eastern')).hour == self.tupleTimeZone[0]:
                    df_ = df_[df_['t'] > ts_init]    
                    df_ = df_[df_['t'] < ts_last]
                    timestamp.extend(df_['t'].tolist())                
                    value.extend(df_['p'].tolist())
                    vol.extend(df_['v'].tolist())
                
                elif init.astimezone(timezone('US/Eastern')).hour == self.tupleTimeZone[1]:
                    df_ = df_[df_['t'] > ts_init2]
                    df_ = df_[df_['t'] < ts_last2]
                    timestamps = df_['t'].tolist()
                    timestamps = [i + 3600000 for i in timestamps]
                    timestamp.extend(timestamps)                
                    value.extend(df_['p'].tolist())
                    vol.extend(df_['v'].tolist())
                
                
            else:
                
            #Si ya no hay data, 
            #comienza a procesar lo descargado para su almacenamiento
            
            #Primero define la longitud de las coordenadas

                length = self.length #1000000

                ts_len =  length - len(timestamp)
                value_len = length - len(value)
                vol_len = length - len(vol)
                ts_zeros = np.zeros(ts_len)
                value_zeros = np.zeros(value_len)
                vol_zeros = np.zeros(vol_len,dtype='int')
                
                timestamp.extend(ts_zeros)
                value.extend(value_zeros)
                vol.extend(vol_zeros)        

                ts_final = np.zeros((1,len(timestamp)))
                value_final = np.zeros((1,len(value)))
                vol_final =np.zeros((1,len(vol)),dtype='int')
                
                date = [date]                
                ts_final[0,:] = timestamp
                value_final[0,:] = value
                vol_final[0,:] = vol
                coords = list(range(length)) 
                
                                                
                ts_arr = xr.DataArray(ts_final,coords=[date,coords],
                                      dims=['date','coords'])

                value_arr = xr.DataArray(value_final,coords=[date,coords],
                                         dims=['date','coords'])
                vol_arr = xr.DataArray(vol_final,coords=[date,coords],
                                       dims=['date','coords'])
                
                ds = xr.Dataset({"vol":vol_arr,
                        "timestamp":ts_arr,
                                "value":value_arr
                                 })  

                # Path construction   
                path_ = self.path + "/" + symbol + ".zarr"   
                symbol_dir = symbol + ".zarr"
                
                if symbol_dir not in os.listdir(self.path) :
                    ds.to_zarr(path_,consolidated=True)
                    print("    >>>> Saving Finished... {}".format(symbol))
                    break
                else:
                    
                    print("    >>>> Saving... ")
                    ds.to_zarr(path_,consolidated=True, append_dim='date')    
                    print("    >>>> Saving Finished... {}".format(symbol))
                    break
                
    #recursive method to ensure data is well-processed                
    def recursive_method(self,func):
        """
        Method that insist to download data.
        If the process gets an error, it wait some time and insist again.
        """
        while True:
            time.sleep(1)
            try:
                func()
            except:
                print(func())
                #print("Error")
                continue
            else:    
                break    
        return 
    
    #iterative Extraction Function from stock-list
    def _extraction(self,symbol_list):
        for stock in symbol_list:
            
            print("Stock: {}".format(stock))
            for date in self.final_dates:
                answer = self._check_dates(self.path,stock,date)
                if answer == 'break':
                    pass
                elif answer == 'continue':
                    print("Date: {}".format(date))
                    self.recursive_method(lambda: 
                        self._data_extraction(stock,date,self.api_key)
                        )
    
    #split lists of stocks based on 'threads'
    def _divide_lists(self,symbol_list):
        threads = self.threads
        list_len = len(symbol_list) // threads
        first = 0
        count = 0
        lists = []
        for i in range(0,threads + 1):
            count += list_len
            new_list = symbol_list[first:count]
            lists.append(new_list)
            first += list_len    
        return lists
    
    #extraction paralelization
    def _parallelize_extraction(self,symbol_list):
        lists = self._divide_lists(symbol_list)
        procs = []
        for i in lists:
            p = threading.Thread(target=self._extraction,args=(i,))
            p.start()
            procs.append(p)
        for i in procs:
            i.join()
        print("Extraction finished")
        
