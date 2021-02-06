"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

from abc import ABCMeta, abstractmethod
from .event import MarketEvent
import numpy as np
import ray 

##################################
from enigmx.utils import compute_vwap_alternative
from datetime import datetime
import pandas as pd
import sys
from numba import njit

@njit
def slicing_data(info_tuple, initialization_step, step):
    ts_ = info_tuple[0]
    range_ = np.where(
                (ts_ >= initialization_step) & 
                (ts_ <initialization_step+step)
            )[0]

    finalTS = ts_[range_[0]:range_[-1]]
    finalPrice = info_tuple[1][range_[0]:range_[-1]]
    finalVol = info_tuple[2][range_[0]:range_[-1]]
    
    return finalTS, finalPrice, finalVol

def pandas_dataset_constructor(variables_tuple, freq):
    
    ts_dt = [datetime.fromtimestamp(t) 
            for t in variables_tuple[0]/10**3]
    
    price_ = variables_tuple[1]
    vol_ = variables_tuple[2]
    
    df_ = pd.DataFrame({
                'value':price_,
                'vol':vol_,
                },index=ts_dt
            )
    
    resampling='{}Min'.format(freq)
    group_data = df_.resample(resampling)
    num_time_bars = group_data.ngroups 
    
    return group_data, num_time_bars, df_

@ray.remote
def gen_bars(lazy_zarr_dataset, #tuple
             date,
             bartype,
             initialization_step,
             step,  
             freq=1,
             actv_dfwap=True):
    
    if lazy_zarr_dataset[0].shape[0] == 0: 
        print('No stored data for this stock/heartbeat') 
        sys.exit()

    
    segmentation = lazy_zarr_dataset #tuple

    slicing = slicing_data(segmentation, 
                           initialization_step, 
                           step) #tuple

    dataframe = pandas_dataset_constructor(slicing, 
                                           freq) #tuple
    
    group_data = dataframe[0] 
    num_time_bars = dataframe[1]

    
    if bartype=='time':
        if actv_dfwap==False:
            #create mean price for each sample of some 'freq'
            result_dataset=group_data.mean()
            
        else:
            #create vwap value for each sample of some 'freq'
            data_pricetime = (
                dataframe[2].value * dataframe[2].vol
                ).resample(
                    "{}Min".format(freq)
                    ).sum() / group_data.vol.sum()
            
            result_dataset = pd.DataFrame({'price':data_pricetime})
            
        #construction of final DataFrame
        return zip(result_dataset.index.values, 
                   result_dataset.price.values) 

    #Tick Bars
    elif bartype=='tick':
            #Get total of ticks and define num ticks per bar
        
        total_ticks = len(slicing[0])
        num_ticks_per_bar = total_ticks/num_time_bars 
        num_ticks_per_bar = round(num_ticks_per_bar, -1)
        n_coo_ = []
        
        for i in range(num_time_bars-1): 
            n_coo_ += [i]*int(num_ticks_per_bar) 
        n_coo_=n_coo_+[i+1]*(total_ticks-len(n_coo_))  
        
            #Conditional output
        if actv_dfwap==False:
            #returns the original xarray.Dataset 
            #not for storage; just for print
            #output format: xarray.Dataset
            
            index = group_data.first().index
            value = group_data.mean().value
            result_dataset = pd.DataFrame(data=value,index=index)

        else:
            #returns the vwap computed from new/vol/bar
            #output format: pd.DataFrame
            
            #'Groupdata' using new/vol/bar coord type 'ts_new=n_coo_' 
            index = group_data.first().index
            #print(dataframe[2])
            df = dataframe[2]
            
            df['group'] = n_coo_
            df = df.groupby(['group'])
            value = df.apply(compute_vwap_alternative)
            result_dataset = pd.DataFrame(data={
                                    'price':value.values
                                    }, 
                                index=index
                                )

        return zip(result_dataset.index.values, 
                   result_dataset.price.values) 



class DataHandler(object): 
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars for each symbol requested. 

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated equally by the rest of the backtesting suite.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")
        


class HistoricDataHandler(DataHandler):
    """
    HistoricDataHandler is designed to read Zarr Files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface. 
    """

    def __init__(self, events,symbol_list,bartype,init,last,step):
        """
        Initialises the historic data handler by requesting
        the location of the NC files and a list of symbols.
        
        IMPORTANT:
        It will be assumed that all files are of the form
        'symbol.nc', where 'symbol' is a string in the list.

        Parameters:
        -----------
        -  events : The Event Queue.
        -  nc_dir : Absolute directory path where zarr files are stored.
        -  symbol_list: A list of symbol strings
        
        ################ from our version ############################
        
        -  need: requirement for the code-structure. 
                 Could be 'backtest' for backtesting.
                 Could be 'live' for live-trading.
        - bartype: type of bar in which the backtesting will work.
        - step: heart-beat timestamp.
        - init: first timestamp period for bar (computed in milisecond)
        - last: last timestamp period for bar (computed in milisecond)
        
        Output:
        --------
        Updated Bar Dataset (pd.DataFrame)
        
        """
        self.events = events
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        
        for symbol in self.symbol_list:
            self.latest_symbol_data[symbol] = [] 
        
        self.bartype=bartype

        
    def _open_convert_csv_files(self, bartype, 
                                heartbeat, step, freq, 
                                vwap, date, dict_symbol):
        """
        Opens the NC files from the path directory, converting
        them into pandas DataFrames within a symbol dictionary.
        """
        #IN CASE OF RAY 'CORE.PY' USE 'ENUMERATE'
        temp_result = [gen_bars.remote(
                                dict_symbol[stock], 
                                date,
                                bartype, 
                                heartbeat, 
                                step, 
                                freq, 
                                vwap
                            ) 
                        for stock in self.symbol_list]
        
        assets_data = ray.get(temp_result)
        
        for stock, datazip in zip(self.symbol_list, assets_data):
            self.latest_symbol_data[stock].extend(datazip)
   
        self.events.put(MarketEvent())
            
            
### Some changes are required
### For ML, create 3 data selection type: 
### simple cross-validation, general kfold, adjusted kfold
        
    def get_latest_bars(self, symbol, N=1):
            """
            Return a list of tuples of time_position and bar price.
            """
            try:
                bars_list = self.latest_symbol_data[symbol]
            except KeyError:
                print ("\
                Symbol not available in the historical data set")
                return ("Process Stopped")
            else:
                return bars_list[-N:] 
    
    def get_latest_bar_datetime(self, symbol, N=1):
            """
            Return a matrix of position bar time.
            
            Se supone que debe retornar un datetime!!!!!
            """
            try:
                bars_list = self.get_latest_bars(symbol, N)
            except KeyError:
                print("\
                Symbol not available in the historical data set.")
                raise
            else:
                return np.array([bar[0] for bar in bars_list]) #already changed
            
    def get_latest_bars_values(self, symbol, N=1):
            """
            Return matrix of bar prices.
            """
            try:
                bars_list = self.get_latest_bars(symbol, N)
            except KeyError:
                print("\
                Symbol not available in the historical data set.")
                raise
            else:
                return np.array([bar[1] for bar in bars_list]) #already changed
            
    def get_bar_dataset(self, symbol, N=1):
            """
            Return a matrix of 2D: (time, price)
            """
            try:
                price_ = self.get_latest_bars_values(symbol, N)
                time_ = self.get_latest_bar_datetime(symbol, N)
            except KeyError:
                print("\
                Symbol not available in the historical data set.")
                raise
            else:
                print("variable")
                print(time_,type(time_[0]))
                print(price_)
                print("**"*15)
                return np.stack([price_, time_])

    
    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            while True:
                try:
                    # Assign on each row the values of 'latest_symbol_data'
                    # as a constant updating process.
                    bar = [s,next(self.symbol_data[s])[1]]
                    self.latest_symbol_data[s].append(bar)
                except StopIteration:
                    break
        #self.events.put(MarketEvent())