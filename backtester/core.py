"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

#import ray
import sys
import queue
import warnings
from dateutil import parser
from datetime import datetime
from enigmx.backtester.iterator import Iterator
from enigmx.utils import sel_days, open_zarr
from enigmx.backtester.portfolio import NaivePortfolio
from enigmx.backtester.data import HistoricDataHandler
from .event import (MarketEvent, SignalEvent, 
                    OrderEvent, FillEvent, IterationEvent)
from enigmx.backtester.metrics import plot_final_diagram
#from enigmx.backtester.strategy import BuyAndHoldStrategy
from enigmx.backtester.execution import SimulatedExecutionHandler

#@ray.remote
#def accelerated_open_zarr(path, date, stock_list):
#    return [open_zarr(path+"/"+ s +".zarr", 
#                      date)
#                for s in stock_list]


class Backtest(object):
    """
    Enscapsulates the settings and components for carrying out
    an event-driven backtest.
    """
    def __init__(self, 
                 data_dir, 
                 symbol_list, 
                 start_date,
                 end_date,
                 data_bundle = None,
                 step_heartbeat = 22500,
                 bartype = "tick",
                 freq = 1,
                 initialization_step = '09:30:00',
                 finalization_step = '16:00:00',
                 initial_capital = 10000,
                 vwap = True,
                 strategy = None,
                 data_handler = HistoricDataHandler, 
                 execution_handler = SimulatedExecutionHandler,
                 portfolio = NaivePortfolio,               
                 iterator = Iterator
                ):
        """
        Initialises the backtest iteration loop.
        
        Parameters:
        -----------
        
        - data_dir (str): the hard root to the data directory.
        - symbol_list (list of str): the list of symbol strings.
        - intial_capital (float): the starting capital for the portfolio.
        - heartbeat (int/float): backtest "heartbeat" in seconds.
        - start_date (datetime): the start datetime of the strategy.
        - data_handler (Class): market data feed HandleProces.
        - execution_handler (Class): orders/fills for trades HandlesProces.
        - portfolio (Class): tracker of portfolio current/prior positions.
        - strategy (Class): Generates signals based on market data.
        
        Output:
        --------
        
        Depends on callable method.
        
        Normally:
        
        'smiulate_trading()'.
        
        Includes:
        - run_backtest
        - out_performance
        
        """        
        #general class definitions
        self.events = queue.Queue()
        
        self.data_bundle = data_bundle
        
        self.data_dir = data_dir
        self.symbol_list = symbol_list
        self.bartype = bartype
        self.heartbeat = step_heartbeat*1000
        self.init = initialization_step
        self.last = finalization_step
        self.freq = freq
        self.start_date = start_date 
        self.end_date = end_date
        self.vwap = vwap
        
        self.range_dates = sel_days(self.start_date,
                                    self.end_date)
        
        
        self.initial_capital = initial_capital

        self.data_handler_cls = data_handler
        self.execution_handler_cls = execution_handler
        self.portfolio_cls = portfolio
        self.strategy_cls = strategy
        self.iterator_cls = iterator

        self.signals = 0
        self.orders = 0
        self.fills = 0
        self.num_strats = 1
        self.keywords = ['init_step',
                         'last_step',
                         'freq',
                         'capital',
                         'frequency',
                         'bartype']
        
    #check input variables
    def add(self,init_step=None, last_step=None, 
            capital=None, frequency=None, bartype=None,
            heartbeat=None):
        
        if init_step is not None:
            self.init = init_step
        if last_step is not None:
            self.last = last_step
        if frequency is not None:
            self.freq = frequency
        if capital is not None:
            self.initial_capital = capital
        if bartype is not None:
            self.bartype = bartype
        if heartbeat is not None:
            self.heartbeat = heartbeat*1000
        
   #main trading system generation 
    def _generate_trading_instances(self):
        """
        Generates the trading instance objects from
        their class types.
        """
        warnings.filterwarnings("ignore")
        
        #Creating Data Handler
        
        self.data_handler = self.data_handler_cls(
            self.events,
            self.symbol_list,
            self.bartype,
            self.init,
            self.last,
            self.heartbeat
        )
        
        #Computing Strategy
        
        self.strategy = self.strategy_cls(
            self.data_handler, 
            self.events
        )
        
        #Constructing Portfolio
        
        self.portfolio = self.portfolio_cls(
            self.data_handler, 
            self.events, 
            self.start_date, 
            self.initial_capital
        )
        
        #Executing Handler Porcess 
        
        self.execution_handler = self.execution_handler_cls(
            self.events
        )
        
        #Iterator process of Heartbeats 
        
        self.iterator = self.iterator_cls(
            self.data_dir,
            self.events
        )
        
    #main backtest process
    def _run_backtest(self):
        """
        Executes the backtest thru an iteration process.
        """
        self._generate_trading_instances()
        
        #mini databundle
        #load_data = self.data_bundle
        
        #load_data = [accelerated_open_zarr.remote(
        #                        self.data_dir,
        #                        date,
        #                        self.symbol_list
        #                        ) 
        #            for date in self.range_dates]        
        #data_bundle = ray.get(load_data)
        
        iteration = 0
        #for idx_date, date in enumerate(self.range_dates):
        for date in self.range_dates:
            dict_symbol = {s: open_zarr(
                                self.data_dir+"/"+ s +".zarr", 
                                date) for s in self.symbol_list
                            }
            
            #selected_data = load_data[idx_date]
            
            print("Processing: {}".format(date))
            
            _initialization = datetime.timestamp(
                    parser.parse(
                            date + " " + self.init
                            )
                    ) * (10**3)
            _finalization = datetime.timestamp(
                    parser.parse(
                            date + " " + self.last
                            )
                    ) * (10**3)

            
            range_steps = range(int(_initialization),
                                int(_finalization),
                                self.heartbeat)
            
            if range_steps[0] > 1:
                range_steps = range_steps[:-1]
            elif range_steps.shape[0] < 0:
                print('Invalid dates range')
                sys.exit()
            else:
                pass
            
            for heartstep in range_steps:               
                iteration += 1
                
                self.data_handler._open_convert_csv_files(
                        self.bartype,
                        heartstep,
                        self.heartbeat,self.freq,
                        self.vwap, date, dict_symbol
                        )
            
                #self.data_handler.update_bars() 
                
                self.iterator.verify_ts_send_orders(
                        date,
                        heartstep,
                        heartstep + self.heartbeat
                        )
                
                while True:
                    
                    try:
                        event = self.events.get(False) #verificar si hay un "evento" 
                    except queue.Empty:    
                        break
                    else:
                        if event is not None:
                            if isinstance(event, MarketEvent):
                                self.strategy.calculate_signals(event) #incluir ray en la construccion enssemble
                                self.portfolio.update_timeindex(event)
                            elif isinstance(event, SignalEvent):
                                self.signals += 1
                                self.portfolio.update_signal(event) 
                            elif isinstance(event, OrderEvent):
                                self.orders += 1
                                self.execution_handler.execute_order(event) 
                            elif isinstance(event, FillEvent):
                                self.fills += 1
                                self.portfolio.update_fill(event)
                            elif isinstance (event, IterationEvent):
                                self.iterator.add_to_dict(event)
                                print("{} added to orders_dict.".format(
                                        event.symbol))
    #output performance
    def _output_performance(self):
        """
        Outputs the strategy performance from the backtest.
        """
        self.portfolio.create_equity_curve_dataframe()
        
        print("\nCreating summary stats...")
        stats = self.portfolio.output_summary_stats()

        print("Creating equity curve...\n")
        
        print("RESULTS")
        print("-"*45)
        
        print("Annualized Sharpe Ratio:{}\n".format(round(stats[1],4)))
        print("Signals: %s" % self.signals)
        print("Orders: %s" % self.orders)
        print("Fills: %s" % self.fills)
        
        plot_final_diagram(stats[0], self.initial_capital)
    
    #request simulated trading
    def simulate_trading(self):
        """
        Simulates the backtest and outputs portfolio performance.
        """
        self._run_backtest()
        self._output_performance()
