"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

from enigmx.backtester.strategy import Strategy, SignalEvent

class UserStrategy(Strategy):
    """
    This is an extremely simple strategy that goes LONG all of the 
    symbols as soon as a bar is received. It will never exit a position.
    It is primarily used as a testing mechanism for the Strategy class
    as well as a benchmark upon which to compare other strategies.
    """

    def __init__(self, bars, events):
        """
        Initialises the buy and hold strategy.
        Parameters:
        -----------
        - bars: The DataHandler object that provides bar information
        - events: The Event Queue object.
        """
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

        # Once buy & hold signal is given, these are set to True
        self.bought = self._calculate_initial_bought()
        self.periods = 0
        
        
# strategy.py

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to False.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought
    
# strategy.py

    def calculate_signals(self, event):
        """
        For "Buy and Hold" we generate a single signal per symbol
        and then no additional signals. This means we are 
        constantly long the market from the date of strategy
        initialisation.
        Parameters:
        ----------
        - Event: A MarketEvent object. 
        """
        strength = 1.0
        
        if event.type == 'MARKET':          
            self.periods+=1 
            #parallelization
            
            # results =  [ray result of parallelization; signals] | execute get | list comp.
            # for s in self.symbol_list: 
            #   send the signal to order event
            
            for s in self.symbol_list: #ray parallelization encapsulating results
                bars = self.bars.get_latest_bars(s, N=1)
                
                if bars is not None and bars != []:
                    
                    if self.bought[s] == False:
                        
                        signal = SignalEvent(s, 
                                             bars[0], 
                                             'LONG', 
                                             strength,
                                             order_type='MKT')
                        self.events.put(signal) #send signal to queue (order event)
                        self.bought[s] = True
                        
                    elif self.bought[s] == True:
                        if self.periods % 3 == 0:
                            signal = SignalEvent(s, 
                                                 bars[0], 
                                                 'EXIT', 
                                                 strength,
                                                 order_type='MKT')
                            self.events.put(signal) #send signal to queue (order event)
                            self.bought[s] = False
                            self.periods=0
                        