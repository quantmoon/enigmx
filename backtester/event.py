"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

class Event(object):
    """
    Event is base class providing an interface for all subsequent 
    (inherited) events, that will trigger further events in the 
    trading infrastructure.   
    """
    pass

class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with 
    corresponding bars.
    """

    def __init__(self):
        """
        Initialises the MarketEvent.
        """
        self.type = 'MARKET'
        
class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """
    
    def __init__(self, symbol, datetime, signal_type,strength,
                 order_type,stop_price=None,limit_price=None):
        """
        Initialises the SignalEvent.

        Parameters:
        -----------
        - symbol: the ticker symbol, e.g. 'GOOG'.
        - datetime: the timestamp at which the signal was generated.
        - signal_type: 'LONG' or 'SHORT'.
        - strength: adjustment factor "suggestion" used to scale values.
                    Useful for pairs strategies and portfolio construction.
        """
        
        self.type = 'SIGNAL'
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        self.strength = strength
        self.order_type=order_type
        self.stop_price=stop_price
        self.limit_price=limit_price
        
class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, symbol, order_type, quantity, 
                 direction,stop_price=None,limit_price=None):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral) and its direction ('BUY' or
        'SELL').

        Parameters:
        ------------
        - symbol: The instrument to trade.
        - order_type: 'MKT' or 'LMT' for Market or Limit.
        - quantity: Non-negative integer for quantity.
        - direction: 'BUY' or 'SELL' for long or short.
        """
        
        self.type = 'ORDER'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.stop_price = None
        self.limit_price = None

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        print("Order: Symbol={0},Type={1},Quantity={2},Direction={3}".format(
            self.symbol, self.order_type, self.quantity, self.direction
        )
             )
            
class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """

    def __init__(self, timeindex, symbol, exchange, quantity, 
                 direction, fill_cost, commission=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional 
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Parameters:
        -----------
        - timeindex: The bar-resolution when the order was filled.
        - symbol: The instrument which was filled.
        - exchange: The exchange where the order was filled.
        - quantity: The filled quantity.
        - direction: The direction of fill ('BUY' or 'SELL')
        - fill_cost: The holdings value in dollars.
        - commission: An optional commission sent from IB.
        """
        
        self.type = 'FILL'
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API, in USD.

        This does not include exchange or ECN fees for data.

        Based on "US API Directed Orders":
        www.interactivebrokers.com/en/index.php?f=commission&p=stocks2
        """
        commission_fees = 1.3
        if self.quantity <= 500:
            commission_fees = max(1.3, 0.013 * self.quantity)
        else: 
            commission_fees = max(1.3, 0.008 * self.quantity)
        return commission_fees
    
    
class IterationEvent():
    """
    This event is similar to OrderEvent, except for the reason that this will 
    insert the order in a dictionary in order to search for a target price, 
    which is going to determine if the order is executed in the current 
    heartbeat.
    """
    
    def __init__(self, symbol, order_type, quantity, 
                 direction,stop_price=None,limit_price=None):
        
        self.type = 'ITERATION'
        self.symbol = symbol
        self.order_type = order_type
        self.quantity = quantity
        self.direction = direction
        self.stop_price = None
        self.limit_price = None