"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import xarray as xr
from .event import OrderEvent

class Iterator():
    """
    PUT TEXT HERE
    """
    
    def __init__(self,path,events):
        self.orders_dict = {}
        self.path = path
        self.events = events
        self.orders_to_execute = {}
        
        
    def add_to_dict(self,event):
        self.orders_dict[event.symbol] = [event.order_type, event.quantity, 
                        event.direction, event.stop_price,event.limit_price]
    
    def verify_ts_send_orders(self,date,init,last):
        for key,val in self.orders_dict.items():
            if(type(key)) == "":
                break
            else:
                ds = xr.open_zarr(self.path+"/"+key+".zarr")
                if self.orders_to_execute[key] == None:
                    if val[1]=='STP':
                        target=val[3]
                        if val[2]=='LONG':
                            ts=ds.ts.where(
                                    ds.cost[0][0]>=target
                                    ).where(
                                            ds.ts>=init
                                            ).where(
                                                    ds.ts<=last
                                                    ).sel(
                                                            dim=date
                                                            ).dropna(
                                                                    dim='ts'
                                                                    )[0]
                        elif val[2]=='SHORT':
                            ts=ds.ts.where(
                                    ds.cost[0][0]<=target
                                    ).where(
                                            ds.ts>=init
                                            ).where(
                                                    ds.ts<=last
                                                    ).sel(
                                                            dim=date
                                                            ).dropna(
                                                                    dim='ts'
                                                                    )[0] 
                        if ts is not None:
                            val=val.append(ts)
                            self.orders_to_execute[key]=val
                            
                    elif val[1]=='LMT':
                        target=val[4]
                        if val[2]=='LONG':
                            ts=ds.ts.where(
                                    ds.cost[0][0]<=target
                                    ).where(
                                            ds.ts>=init
                                            ).where(
                                                    ds.ts<=last
                                                    ).sel(
                                                            dim=date
                                                            ).dropna(
                                                                    dim='ts'
                                                                    )[0]
                        elif val[2]=='SHORT':
                            ts=ds.ts.where(
                                    ds.cost[0][0]>=target
                                    ).where(
                                            ds.ts>=init
                                            ).where(
                                                    ds.ts<=last
                                                    ).sel(
                                                            dim=date
                                                            ).dropna(
                                                                    dim='ts'
                                                                    )[0] 
                            
                        if ts is not None:
                            
                            val=val.append(ts)
                            self.orders_to_execute[key]=val
                            
            self._create_orders()
    
    def _sorting_orders(self):
        
        self.orders_to_execute = sorted(
                self.orders_to_execute.items(), 
                key=lambda x: x[1][5]
                )

    def _create_orders(self):
        
        self._sorting_orders()
        for key,val in self.orders_to_execute.items():
            if val[5] != None:
                order=OrderEvent(val[0],val[1],val[2],val[3],val[4])
                self.events.put(order)
                

