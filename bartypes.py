"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pandas as pd
from datetime import datetime


#pandas dataset construction: based on frequency
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