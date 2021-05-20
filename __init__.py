"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import warnings
from pandas.core.common import SettingWithCopyWarning
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

#from enigmx.backtester.core import Backtest
#from enigmx.extractor import Extractor
