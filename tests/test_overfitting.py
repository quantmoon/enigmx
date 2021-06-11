"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

from enigmx.PBO import OverfittingTesting 

feature = OverfittingTesting(
    path_backtest ="C:/data/BACKTEST_TRIAL_001.csv",
    path_metrics = "C:/data/METRICS_TRIAL_001.csv").get_test(pol_threshold = 1.20)

print(feature)

