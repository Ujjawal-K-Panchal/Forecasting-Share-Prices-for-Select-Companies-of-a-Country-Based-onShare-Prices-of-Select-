# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 21:09:00 2018

@author: uchih
"""

import pandas as pd

dataset = pd.read_csv('MergedCombined.csv')
dataset[:].fillna(0, inplace = True)
dataset.to_csv('MergedCombinedNA0d.csv')
