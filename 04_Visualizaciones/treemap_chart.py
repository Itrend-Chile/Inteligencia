# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 16:11:41 2021

@author: Inteligencia 1
"""

import os
import pandas as pd
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
import math

os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia\\Patentes')

data = pd.read_excel(".\\analisis_nota.xlsx")
data = data[['fuente datos', 'id', 'N INAPI', 'classifications']]

classifications = data.classifications.str.split(';', expand=True).stack().str.strip().reset_index(level=1, drop=True)
classifications = pd.DataFrame(data=classifications, columns=['class'])

classifications['L1'] = classifications['class'].str[:1]
classifications['L2'] = classifications['class'].str[:3]
classifications['L3'] = classifications['class'].str[:4]


classifications['index1'] = classifications.index
classifications = classifications.drop_duplicates(subset=['index1', 'L2'])

classifications['N_L1'] = classifications.groupby(['L1'])['L1'].transform('count')
classifications['N_L2'] = classifications.groupby(['L2'])['L2'].transform('count')
classifications['N_L3'] = classifications.groupby(['L3'])['L3'].transform('count')

classifications = classifications.drop_duplicates(subset=['L2'])