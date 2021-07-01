# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:52:27 2020

@author: Inteligencia 1
"""

import matplotlib.pyplot as plt
import pandas as pd
from math import pi


fig = plt.figure(figsize=(6,6))
ax = plt.subplot(polar="True")



categorias = ['Riesgo', 'Humo', 'Impacto', 'Históricos', 'Combustible']
N  =len(categorias)

values_2000 = [1.9174, 0.2157, 4.5568, 6.4908, 5.8197]
values_2000 += values_2000[:1]



values_2010 = [8.6105, 6.9496, 14.1251, 40.004, 16.3107]
values_2020 = [63.2565, 29.3063, 40.9525, 97.8812, 42.6017]

angles = [n/float(N)*2*pi for n in range(N)]

plt.polar(angles, values, marker='.')
plt.fill(angles, vlaues, alpha=0.3)