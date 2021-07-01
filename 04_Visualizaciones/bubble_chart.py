# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 13:29:28 2021

@author: Inteligencia 1
"""

import os
import pandas as pd
import plotly.offline as py
import plotly.graph_objects as go
import plotly.express as px
import math

os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia\\Patentes')

data = pd.read_excel(".\\patentes ocde.xlsx")
data = data.sort_values(['continente', 'pais']).reset_index()


hover_text = []
bubble_size = []

for index, row in data.iterrows():
    hover_text.append(('País: {pais}<br>'+
                      'Patentes: {patentes}<br>'+
                      'PIB: {PIB}<br>'+
                      'Gasto I+D: {ID}<br>'+
                      'Año: {year}').format(pais=row['pais'],
                                            patentes=row['patentes'],
                                            PIB=row['PIB'],
                                            ID =row['ID'],
                                            year=row['year1']))
    bubble_size.append(math.sqrt(row['ID']))



data['text'] = hover_text
data['size'] = bubble_size
sizeref = 2.*max(data['size'])/(100**2)

# Dictionary with dataframes for each continent
continent_names = ['Africa', 'América', 'Asia', 'Europa', 'Oceanía']
continent_data = {continente:data.query("continente == '%s'" %continente)
                              for continente in continent_names}

# Create figure
fig = go.Figure()

for continent_name, continent in continent_data.items():
    fig.add_trace(go.Scatter(
        x=continent['PIB'], y=continent['patentes'],
        name=continent_name, text=continent['text'],
        marker_size=continent['size'],
        ))

# Tune marker appearance and layout
fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                              sizeref=sizeref, line_width=2))

fig.update_layout(
    title='Patentes Solicitadas  v. Producto Interno Bruto, 2019',
    xaxis=dict(
        title='PIB (PPP dollars)',
        gridcolor='white',
        type='log',
        gridwidth=2,
    ),
    yaxis=dict(
        title='Patentes solicitadas',
        gridcolor='white',
        type='log',
        gridwidth=2,
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)',
)
fig.show()

py.plot(fig, filename='graph.html')