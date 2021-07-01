# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:37:51 2020

@author: Inteligencia 1
"""

import os
import pandas as pd
from itertools import combinations
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx

os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia\\Networks')

df = pd.read_excel(".\\Clas_auto_manual_v3.xlsx")
df = df[df['author_ids'].notna()]

df_fire = df[(df['Ame1']=='Incendio')|(df['Ame2']=='Incendio')|(df['Ame3']=='Incendio')]

df_fire = df_fire.reset_index()
del df_fire['index']


aff_ids = df_fire.afid.str.split(';', expand=True).stack().str.strip().reset_index(level=1, drop=True)
df_aff = pd.DataFrame(data=aff_ids, columns=['aff_id'])


#EDGES
df_edges = df_aff.groupby(df_aff.index)['aff_id'].apply(lambda x : list(combinations(x.values,2))).apply(pd.Series).stack().reset_index(level=0,name='pares')


df_edges[['aff1','aff2']] = pd.DataFrame(df_edges['pares'].tolist(), index=df_edges.index)


df_edges[['aff1','aff2']] = df_edges[['aff1','aff2']].astype(int)


#NODOS
df_cruce = pd.read_excel(".\\Resultados_afiliaciones_corregido.xlsx")
df_cruce = df_cruce[df_cruce['N_papers']>=5]


df_edges= df_edges.merge(df_cruce[['aff_id','short_name']], how='left', left_on = 'aff1', right_on='aff_id')
df_edges= df_edges.merge(df_cruce[['aff_id','short_name']], how='left', left_on = 'aff2', right_on='aff_id')

df_edges['aff_name_1'] = df_edges['short_name_x']
df_edges['aff_name_2'] = df_edges['short_name_y']
df_edges = df_edges[['level_0','aff1','aff2', 'aff_name_1', 'aff_name_2']]

df_edges = df_edges[df_edges['aff_name_1'].notna()]
df_edges = df_edges[df_edges['aff_name_2'].notna()]

df_edges = df_edges.reset_index()
del df_edges['index']

df_edges = df_edges.groupby(['aff_name_1', 'aff_name_2']).size().reset_index(name = 'weight')

df_nodos = df_cruce[['short_name', 's_aff_country', 'N_papers']]
aggregation_functions = {'short_name': 'first', 's_aff_country': 'first', 'N_papers': 'sum'}
df_nodos = df_nodos.groupby(df_nodos['short_name']).aggregate(aggregation_functions)
df_nodos = df_nodos.reset_index(drop=True)



# 1. CREAR EL GRAFO

graph = nx.convert_matrix.from_pandas_edgelist(df_edges,
                                               source = 'aff_name_1',
                                               target = 'aff_name_2',
                                               edge_attr = 'weight')

from random import sample
sampled_edges = sample(graph.edges, 10)

node_dict = df_nodos.set_index('short_name').to_dict(orient = 'index')
nx.set_node_attributes(graph, node_dict)

sampled_nodes = sample(graph.nodes, 10)

# 2. GET POSITIONS FOR THE NODES IN G

pos_ = nx.spring_layout(graph)

def make_edge(x, y, text, width):
    
    '''Creates a scatter trace for the edge between x's and y's with given width

    Parameters
    ----------
    x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])
    
    y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])
    
    width: the width of the line

    Returns
    -------
    An edge trace that goes between x0 and x1 with specified width.
    '''
    return  go.Scatter(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'lightgrey'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')

# For each edge, make an edge_trace, append to list
edge_trace = []

for edge in graph.edges():
    
    if graph.edges()[edge]['weight'] > 0:
        char_1 = edge[0]
        char_2 = edge[1]

        x0, y0 = pos_[char_1]
        x1, y1 = pos_[char_2]

        text   = char_1 + '--' + char_2 + ': ' + str(graph.edges()[edge]['weight'])
        
        trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                           0.1*graph.edges()[edge]['weight']**1.75)

        edge_trace.append(trace)


# Make a node trace
node_trace = go.Scatter(x         = [],
                        y         = [],
                        text      = [],
                        textposition = "top center",
                        textfont_size = 10,
                        mode      = 'markers',
                        hoverinfo = 'text',
                        marker    = dict(color = [],
                                         size  = [],
                                         line  = None,
                                         colorscale='Bluyl',
                                         colorbar=dict(thickness=15,
                                                       title='N° Colaboraciones',
                                                       xanchor='left',
                                                       titleside='right')))





# For each node in midsummer, get the position and size and add to the node_trace


for node in graph.nodes():
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    #node_trace['marker']['color'] += tuple(['cornflowerblue'])
    node_trace['marker']['size'] += tuple([1*graph.nodes()[node]['N_papers']])
    #node_trace['text'] += tuple(['<b>' + node + '</b>'])
    
#node_adjacencies = []
#node_text = []
#for node, adjacencies in enumerate(graph.adjacency()):
#    node_adjacencies.append(len(adjacencies[1]))
#    node_text.append('# of connections: '+str(len(adjacencies[1])))

#node_trace.marker.color = node_adjacencies
#node_trace.text = node_text

annotations = []

# hover text and color group
for node, adjacencies in enumerate(graph.adjacency()):
    # adjacencies[1] : tuple
    # adjacencies[1][0] : origin node
    # adjacencies[1][1] : targets nodes     

    # setting the text that will be display on hover
    # (I have originally more informations involved but I made it shorter here)
    node_info = '{} - {} colaboraciones'.format(
        adjacencies[0],
        str(len(adjacencies[1]))
        )

    node_trace['text'] += tuple([node_info])

    # THERE : Annotations is a list of dictionaries with every needed parameter for each node annotation
    annotations.append(
        dict(x=pos_[adjacencies[0]][0],
             y=pos_[adjacencies[0]][1],
             #text= dict(text=adjacencies[0], textposition='top center'),
             text=adjacencies[0], # node name that will be displayed
             xanchor='center',
             yanchor='top',
             xshift=10,
             font=dict(color='black', size=10),
             showarrow=False, arrowhead=1, ax=-10, ay=-10)
        )


node_adjacencies = []
for node, adjacencies in enumerate(graph.adjacency()):
    node_adjacencies.append(len(adjacencies[1]))

node_trace.marker.color = node_adjacencies


    #VISUALIZACIÓN
    
layout = go.Layout(
    annotations=annotations,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)


fig = go.Figure(layout = layout)

for trace in edge_trace:
    fig.add_trace(trace)

fig.add_trace(node_trace)

fig.update_traces(textposition='top center')

fig.update_layout(showlegend = False)

fig.update_xaxes(showticklabels = False)

fig.update_yaxes(showticklabels = False)

fig.show()
py.plot(fig, filename='graph1.html')