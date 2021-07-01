# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 09:42:05 2020

@author: belen
"""
import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
from bokeh.palettes import brewer
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.plotting import figure
from bokeh.io import output_notebook, show, output_file

os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\analisis grafico')
map_data = gpd.read_file('regiones3.geojson')

#plt.rcParams['figure.figsize'] = (40, 40)
#map_data.plot()

#arreglar nombres de las comunas

map_data['REGION'][3] = 'Aysén'
map_data['REGION'][4] = 'Biobío'
map_data['REGION'][6] = 'La Araucanía'
map_data['REGION'][7] = 'O Higgins'
map_data['REGION'][9] = 'Los Ríos'
map_data['REGION'][10] = 'Magallanes'
map_data['REGION'][11] = 'Magallanes'
map_data['REGION'][15] = 'Tarapacá'
map_data['REGION'][14] = 'Ñuble'
map_data['REGION'][16] = 'Valparaíso'

auth_region = pd.read_excel(".\\autores_regiones.xlsx")



author_data = pd.merge(map_data, auth_region, left_on='REGION', right_on='region', how= 'outer')
author_data['N'].fillna(0, inplace=True)


variable = 'N'

vmin, vmax = 0, 300

#fig, ax = plt.subplots(1, figsize=(40, 20))

ax = author_data.plot(column='N', cmap = 'Blues', figsize=(30,15),  k=5, legend = True);
ax.set_title('Académicos en DDNN, por región', fontdict={'fontsize': '25', 'fontweight' : '3'})
ax.set_axis_off()

ax.legend(bbox_to_anchor=(0.5, -0.05))

ax.get_figure()


author_data.plot(column=variable, cmap='Blues', linewidth=0.8, ax=ax, edgecolor='0.8')
sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm._A =
ax.axis('off')

#os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\analisis grafico\\regiones_ufro2')
#sf = shp.Reader('cl_regiones_geo')

def plot_map(sf, x_lim = None, y_lim = None, figsize = (20,10)):
    '''
    Plot map with lim coordinates
    '''
    plt.figure(figsize = figsize)
    id=0
    for shape in sf.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k')
        
        if (x_lim == None) & (y_lim == None):
            x0 = np.mean(x)
            y0 = np.mean(y)
            plt.text(x0, y0, id, fontsize=10)
        id = id+1
    
    if (x_lim != None) & (y_lim != None):     
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        
plot_map(map_data)

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    return df

df = read_shapefile(sf)
df.shape


#Traer info de interés:
d = gpd.read_file('DPA_Regional.shp')
x = d.buffer(0.0001)

tolerance = 0.00005
simplified = x.simplify(tolerance, preserve_topology=True)


