# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:15:09 2020

@author: belen
"""

# Script para distinguir papers en DDNN a partir de palabras clave.


import os
import pandas as pd

os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\bases')
pd.set_option('precision', 15)


docs = pd.read_excel(".\\Resultados_scopus_docus.xlsx") # importar resultados web scrapping
docs['title'] = docs['title'].str.lower() # título a minúsculas
docs['description'] = docs['description'].str.lower() # abstract a minúsculas
docs['year'] = docs['date'].str.extract(r'(^\w{4})') # varible con el año de publicación
docs = docs.drop_duplicates(subset=['eid']) # eliminar duplicados

#Terremotoc (en título o abstract)
docs['etq_ttl'] = docs['title'].str.count('earthquake')
docs['etq_ab'] = docs['description'].str.count('earthquake')
docs['seismic_ttl'] = docs['title'].str.count('seism')
docs['seismic_ab'] = docs['description'].str.count('seism')
docs['sub_ttl'] = docs['title'].str.count('subduct')
docs['sub_ab'] = docs['description'].str.count('subduct')
docs['tell_ttl'] = docs['title'].str.count('telluric')
docs['tell_ab'] = docs['description'].str.count('telluric')
docs['ground_ttl'] = docs['title'].str.count('ground motion')
docs['ground_ab'] = docs['description'].str.count('ground motion')
docs['fault_ttl'] = docs['title'].str.count('fault')
docs['liq_ttl'] = docs['title'].str.count('liquefaction')
docs['epi_ttl'] = docs['title'].str.count('epicenter')
docs['hypo_ttl'] = docs['title'].str.count('hypocenter')


docs['earthquake_ttl'] = docs['etq_ttl'] + docs['seismic_ttl'] + docs['sub_ttl']\
 + docs['tell_ttl'] + docs['ground_ttl'] + docs['fault_ttl'] + docs['liq_ttl']\
+ docs['epi_ttl'] + docs['hypo_ttl']
 
docs['earthquake_ab'] = docs['etq_ab'] + docs['seismic_ab'] + docs['sub_ab']\
+ docs['tell_ab'] + docs['ground_ab']               
            
del docs['etq_ttl'], docs['seismic_ttl'], docs['sub_ttl'], docs['tell_ttl'], docs['ground_ttl'], docs['fault_ttl'], docs['liq_ttl'], docs['epi_ttl'], docs['hypo_ttl']
del docs['etq_ab'], docs['seismic_ab'], docs['sub_ab'], docs['tell_ab'], docs['ground_ab']

#Tsunami
docs['tsnm_ttl'] = docs['title'].str.count('tsunami')
docs['tsnm_ab'] = docs['description'].str.count('tsunami')
docs['sq_ttl'] = docs['title'].str.count('seaquake')
docs['sq_ab'] = docs['description'].str.count('seaquake')
docs['runup_ttl'] = docs['title'].str.count('runup')
docs['runup_ab'] = docs['description'].str.count('runup')

docs['tsunami_ttl'] = docs['tsnm_ttl'] + docs['runup_ttl'] + docs['sq_ttl']
docs['tsunami_ab'] = docs['tsnm_ab'] + docs['runup_ab'] + docs['sq_ab']
del docs['tsnm_ttl'], docs['runup_ttl'], docs['sq_ttl'], docs['tsnm_ab'], docs['runup_ab'], docs['sq_ab']

#Incendio
docs['fire_ttl'] = docs['title'].str.count('fire\s')
docs['fire_ab'] = docs['description'].str.count('fire\s')

#Remoción en masa
docs['ldsl_ttl'] = docs['title'].str.count('landslide')
docs['ldsl_ab'] = docs['description'].str.count('landslide')
docs['rcksl_ttl'] = docs['title'].str.count('rockslide')
docs['rcksl_ab'] = docs['description'].str.count('rockslide')
docs['all_ttl'] = docs['title'].str.count('alluvium')
docs['all_ab'] = docs['description'].str.count('alluvium')
docs['debris_ttl'] = docs['title'].str.count('debris flow')
docs['debris_ab'] = docs['description'].str.count('debris flow')

docs['rem_ttl'] = docs['ldsl_ttl'] + docs['rcksl_ttl'] + docs['all_ttl'] + docs['debris_ttl'] 
docs['rem_ab'] = docs['ldsl_ab'] + docs['rcksl_ab'] + docs['all_ab'] + docs['debris_ab'] 
del docs['ldsl_ttl'], docs['all_ttl'], docs['ldsl_ab'], docs['all_ab'], docs['rcksl_ttl'], docs['rcksl_ab'], docs['debris_ttl'], docs['debris_ab']

#Erupciones volcanicas
docs['vcn_ttl'] = docs['title'].str.count('volcan')
docs['vcn_ab'] = docs['description'].str.count('volcan')
docs['lava_ttl'] = docs['title'].str.count(' lava\s')
docs['lava_ab'] = docs['description'].str.count(' lava\s')
docs['lahar_ttl'] = docs['title'].str.count('lahar')
docs['lahar_ab'] = docs['description'].str.count('lahar')
docs['pyro_ttl'] = docs['title'].str.count('pyroclas')
docs['pyro_ab'] = docs['description'].str.count('pyroclas')
docs['tephra_ttl'] = docs['title'].str.count('tephra')
docs['tephra_ab'] = docs['description'].str.count('tephra')
docs['eruption_ttl'] = docs['title'].str.count('eruption')
docs['eruption_ab'] = docs['description'].str.count('eruption')

docs['volcan_ttl'] = docs['vcn_ttl'] + docs['lava_ttl'] + docs['lahar_ttl'] + docs['pyro_ttl'] + docs['tephra_ttl'] + docs['eruption_ttl']
docs['volcan_ab'] = docs['vcn_ab'] + docs['lava_ab'] + docs['lahar_ab'] + docs['pyro_ab'] + docs['tephra_ab'] + docs['eruption_ab'] 
del docs['vcn_ttl'], docs['lava_ttl'], docs['vcn_ab'], docs['lava_ab'], docs['lahar_ttl'], docs['pyro_ttl'], docs['lahar_ab'], docs['pyro_ab'], docs['eruption_ttl'], docs['eruption_ab'], docs['tephra_ttl'], docs['tephra_ab']


#Climáticos
#Inundación
docs['flood_ttl'] = docs['title'].str.count('flood')
docs['flood_ab'] = docs['description'].str.count('flood')
docs['oflw_ttl'] = docs['title'].str.count('overflowing')
docs['oflw_ab'] = docs['description'].str.count('overflowing')
docs['logg_ttl'] = docs['title'].str.count('waterlogging')
docs['logg_ab'] = docs['description'].str.count('waterlogging')
#Sequía y desertificación
docs['drought_ttl'] = docs['title'].str.count('drought')
docs['drought_ab'] = docs['description'].str.count('drought')
docs['desert_ttl'] = docs['title'].str.count('desertification')
docs['desert_ab'] = docs['description'].str.count('desertification')
#Otros
docs['ex_ttl'] = docs['title'].str.count('extreme event')
docs['ex_ab'] = docs['description'].str.count('extreme event')
docs['tromba_ttl'] = docs['title'].str.count('watersprout')
docs['tromba_ab'] = docs['description'].str.count('watersprout')
docs['weather_ttl'] = docs['title'].str.count('extreme weather')
docs['weather_ab'] = docs['description'].str.count('extreme weather')


docs['climate_ttl'] = docs['flood_ttl'] + docs['oflw_ttl'] + docs['logg_ttl'] + docs['drought_ttl'] + docs['ex_ttl'] + docs['tromba_ttl'] + docs['weather_ttl'] + docs['desert_ttl']
docs['climate_ab'] = docs['flood_ab'] + docs['oflw_ab'] + docs['logg_ab'] + docs['drought_ab'] + docs['ex_ab'] + docs['tromba_ab'] + docs['weather_ab'] + docs['desert_ab']

del docs['flood_ttl'], docs['oflw_ttl'], docs['logg_ttl'], docs['drought_ttl'], docs['ex_ttl'], docs['tromba_ttl'], docs['weather_ttl'], docs['desert_ttl'],\
    docs['flood_ab'], docs['oflw_ab'], docs['logg_ab'], docs['drought_ab'], docs['ex_ab'], docs['tromba_ab'], docs['weather_ab'], docs['desert_ab']

#Generales/Otras
#docs['multihzd_ttl'] = docs['title'].str.count('multi-hazard')
#docs['nathzd_ttl'] = docs['title'].str.count('natural hazard')
#docs['dst_ttl'] = docs['title'].str.count('disaster')
#docs['other_ttl'] = docs['multihzd_ttl'] + docs['nathzd_ttl'] + docs['dst_ttl']


docs['earthquake'] = 0
docs['tsunami'] = 0
docs['volcan'] = 0
docs['wildfire'] = 0
docs['landslide'] = 0
docs['climate'] = 0


#Asignar tipo de desastre por keywords en el título:

docs.loc[docs['earthquake_ttl'] > 0, 'earthquake'] = 1
docs.loc[docs['tsunami_ttl'] > 0, 'tsunami'] = 1
docs.loc[docs['volcan_ttl'] > 0, 'volcan'] = 1
docs.loc[docs['fire_ttl'] > 0, 'wildfire'] = 1
docs.loc[docs['rem_ttl'] > 0, 'landslide'] = 1
docs.loc[docs['climate_ttl'] > 0, 'climate'] = 1

#Corrección Climáticos: si es que tiene más palabras claves asociadas a otros desastres, de da prioridad al otro desastre
docs['any'] = docs[['earthquake', 'tsunami', 'volcan', 'wildfire', 'landslide']].any(axis='columns').astype(int)
docs.loc[docs['any'] == 1, 'climate'] = 0
del docs['any']


#Corrección Terremotos: para terremotos, asignar por key words en el abstract si es que no está asignado a otra clasificación (más variedad de términos)
docs['any'] = docs[['tsunami', 'volcan', 'wildfire', 'landslide', 'climate']].any(axis='columns').astype(int)
docs.loc[(docs['any'] == 1), 'earthquake'] = 0
docs.loc[(docs['any'] == 0) & (docs['earthquake_ab'] >= 1), 'earthquake'] = 1
del docs['any']


# Variable "nat_dis": binaria. Verdadero si la observación corresponde a publ. en DDNN:
docs['nat_dis'] = docs[['earthquake', 'tsunami', 'volcan', 'wildfire', 'landslide', 'climate']].any(axis='columns').astype(int)
docs['nat_dis_ttlabs'] = docs[['earthquake_ab', 'earthquake_ttl', 'tsunami_ab', 'tsunami_ttl',\
    'fire_ttl', 'fire_ab', 'rem_ttl', 'rem_ab', 'volcan_ttl', 'volcan_ab', 'climate_ttl', 'climate_ab']].any(axis='columns').astype(int)

# Eliminar  variables auxiliares 
del docs['earthquake_ttl'], docs['earthquake_ab'], docs['tsunami_ttl'], docs['tsunami_ab'], docs['volcan_ttl'], docs['volcan_ab'],\
     docs['fire_ttl'], docs['fire_ab'], docs['rem_ttl'], docs['rem_ab'], docs['climate_ttl'], docs['climate_ab']
    
docs.to_excel('.\\base_papers.xlsx', index=False) #Guardar base final, método búsqueda por autor









