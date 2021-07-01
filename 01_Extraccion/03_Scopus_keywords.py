# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 15:41:37 2020

@author: belen
"""

#Este código realiza búsquedas en Scopus por palabras clave

import os
import pandas as pd
from pybliometrics.scopus import ScopusSearch

os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\bases')
pd.set_option('precision', 15)

#Palabras clave

#1. Terremoto
etq_search = ScopusSearch('(TITLE-ABS-KEY(earthquake) OR TITLE-ABS-KEY(seismic) OR TITLE-ABS-KEY(seismicity) OR TITLE-ABS-KEY(subduction) OR TITLE-ABS-KEY(ground motion) OR TITLE-ABS-KEY(fault) OR TITLE-ABS-KEY(liquefaction) OR TITLE-ABS-KEY(coseismic)) AND  AFFILCOUNTRY(chile)')

df_etq = pd.DataFrame(pd.DataFrame(etq_search.results))

#2. Tsunami

tsunami_search = ScopusSearch('(TITLE-ABS-KEY(tsunami) OR TITLE-ABS-KEY (runup) OR TITLE-ABS-KEY (run-up) OR TITLE-ABS-KEY(coastal flood) OR TITLE-ABS-KEY(seaquake))   AND  AFFILCOUNTRY (chile)')

df_tsunami = pd.DataFrame(pd.DataFrame(tsunami_search.results))


#3. Erupciones volcanicas

volcan_search = ScopusSearch('(TITLE-ABS-KEY(volcan) OR TITLE-ABS-KEY(volcano) OR TITLE-ABS-KEY(volcanic) OR TITLE-ABS-KEY (volcanism) OR TITLE-ABS-KEY(pyroclas) OR TITLE-ABS-KEY(teprha) OR TITLE-ABS-KEY(lava) OR TITLE-ABS-KEY(lahar))  AND  AFFILCOUNTRY (chile)')

df_volcan = pd.DataFrame(pd.DataFrame(volcan_search.results))

#4. incendios forestales

fire_search = ScopusSearch('(TITLE-ABS-KEY(fire) OR TITLE-ABS-KEY(wildfire) OR TITLE-ABS-KEY(forest fire))  AND  AFFILCOUNTRY (chile)')

df_fire = pd.DataFrame(pd.DataFrame(fire_search.results))

#5. remoción en masa

rem_search = ScopusSearch('(TITLE-ABS-KEY(landslide) OR TITLE-ABS-KEY(debris flow) OR TITLE-ABS-KEY(alluvium) OR TITLE-ABS-KEY(avalanche))  AND  AFFILCOUNTRY(chile)')

df_rem = pd.DataFrame(pd.DataFrame(rem_search.results))

#6. Inundaciones

flood_search = ScopusSearch('(TITLE-ABS-KEY(flood) OR TITLE-ABS-KEY(waterlogging) OR TITLE-ABS-KEY(river overflowing) OR TITLE-ABS-KEY(fluvial hazard) OR TITLE-ABS-KEY(stormwater))  AND  AFFILCOUNTRY (chile)')

df_flood = pd.DataFrame(pd.DataFrame(flood_search.results))

#7. Sequía y desertificación

drough_search = ScopusSearch('(TITLE-ABS-KEY(drought) OR TITLE-ABS-KEY(desertification))  AND  AFFILCOUNTRY(chile)')

df_drough = pd.DataFrame(pd.DataFrame(drough_search.results))

#8. Multi amenaza

multi_search = ScopusSearch('(TITLE-ABS-KEY(multi-hazard) OR TITLE-ABS-KEY(multi-risk))  AND  AFFILCOUNTRY (chile)')

df_multi = pd.DataFrame(pd.DataFrame(multi_search.results))

#9. Generales

general_search = ScopusSearch('(TITLE-ABS-KEY(natural hazard) OR TITLE-ABS-KEY (disaster) OR TITLE-ABS-KEY (extreme event)) AND  AFFILCOUNTRY(chile)')

df_general = pd.DataFrame(pd.DataFrame(general_search.results))

#10. weather
weather_search = ScopusSearch('(TITLE-ABS-KEY(cold wave) OR TITLE-ABS-KEY(heat wave) OR TITLE-ABS-KEY(extreme weather) OR TITLE-ABS-KEY(weather event) OR TITLE-ABS-KEY(climate event) OR TITLE-ABS-KEY(climate variability) OR TITLE-ABS-KEY(extreme temperature)) AND  AFFILCOUNTRY(chile)')

df_weather = pd.DataFrame(pd.DataFrame(weather_search.results))

#11. otras
otras_search = ScopusSearch('(TITLE-ABS-KEY(structural damage) OR TITLE-ABS-KEY(shake table) OR TITLE-ABS-KEY(seismogram) OR TITLE-ABS-KEY(seismograph) OR TITLE-ABS-KEY(tornado) OR TITLE-ABS-KEY(waterspout)) AND  AFFILCOUNTRY(chile)')

df_otras = pd.DataFrame(pd.DataFrame(otras_search.results))

#Juntar Bases
dflist =[df_etq, df_tsunami, df_fire, df_volcan, df_flood, df_drough, df_rem, df_multi, df_general, df_weather, df_otras]
df = pd.concat(dflist, ignore_index=True)
df = df.drop_duplicates(subset=['eid'])
id_data = df['eid'].str.split('-', n = 2, expand = True) 

df.to_excel('.\\backwards_search\base_keywords.xlsx', index=False) #Guardar resultados


