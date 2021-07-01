# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:02:51 2020

@author: Inteligencia 1
"""

import os
import pandas as pd

os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia\\Autores')

df = pd.read_excel(".\\autores_etq_temp.xlsx")
df_tsu = pd.read_excel(".\\autores_tsu_temp.xlsx")
df_vol = pd.read_excel(".\\autores_vol_temp.xlsx")
df_multi = pd.read_excel(".\\autores_multi_temp.xlsx")
df_fire = pd.read_excel(".\\autores_incendios_temp.xlsx")

df_scopus_geo = pd.read_excel(".\\Resultados_autores.xlsx")
df_scopus_fire = pd.read_excel(".\\Resultados_autores_incendios.xlsx")
df_scopus_geo1 = pd.read_excel(".\\Resultados_autores_geo_1.xlsx")
df_scopus_fire1 = pd.read_excel(".\\Resultados_autores_incendios_1.xlsx")
df_scopus = pd.concat([df_scopus_geo, df_scopus_fire, df_scopus_geo1, df_scopus_fire1], ignore_index=True)
df_scopus = df_scopus.drop_duplicates(subset=['author_ids'])


df= df.merge(df_tsu, how='outer', on='author_ids')
df= df.merge(df_vol, how='outer', on='author_ids')
df= df.merge(df_multi, how='outer', on='author_ids')
df= df.merge(df_fire, how='outer', on='author_ids')
df = df.fillna(0)
df = df.drop_duplicates(subset=['author_ids'])

df= df.merge(df_scopus, how='left', on='author_ids')

df = df.drop_duplicates()

#df.to_excel('.\\autores_geo_incendios.xlsx')

df_descargar = df[df['s_name'].isnull()]



import time
from pybliometrics.scopus import ContentAffiliationRetrieval
from pybliometrics.scopus import AuthorRetrieval

os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia\\Autores\\Descarga NANs')

appended_aff = []

#df_descargar =df_descargar[:10]

start_time = time.time()
for index, row in df_descargar.iterrows():
    try:
        au = AuthorRetrieval(row['author_ids'])
        df_descargar.loc[index,'s_name'] = au.indexed_name
        df_descargar.loc[index,'s_surname'] = au.surname
        df_descargar.loc[index,'s_given_name'] = au.given_name
        df_descargar.loc[index,'s_hindex'] =  au.h_index
        df_descargar.loc[index,'s_ncit'] = au.citation_count
        df_descargar.loc[index,'s_ndoc'] = au.document_count
        df_descargar.loc[index,'s_pubyear1'] = au.publication_range[0]
        df_descargar.loc[index,'s_pubyear2'] = au.publication_range[1]
        df_descargar.loc[index,'s_coauthor'] = au.coauthor_count
        aff = ContentAffiliationRetrieval(au.affiliation_current[0][0])
        df_descargar.loc[index,'s_aff_current'] = aff.affiliation_name
        df_descargar.loc[index,'s_aff_id'] = au.affiliation_current[0][0]
        df_descargar.loc[index,'s_aff_city'] = aff.city
        df_descargar.loc[index,'s_aff_country'] = aff.country
        df_descargar.loc[index,'s_aff_type'] = aff.org_type
        df_descargar.loc[index,'s_org_domain'] = aff.org_domain
        df_descargar.loc[index, 's_org_url'] = aff.org_URL
        #Lista afiliaciones 
        aff_history = pd.DataFrame(au.affiliation_history)
        aff_history.shape
        aff_history.rename(columns={0:'affiliation_eid'}, inplace=True)
        aff_history['author'] = au.indexed_name
        aff_history['author_eid'] = row['author_ids']
        aff_history['current'] = 0
        aff_history.loc[aff_history['id'] == au.affiliation_current[0][0], 'current'] = 1
        aff_current = aff_history[aff_history['current'] == 1]
        df_descargar.loc[index, 's_parent'] = aff_history.loc[0, 'parent_preferred_name']    
        appended_aff.append(aff_history)
       
    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))

afiliaciones = pd.concat(appended_aff, ignore_index=True)
afiliaciones = afiliaciones.reset_index(drop=True)

df_descargar.loc[df_descargar['s_parent'].isnull(), 's_parent'] = df_descargar['s_aff_current']

df_descargar.to_excel('.\\Resultados_autores.xlsx')
afiliaciones.to_excel('.\\Resultados_autores_aff.xlsx')

# de aquí para abajo solo sirve para identificar qué autores han publicado alguna vez con afiliación chilena

afiliaciones.loc[afiliaciones['type'] == 'parent', 'parent_preferred_name'] = afiliaciones['preferred_name']
afiliaciones.loc[afiliaciones['type'] == 'parent', 'parent'] = afiliaciones['id']

afiliaciones = afiliaciones.drop_duplicates(subset=['parent','author_eid'], keep="last")


afiliaciones = afiliaciones.dropna(subset=['preferred_name'])
afiliaciones.loc[afiliaciones['city'].isnull(), 'city'] = 'S/I'
afiliaciones.loc[afiliaciones['country'].isnull(), 'country'] = 'S/I'

#unstack

df_descargar_aff =afiliaciones.groupby('author_eid')['parent_preferred_name'].agg([('parent_preferred_name', '; '.join)]).reset_index()
df_descargar_aff2 =afiliaciones.groupby('author_eid')['country'].agg([('country', '; '.join)]).reset_index()
df_descargar_aff3 =afiliaciones.groupby('author_eid')['city'].agg([('city', '; '.join)]).reset_index()


#
df_descargar= df_descargar.merge(df_descargar_aff, how='left', left_on='author_ids', right_on='author_eid')
df_descargar= df_descargar.merge(df_descargar_aff2, how='left', left_on='author_ids', right_on='author_eid')
df_descargar= df_descargar.merge(df_descargar_aff3, how='left', left_on='author_ids', right_on='author_eid')

del df_descargar_aff, df_descargar_aff2, df_descargar_aff3
del df_descargar['author_eid'], df_descargar['author_eid_x'], df_descargar['author_eid_y']

df_descargar['aff_chile'] = 0

df_descargar.loc[df_descargar['country'].isnull(), 'country'] = 'S/I'
df_descargar.loc[df_descargar.country.str.contains('Chile', regex=False), 'aff_chile'] = 1

df_descargar.to_excel('.\\Resultados_autores_cl.xlsx')


######### unir ambas bases
os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia\\Autores\\Descarga NANs')
df_descargar= pd.read_excel('.\\Resultados_autores.xlsx')
#quitar NAs de la base principal

df = df[df['s_name'].notna()]
#df_descargar = df_descargar.iloc[:,:-4]

df = pd.concat([df, df_descargar], ignore_index=True)

df.to_excel('.\\Autores_total_geo_incendios.xlsx')





