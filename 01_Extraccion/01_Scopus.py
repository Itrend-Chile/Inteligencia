# -*- coding: utf-8 -*-

"""
Created on Thu Jan 30 17:21:13 2020


@author: belen
"""

import os
import pandas as pd
import time
import random
from pybliometrics.scopus import ContentAffiliationRetrieval
from pybliometrics.scopus import AuthorRetrieval
from pybliometrics.scopus import AbstractRetrieval


os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\bases')
pd.set_option('precision', 15)

# 1) Base de dato autores:

df = pd.read_excel(".\\Base_academicos.xlsx")
del df['scopus_eid2'], df['scopus_eid3']
df = df.loc[df['Academico_Investigador'] == 'X']
df['scopus_eid'] = df['scopus_eid'].astype('Int64')

df['s_name'] = ""
df['s_surname'] = ""
df['s_given_name'] = ""
df['s_hindex'] = ""
df['s_ncit'] = ""
df['s_ndoc'] = ""
df['s_pubyear1'] = ""
df['s_pubyear2'] = ""
df['s_aff_current'] = ""
df['s_aff_id'] = ""
df['s_aff_city'] = ""
df['s_aff_country'] = ""
df['s_aff_type'] = ""
df['s_org_domain'] = ""
df['s_org_url'] = ""


appended_eids = []
appended_sub = []
appended_aff = []

# 2) Loop busca la info del perfil de cada autor y publicaciones asociadas:

start_time = time.time()
for index, row in df.iterrows():
    try:
        time.sleep(3+random.uniform(0, 1))
        au = AuthorRetrieval(row['scopus_eid'])
        aff = ContentAffiliationRetrieval(au.affiliation_current)
        df.loc[index,'s_name'] = au.indexed_name
        df.loc[index,'s_surname'] = au.surname
        df.loc[index,'s_given_name'] = au.given_name
        df.loc[index,'s_hindex'] =  au.h_index
        df.loc[index,'s_ncit'] = au.citation_count
        df.loc[index,'s_ndoc'] = au.document_count
        df.loc[index,'s_pubyear1'] = au.publication_range[0]
        df.loc[index,'s_pubyear2'] = au.publication_range[1]
        df.loc[index,'s_aff_current'] = aff.affiliation_name
        df.loc[index,'s_aff_id'] = au.affiliation_current
        df.loc[index,'s_aff_city'] = aff.city
        df.loc[index,'s_aff_country'] = aff.country
        df.loc[index,'s_aff_type'] = aff.org_type
        df.loc[index,'s_org_domain'] = aff.org_domain
        df.loc[index,'s_org_url'] = aff.org_URL
        #Lista papers
        eids = pd.DataFrame(au.get_document_eids(refresh=False))
        eids.shape
        eids.rename(columns={0:'doc_eid'}, inplace=True)
        #Lista áreas de interés
        subjects = pd.DataFrame(au.subject_areas)
        subjects.shape
        subjects['author'] = au.indexed_name
        subjects['author_eid'] = row['scopus_eid']
        #Lista afiliaciones 
        aff_history = pd.DataFrame(au.affiliation_history)
        aff_history.shape
        aff_history.rename(columns={0:'affiliation_eid'}, inplace=True)
        aff_history['author'] = au.indexed_name
        aff_history['author_eid'] = row['scopus_eid']
        aff_history['current'] = 0
        aff_history.loc[aff_history['affiliation_eid'] == au.affiliation_current, 'current'] = 1
        
        appended_eids.append(eids)
        appended_sub.append(subjects)
        appended_aff.append(aff_history)
       
    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))
df.to_excel('.\\Resultados_scopus_académicos.xlsx')

#Bases de datos subjects y aff history
areas = pd.concat(appended_sub, ignore_index=True)
afiliaciones = pd.concat(appended_aff, ignore_index=True)
areas.to_excel('.\\Resultados_scopus_areas.xlsx')
afiliaciones.to_excel('.\\Resultados_scopus_afiliaciones.xlsx')

# 3) Base de datos papers:
        
docs = pd.concat(appended_eids, ignore_index=True)
# variable ID de búsqueda:
id_data = docs['doc_eid'].str.split('-', n = 2, expand = True) 
docs['eid'] = id_data[2]
docs = docs.drop_duplicates(subset=['eid'])
docs = docs.reset_index(drop=True)


master_docs = pd.read_excel(".\\Resultados_scopus_docus.xlsx")
id_data = master_docs['doc_eid'].str.split('-', n = 2, expand = True) 
master_docs['eid'] = id_data[2]
master_docs = master_docs[~master_docs['eid'].astype(str).str.startswith('0')]
master_docs = master_docs[master_docs['title'].notna()]


id_list1 = master_docs['eid'].tolist()
id_list2 = docs['eid'].tolist()

update = list(set(id_list2) - set(id_list1))

docs = docs[docs['eid'].isin(update)]
#docs = pd.DataFrame(docs)
#docs.rename(columns={0:'eid'}, inplace=True)
#del id_list1, id_list2, docs0, appended_eids, appended_sub, appended_aff
# variables a buscar en scopus:

docs['title'] = '' #título artículo
docs['pub_name'] = '' # nombre publicación
docs['agg_type'] = '' # tipo de agregación (ex: journal, conference)
docs['date'] =  '' # fecha de publicación
docs['first_auth'] =  '' # autor 1
docs['first_eid'] =  '' # ID autor 1
docs['description'] =  '' # descripción
docs['page_range'] =  '' # rango de páginas
docs['ncitas'] =  '' # numero de citas
docs['doi'] =  '' # DOI
docs['issn'] =  '' # ISSN

# 4) Loop busca toda la info de cada paper:
appended_au = []
appended_aff1 = []


start_time = time.time()
for index, row in docs.iterrows():
    try:
        time.sleep(3+random.uniform(0, 1))
        ab = AbstractRetrieval(row['eid'])
        docs.loc[index,'title'] = ab.title
        docs.loc[index,'pub_name'] = ab.publicationName
        docs.loc[index,'agg_type'] = ab.aggregationType
        docs.loc[index,'date'] = ab.coverDate
        docs.loc[index,'first_auth'] = ab.authors[0][1]
        docs.loc[index,'first_eid'] = ab.authors[0][0]
        docs.loc[index,'description'] = ab.description
        docs.loc[index,'page_range'] = ab.pageRange   
        docs.loc[index,'ncitas'] = ab.citedby_count
        docs.loc[index,'doi'] = ab.doi
        docs.loc[index,'issn'] = ab.issn
        autores = pd.DataFrame(ab.authors)
        autores['eid'] = row['eid']
        autores['title'] = ab.title
        affil = pd.DataFrame(ab.affiliation)
        affil['eid'] = row['eid']
        affil['title'] = ab.title
        appended_au.append(autores)
        appended_aff1.append(affil)

    except:
        pass
print("--- %s seconds ---" % (time.time() - start_time))

#docs = pd.concat([master_docs, docs], ignore_index=True)

docs.to_excel('.\\Resultados_scopus_docus.xlsx')
# 5) 

autores = pd.concat(appended_au, ignore_index=True)
autores.to_excel('.\\Resultados_scopus_docus_autores.xlsx')

afiliaciones = pd.concat(appended_aff1, ignore_index=True)
afiliaciones.to_excel('.\\Resultados_scopus_docus_afiliaciones.xlsx')

docs = pd.read_excel(".\\Resultados_scopus_docus.xlsx")
docs.dropna(subset=['description'], inplace=True)
docs.reset_index(drop=True)

#sample = docs.sample(frac=0.3, replace=True, random_state=1)
#sample1 = sample[:1202]
#sample2 = sample[1203:2405]
#sample1 = sample1.reset_index(drop=True)
#sample2 = sample2.reset_index(drop=True)

#sample1.to_excel('.\\etiquetado\etiquetado1.xlsx', index=False)
#sample2.to_excel('.\\etiquetado\etiquetado2.xlsx', index=False)

