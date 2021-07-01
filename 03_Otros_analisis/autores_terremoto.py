# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 15:45:02 2020

@author: Inteligencia 1
"""

import os
import pandas as pd


os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia')

df = pd.read_excel(".\\Clas_auto_manual_v3.xlsx")
df = df[df['author_ids'].notna()]

df['year'] = df.coverDate.astype(str).str[:4].astype(int)

df_earth = df[(df['Ame1']=='Terremoto')|(df['Ame2']=='Terremoto')|(df['Ame3']=='Terremoto')]

df_earth = df_earth.reset_index()
del df_earth['index']


#Autores:
    
author_ids = df_earth.author_ids.str.split(';', expand=True).stack().str.strip().reset_index(level=1, drop=True)

df_aut = pd.DataFrame(data=author_ids, columns=['author_ids'])

df_earth['first_aut'] = df_earth['author_ids'].str.rsplit(';').str[0]
df_earth = df_earth.drop(['author_ids','author_names'], axis=1).join(df_aut).reset_index(drop=True)
df_earth['first_author'] = 0
df_earth.loc[(df_earth.first_aut == df_earth.author_ids), 'first_author'] = 1
df_earth['first_isi'] = 0
df_earth.loc[((df_earth.aggregationType == 'Journal')|(df_earth.aggregationType == 'Trade Journal'))&(df_earth.first_author == 1), 'first_isi'] = 1


df_earth['N_papers'] = df_earth.groupby(['author_ids'])['author_ids'].transform('count')
df_earth['N_first'] = df_earth.groupby(['author_ids'])['first_author'].transform('sum')
df_earth['N_first_isi'] = df_earth.groupby(['author_ids'])['first_isi'].transform('sum')


df_aux_isi = df_earth.groupby('author_ids')['aggregationType'].apply(lambda x: ((x=='Journal')|(x=='Trade Journal')).sum()).reset_index(name='N_isi')
df_aux_5 = df_earth.groupby('author_ids')['year'].apply(lambda x: (x>=2015).sum()).reset_index(name='N_5_anos')
df_aux_10 = df_earth.groupby('author_ids')['year'].apply(lambda x: (x>=2010).sum()).reset_index(name='N_10_Anos')

df_aut = df_earth[['author_ids', 'N_papers', 'N_first', 'N_first_isi']]
df_aut = df_aut.drop_duplicates(subset='author_ids', keep="last")

df_aut= df_aut.merge(df_aux_isi, how='left', on='author_ids')
df_aut['N_otros'] = df_aut['N_papers'] - df_aut['N_isi']
df_aut= df_aut.merge(df_aux_5, how='left', on='author_ids')
df_aut= df_aut.merge(df_aux_10, how='left', on='author_ids')

del df_aux_isi, df_aux_10, df_aux_5
df_aut = df_aut[df_aut['author_ids'].notna()]

df_aut.to_excel('.\\autores_etq_temp.xlsx')
