# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:28:09 2020

@author: belen
"""

import os
import pandas as pd
import numpy as np

#NLTK
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import word_tokenize

import gensim
from gensim.utils import simple_preprocess

import pyLDAvis
import pyLDAvis.sklearn

import spacy
import en_core_web_sm

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

os.chdir('C:\\Users\\Inteligencia 1\\Documents\\Inteligencia')

df = pd.read_excel(".\\Clas_auto_manual_v2.xlsx")

df['year'] = df.coverDate.astype(str).str[:4]


#- tokenización

df['tok_text'] = df['text'].apply(word_tokenize)


df_geo = df[(df['earth_topic']>=0)|(df['tsu_topic']>=0)|(df['vol_topic']>=0)]
df_geo = df_geo[(df_geo['Ame1']=='Terremoto')|(df_geo['Ame1']=='Multi')|(df_geo['Ame1']=='Tsunami')|(df_geo['Ame1']=='Volcanismo')]

#- Solo terremoto

df_earth = df[df['earth_topic']>=0]
df_earth = df_earth[(df_earth['Ame1']=='Terremoto')|(df_earth['Ame1']=='Multi')|(df_earth['Ame2']=='Terremoto')|(df_earth['Ame3']=='Terremoto')]

#- Solo tsunami

df_tsu = df[df['tsu_topic']>=0]
df_tsu = df_tsu[(df_tsu['Ame1']=='Tsunami')|(df_tsu['Ame1']=='Multi')|(df_tsu['Ame2']=='Tsunami')|(df_tsu['Ame3']=='Tsunami')]

#- Solo volcán

df_vol = df[df['vol_topic']>=0]
df_vol = df_vol[(df_vol['Ame1']=='Volcanismo')|(df_vol['Ame1']=='Multi')|(df_vol['Ame2']=='Volcanismo')|(df_vol['Ame3']=='Volcanismo')]


#Seleccionar amenaza
df= df_vol
df = df.reset_index()
del df['index']

# Crear BIGRAMAS Y TRIGRAMAS
data_words = df.tok_text.values.tolist()
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
bigram_mod = gensim.models.phrases.Phraser(bigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Form Bigrams
data_words_bigrams = make_bigrams(data_words)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = en_core_web_sm.load()
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

corpus = list(map(' '.join, data_lemmatized))

#LDA - Sklearn

# 1. Create the Document-Word matrix

#lemmatizer = nltk.stem.WordNetLemmatizer()
#df['lem_text'] = df['tok_text'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])
#df['lem_text']=[" ".join(words) for words in df['lem_text'].values]
#corpus = df.lem_text.values.tolist()


TOKENS_BASIC = '\\S+(?=\\s+)'

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # n° mínimo de ocurrencias por palabra
                             stop_words='english',             # remover stop words
                             token_pattern=TOKENS_BASIC)

data_vectorized = vectorizer.fit_transform(corpus)

lda_model = LatentDirichletAllocation(n_components=4, # Número de tópicos
                                      learning_method='online',
                                      random_state=0,       
                                      n_jobs = -1  # Usar todas las CPUs disponibles
                                     )

lda_output = lda_model.fit_transform(data_vectorized)

#pyLDAvis.enable_notebook()
#panel =pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
#pyLDAvis.show(panel)


# Show top 20 keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]

# Asignar tópico dominante a cada documento
     
      #### Document-topic matrix
     
     # column names
topicnames = ['Topic' + str(i) for i in range(lda_model.n_components)]
# index names
docnames = [str(i) for i in range(len(df))]
# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)

df_document_topic['dominant_topic'] = dominant_topic
#df_document_topic['x_1_topic_probability'] = dominant_topic

df_document_topic['x_1_topic_probability'] = df_document_topic.max(axis=1)

#cruce con df
df_document_topic = df_document_topic.reset_index()
del df_document_topic['index']
df_cruce = df.filter(['eid','title','description', 'text'], axis=1)

#df = pd.concat([df_cruce, df_document_topic], axis=1)
df = df_cruce.merge(df_document_topic, left_index=True, right_index=True)

df.to_excel('.\\tsunami4topicos.xlsx', index=False) 


########################
# Reducción de dimensionalidad 

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, verbose=1, random_state=7, angle=.99, init='pca')
# 13-D -> 2-D
tsne_lda = tsne_model.fit_transform(lda_output) # doc_topic is document-topic matrix from LDA or GuidedLDA

df_document_topic['x_tsne'] = tsne_lda[:,0]
df_document_topic['y_tsne'] = tsne_lda[:,1]

df_document_topic.to_excel('.\\document_topic_tsu.xlsx', index=False) 


########################















