# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:04:23 2020

@author: belen
"""
# Este código hace una limpieza del texto de los registros de Scopus y genera un modelo de tópicos
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 16:02:06 2020

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
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.collocations import *
from collections import Counter

#Herramientas de Gemsim
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
# Sklearn
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

# Plotting tools
import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt
import pyLDAvis.gensim

# Spacy
import spacy

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\bases')

df = pd.read_excel(".\\base_final.xlsx")

#0- quitar ruido al abstract :

df['description'] = df.description.str.replace('^\W+', '')
df['description'] = df.description.str.replace('\N{COPYRIGHT SIGN}', '')
df['description'] = df.description.str.replace('Copyright+', '')
df['description'] = df.description.str.replace('^\W+', '')
df['description'] = df.description.str.replace('^\d+', '')
df['description'] = df.description.str.replace('^\W+', '')
df['descrpition'] = df.description.str.replace('^\d+', '')
df['description'] = df.description.str.replace('^\W+', '')

#1- Juntar título y abstract

df['text'] = df['title'].str.cat(df['description'],sep=" ")

#2- todo a minúscula

df['text'] = df['text'].str.lower()

#3- quitar puntuaciones

df['text'] = df['text'].str.replace('-+', ' ')
df['text'] = df['text'].str.replace('[^\w\s]','')
df['text'] = df['text'].str.replace(' +', ' ')


#### Eliminar bigramas/trigramas comunes
df['text'] = df.text.str.replace('copyright+', '')
df['text'] = df.text.str.replace('by the authors+', '')
df['text'] = df.text.str.replace('the authors+', '')
df['text'] = df.text.str.replace('the author+', '')
df['text'] = df.text.str.replace('authors+', '')
df['text'] = df.text.str.replace('author+', '')
df['text'] = df.text.str.replace('los autores+', '')
df['text'] = df.text.str.replace('published by+', '')
df['text'] = df.text.str.replace('ltd+', '')
df['text'] = df.text.str.replace('llc+', '')
df['text'] = df.text.str.replace('bv+', '')
df['text'] = df.text.str.replace('ieee+', '')
df['text'] = df.text.str.replace('inc+', '')
df['text'] = df.text.str.replace('elsevier science+', '')
df['text'] = df.text.str.replace('elsevier+', '')
df['text'] = df.text.str.replace('all rights reserved+', '')
df['text'] = df.text.str.replace('rights reserved+', '')
df['text'] = df.text.str.replace('john wiley sons+', '')
df['text'] = df.text.str.replace('american geophysical union+', '')
df['text'] = df.text.str.replace('springer sciencebusiness media dordrecht', '')
df['text'] = df.text.str.replace('springer sciencebusiness media new york', '')
df['text'] = df.text.str.replace('springer sciencebusiness media singapore', '')
df['text'] = df.text.str.replace('springer sciencebussines media+', '')
df['text'] = df.text.str.replace('taylor francis group+', '')
df['text'] = df.text.str.replace('american geophysical union+', '')
df['text'] = df.text.str.replace('springerverlag berlin heidelberg+', '')
df['text'] = df.text.str.replace('verlag berlin heidelberg+', '')
df['text'] = df.text.str.replace('springervelag+', '')
df['text'] = df.text.str.replace('springer international publishing ag+', '')
df['text'] = df.text.str.replace('springer international publishing switzerland+', '')
df['text'] = df.text.str.replace('springer international publishing+', '')
df['text'] = df.text.str.replace('springer international+', '')
df['text'] = df.text.str.replace('springer nature switzerland+', '')
df['text'] = df.text.str.replace('springer nature+', '')
df['text'] = df.text.str.replace('springer+', '')
df['text'] = df.text.str.replace('american astronomical society+', '')
df['text'] = df.text.str.replace('geological society of america+', '')
df['text'] = df.text.str.replace('oxford university press+', '')
df['text'] = df.text.str.replace('sociedad chilena de la ciencia del suelo', '')
df['text'] = df.text.str.replace('earthquake engineering research institute', '')
df['text'] = df.text.str.replace('american society of civil engineering', '')
df['text'] = df.text.str.replace('verlag gmbh germany part of', '')
df['text'] = df.text.str.replace('verlag gmbh', '')
df['text'] = df.text.str.replace('gmbh germany part of', '')
df['text'] = df.text.str.replace('sociedad de biología de chile', '')
df['text'] = df.text.str.replace('pontificia universidad catolica de chile', '')
df['text'] = df.text.str.replace('instituto de investigaciones agropecuarias', '')
df['text'] = df.text.str.replace('inia', '')
df['text'] = df.text.str.replace('associazione geotecnica italiana', '')
df['text'] = df.text.str.replace('escuela de ciencias del mar', '')
df['text'] = df.text.str.replace('servicio nacional de geologia y mineria', '')
df['text'] = df.text.str.replace('this is an open access article distributed under the terms of the creative commons attribution license', '')
df['text'] = df.text.str.replace('this is an open access article distributed under the terms of the creative commons attribution', '')
df['text'] = df.text.str.replace('which permits unrestricted use distribution and reproduction in any medium', '')
df['text'] = df.text.str.replace('provided the original source', '')
df['text'] = df.text.str.replace('credited', '')
df['text'] = df.text.str.replace('et al', '')
df['text'] = df['text'].str.replace(' +', ' ')


#Sacar stopwords y palabras con menos de 4 caracteres
STOPWORDS = set(stopwords.words('english'))
def stop(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS and len(word)>3])
df["text"] = df["text"].apply(stop)


all_text = ' '.join(df['text']).split()

# número de tokens (palabras):
len(all_text)
# número de tokens únicos (palabras):
len(set(all_text))

freq = pd.Series(all_text, name='frecuencia').value_counts().rename_axis('frecuencia')

#quitar tokens con 1 o repeticiones 
#Cost0 = brl[blr['Renewal % Change']=='0%']['Account Name'].tolist()

least_freq = freq.loc[freq==1]
least_freq = list(least_freq.index)

# eliminar palabras que se repiten una sola vez (23 022 de 54 433)
def unicas(text):
    return " ".join([word for word in str(text).split() if word not in least_freq])

df["text"] = df["text"].apply(unicas)

#- tokenización

df['tok_text'] = df['text'].apply(word_tokenize)

# Stemming

stemmer = SnowballStemmer("english")
df['stemmed_text'] = df['tok_text'].apply(lambda x: [stemmer.stem(y) for y in x]) 


# Lematización

lemmatizer = nltk.stem.WordNetLemmatizer()
df['lem_text'] = df['stemmed_text'].apply(lambda x: [lemmatizer.lemmatize(y) for y in x])

# Quitar palabras comunes en textos e investigación científica

#paper_terms = ['paper', 'present',1 'presents','work', 'research', 'study', 'article', 'result', 'results']
paper_terms = ['paper', 'present', 'present','work', 'research', 'studi', 'article', 'result', 'approach', 'method', 'analysi', ]
lugares = ['chile']

#df["tok_text"] = df["tok_text"].apply(lambda x: [item for item in x if item not in paper_terms])
#df["tok_text"] = df["tok_text"].apply(lambda x: [item for item in x if item not in lugares])

df["lem_text"] = df["lem_text"].apply(lambda x: [item for item in x if item not in paper_terms])
df["lem_text"] = df["lem_text"].apply(lambda x: [item for item in x if item not in lugares])


# Crear BIGRAMAS Y TRIGRAMAS

data_words = df.lem_text.values.tolist()
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

data_words_bigrams = make_bigrams(data_words)


#LDA - Sklearn

# 1. Create the Document-Word matrix

df['lem_text']=[" ".join(words) for words in df['lem_text'].values]
corpus = df.lem_text.values.tolist()


TOKENS_BASIC = '\\S+(?=\\s+)'

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,                        # n° mínimo de ocurrencias por palabra
                             stop_words='english',             # remover stop words
                             token_pattern=TOKENS_BASIC)

data_vectorized = vectorizer.fit_transform(corpus)

lda_model = LatentDirichletAllocation(n_components=25, # Número de tópicos
                                      learning_method='online',
                                      random_state=0,       
                                      n_jobs = -1  # Usar todas las CPUs disponibles
                                     )

lda_output = lda_model.fit_transform(data_vectorized)

pyLDAvis.enable_notebook()
panel =pyLDAvis.sklearn.prepare(lda_model, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.show(panel)


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




#GENSIM
# Create Dictionary
#id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
#texts = data_lemmatized

# Term Document Frequency
#corpus = [id2word.doc2bow(text) for text in texts]

#lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
 #                                          id2word=id2word,
  #                                         num_topics=25, 
 #                                         random_state=100,
  #                                         update_every=1,
   #                                        chunksize=100,
    #                                       passes=10,
     ##                                     per_word_topics=True)

#pprint(lda_model.print_topics())
#doc_lda = lda_model[corpus]

# Compute Perplexity
#print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
#coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
#coherence_lda = coherence_model_lda.get_coherence()
#print('\nCoherence Score: ', coherence_lda)


# Visualize the topics
#pyLDAvis.enable_notebook()
#panel =pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
#pyLDAvis.show(panel)
     
     
     
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


#cruce con df
df_document_topic = df_document_topic.reset_index()
del df_document_topic['index']

df_cruce = df.filter(['eid','title','description'], axis=1)

df_new = pd.concat([df_cruce, df_document_topic], axis=1)


#Clasificaciones manuales:

df_eti = pd.read_excel(".\\etiquetado_manual.xlsx")

df_labels_topics = df_eti.merge(df_new, how="inner", on='eid')
df_labels_topics.to_excel('.\\df_labels_topics.xlsx', index=False) #Guardar base final, método búsqueda por autor
