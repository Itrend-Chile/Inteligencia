# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:40:06 2020

@author: belen
"""
# este codigo incluye algunos modelos de clasificacion supervisada para analisis de texto 


import os
import pandas as pd
#import numpy as np
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize, FreqDist
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('averaged_perceptron_tagger')
from collections import Counter

from gensim.models import Phrases

'''Features'''
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize

'''Classifiers'''

from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

'''Metrics/Evaluation'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from scipy import interp
from itertools import cycle

os.chdir('C:\\Users\\belen\\Proyectos\\Mapeos\\clasificacion supervisada')

#papers preprocesados:

df_labels = pd.read_excel(".\\labels.xlsx")
df_papers = pd.read_excel(".\\papers_desastres.xlsx")
df = pd.read_excel(".\\papers_preprocesados.xlsx")

#df_labels = pd.concat([df_labels, df_papers], axis=1)

#df_labels = df_labels([df_labels, df_papers], axis=1)

df_labels.drop_duplicates(subset ="eid", 
                     keep = False, inplace = True)

df_labels = df_labels[df_labels['dis'] == 1] 

#crear columna con clases:

df_labels['label'] = 0
df_labels.loc[df_labels['tsu'] > 0, 'label'] = 1
df_labels.loc[df_labels['vol'] > 0, 'label'] = 2
df_labels.loc[df_labels['fire'] > 0, 'label'] = 3
df_labels.loc[df_labels['land'] > 0, 'label'] = 4
df_labels.loc[df_labels['clim'] > 0, 'label'] = 5
df_labels.loc[df_labels['multi'] > 0, 'label'] = 6

del df_labels['earth'],df_labels['tsu'],df_labels['vol'],df_labels['fire'],df_labels['land'],df_labels['clim'],df_labels['multi']

df_labels = pd.merge(df_labels,df[['eid','text']], on='eid', how='inner')
#df_labels = pd.merge(df_labels,df2[['eid','text']], on='eid', how='inner')


desastres = df_labels.groupby(['dis']).size()
amenazas = df_labels.groupby(['label']).size()


#- tokenización

df_labels['tok_text'] = df_labels['text'].apply(word_tokenize)

lemmatizer = nltk.stem.WordNetLemmatizer()


#lem_text_list =df_labels['tok_text'].tolist()
#wf = word_freq(lem_text_list,20)
#wf.head(20)

def get_word_net_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
    
def lemma_wordnet(tagged_text):
    final = []
    for word, tag in tagged_text:
        wordnet_tag = get_word_net_pos(tag)
        if wordnet_tag is None:
            final.append(lemmatizer.lemmatize(word))
        else:
            final.append(lemmatizer.lemmatize(word, pos=wordnet_tag))
    return final


df_labels['tagged'] = df_labels['tok_text'].apply(nltk.pos_tag)
df_labels['lem_text'] = df_labels['tagged'].apply(lemma_wordnet)


def word_count(text):
    return len(str(text).split(' '))

def word_freq(clean_text_list, top_n):
    """
    Word Frequency
    """
    flat = [item for sublist in clean_text_list for item in sublist]
    with_counts = Counter(flat)
    top = with_counts.most_common(top_n)
    word = [each[0] for each in top]
    num = [each[1] for each in top]
    return pd.DataFrame([word, num]).T


cl_text_list = df_labels['lem_text'].tolist()
wf = word_freq(cl_text_list, 20)
wf.head(20)

def word_freq_bigrams(clean_text_list, top_n):
    """
    Word Frequency With Bigrams
    """
    bigram_model = Phrases(clean_text_list, min_count=2, threshold=1)
    w_bigrams = list(bigram_model[clean_text_list])
    flat_w_bigrams = [item for sublist in w_bigrams for item in sublist]
    with_counts = Counter(flat_w_bigrams)
    top = with_counts.most_common(top_n)
    word = [each[0] for each in top]
    num = [each[1] for each in top]
    return pd.DataFrame([word, num]).T


def bigram_freq(clean_text_list, top_n):
    bigram_model = Phrases(clean_text_list, min_count=2, threshold=1)
    w_bigrams = list(bigram_model[clean_text_list])
    flat_w_bigrams = [item for sublist in w_bigrams for item in sublist]
    bigrams = []
    for each in flat_w_bigrams:
        if '_' in each:
            bigrams.append(each)
    counts = Counter(bigrams)
    top = counts.most_common(top_n)
    word = [each[0] for each in top]
    num = [each[1] for each in top]
    return pd.DataFrame([word, num]).T

bf = bigram_freq(cl_text_list, 20)
bf.head(20)

#Creating the features (tf-idf weights) for the processed text

texts = df_labels['lem_text'].astype('str')

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), 
                                   min_df = 2, 
                                   max_df = .95)

X = tfidf_vectorizer.fit_transform(texts) #features
y = df_labels['label'].values #target

#Dimenionality reduction. Only using the 100 best features per category

lsa = TruncatedSVD(n_components=100, 
                   n_iter=10, 
                   random_state=3)

X = lsa.fit_transform(X)
X.shape
y.shape

# MODELO DESASTRE . NO - DESASTRE

#Creating a dict of the models

model_dict = {'SVM' : LinearSVC(),
              'Random Forest': RandomForestClassifier(random_state=3),
              'Decsision Tree': DecisionTreeClassifier(random_state=3),
              'AdaBoost': AdaBoostClassifier(random_state=3),
              'K Nearest Neighbor': KNeighborsClassifier()}

#Train test split with stratified sampling for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .3, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 3)

#Function to get the scores for each model in a df
def model_score_df(model_dict):   
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k,v in model_dict.items():   
        model_name.append(k)
        v.fit(X_train, y_train)
        y_pred = v.predict(X_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
    return model_comparison_df

model_score_df(model_dict)

#Preliminary model evaluation using default parameters

#Creating a dict of the models

model_dict = {'SVM' : LinearSVC(),
              'Random Forest': RandomForestClassifier(random_state=3),
              'Decsision Tree': DecisionTreeClassifier(random_state=3),
              'AdaBoost': AdaBoostClassifier(random_state=3),
              #'Gaussian Naive Bayes': MultinomialNB(),
              'K Nearest Neighbor': KNeighborsClassifier()}

#Train test split with stratified sampling for evaluation
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = .3, 
                                                    shuffle = True, 
                                                    stratify = y, 
                                                    random_state = 3)

#Function to get the scores for each model in a df
def model_score_df(model_dict):   
    model_name, ac_score_list, p_score_list, r_score_list, f1_score_list = [], [], [], [], []
    for k,v in model_dict.items():   
        model_name.append(k)
        v.fit(X_train, y_train)
        y_pred = v.predict(X_test)
        ac_score_list.append(accuracy_score(y_test, y_pred))
        p_score_list.append(precision_score(y_test, y_pred, average='macro'))
        r_score_list.append(recall_score(y_test, y_pred, average='macro'))
        f1_score_list.append(f1_score(y_test, y_pred, average='macro'))
        model_comparison_df = pd.DataFrame([model_name, ac_score_list, p_score_list, r_score_list, f1_score_list]).T
        model_comparison_df.columns = ['model_name', 'accuracy_score', 'precision_score', 'recall_score', 'f1_score']
        model_comparison_df = model_comparison_df.sort_values(by='f1_score', ascending=False)
    return model_comparison_df

model_score_df(model_dict)

rf = RandomForestClassifier()
svm = LinearSVC()
knn = KNeighborsClassifier()
rf.fit(X, y)
svm.fit(X, y)
knn.fit(X, y)

#from sklearn import metrics
#print(metrics.classification_report(y_test, y_pred, target_names=df_labels['label'].unique()))

#Predecir clase del resto de la base

df = df.iloc[:, : 36]

df['tok_text'] = df['text'].apply(word_tokenize)
df['tagged'] = df['tok_text'].apply(nltk.pos_tag)
df['lem_text'] = df['tagged'].apply(lemma_wordnet)

#Creating the features (tf-idf weights) for the processed text
texts_unlab = df['lem_text'].astype('str')

tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), 
                                   min_df = 2, 
                                   max_df = .95)

X_unlab = tfidf_vectorizer.fit_transform(texts_unlab) #features
#y = df2['label_num'].values #target

#Dimenionality reduction. Only using the 100 best features
lsa = TruncatedSVD(n_components=100, 
                   n_iter=10, 
                   random_state=3)

X_unlab = lsa.fit_transform(X_unlab)


df['pred_svm'] = svm.predict(X_unlab)
df['pred_rf'] = rf.predict(X_unlab)
df['pred_knn'] = knn.predict(X_unlab)


df_labels['pred_svm'] = svm.predict(X)
df_labels['pred_rf'] = rf.predict(X)
df_labels['pred_knn'] = knn.predict(X)


##################################################

df_labels['pred_svm'] = svm.predict(X)
df['pred_knn'] = KNeighborsClassifier.predict(X)
