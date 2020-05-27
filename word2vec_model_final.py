import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from livelossplot import PlotLossesKeras
np.random.seed(7)
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LSTM
from keras.layers.embeddings import Embedding
from keras.utils import np_utils
from keras.preprocessing import sequence
from gensim.models import Word2Vec, KeyedVectors, word2vec
import gensim
from gensim.utils import simple_preprocess
from keras.utils import to_categorical
import pickle
import h5py
from time import time
import collections
import random
import re
from collections import Counter
from itertools import islice
import nltk
from nltk.corpus import stopwords
import numpy as np 
import pandas as pd 
pd.set_option('display.max_colwidth', -1)
from time import time
import re
import string
import os
import emoji
from pprint import pprint
import collections
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
sns.set(font_scale=1.3)
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
import gensim
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings('ignore')
np.random.seed(37)  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder   
import imblearn
#!pip install imblearn
from imblearn.over_sampling import SMOTE                        
                           
arabic_diacritics = re.compile(""" ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)


def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', str(text))
    return text



def remove_repeating_char(text):
    # return re.sub(r'(.)\1+', r'\1', text)     # keep only 1 repeat
    return re.sub(r'(.)\1+', r'\1\1', text)  # keep 2 repeat

def process_text(text, grams=False):
    clean_text = remove_diacritics(text)
    clean_text = remove_repeating_char(clean_text)
    
    if grams is False:
        return nltk.tokenize.sent_tokenize(clean_text)
    else:
        tokens = nltk.tokenize.sent_tokenize(clean_text)
        grams = list(window(tokens))
        grams = [' '.join(g) for g in grams]
        grams = grams + tokens
        return grams


def window(words_seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(words_seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
def document_features(document, corpus_features):
    document_words = set(document)
    features = {}
    for word in corpus_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


all_features = list()
texts = list()
data_labels = list()

negative_file = open("C:/Users/121/.spyder-py3/text_mining/negative_tweets.txt", encoding ="utf8")
positive_file = open("C:/Users/121/.spyder-py3/text_mining/positive_tweets.txt", encoding ="utf8")

n_grams_flag = False
min_freq = 13

print('read data ...')
print('read data ...')
# read positive data
for line in positive_file:
    
    text_features = process_text(line, grams=n_grams_flag)
    stop_words = set(stopwords.words('arabic'))
    text_features = [w for w in text_features if not w in stop_words]
    all_features += text_features
    texts.append(text_features)
    data_labels.append('pos')

for line in negative_file:
    
    text_features = process_text(line, grams=n_grams_flag)
    stop_words = set(stopwords.words('arabic'))
    text_features = [w for w in text_features if not w in stop_words]
    all_features += text_features
    texts.append(text_features)
    data_labels.append('neg')
   
df1 = pd.DataFrame(list(zip(texts, data_labels)),
              columns=['texts','data_labels'])

#df1.to_excel ('C:/Users/121/.spyder-py3/text_mining/train.xlsx', index = False, header=True,encoding="utf-8-sig")
df = pd.read_excel('C:/Users/121/.spyder-py3/text_mining/tun.xlsx' , encoding = "utf8")   
#df = pd.read_excel('C:/Users/121/.spyder-py3/text_mining/tun.xlsx' , encoding = "utf8")
X = df['texts']
y = df['data_labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000, binary=True)


X_1 = vect.fit_transform(df['texts']) 
list_key_words = list(vect.vocabulary_.keys())
#TfidfTransformer to Compute Inverse Document Frequency (IDF)
from sklearn.feature_extraction.text import TfidfTransformer
 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(X_1)
#Computing TF-IDF and Extracting Keywords
from scipy.sparse import coo_matrix
from collections import defaultdict
feature_names=vect.get_feature_names()
doc=df['texts'][0] 
tf_idf_vector=tfidf_transformer.transform(vect.transform([doc]))

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        #results[feature_vals[idx]]=score_vals[idx]
         results[feature_vals[idx]]=feature_vals[idx]
    return results



sorted_items=sort_coo(tf_idf_vector.tocoo())
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
key_words_text =[] 
for d in df['texts'] :
    feature_names=vect.get_feature_names()
     
    tf_idf_vector=tfidf_transformer.transform(vect.transform([d]))
    sorted_items=sort_coo(tf_idf_vector.tocoo())
    #extract only the top n; n here is 10
    keywords=extract_topn_from_vector(feature_names,sorted_items,5)
    #key_words_text.append(feature_names)
    key_words_text.append(keywords) 
    
df['key_words_text'] = key_words_text 
df.to_excel ('C:/Users/121/.spyder-py3/text_mining/train2.xlsx', index = False, header=True,encoding="utf-8-sig")
data = pd.read_excel('C:/Users/121/.spyder-py3/text_mining/train2.xlsx' , encoding = "utf8")

#from collections import OrderedDict
#data['key_words_text'].str.split().apply(lambda x: ','.join(OrderedDict.fromkeys(x).keys()))
#data['key_words_text'].str.split().apply(lambda x: ','.join(list(set(x))))

data = data[data['key_words_text'].notnull()]
#remove repeated words in avery row in the column
data['summary'] = data['key_words_text'].apply(lambda x: ' '.join(sorted(set(x.split()), key=x.index)))

#df.drop('column_name', axis=1, inplace=True)

data.to_excel ('C:/Users/121/.spyder-py3/text_mining/train4.xlsx', index = False, header=True,encoding="utf-8-sig")
 
    


































X_train_vect = vect.fit_transform(X_train)
sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)

from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)
print('the score is')
print(nb.score(X_train_res, y_train_res)) 







"""
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print("X_train shape: " + str(X_train.shape))
print("X_test shape: " + str(X_test.shape))
print("X_val shape: " + str(X_val.shape))
print("y_train shape: " + str(y_train.shape))
print("y_test shape: " + str(y_test.shape))
print("y_val shape: " + str(y_val.shape))
#word2vec_model = gensim.models.Word2Vec.load('models/full_grams_cbow_100_twitter.mdl')
model = gensim.models.Word2Vec.load('Twittert-CBOW/tweets_cbow_300')


# Create new observation
new_observation = [[0, 0, 0, 1, 0, 1, 0]]
Predict Observation’s Class
# Predict new observation's class
model.predict(new_observation)


"""













