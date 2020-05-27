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
        return clean_text.split()
    else:
        tokens = clean_text.split()
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

data = pd.DataFrame(list(zip(texts, data_labels)),
              columns=['texts','data_labels'])
data.to_excel ('C:/Users/121/.spyder-py3/text_mining/tun.xlsx', index = False, header=True,encoding="utf-8-sig")
print('data size', len(data_labels))
print('# of positive', data_labels.count('pos'))
print('# of negative', data_labels.count('neg'))  
  
tweets = [(t, l) for t, l in zip(texts, data_labels)]

random.shuffle(tweets)
print('sample tweets')






for t in tweets[:10]: print(t)  # see the first 10 instances
print('all words sample')
print(all_features[:20])
print('len(all_words):', len(all_features))
all_features_freq = nltk.FreqDist(w for w in all_features)
df_fdist = pd.DataFrame.from_dict(all_features_freq, orient='index')
df_fdist.columns = ['Frequency']
df_fdist.index.name = 'Term'

df_fdist.to_excel ('C:/Users/121/.spyder-py3/text_mining/freq.xlsx', index = False, header=True,encoding="utf-8-sig")


print(all_features_freq)
print('sample frequencies')
print(all_features_freq.most_common(100))

print('freq of في', all_features_freq.freq('في'))
print('features frequencies are computed')
thr = min_freq / len(all_features)
print('selecting features')

# remove features that have frequency below the threshold
my_features = set([word for word in all_features if all_features_freq.freq(word) > thr])

my_features_ = list(all_features_freq)[:500]  # top 3k features

print(len(my_features), 'are kept out of', len(all_features))
print('features are selected')
print('------------------------------------')
print('sample features:')
print(list(my_features)[:100])
print('------------------------------------')


print('generating features for documents ...')
feature_sets = [(document_features(d, my_features), c) for (d, c) in tweets]

print('splitting documents into train and test ...')
print('data set size', len(data_labels))
train_percentage = 0.8
splitIndex = int(len(tweets) * train_percentage)
train_set, test_set = feature_sets[:splitIndex], feature_sets[splitIndex:]
y_train = [l for t, l in train_set]
y_test = [l for t, l in test_set]

print('data set:', Counter(data_labels))
print('train:', Counter(y_train))
print('test:', Counter(y_test))

print('training NaiveBayes classifier ...')
classifier = nltk.NaiveBayesClassifier.train(train_set)
print('training is done')

ref_sets = collections.defaultdict(set)
test_sets = collections.defaultdict(set)

for i, (feats, label) in enumerate(test_set):
    ref_sets[label].add(i)
    observed = classifier.classify(feats)
    test_sets[observed].add(i)
# calculates f1 for 1:100 dataset with 95tp, 5fn, 55fp

print('accuracy:', nltk.classify.accuracy(classifier, test_set))
#print('positive f-score:', f_Measure(ref_sets['pos'], test_sets['pos']))
#print('negative f-score:', f_Measure(ref_sets['neg'], test_sets['neg']))

classifier.show_most_informative_features(20)
 
from gensim import corpora, models

list_of_list_of_tokens = texts 
dictionary_LDA = corpora.Dictionary(list_of_list_of_tokens)
dictionary_LDA.filter_extremes(no_below=3)
corpus = [dictionary_LDA.doc2bow(list_of_tokens) for list_of_tokens in list_of_list_of_tokens]
num_topics = 20
%time lda_model = models.LdaModel(corpus, num_topics=num_topics,id2word=dictionary_LDA,passes=4, alpha=[0.01]*num_topics, eta=[0.01]*len(dictionary_LDA.keys())) 

for i,topic in lda_model.show_topics(formatted=True, num_topics=num_topics, num_words=10):
    print(str(i)+": "+ topic)
    print()                                  
                                   
lda_model[corpus[0]]                                 
import pyLDAvis
import pyLDAvis.gensim
vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=dictionary_LDA)
pyLDAvis.enable_notebook()
pyLDAvis.display(vis)




"""

cv = CountVectorizer()
bow = cv.fit_transform(my_features)
word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
word_counter = collections.Counter(word_freq)
word_counter_df = pd.DataFrame(word_counter.most_common(10), columns = ['word', 'freq'])
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
plt.show();

import multiprocessing

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
phrases = Phrases(texts, min_count=30, progress_per=10000) 
bigram = Phraser(phrases)
sentences = bigram[texts]
cores = multiprocessing.cpu_count() # Count the number of cores in a computer
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     size=300,
                     sample=6e-5, 
                     alpha=0.03, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)
"""

from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import matplotlib
import matplotlib.pyplot as plt
model = Word2Vec(texts, min_count=50,) 
print(model) 
words = list(model.wv.vocab)
print(words)

print(model['الصحة'])
model.save('model.bin')
new_model = Word2Vec.load('model.bin')
print(new_model)
	
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
plt.figure(figsize=(16,16))	
matplotlib.pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	matplotlib.pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    
model.wv.most_similar(positive=["الصحة"])
model.wv.similarity('الناس', 'الصحة')

import numpy as np
import matplotlib.pyplot as plt

 
import seaborn as sns
sns.set_style("darkgrid")

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

"""
def tsnescatterplot(model, word, list_names):
    #Plot in seaborn the results from the t-SNE dimensionality reduction algorithm of the vectors of a query word,
    #its list of most similar words, and a list of words.
   
    arrays = np.empty((0, 300), dtype='f')
    word_labels = [word]
    color_list  = ['red']

    # adds the vector of the query word
    arrays = np.append(arrays, model.wv.__getitem__([word]), axis=0)
    
    # gets list of most similar words
    close_words = model.wv.most_similar([word])
    
    # adds the vector for each of the closest words to the array
    for wrd_score in close_words:
        wrd_vector = model.wv.__getitem__([wrd_score[0]])
        word_labels.append(wrd_score[0])
        color_list.append('blue')
        arrays = np.append(arrays, wrd_vector, axis=0)
    
    # adds the vector for each of the words from list_names to the array
    for wrd in list_names:
        wrd_vector = model.wv.__getitem__([wrd])
        word_labels.append(wrd)
        color_list.append('green')
        arrays = np.append(arrays, wrd_vector, axis=0)
    # Reduces the dimensionality from 300 to 50 dimensions with PCA
    reduc = PCA(n_components=50).fit_transform(arrays)
    
    # Finds t-SNE coordinates for 2 dimensions
    np.set_printoptions(suppress=True)
    
    Y = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(reduc)
    
    # Sets everything up to plot
    df = pd.DataFrame({'x': [x for x in Y[:, 0]],
                       'y': [y for y in Y[:, 1]],
                       'words': word_labels,
                       'color': color_list})
    fig, _ = plt.subplots()
    fig.set_size_inches(9, 9)
    
    # Basic plot
    p1 = sns.regplot(data=df,
                     x="x",
                     y="y",
                     fit_reg=False,
                     marker="o",
                     scatter_kws={'s': 40,
                                  'facecolors': df['color']
                                 }
                    )
    # Adds annotations one by one with a loop
    for line in range(0, df.shape[0]):
         p1.text(df["x"][line],
                 df['y'][line],
                 '  ' + df["words"][line].title(),
                 horizontalalignment='left',
                 verticalalignment='bottom', size='medium',
                 color=df['color'][line],
                 weight='normal'
                ).set_size(15)

    
    plt.xlim(Y[:, 0].min()-50, Y[:, 0].max()+50)
    plt.ylim(Y[:, 1].min()-50, Y[:, 1].max()+50)
            
    plt.title('t-SNE visualization for {}'.format(word.title()))
     
model = model 
word = 'العالمي'
list_names = [('الخير','العالم','الراجحي','كوبون','درهم','بمناسبة','اليوم','الناس')]

tsnescatterplot(model, word, list_names) 
"""

from textblob import TextBlob
text = nltk.Text(all_features)
type(text)
text[1024:1062]
#text.collocations()


# Python program to convert a list 
# to string using join() function 
    
# Function to convert   
def listToString(s):  
    
    # initialize an empty string 
    str1 = " " 
    
    # return string   
    return (str1.join(s)) 
        
        
# Driver code     

text_ = listToString(all_features) 
  

blob = TextBlob(text_)
blob.tags
blob.noun_phrases
for sentence in blob.sentences:
    print(sentence.sentiment.polarity)
blob.sentiment
df['tb_Pol'] = [b.sentiment.polarity for b in blob]
df['tb_Subj'] = [b.sentiment.subjectivity for b in blob]

"""
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, RegressorMixin

class PseudoLabeler(BaseEstimator, RegressorMixin):
    '''
    Sci-kit learn wrapper for creating pseudo-lebeled estimators.
    '''
    
    def __init__(self, model, unlabled_data, features, target, sample_rate=0.2, seed=42):
        '''
        @sample_rate - percent of samples used as pseudo-labelled data
                       from the unlabled dataset
        '''
        assert sample_rate <= 1.0, 'Sample_rate should be between 0.0 and 1.0.'
        
        self.sample_rate = sample_rate
        self.seed = seed
        self.model = model
        self.model.seed = seed
        
        self.unlabled_data = unlabled_data
        self.features = features
        self.target = target
    def get_params(self, deep=True):
        return {
            "sample_rate": self.sample_rate,
            "seed": self.seed,
            "model": self.model,
            "unlabled_data": self.unlabled_data,
            "features": self.features,
            "target": self.target
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

        
    def fit(self, X, y):
        '''
        Fit the data using pseudo labeling.
        '''

        augemented_train = self.__create_augmented_train(X, y)
        self.model.fit(
            augemented_train[self.features],
            augemented_train[self.target]
        )
        
        return self
    def __create_augmented_train(self, X, y):
        '''
        Create and return the augmented_train set that consists
        of pseudo-labeled and labeled data.
        '''        
        num_of_samples = int(len(self.unlabled_data) * self.sample_rate)
        
        # Train the model and creat the pseudo-labels
        self.model.fit(X, y)
        pseudo_labels = self.model.predict(self.unlabled_data[self.features])
        
        # Add the pseudo-labels to the test set
        pseudo_data = self.unlabled_data.copy(deep=True)
        pseudo_data[self.target] = pseudo_labels
        
        # Take a subset of the test set with pseudo-labels and append in onto
        # the training set
        sampled_pseudo_data = pseudo_data.sample(n=num_of_samples)
        temp_train = pd.concat([X, y], axis=1)
        augemented_train = pd.concat([sampled_pseudo_data, temp_train])

        return shuffle(augemented_train)
        
    def predict(self, X):
        '''
        Returns the predicted values.
        '''
        return self.model.predict(X)
    
    def get_model_name(self):
        return self.model.__class__.__name__

test = pd.read_excel('C:/Users/121/.spyder-py3/text_mining/train.xlsx')
target ='data_labels'

model = PseudoLabeler() 
""" 

