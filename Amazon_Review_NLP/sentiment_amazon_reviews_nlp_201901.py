# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 11:48:13 2019

# Sentiment analysis using NLP: tokenizing amazon reviews and fitting different models. 
Files used: 1. positive  reviews, 2. negative reviews, 3. stopwords 
@author: etl_p
"""

import pandas as pd 
import numpy as np
import os
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup as bfs

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.naive_bayes import MultinomialNB as MNB, BernoulliNB as BNB, GaussianNB as GNB
from sklearn.ensemble import AdaBoostClassifier as ABC

from wordcloud import WordCloud 

import nltk 
# Lemmatizer reduces words into their root form: wolves -> wolf, jumping -> jump, etc. 
from nltk.stem import WordNetLemmatizer as WNL    
from nltk.corpus import wordnet 
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer



os.chdir('C:\Python_Practice\\NLP\spambase')
wlem = WNL()  # Inititate Lemmatizer

#------- simple tokenization using vectorizers on amazon reviews ----- 

    # import positive and negative reviews
positive_rev = bfs(open('positive.review').read())
positive_rev = positive_rev.findAll("review_text")
negative_rev = bfs(open('negative.review').read())
negative_rev = negative_rev.findAll("review_text")
    #combine reviews into one file and labels 
"""rev_data = positive_rev.text + negative_rev.text #combine positive & negative reviews
rev_label = np.zeros(len(rev_data))
rev_label[0:1000]=1   #set labels 
"""
    # extract positive_rev text into array 
rev_data = []
for prev in positive_rev:
    review = prev.text
    rev_data.append(review) 
    #Vectorize review data 
for nrev in negative_rev:
    review = nrev.text
    rev_data.append(review)     
cvect = CountVectorizer() 
X = cvect.fit_transform(rev_data)
    # see number of features
len(cvect.get_feature_names())
Y = np.zeros((len(positive_rev) + len(negative_rev)))
Y[0:1000] = 1   # label data

nmod_MNB = MNB()
nmod_BNB = BNB()
logmod = LogReg()
    
    # Cross-Validation (10-fold) instead of simple train_test_split 
kf=KFold(n_splits=10, shuffle=True)   #default value is 5
cv_scores_MNB = cross_val_score(nmod_MNB, X, Y, cv=kf)
cv_scores_BNB = cross_val_score(nmod_BNB, X, Y, cv=kf)
cv_scores_LogReg = cross_val_score(logmod, X, Y, cv=kf)
print("Multinomial NB / CountVectorizer CV Score is %s" % "{0:.3%}".format(cv_scores_MNB.mean()))
print("Bernoulli NB / CountVectorizer CV Score is %s" % "{0:.3%}".format(cv_scores_BNB.mean()))
print("LogReg / CountVectorizer CV Score is %s" % "{0:.3%}".format(cv_scores_LogReg.mean()))

#----------- Preprocessing and tokenization--------------
    """ Whiie both vectorizers allow for lemmatizers and stopwords, processing the tokens first 
    allows for further customization. For this exercise, we will lemmatize, distinguish parts-of-speech (pos-tagging), 
    and remove digits and stopwords. Note: we run pos-tagging first, to pass as an argument to the lemmatizer.
    """
import string

pos1 = positive_rev[1].get_text() # pulls one element from the positive_rev set 
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def tokenize_this(tk):
    tokens = tk.lower()
    #tokens = nltk.tokensize.word_tokensize(tk)   # does not take out apostrophes
    # tokens = [tk.strip(string.punctuation) for tk in tokens.split(" ")]  #uses string library
    tokens = [tk.strip() for tk in tokens.split(" ")]  #uses string library
    tokens = [t for t in tokens if len(t)>2]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    pos_tags = pos_tag(tokens)
    tokens = [wlem.lemmatize(t[0],get_wordnet_pos(t[1])) for t in pos_tags]
    tokens = [tk for tk in tokens if tk not in stopwords]
    tokens = " ".join(tokens)
    return tokens

test_token = tokenize_this(pos1)
#------------- Parameter Tune with Gridsearch CV and Pipeline---------- 
""" 
Use Pipeline. will only focus on Bernoulli NB vs. Logistic Regression using CountVectorizer and TFIDF
Note: since sample is relatively small: only 2k reviews, we can use GridSearch.
If dataset is very large, it makes sense to use RandomizedCV for parameter tuning. 
"""
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV 

# initialize vectorizers
cvect = CountVectorizer() 
tfidf = TfidfVectorizer(decode_error='ignore')

    # set up pipeline for CountVectorizer and Bernoulli NB
rev_pipe_CTNB = Pipeline([('Count_Vectorizer', cvect),
                     ('BernoulliNB', nmod_BNB)                    
                     ])

params_CTNB = {'Count_Vectorizer__ngram_range':[(1,1),(1,2),(2,2)],
            'Count_Vectorizer__min_df': [0.001, .005, .01],
            'BernoulliNB__alpha': [0.2,0.4,0.5,0.6], 
            'BernoulliNB__fit_prior':[True,False]} 

grid_cvmod_CTNB = GridSearchCV(rev_pipe_CTNB,param_grid = params_CTNB, cv=5, scoring='accuracy',return_train_score=False)
grid_cvmod_CTNB.fit(rev_data, Y)
grid_cvresults_CTNB = pd.DataFrame(grid_cvmod_CTNB.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
# split out param column into individual columns 
grid_cvresults_CTNB = pd.concat([grid_cvresults_CTNB.drop(['params'], axis=1),
                                 grid_cvresults_CTNB['params'].apply(pd.Series)], axis=1)

    # set up pipeline for TFIDFVectorizer and Bernoulli NB
rev_pipe_TFNB = Pipeline([('TFIDF', tfidf),
                     ('BernoulliNB', nmod_BNB)                    
                     ])

params_TFNB = {'TFIDF__ngram_range':[(1,1),(1,2),(2,2)],
            'TFIDF__min_df': [0.001, .005, .01],
            'BernoulliNB__alpha': [0.2,0.4,0.5,0.6], 
            'BernoulliNB__fit_prior':[True,False]} 

grid_cvmod_TFNB = GridSearchCV(rev_pipe_TFNB,param_grid = params_TFNB, cv=5, scoring='accuracy',return_train_score=False)
grid_cvmod_TFNB.fit(rev_data, Y)
grid_cvresults_TFNB = pd.DataFrame(grid_cvmod_TFNB.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_cvresults_TFNB = pd.concat([grid_cvresults_TFNB.drop(['params'], axis=1),
                                 grid_cvresults_TFNB['params'].apply(pd.Series)], axis=1)


        # Pipeline/Gridsearch for LogReg + Countvectorizer 
logmod = LogReg(max_iter=4000)
rev_pipe_CTLR = Pipeline([('Count_Vectorizer', cvect),
                     ('LogReg',logmod)                    
                     ])

params_CTLR = {'Count_Vectorizer__ngram_range':[(1,1),(1,2),(2,2)],
            'Count_Vectorizer__min_df': [0.001, .005, .01],
            'LogReg__solver':['newton-cg', 'liblinear', 'saga']   # can't include all: non-convergence
            # 'LogReg__class_weight':[None, 'balanced']               # no difference in scores, so will comment out 
            } 

grid_cvmod_LR = GridSearchCV(rev_pipe_CTLR,param_grid = params_CTLR, cv=5, scoring='accuracy',return_train_score=False)
grid_cvmod_LR.fit(rev_data, Y)
grid_cvresults_CTLR = pd.DataFrame(grid_cvmod_LR.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_cvresults_CTLR = pd.concat([grid_cvresults_CTLR.drop(['params'], axis=1),
                                 grid_cvresults_CTLR['params'].apply(pd.Series)], axis=1)

    
    # Pipeline/Gridsearch for LogReg + TFIDF 
rev_pipe_TFLR = Pipeline([('TFIDF', tfidf),
                     ('LogReg',logmod)                    
                     ])

params_TFLR = {'TFIDF__ngram_range':[(1,1),(1,2),(2,2)],
            'TFIDF__min_df': [0.001, .005, .01],
            'LogReg__solver':['newton-cg', 'liblinear', 'saga']   # can't include all: non-convergence
            # 'LogReg__class_weight':[None, 'balanced']               # no difference in scores, so will comment out 
            } 

grid_cvmod_TFLR = GridSearchCV(rev_pipe_TFLR,param_grid = params_TFLR, cv=5, scoring='accuracy',return_train_score=False)
grid_cvmod_TFLR.fit(rev_data, Y)
grid_cvresults_TFLR = pd.DataFrame(grid_cvmod_TFLR.cv_results_)[['mean_test_score', 'std_test_score', 'params']]
grid_cvresults_TFLR = pd.concat([grid_cvresults_TFLR.drop(['params'], axis=1), 
                                 grid_cvresults_TFLR['params'].apply(pd.Series)], axis=1)


""" comparing results, we see (1,2) n-gram performs the best regardless of vectorizer or model used. 
keeping the minimum document requiremenet low also helps. 
We illustrate the effect of n-grams using the seaborn package. 
This example will use the combination of Bernoulli NB against both vectorizers, as NB performed slightly better
"""
import seaborn as sns 
ngram_ranges = [(1,1), (1,2), (1,3)]
ngram_cv_scores = [] 
ngram_tf_scores = [] 

def cvec_ngram_crval(nrange, X, Y):
    cvec = CountVectorizer(ngram_range=(nrange))
    X_cv = cvec.fit_transform(X)
    kf=KFold(n_splits=10, shuffle=True)   #default value is 5
    cv_scores_BNB = cross_val_score(nmod_BNB, X_cv, Y, cv=kf)
    return cv_scores_BNB.mean()

def tfvec_ngram_crval(nrange, X, Y):
    tfvec = TfidfVectorizer(ngram_range=(nrange))
    X_tf = tfvec.fit_transform(X)
    kf=KFold(n_splits=10, shuffle=True)   #default value is 5
    cv_scores_BNB = cross_val_score(nmod_BNB, X_tf, Y, cv=kf)
    return cv_scores_BNB.mean()

for ngram in ngram_ranges:
    # print(ngram)
    # mu_score = "%s"% "{0:.3%}".format(cvec_ngram_crval(ngram, rev_data, Y))   : seaborn can't read %'s 
    ngram_cv_scores.append(cvec_ngram_crval(ngram, rev_data, Y))
    ngram_tf_scores.append(tfvec_ngram_crval(ngram,rev_data,Y))


# create dataframe for seaborn line graphs 
ngram_scores = ngram_cv_scores + ngram_tf_scores 
ngram_xlabels = ['ngram1','ngram2','ngram3', 'ngram1','ngram2','ngram3']
cvec_ngram_df = pd.DataFrame({'ngram':ngram_xlabels, 'cv_scores':ngram_scores})
cvec_ngram_df['vectorizer'] = 'CVEC'
cvec_ngram_df.vectorizer[3:] = 'TFIDF'
    # plot lines 
b = sns.pointplot(x = 'ngram',y='cv_scores',hue='vectorizer',data=cvec_ngram_df)
b.set_ylabel("10-fold CV Scores", fontsize=15)
b.set_xlabel("N-gram",fontsize=15)
b.tick_params(labelsize=15)
b.legend(fontsize=20)
b.axes.set_title("N-Gram vs. Cross-Val Scores (Bernoulli Naive Bayes)", fontsize=20)
# plt.title("Cvec: N-Gram Cross-Val Scores (Bernoulli Naive Bayes) ") 
    # shows outperformance for ngram (1,2) for both vectorizers 
    # compare models with more stability on countvectorizer.  
    # compare models for countvectorizer: ngram of (1,2) 


#----- Word Proportion by line----------
"""Traditional vectorizers use every word when tokenizing. going to customize our tokenizer 
by leveraging the nltk package and adjusting the problem to look at word proportions instead of total word counts
"""
    #---- Start with bigrams tokenization to existing tokenize_this function--   
pos1 = positive_rev[1].get_text() 

def tokenize_this(tk):
    tokens = tk.lower()
    tokens = tokens.replace('\n', '')
    #tokens = nltk.tokensize.word_tokensize(tk)   # does not take out apostrophes
    tokens = [tk.strip(string.punctuation) for tk in tokens.split(" ")]  #uses string library
    # tokens = [tk.strip() for tk in tokens.split(" ")]  
    tokens = [t for t in tokens if len(t)>2]
    tokens = [t for t in tokens if not any(c.isdigit() for c in t)]
    pos_tags = pos_tag(tokens)
    tokens = [wlem.lemmatize(t[0],get_wordnet_pos(t[1])) for t in pos_tags]
    tokens = [tk for tk in tokens if tk not in stopwords]
    tokens = list(nltk.bigrams(tokens))  #Bigrams produces a generator function. List turns the generated items into a list. 
    #tokens = " ".join(tokens)
    return tokens

# test the one-line code 
token_test = tokenize_this(pos1) 
for a in token_test:
    print(a)
print(*map(' '.join,token_test), sep=', ')   

"""
Create tew
"""   
tk_index_map = {}  #dictionary assigns number (index) to each token
current_index = 0  #will track # of unique tokens  
positive_token_lists = []   #house lists of tokens for positive reviews (one list per review)
negative_token_lists = []   #house lists of tokens for negative reviews
review_text = [] 

# populate token-index dictionary with unique universe of words and keep count. 
# populate positive and negative reviews arrays with tokenized lists from positive and negative reviews 
for review in positive_rev:
    # review = rev.get_text()
    token_list=tokenize_this(review.text)
    positive_token_lists.append(token_list)
    for token in token_list:
        if token not in tk_index_map:
            tk_index_map[token] = current_index
            current_index += 1

for review in negative_rev:
    # review = rev.get_text()
    token_list=tokenize_this(review.text)
    negative_token_lists.append(token_list)
    for token in token_list:
        if token not in tk_index_map:
            tk_index_map[token] = current_index
            current_index += 1

""" Token Probs is used to create each row in the feature vector.
Each columns will represent a distinct token in the universe: tk_index_map.
values will the proportion that token makes up within the review. 
"""
#token_probs function will be used to create line for each review, computes word proportion, and adds label
def token_probs(token_list, label):
    x = np.zeros(len(tk_index_map)+1)
    for token in token_list:
        i = tk_index_map[token]     #retrieves index number from tk_index_map
        x[i] += 1                   # adds one to index number in x everytime word appears
    x = x/x.sum()                   # compute probability of every word within positive reviews
    x[-1] = label                   # set last record to label: positive or negative 
    return x 

#------------- De-bug token_probs for bigram----------

token_test = tokenize_this(pos1)  # list of 2-element tuples 
x= np.zeros(len(tk_index_map)+1)

for a in positive_token_lists:
    print(a)

pos1 = positive_token_lists[1]
tkprobs = token_probs(pos1, 1)  
del(tkprobs, pos1)
    
# ------------------------
    

N = len(positive_token_lists) + len(negative_token_lists)  

dt = np.zeros((N,len(tk_index_map)+1))   
i=0
    # populate dt for positive reviews 
for p_rev in positive_token_lists:
    tkprops = token_probs(p_rev, 1)
    dt[i,:] = tkprops
    i+=1
    #populate dt for negative reviews 
for n_rev in negative_token_lists:
    tkprops = token_probs(n_rev, 0)
    dt[i,:] = tkprops
    i+=1

# check for null values
np.count_nonzero(np.isnan(dt))  #looks like one row 
    # grab indices for null values
na_ind = np.isnan(dt)
na_test = dt[na_ind]
dt[na_ind] = 0   # replace null value with 0 

    # fit models
np.random.shuffle(dt)
X = dt[:, :-1]
Y = dt[:, -1]

    # initialize models: 
    
nmod_MNB = MNB()
nmod_BNB = BNB()
logmod = LogReg()

kf=KFold(n_splits=5, shuffle=True)   #default value is 5
cv_scores_MNB = cross_val_score(nmod_MNB, X, Y, cv=kf)
cv_scores_BNB = cross_val_score(nmod_BNB, X, Y, cv=kf)
cv_scores_LogReg = cross_val_score(logmod, X, Y, cv=kf)
print("Multinomial NB CV Score is %s" % "{0:.3%}".format(cv_scores_MNB.mean()))
print("Bernoulli NB CV Score is %s" % "{0:.3%}".format(cv_scores_BNB.mean()))
print("LogReg CV Score is %s" % "{0:.3%}".format(cv_scores_LogReg.mean()))

# graph metrics
metrics = pd.DataFrame(index=['accuracy','precision','recall'],
                       columns=['lr','dec','nb'])

    # visualize positive and negative words with WordCloud
from wordcloud import WordCloud 

def wordcloud_vis(lab_file):
    wc_string = ''
    for review in lab_file:
        rev = review.get_text()
        rev = rev.lower()
        for msg in rev.split(" "):
            wc_string += msg + ' '
    wcloud = WordCloud(width=600, height=400).generate(wc_string)
    plt.imshow(wcloud)
    plt.axis('off')
    plt.show()

wordcloud_vis(positive_rev) 
wordcloud_vis(negative_rev)


or tk in tokens.split(" ")
for review in positive_rev:
    # review = rev.get_text()
    token_list=tokenize_this(review.text)
    positive_token_lists.append(token_list)
    for token in token_list:
        if token not in tk_index_map:
            tk_index_map[token] = current_index
            current_index += 1
