# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 07:08:08 2019

@author: Sreeju
"""

import nltk
import random
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
import re
import os

files_pos = os.listdir('D:/sentiment/aclImdb/train/pos')
files_pos = [open('D:/sentiment/aclImdb/train/pos'+f, 'r', encoding='utf8').read() for f in file_pos]
files_neg = os.listdir('D:/sentiment/aclImdb/train/neg')
files_neg = [open('D:/sentiment/aclImdb/train/neg'+f, 'r', encoding='utf8').read() for f in file_pos]

len(files_pos), len(files_neg)

all_words = []
documents = []

from nltk.corpus import stopwords

stop_words = list(set(stopwords.words('english')))

# j is adjective, r is adverb, v is verb
allowed_word_types = ["J"]

for p in files_pos:
    # create list of tuples where the first element is review
    # second element is the label
    document.append(p,'pos')
    
    # remove punctuation
    cleaned = re.sub(r'[^(a-zA-Z)\s'),'',p)
    
    # tokenize
    tokenized = word_tokenize(cleaned)
    
    #remove stop words
    stopped = [w for w in tokenized if not w in stop_words]
    
    #part of speech tagging of each words
    pos = nltk.pos_tag(stopped)
    
    # make list of all adjectives identified by allowed word types
    for w in pos:
        if w[0] in allowed_types:
            all_words.append(w[0].lower())
            
for p in files_neg:
    # create list of tuples where the first element is review
    # second element is the label
    document.append(p,'neg')
    
    # remove punctuation
    cleaned = re.sub(r'[^(a-zA-Z)\s'),'',p)
    
    # tokenize
    tokenized = word_tokenize(cleaned)
    
    #remove stop words
    stopped = [w for w in tokenized if not w in stop_words]
    
    #part of speech tagging of each words
    neg = nltk.pos_tag(stopped)
    
    # make list of all adjectives identified by allowed word types
    for w in neg:
        if w[0] in allowed_types:
            all_words.append(w[0].lower())            
        
    
    # creating feequency distribution of each words
    all_words = nltk.FreqDist(all_words)
    
    import matplotlib.pyplot as plt
    all_words.plot(30,cumulative=False)
    plt.show()
    
    # list the 1000 most frequent words 
    word_features = list(all_words.keys())[:1000]
    
    # function to create a dictionary of features for each review in the list document
    # keys are the words in the word_features 
    # the value of each key are either true or false for whether that feature appears in the review or not
    
    def find_features(document):
        words = word_tokenize(document)
        features = {}
        for w in word_features:
            features[w] = (w in words)
        return features
    
    # create features for each review
    featuresets = [(find_features(rev),category) for (rev, category) in documents]
    
    # shuffling the documents
    random.shuffle(featuresets)
    
    training_set = featuresets[:800]
    testing_set = featuresets[800:]
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    
    print("NaiveBayes Classifier accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*1000)
    
    classifier.show_most_informative__features(15)
    
    from sklearn import metrics
    
    MNB_clf = SKlearnClassifer(MultinomialNB())
    mnb_cls = MNB_clf_train(training_set)
    
    print("MultinomialNB Classifier accuracy percent:", (nltk.classify.accuracy(mnb_cls, testing_set))*100)
    
    BNB_clf = SKlearnClassifer(BernoulliNB())
    bnb_cls = BNB_clf.clf.train(training_set)
    
    print("BernoulliNB Classifier accuracy percent:", (nltk.classify.accuracy(bnb_cls, testing_set))*100)
    
    
    LogReg_clf = SKlearnClassifer(LogisticRegression())
    logReg_cls = LogReg_clf.clf.train(training_set)
    
    print("LogisticRegression Classifier accuracy percent:", (nltk.classify.accuracy(logReg_cls, testing_set))*100)
    
    SGD_clf = SKlearnClassifer(SGDClassifier())
    sgd_cls = SGD_clf.clf.train(training_set)
    
    print("SGD Classifier accuracy percent:", (nltk.classify.accuracy(sgd_cls, testing_set))*100)
    
    SVC_clf = SKlearnClassifer(SVCClassifier())
    sgd_cls = SVC_clf.clf.train(training_set)
    
    print("SCV Classifier accuracy percent:", (nltk.classify.accuracy(sgd_cls, testing_set))*100)
