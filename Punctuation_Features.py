from __future__ import division
from nltk.tokenize import word_tokenize

def feature_exclamations(sent):
    features = {}
    letter = '!'
	#Calculating ratio of no. of exclamation marks and sentence length
    features["count(%s)" % letter] = sent.count(letter) / len(sent)
    return features
	
def feature_questionmarks(sent):
    features = {}
    letter = '?'
	#Calculating ratio of no. of exclamation marks and sentence length
    features["count(%s)" % letter] = sent.count(letter) / len(sent)
    return features
	
def feature_uppercase(sent):
    features = {}
    capscount = 0
    sent_words = set(word_tokenize(sent))
    for word in sent_words:
        if word.isupper():
            capscount += 1
    features["capscount(%s)" % capscount] = capscount;
    return features
