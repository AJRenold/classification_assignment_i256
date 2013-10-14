from __future__ import division
from nltk.tokenize import word_tokenize

# Description - This file has the implementation for following functions:
# 1. Syntactic features
#  1.1 Exclamation marks
#  1.1 Question marks
#  1.1 Commas
#  1.1 Semicolons
# 2. Word based features
#  2.1 Uppercase words
#  2.2 Length of the review
# 3. Emoticons - Combined and improved use of punctuations and other characters
#
# Input : Sentence from training set after sanitization is passed as input.
#
# Output : Functions return a dictionary, known as feature set, which maps 
#          features' names to their values.

#This function returns the ratio of no. of exclamation marks and sentence length
def feature_exclamations(sent):
    features = {}
    letter = '!'
    features["count(%s)" % letter] = sent.count(letter) / len(sent)
    return features
	
#This function returns the ratio of no. of question marks and sentence length
def feature_questionmarks(sent):
    features = {}
    letter = '?'
    features["count(%s)" % letter] = sent.count(letter) / len(sent)
    return features
	
#This function returns the ratio of no. of commas marks and sentence length
def feature_commas(sent):
    features = {}
    letter = ','
    features["count(%s)" % letter] = sent.count(letter) / len(sent)
    return features
	
#This function returns the ratio of no. of semicolons marks and sentence length
def feature_semicolons(sent):
    features = {}
    letter = ';'
    features["count(%s)" % letter] = sent.count(letter) / len(sent)
    return features
	
#This function returns the count of all uppercase words in the sentence.
def feature_uppercase(sent):
    features = {}
    capscount = 0
    sent_words = set(word_tokenize(sent))
    for word in sent_words:
        if word.isupper():
            capscount += 1
    features["capscount(%s)" % capscount] = capscount
    return features
	
#This function returns the total length of the input sentence
def feature_sentlength(sent):
    features = {}
    sent_len = len(sent)
    features["sent_len"] = sent_len
    return features
	
#This function returns the presence of emoticons in sentence
def feature_emoticons(sent):
    features = {}
    emoticons = [':)', ':-)', ':(', ':o', ':/', ":'(", '>:o', '(:', '>.<', 'XD',\
                '-__-', 'o.O', ';D', '@_@', ':P', '8D', ':1', '>:(', ':D', '=|',\
                '")', ':>', ':*', ';)']
    words = sent.split()
    for word in words:
        for index in range(len(emoticons)):
            if emoticons[index] in word:
                features["emoticon(%s)" % emoticons[index]] = True;
            else:
                features["emoticon(%s)" % emoticons[index]] = False;
    return features