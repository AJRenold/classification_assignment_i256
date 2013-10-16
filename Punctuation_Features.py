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


def feature_exclamations(sent):
    # This function returns the ratio of no. of exclamation marks and sentence
    # length
    features = {}
    letter = '!'
    features["count(%s)" % letter] = sent.count(letter)# / len(sent)
    return features


def feature_questionmarks(sent):
    # This function returns the ratio of no. of question marks and sentence
    # length

    features = {}
    letter = '?'
    features["count(%s)" % letter] = sent.count(letter)# / len(sent)
    return features


def feature_commas(sent):
    # This function returns the ratio of no. of commas marks and sentence
    # length

    features = {}
    letter = ','
    features["count(%s)" % letter] = sent.count(letter)# / len(sent)
    return features


def feature_semicolons(sent):
    # This function returns the ratio of no. of semicolons marks and sentence
    # length
    features = {}
    letter = ';'
    features["count(%s)" % letter] = sent.count(letter)# / len(sent)
    return features


def feature_uppercase(sent):
    # This function returns the count of all uppercase words in the sentence.

    features = {}
    capscount = 0
    sent_words = set(word_tokenize(sent))
    for word in sent_words:
        if word.isupper():
            capscount += 1
    features["capscount(%s)" % capscount] = capscount# / len(sent)
    return features


def feature_sentlength(sent):
    # This function returns the total length of the input sentence
    features = {}
    sent_len = len(sent)
    features["sent_len"] = sent_len
    return features


def feature_emoticons(sent):
    # This function returns the presence of emoticons in sentence
    features = {}
    emoticons = [':)', ':-)', '=)']
    for emoticon in emoticons:
        features["emoticon(%s)" % emoticon] = emoticon in sent
    return features
