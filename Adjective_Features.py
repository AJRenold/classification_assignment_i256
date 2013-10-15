
__author__ = 'Jeff Tsui'

import string
import nltk
from nltk.tokenize import word_tokenize


def get_adjectives(tagged_sents):
    adjectives = []
    for tag, pos_tags in tagged_sents:
        adj = [word.lower() for word, pos in pos_tags if pos == 'JJ']
        if adj:
            adjectives.append((tag, adj))
    pos_adj = nltk.FreqDist(
        [a for x, y in adjectives if x > 0 and y for a in y]).keys()[:20]
    neg_adj = nltk.FreqDist(
        [a for x, y in adjectives if x < 0 and y for a in y]).keys()[:20]
    best_adj = list(set(pos_adj).union(set(neg_adj)))
    return best_adj, pos_adj, neg_adj


def feature_adjectives(sent, words):
    sent_words = set([x.lower()
                     for x in word_tokenize(sent) if x not in string.punctuation])
    features = {}
    for word in words:
        features['contains(%s)' % word] = word in sent_words
    return features


def feature_adjectives_count(sent, pos_words, neg_words):
    sent_words = set([x.lower()
                     for x in word_tokenize(sent) if x not in string.punctuation])
    pos, neg = 0, 0
    for word in sent_words:
        if word.lower() in pos_words:
            pos += 1
        elif word.lower() in neg_words:
            neg += 1
    features = {}
    pos = pos * 1.0 / len(sent_words)
    neg = neg * 1.0 / len(sent_words)
    features['pos_feature_adjectives_count'] = pos
    features['neg_feature_adjectives_count'] = neg
    features['diff_feature_adjectives_count'] = pos - neg
    features['sum_feature_adjectives_count'] = pos + neg
    return features


def feature_adjectives_curated(sent, pos_words, neg_words):
    sent_words = set([x.lower()
                     for x in word_tokenize(sent) if x not in string.punctuation])
    pos, neg = 0, 0
    for word in sent_words:
        if word.lower() in pos_words:
            pos += 1
        elif word.lower() in neg_words:
            neg += 1
    features = {}
    pos = pos * 1.0 / len(sent_words)
    neg = neg * 1.0 / len(sent_words)
    features['pos_feature_adjectives_curated'] = pos
    features['neg_feature_adjectives_curated'] = neg
    features['diff_feature_adjectives_curated'] = pos - neg
    features['sum_feature_adjectives_curated'] = pos + neg
    return features
