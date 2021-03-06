
__author__ = 'AJ Renold'

import string
import re

import nltk
from nltk.tokenize import word_tokenize


def get_unigram_features(sent):
    ## all unigrams
        
    features = {}
    for word in sent.lower().split(' '):
        if word not in string.punctuation:
            features['contains({})'.format(word)] = True
    
    return features

def words_maximizing_prob_diff(tagged_sents, n, stopwords):
    ## extracts n unigrams that maximize class probablity diff
    
    ex = set(['router','ipod','norton','jack','apex','diaper','canon',\
            'two','radio','nomad','nokia','phone','disc','month','apple', 'linksys', \
        'nikon', 'windows', 'wireless'])

    def get_max_diff(words, pos, neg):
        prob_diff_words = []
        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()
        
        for word in words:
            p = pos.prob(word)
            n = neg.prob(word)
            
            if pos_fd[word] >= 10 or neg_fd[word] >= 10:
                if len(word) > 3 and word not in stopwords and word not in ex:
                    if not re.findall(r'[\W]|[\d]', word):
                        prob_diff_words.append((abs(p - n), word))

        return sorted(prob_diff_words, reverse=True)
    
    cfd = nltk.ConditionalFreqDist((label, re.sub(r'\W\s','',word.lower()))
                               for label, sent in tagged_sents
                               for word, pos in sent
                               if word not in string.punctuation 
                               and label != 0 and not pos.startswith('N'))
    
    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]

    words = list(set(pos.samples()).union(set(neg.samples())))
    
    return get_max_diff(words, pos, neg)[:n]

def feature_unigram_probdiff(sent, max_diff_words):
    negation_words = set(['but','however','although'])

    features = {}
    #for word in word_tokenize(sent):
    #    if word in max_diff_words:
    #        features['contains({})'.format(word)] = True
    sent_words = sent_words = [x.lower()
                     for x in word_tokenize(sent) if x not in string.punctuation]
    """
    len_sent = len(sent_words)
    for i, word in enumerate(sent_words):
        lidx = 0 if i - 3 < 0 else i - 3
        uidx = len_sent if i + 3 > len_sent else i + 3

        if word in max_diff_words:
            if any( w == 'not' for w in sent_words[lidx:i+1] ):
                pass
            elif any( w in negation_words for w in sent[lidx:uidx+1]):
                pass
            else:
                features['contains(%s)' % word] = True
    """
    for word in max_diff_words:
        features['contains(%s)' % word] = word in sent_words

    return features
