import os
import pickle
from itertools import islice
import json

from driver import sanitize, get_tagged_sents, get_stopwords


from corpus_reader import AssignmentCorpus
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Adjective_Features import get_adjectives, feature_adjectives, feature_adjectives_curated, feature_adjectives_count, feature_adjectives_curated_with_negation
from Bigrams_Features import bigrams_maximizing_prob_diff, best_bigrams, feature_bigrams
from Unigram_Features import words_maximizing_prob_diff, feature_unigram_probdiff
from Bigram_Pattern_Features import patterns_maximizing_prob_diff, feature_patterns, feature_patterns_count, pos_and_neg_patterns_maximizing_prob_diff
from Punctuation_Features import feature_exclamations, feature_questionmarks, feature_uppercase, feature_emoticons, feature_sentlength



def getFiles(source_dir):

    files = [ f for (dirpath, dirnames, filenames) in os.walk(source_dir) 
                for f in filenames if f[-4:] == '.txt' ]
    return files
        
def readFiles(files, source_dir):
    sents = []
    for fname in files:
        with open(source_dir + fname) as infile:
            for line in islice(infile.readlines(),None):
                line = line.strip()
                for line, sent in cleanLine(line):
                    sents.append((fname, line, sent))

    return sents

def cleanLine(line):
    line_no, sent = line.split('\t')
    
    if sent == '[t]':
        yield line_no, sent
    else:
        yield line_no, sent[2:]


def getModel(fname):
    with open(fname, 'rb') as infile:
        model = pickle.load(infile)
        
    return model


def labelData(test_data, model):



    training_corpus = AssignmentCorpus(
        'product_data_training_heldout/training/')
    heldout_corpus = AssignmentCorpus('product_data_training_heldout/heldout/')
    # Perform cross validation on all the data given
    all_sents = training_corpus.sents + heldout_corpus.sents

    # This is the data we extract features and do cross validation on.
    sents = sanitize(all_sents)

    # Do preprocessing for feature extraction
    tagged_sents = get_tagged_sents(sents)
    adjectives, pos_adj, neg_adj = get_adjectives(tagged_sents)
    pos_words = set([str(x) for x in json.load(open('positive_words.json'))])
    neg_words = set([str(x) for x in json.loads(open('negative_words.json')
                                                .read().decode('utf-8', 'ignore'))])

    stopwords = get_stopwords()
    max_prob_diff_words = set(
        [word for diff, word in words_maximizing_prob_diff(tagged_sents, 150, stopwords)])
    max_patterns_pos, max_patterns_neg = pos_and_neg_patterns_maximizing_prob_diff(
        tagged_sents, 100)
    max_prob_diff_patterns = patterns_maximizing_prob_diff(tagged_sents, 150)

    bigrams_best = best_bigrams(sents, stopwords, n=50)

    labelled = []
    for fname, line_num, sent in test_data:
        features = {}
        features.update(feature_adjectives_count(sent, pos_adj, neg_adj))

        features.update(feature_adjectives(sent, adjectives))
        #features.update(feature_adjectives_curated(sent, pos_words, neg_words))
        features.update(feature_adjectives_curated_with_negation(sent, pos_words, neg_words))

        features.update(feature_unigram_probdiff(sent, max_prob_diff_words))
        #features.update(feature_bigrams(sent, bigrams_best))
        features.update(feature_patterns(sent, max_prob_diff_patterns))

        features.update(feature_patterns_count(
            sent, max_patterns_pos, max_patterns_neg))
        features.update(feature_exclamations(sent))
        #features.update(feature_questionmarks(sent))
        features.update(feature_emoticons(sent))
        #features.update(feature_uppercase(sent))
        #features.update(feature_sentlength(sent))

        #print features
        #print model.classify(features)
        if sent == '[t]':
            labelled.append((fname, line_num, 0))
        else:
            score = model.classify(features)
            labelled.append((fname, line_num, score))

    return labelled

def main():

    files = getFiles('testdata/')
    test_data = readFiles(files, 'testdata/')

    model = getModel('nltk_model0.pkl')
    model.show_most_informative_features(50)

    labelled_data = labelData(test_data, model)

    with open('output_G9.txt','w') as outfile:
        for fname, line_num, score in labelled_data:
            line = fname+'\t'+str(line_num)+'\t'+str(score)+'\n'
            outfile.write(line)
        #print fname, line_num, score

if __name__ == '__main__':
    main()
