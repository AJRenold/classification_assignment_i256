
__author__ = 'group G9'

from corpus_reader import AssignmentCorpus
import pprint
from random import shuffle
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import progressbar
import json
import string
import pickle
import re
from itertools import islice
from nltk.corpus import stopwords

## FEATURES ##
from Adjective_Features import get_adjectives, feature_adjectives, feature_adjectives_curated
from Bigrams_Features import bigrams_maximizing_prob_diff, best_bigrams, feature_bigrams
from Unigram_Features import words_maximizing_prob_diff, feature_unigram_probdiff
from Bigram_Pattern_Features import patterns_maximizing_prob_diff, feature_patterns
from Punctuation_Features import feature_exclamations, feature_questionmarks, feature_uppercase

pr = pprint.PrettyPrinter(indent=2)


def pbar(size):
    bar = progressbar.ProgressBar(maxval=size,
                                  widgets=[progressbar.Bar('=', '[', ']'),
                                           ' ', progressbar.Percentage(),
                                           ' ', progressbar.ETA(),
                                           ' ', progressbar.Counter(),
                                           '/%s' % size])
    return bar


def k_fold_cross_validation(items, k):
    shuffle(items)
    slices = [items[i::k] for i in xrange(k)]
    for i in xrange(k):

        validation = [ (feat, tag) for feat, tag, sent in slices[i] ]
        validation_sents = [ sent for feat, tag, sent in slices[i] ]

        training = [ (feat, tag)
                    for s in slices if s is not slices[i]
                    for feat, tag, sent in s]
        yield training, validation, validation_sents


def evaluate(classifier, data, k, verbose_errors=False):
    accuracies = []
    ref = []
    test = []

    print 'evaluating %s using k fold cross validation. k = %s' % (classifier, k)
    i, bar = 0, pbar(k)
    bar.start()
    for training, validation, val_sents in k_fold_cross_validation(data, k):
        model = classifier.train(training)
        #model.show_most_informative_features(20)
        accuracies.append(nltk.classify.accuracy(model, validation))
        for j, (feat, tag) in enumerate(validation):
            guess = model.classify(feat)
            ref.append(tag)
            test.append(guess)
            if guess != tag and verbose_errors:
                print 'guess:', guess, 'actual:', tag, 'SENT:', val_sents[j]

        i += 1
        bar.update(i)
    bar.finish()
    print nltk.ConfusionMatrix(ref, test)
    return model, sum(accuracies) * 1.0 / len(accuracies)


def sanitize(sents):
    # ignore all sents with no tag or no sent
    sents = [(tag, sent) for tag, sent in sents if sent]
    # simplify each tags list to an average of ratings
    
    ### Alternative if we want to exclude sents with positive and negative sentiment
    new_sents = []
    for tag, sent in sents:
        if not tag:
            new_sents.append((0,sent.strip()))
        else:
            pos_sentiment = all([ x[1] > 0 for x in tag])
            neg_sentiment = all([ x[1] < 0 for x in tag])
            if pos_sentiment:
                new_sents.append((1, sent.strip()))
            elif neg_sentiment:
                new_sents.append((-1, sent.strip()))
            else:
                pass
                #print tag, sent

    return new_sents

    # does it make sense to simplify tags to 1, 0 and -1?
    #for i in range(len(sents)):
    #    tag, sent = sents[i]
    #    if not tag:
    #        sents[i] = (0, sent)
    #    else:
    #        ratings = [x[1] for x in tag]
    #        avg_rating = sum(ratings) * 1.0 / len(ratings)
    #        
    #        if avg_rating > 0:
    #            sents[i] = (1, sent)
    #        elif avg_rating < 0:
    #            sents[i] = (-1, sent)
    #        else:
    #            sents[i] = (0, sent)
    # convert sents to strings instead of lists
    #for i in range(len(sents)):
    #    tag, sent = sents[i]
    #    if isinstance(sent, list):
    #        print 'list'
    #        sents[i] = (tag, sent[0])
    # remove whitespace in sents
    #sents = [(tag, sent.strip()) for tag, sent in sents]
    #return sents

def get_stopwords():
    return { word: True for word in stopwords.words('english') }

# this can be an AssignmentCorpus method
def get_tagged_sents(sents):

    try:
        with open('tagged_sents.pkl', 'rb') as infile:
            print 'Tagged_sentences pickle found, delete tagged_sents.pkl to reset cache'
            tagged_sents = pickle.load(infile)
            return tagged_sents
    except:
        print 'No tagged sentences stored'

    print 'tagging sentences'
    tagged_sents = []
    i, bar = 0, pbar(len(sents))
    bar.start()
    for tag, sent in sents:
        tagged_sents.append((tag, pos_tag(word_tokenize(sent))))
        i += 1
        bar.update(i)
    bar.finish()

    print 'caching tagged_sents as tagged_sents.pkl'
    with open('tagged_sents.pkl', 'wb') as outfile:
        pickle.dump(tagged_sents, outfile)

    return tagged_sents


def main():
    training_corpus = AssignmentCorpus(
        'product_data_training_heldout/training/')
    heldout_corpus = AssignmentCorpus('product_data_training_heldout/heldout/')
    # We should do cross validation on all the data given
    all_sents = training_corpus.sents + heldout_corpus.sents

    # This is the data we extract features and do cross validation on.
    sents = sanitize(all_sents)

    # Do preprocessing for feature extraction
    tagged_sents = get_tagged_sents(sents)
    adjectives = get_adjectives(tagged_sents)
    # adjectives = json.load(open('adjectives.json'))
    pos_words = set([str(x) for x in json.load(open('positive_words.json'))])
    neg_words = set([str(x) for x in json.loads(open('negative_words.json')
                                                .read().decode('utf-8', 'ignore'))])

    stopwords = get_stopwords()

    max_prob_diff_words = set([ word for diff, word in words_maximizing_prob_diff(tagged_sents, 150, stopwords)])
    max_prob_diff_patterns = patterns_maximizing_prob_diff(tagged_sents, 150)

    # max_prob_diff_bigrams = set([bigram for diff, bigram in bigrams_maximizing_prob_diff(sents, 500, stopwords)])
    bigrams_best = best_bigrams(sents, stopwords)

    # Extract features. All feature extraction methods are expected to return a dictionary.
    data = []
    for tag, sent in islice(sents,None):
        features = {}
        feat1 = feature_adjectives(sent, adjectives)
        feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
        feat3 = feature_unigram_probdiff(sent, max_prob_diff_words)
        feat4 = feature_patterns(sent, max_prob_diff_patterns)
        #feat5 = feature_bigrams(sent, max_prob_diff_bigrams)
        #feat5 = feature_bigrams(sent, bigrams_best)
        #feat6 = feature_exclamations(sent)
        #feat7 = feature_questionmarks(sent)
        #feat8 = feature_uppercase(sent)
        
        # Update features with extracted features
        features.update(feat1)
        features.update(feat2)
        features.update(feat3)
        features.update(feat4)
        #features.update(feat5)
        #features.update(feat6)
        #features.update(feat7)
        #features.update(feat8)


        ## Include sent for error analysis
        data.append((features, tag, sent))

    model, accuracy = evaluate(nltk.NaiveBayesClassifier, data, 10, verbose_errors=False)
    print 'Naive Bayes:\t%s' % accuracy
    #print 'Decision Tree:\t%s' % evaluate(nltk.DecisionTreeClassifier, data, 10)

if __name__ == '__main__':
    main()
