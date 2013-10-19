
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
from Adjective_Features import get_adjectives, feature_adjectives, feature_adjectives_curated, feature_adjectives_count, feature_adjectives_curated_with_negation
from Bigrams_Features import bigrams_maximizing_prob_diff, best_bigrams, feature_bigrams
from Unigram_Features import words_maximizing_prob_diff, feature_unigram_probdiff
from Bigram_Pattern_Features import patterns_maximizing_prob_diff, feature_patterns, feature_patterns_count, pos_and_neg_patterns_maximizing_prob_diff
from Punctuation_Features import feature_exclamations, feature_questionmarks, feature_uppercase, feature_emoticons, feature_sentlength

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

        validation = [(feat, tag) for feat, tag, sent in slices[i]]
        validation_sents = [sent for feat, tag, sent in slices[i]]

        training = [(feat, tag)
                    for s in slices if s is not slices[i]
                    for feat, tag, sent in s]
        yield training, validation, validation_sents


def evaluate(classifier, data, k, verbose_errors=False):

    pos_words = set([str(x) for x in json.load(open('positive_words.json'))])

    accuracies, ref, test = [], [], []
    print 'Using %s with k = %s' % (classifier, k)
    i, bar = 0, pbar(k)
    bar.start()
    for training, validation, val_sents in k_fold_cross_validation(data, k):
        model = classifier.train(training)
        # model.show_most_informative_features(50)
        accuracies.append(nltk.classify.accuracy(model, validation))
        for j, (feat, tag) in enumerate(validation):
            guess = model.classify(feat)
            ref.append(tag)
            test.append(guess)
            if guess != tag and verbose_errors:
                if guess == 1 and tag == -1:
                    if any([word.lower() in pos_words for word in val_sents[j].split(' ')]):
                        print 'guess:', guess, 'actual:', tag, 'SENT:', val_sents[j]

        i += 1
        bar.update(i)
    bar.finish()
    print 'Accuracy: %s' % (sum(accuracies) * 1.0 / len(accuracies))
    print nltk.ConfusionMatrix(ref, test)
    return model


def evaluate_ensemble(classifiers, data, k):
    accuracies, ref, test = [], [], []
    print 'Using ensemble with k = %s' % k
    i, bar = 0, pbar(k)
    bar.start()
    for training, validation, val_sents in k_fold_cross_validation(data, k):
        models = [classifier.train(training) for classifier in classifiers]
        correct_tags = 0
        for sent, tag in validation:
            predictions = [model.classify(sent) for model in models]
            combined_prediction = 0
            if len(set(predictions)) == 1:
                combined_prediction = predictions[0]
            elif predictions[0] == -1 and predictions[0] == 1 or predictions[0] == 1 and predictions[0] == -1:
                combined_prediction = 0
            else:
                combined_prediction = predictions[0] ^ predictions[1]
            if combined_prediction == tag:
                correct_tags += 1
        accuracy = correct_tags * 1.0 / len(validation)
        accuracies.append(accuracy)
        print accuracy
        # for j, (feat, tag) in enumerate(validation):
        #     guess = model.classify(feat)
        #     ref.append(tag)
        #     test.append(guess)
        #     if guess != tag and verbose_errors:
        #         print 'guess:', guess, 'actual:', tag, 'SENT:', val_sents[j]
        i += 1
        bar.update(i)
    bar.finish()
    print 'Accuracy: %s' % (sum(accuracies) * 1.0 / len(accuracies))
    # print nltk.ConfusionMatrix(ref, test)
    return sum(accuracies) * 1.0 / len(accuracies), model


def removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


def sanitize(sents):
    # ignore all sents with no tag or no sent
    sents = [(tag, sent) for tag, sent in sents if sent]
    # We exclude sents with positive and negative sentiment
    new_sents = []
    for tag, sent in sents:
        sent = sent.strip()
        sent = removeNonAscii(sent)
        if len(sent) < 20:
            continue
        if not tag:
            new_sents.append((0, sent))
        else:
            pos_sentiment = all([x[1] > 0 for x in tag])
            neg_sentiment = all([x[1] < 0 for x in tag])
            if pos_sentiment:
                new_sents.append((1, sent))
            elif neg_sentiment:
                new_sents.append((-1, sent))
    return new_sents


def get_stopwords():
    return {word: True for word in stopwords.words('english')}


def get_tagged_sents(sents):
    try:
        with open('tagged_sents.json', 'rb') as infile:
            print 'tagged_sentences.json found, delete tagged_sents.json to reset cache'
            tagged_sents = json.load(infile)
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
    print 'caching tagged_sents as tagged_sents.json'
    with open('tagged_sents.json', 'wb') as outfile:
        json.dump(tagged_sents, outfile, indent=2)
    return tagged_sents


def main():
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
    max_prob_diff_patterns = patterns_maximizing_prob_diff(tagged_sents, 300)
    bigrams_best = best_bigrams(sents, stopwords, n=50)

    # Extract features. All feature extraction methods are expected to return
    # a dictionary with distinct keys.
    data = []
    for tag, sent in islice(sents, None):
        features = {}
        #features.update(feature_adjectives_count(sent, pos_adj, neg_adj))
        features.update(feature_adjectives(sent, adjectives))
        #features.update(feature_adjectives_curated(sent, pos_words, neg_words))
        features.update(
            feature_adjectives_curated_with_negation(sent, pos_words, neg_words))
        features.update(feature_unigram_probdiff(sent, max_prob_diff_words))
        #features.update(feature_bigrams(sent, bigrams_best))
        features.update(feature_patterns(sent, max_prob_diff_patterns))
        features.update(feature_patterns_count(
            sent, max_patterns_pos, max_patterns_neg))
        features.update(feature_exclamations(sent))
        # features.update(feature_questionmarks(sent))
        features.update(feature_emoticons(sent))
        # features.update(feature_uppercase(sent))
        # features.update(feature_sentlength(sent))
        data.append((features, tag, sent))

    print 'Gathered %s features.' % len(data[0][0])
    classifiers = [nltk.NaiveBayesClassifier,
                   # nltk.DecisionTreeClassifier,
                   ]
    for i, classifier in enumerate(classifiers):
        model = evaluate(classifier, data, 10, verbose_errors=False)
        with open('nltk_model' + str(i) + '.pkl', 'wb') as outfile:
            pickle.dump(model, outfile)
    # evaluate_ensemble(classifiers, data, 10)

if __name__ == '__main__':
    main()
