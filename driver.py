from corpus_reader import AssignmentCorpus
import pprint
from random import shuffle
import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import progressbar
import json
import string

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
        validation = slices[i]
        training = [item
                    for s in slices if s is not validation
                    for item in s]
        yield training, validation


def evaluate(classifier, data, k):
    accuracies = []
    print 'evaluating %s using k fold cross validation. k = %s' % (classifier, k)
    i, bar = 0, pbar(k)
    bar.start()
    for training, validation in k_fold_cross_validation(data, k):
        model = classifier.train(training)
        accuracies.append(nltk.classify.accuracy(model, validation))
        i += 1
        bar.update(i)
    bar.finish()
    return sum(accuracies) * 1.0 / len(accuracies)


def sanitize(sents):
    # ignore all sents with no tag or no sent
    sents = [(tag, sent) for tag, sent in sents if sent]
    # simplify each tags list to an average of ratings
    for i in range(len(sents)):
        tag, sent = sents[i]
        if not tag:
            sents[i] = (0, sent)
        else:
            ratings = [x[1] for x in tag]
            avg_rating = sum(ratings) * 1.0 / len(ratings)
            if avg_rating > 0:
                sents[i] = (1, sent)
            elif avg_rating < 0:
                sents[i] = (-1, sent)
            else:
                sents[i] = (0, sent)
    # convert sents to strings instead of lists
    for i in range(len(sents)):
        tag, sent = sents[i]
        if isinstance(sent, list):
            sents[i] = (tag, sent[0])
    # remove whitespace in sents
    sents = [(tag, sent.strip()) for tag, sent in sents]
    return sents


def get_tagged_sents(sents):
    print 'tagging sentences'
    tagged_sents = []
    i, bar = 0, pbar(len(sents))
    bar.start()
    for tag, sent in sents:
        tagged_sents.append((tag, pos_tag(word_tokenize(sent))))
        i += 1
        bar.update(i)
    bar.finish()
    return tagged_sents


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
    json.dump(best_adj, open('adjectives.json', 'wb'), indent=2)
    return best_adj


def feature_adjectives(sent, words):
    sent_words = set([x.lower()
                     for x in word_tokenize(sent) if x not in string.punctuation])
    features = {}
    for word in words:
        features['contains(%s)' % word] = word in sent_words
    return features


def feature_adjectives_curated(sent, pos_words, neg_words):
    sent_words = set([x.lower()
                     for x in word_tokenize(sent) if x not in string.punctuation])
    features = {'pos': 0, 'neg': 0}
    for word in sent_words:
        if word.lower() in pos_words:
            features['pos'] += 1
        elif word.lower() in neg_words:
            features['neg'] += 1
    return features


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

    # Extract features.
    data = []
    for tag, sent in sents:
        feat1 = feature_adjectives(sent, adjectives)
        feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
        feat1.update(feat2)
        data.append((feat1, tag))
    print 'Naive Bayes:\t%s' % evaluate(nltk.NaiveBayesClassifier, data, 10)
    print 'Decision Tree:\t%s' % evaluate(nltk.DecisionTreeClassifier, data, 10)


if __name__ == '__main__':
    main()
