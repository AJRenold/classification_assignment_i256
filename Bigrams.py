from corpus_reader import AssignmentCorpus
import nltk
from driver import *
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import itertools


def get_bigrams(sents):
    bigrams = []
    for tag, sent in sents:
        words = [w.lower() for w in word_tokenize(sent)]
        bigrams.append(nltk.bigrams(words))

    return bigrams

def bigram_word_feats(sent, bigram_features):
    words = [word.lower() for word in word_tokenize(sent) if word not in string.punctuation]
    words_bigrams = nltk.bigrams(words)
    features = {}
    for bigram in bigram_features:
        features[bigram] = (bigram in words_bigrams)
    return features

def bigram_word_features(sent, bigram_features, score_fn=BigramAssocMeasures.likelihood_ratio, n=10):
    words = [word.lower() for word in word_tokenize(sent) if word not in string.punctuation]

    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams_best = bigram_finder.nbest(score_fn, n)

    #features = {}
    #for bigram in bigram_features:
    #    features[bigram] = (bigram in bigrams_best)
    ##for bigram in bigrams_best:
    ##    features[bigram] = True
    #return features
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams_best)])

def main():
    corpus = AssignmentCorpus('product_data_training_heldout/training/')
    sents = sanitize(corpus.sents)
    tagged_sents = get_tagged_sents(sents)
    adjectives = get_adjectives(tagged_sents)
    max_prob_diff_words = set([ word for diff, word in words_maximizing_prob_diff(sents, 200)])

    pos_words = set([str(x) for x in json.load(open('positive_words.json'))])
    neg_words = set([str(x) for x in json.loads(open('negative_words.json')
                                                .read().decode('utf-8', 'ignore'))])

    bigrams = get_bigrams(sents)
    bigrams_fd = nltk.FreqDist(b for bi in bigrams for b in bi)
    bigram_features = bigrams_fd.keys()[:500]

    #data = []
    #for tag, sent in sents:
    #    feat1 = feature_adjectives(sent, adjectives)
    #    feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
    #    feat3 = feature_unigram_probdiff(sent, max_prob_diff_words)
    #    feat1.update(feat2)
    #    feat1.update(feat3)
    #    data.append((feat1, tag))
    #
    #print 'Naive Bayes:\t%s' % evaluate(nltk.NaiveBayesClassifier, data, 10)

    data = []
    for tag, sent in sents:
        feat1 = feature_adjectives(sent, adjectives)
        feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
        feat3 = feature_unigram_probdiff(sent, max_prob_diff_words)
        feat4 = bigram_word_feats(sent, bigram_features)
        feat1.update(feat2)
        feat1.update(feat3)
        feat1.update(feat4)
        data.append((feat1, tag))

    print 'Naive Bayes:\t%s' % evaluate(nltk.NaiveBayesClassifier, data, 10)


if __name__ == '__main__':
    main()
