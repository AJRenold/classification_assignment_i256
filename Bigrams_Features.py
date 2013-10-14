from corpus_reader import AssignmentCorpus
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize
import string
import itertools


def best_bigrams(sents_tagged, stopwords, score_fn=BigramAssocMeasures.likelihood_ratio, n=100):
    sents_pos = []
    sents_neg = []
    for tag, sent in sents_tagged:
        if tag == 1:
            sents_pos.append(sent)
        elif tag == -1:
            sents_neg.append(sent)

    words_pos = [word.lower() for s in sents_pos for word in word_tokenize(s) if word not in string.punctuation]
    words_neg = [word.lower() for s in sents_neg for word in word_tokenize(s) if word not in string.punctuation]

    bigram_finder1 = BigramCollocationFinder.from_words(words_pos)
    bigrams_best_pos = bigram_finder1.nbest(score_fn, n)

    bigram_finder2 = BigramCollocationFinder.from_words(words_neg)
    bigrams_best_neg = bigram_finder2.nbest(score_fn, n)

    bigrams_all = list(set(bigrams_best_pos).union(set(bigrams_best_neg)))
    # bigrams_all = bigrams_best_pos + bigrams_best_neg

    bigrams_best = [bigram for bigram in bigrams_all if (len(bigram[0]) > 3 or len(bigram[1]) > 3)]

    return bigrams_best


def bigrams_maximizing_prob_diff(sents, n, stopwords):

    def get_max_diff_bigrams(bigrams, pos, neg):
        prob_diff_bigrams = []
        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()

        for bigram in bigrams:
            p = pos.prob(bigram)
            n = neg.prob(bigram)

            if pos_fd[bigram] >= 15 or neg_fd[bigram] >= 15:
                prob_diff_bigrams.append((abs(p - n), bigram))

        return sorted(prob_diff_bigrams, reverse=True)

    bigrams_tagged = get_bigrams_tagged(sents)

    cfd = nltk.ConditionalFreqDist((tag, bigram)
                                for tag, bigrams in bigrams_tagged
                                for bigram in bigrams
                                if tag != 0
                                and (bigram[0] not in stopwords or bigram[1] not in stopwords)
                                and (len(bigram[0]) > 3 or len(bigram[1]) > 3))

    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]

    bigrams = list(set(pos.samples()).union(set(neg.samples())))

    return get_max_diff_bigrams(bigrams, pos, neg)[:n]

def feature_bigrams(sent, best_bigrams):
    features = {}
    sent_bigrams = set(nltk.bigrams(sent.lower()))
    for bigram in best_bigrams:
        if bigram in sent_bigrams:
            features['contains({0},{1})'.format(bigram[0], bigram[1])] = True
        else:
            features['contains({0},{1})'.format(bigram[0], bigram[1])] = False

    return features


def get_bigrams_tagged(sents):
    bigrams_tagged = []
    for tag, sent in sents:
        words = [w.lower() for w in word_tokenize(sent) if w not in string.punctuation]
        bigrams_tagged.append((tag, nltk.bigrams(words)))

    return bigrams_tagged

def bigram_word_feats(sent, bigram_features):
    words = [word.lower() for word in word_tokenize(sent) if word not in string.punctuation]
    words_bigrams = nltk.bigrams(words)
    features = {}
    for bigram in bigram_features:
        features[bigram] = (bigram in words_bigrams)
    return features


def local_main():
    corpus = AssignmentCorpus('product_data_training_heldout/training/')
    sents = sanitize(corpus.sents)
    tagged_sents = get_tagged_sents(sents)
    adjectives = get_adjectives(tagged_sents)

    stopwords = get_stopwords()
    bigrams_best = best_bigrams(sents, stopwords)

    max_prob_diff_words = set([ word for diff, word in words_maximizing_prob_diff(sents, 200, stopwords)])

    pos_words = set([str(x) for x in json.load(open('positive_words.json'))])
    neg_words = set([str(x) for x in json.loads(open('negative_words.json')
                                                .read().decode('utf-8', 'ignore'))])

    #bigrams_tagged = get_bigrams_tagged(sents)
    #bigrams_fd = nltk.FreqDist(b for bi in bigrams_tagged for b in bi[1])
    #bigram_features = bigrams_fd.keys()[:500]

    max_prob_diff_bigrams = set([bigram for diff, bigram in bigrams_maximizing_prob_diff(sents, 200, stopwords)])

    data = []
    for tag, sent in sents:
        feat1 = feature_adjectives(sent, adjectives)
        feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
        feat3 = feature_unigram_probdiff(sent, max_prob_diff_words)
        #feat4 = bigram_word_feats(sent, bigram_features)
        #feat4 = feature_bigrams(sent, max_prob_diff_bigrams)
        feat4 = feature_bigrams(sent, bigrams_best)
        feat1.update(feat2)
        feat1.update(feat3)
        feat1.update(feat4)
        data.append((feat1, tag, sent))

    print 'Naive Bayes:\t%s' % evaluate(nltk.NaiveBayesClassifier, data, 10)


if __name__ == '__main__':
    from driver import *
    local_main()
