from corpus_reader import AssignmentCorpus
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.tokenize import word_tokenize
import string
import itertools


def best_bigrams(sents_tagged, stopwords, score_fn=BigramAssocMeasures.likelihood_ratio, n=300):
    sents_pos = []
    sents_neg = []

    # Separate positive and negative sentences.
    for tag, sent in sents_tagged:
        if tag == 1:
            sents_pos.append(sent)
        elif tag == -1:
            sents_neg.append(sent)

    # Extract words from positive and negative sentences.
    words_pos = [word.lower() for s in sents_pos for word in word_tokenize(s) if word not in string.punctuation]
    words_neg = [word.lower() for s in sents_neg for word in word_tokenize(s) if word not in string.punctuation]

    # Find the best bigrams for positive sentences based on informative collocations
    bigram_finder1 = BigramCollocationFinder.from_words(words_pos)
    bigrams_best_pos = bigram_finder1.nbest(score_fn, n)

    # Find the best bigrams for negative sentences based on informative collocations
    bigram_finder2 = BigramCollocationFinder.from_words(words_neg)
    bigrams_best_neg = bigram_finder2.nbest(score_fn, n)

    bigrams_all = list(set(bigrams_best_pos).union(set(bigrams_best_neg)))

    # Select only the bigrams that have either one of the word greater than length 3
    bigrams_best = [bigram for bigram in bigrams_all if (len(bigram[0]) > 3 or len(bigram[1]) > 3)]

    return bigrams_best


def bigrams_maximizing_prob_diff(sents, n, stopwords):

    def get_max_diff_bigrams(bigrams, pos, neg):
        prob_diff_bigrams = []
        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()

        # For each bigram, find the positive and negative probabilities
        for bigram in bigrams:
            p = pos.prob(bigram)
            n = neg.prob(bigram)

            # Check if the frequency of occurrence of the bigram in either positive or negative sentences is > 15
            if pos_fd[bigram] >= 15 or neg_fd[bigram] >= 15:
                # Find the absolute difference in probability for each bigram
                prob_diff_bigrams.append((abs(p - n), bigram))

        # Return the bigram list sorted based on the absolute difference between positive and negative probabilities.
        return sorted(prob_diff_bigrams, reverse=True)

    # Get all the bigrams along with the positive or negative tag
    bigrams_tagged = get_bigrams_tagged(sents)

    # Calculate the conditional frequency distribution for positive and negative bigrams.
    cfd = nltk.ConditionalFreqDist((tag, bigram)
                                for tag, bigrams in bigrams_tagged
                                for bigram in bigrams
                                if tag != 0
                                and (bigram[0] not in stopwords or bigram[1] not in stopwords)
                                and (len(bigram[0]) > 3 or len(bigram[1]) > 3))

    # Calculate the conditional probability distribution for the computed conditional frequency distribution
    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]

    bigrams = list(set(pos.samples()).union(set(neg.samples())))

    return get_max_diff_bigrams(bigrams, pos, neg)[:n]


def feature_bigrams(sent, best_bigrams):
    features = {}
    sent_bigrams = set(nltk.bigrams(sent.lower().split(' ')))
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
    training_corpus = AssignmentCorpus(
        'product_data_training_heldout/training/')
    heldout_corpus = AssignmentCorpus('product_data_training_heldout/heldout/')
    # We should do cross validation on all the data given
    all_sents = training_corpus.sents + heldout_corpus.sents

    # This is the data we extract features and do cross validation on.
    sents = sanitize(all_sents)
    stopwords = get_stopwords()

    tagged_sents = get_tagged_sents(sents)
    adjectives = get_adjectives(tagged_sents)

    pos_words = set([str(x) for x in json.load(open('positive_words.json'))])
    neg_words = set([str(x) for x in json.loads(open('negative_words.json')
                                                .read().decode('utf-8', 'ignore'))])

    #max_prob_diff_bigrams = set([bigram for diff, bigram in bigrams_maximizing_prob_diff(sents, 500, stopwords)])
    bigrams_best = best_bigrams(sents, stopwords)

    #max_prob_diff_bigrams = set([bigram for diff, bigram in bigrams_maximizing_prob_diff(sents, 200, stopwords)])

    # Extract features.
    data = []
    for tag, sent in sents:
        #feat1 = feature_adjectives(sent, adjectives)
        #feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
        #feat3 = feature_unigram_probdiff(sent, max_prob_diff_words)
        #feat4 = bigram_word_feats(sent, bigram_features)
        #feat4 = feature_bigrams(sent, max_prob_diff_bigrams)
        features = feature_bigrams(sent, bigrams_best)

        ## Include sent for error analysis
        data.append((features, tag, sent))

    print 'Naive Bayes:\t%s' % evaluate(nltk.NaiveBayesClassifier, data, 10, verbose_errors=False)

if __name__ == '__main__':
    from driver import *
    local_main()
