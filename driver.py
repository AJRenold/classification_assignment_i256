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
    ref = []
    test = []

    print 'evaluating %s using k fold cross validation. k = %s' % (classifier, k)
    i, bar = 0, pbar(k)
    bar.start()
    for training, validation in k_fold_cross_validation(data, k):
        model = classifier.train(training)
        #model.show_most_informative_features(15)
        accuracies.append(nltk.classify.accuracy(model, validation))
        for feat, tag in validation:
            guess = model.classify(feat)
            ref.append(tag)
            test.append(guess)

        i += 1
        bar.update(i)
    bar.finish()
    print nltk.ConfusionMatrix(ref, test)
    return sum(accuracies) * 1.0 / len(accuracies)


def sanitize(sents):
    # ignore all sents with no tag or no sent
    sents = [(tag, sent) for tag, sent in sents if sent]
    # simplify each tags list to an average of ratings
    
    # does it make sense to simplify tags to 1, 0 and -1?
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
            print 'list'
            sents[i] = (tag, sent[0])
    # remove whitespace in sents
    sents = [(tag, sent.strip()) for tag, sent in sents]
    return sents

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

def get_unigram_features(sent):
    ## all unigrams
        
    features = {}
    for word in sent.lower().split(' '):
        if word not in string.punctuation:
            features['contains({})'.format(word)] = True
    
    return features

def words_maximizing_prob_diff(sents, n, stopwords):
    ## extracts n unigrams that maximize class probablity diff
    
    ex = set(['router','ipod','norton','jack','apex','diaper','canon',\
            'two','radio','nomad','nokia','phone','disc','month','apple', 'linksys', \
        'nikon'])

    def get_max_diff(words, pos, neg):
        prob_diff_words = []
        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()
        
        for word in words:
            p = pos.prob(word)
            n = neg.prob(word)
            
            if pos_fd[word] >= 15 or neg_fd[word] >= 15:
                if len(word) > 3 and word not in stopwords: #and word not in ex:
                    if not re.findall(r'[\W]|[\d]', word):
                        prob_diff_words.append((abs(p - n), word))

        return sorted(prob_diff_words, reverse=True)
    
    cfd = nltk.ConditionalFreqDist((label, re.sub(r'\W\s','',word))
                               for label, sent in sents
                               for word in sent.lower().split(' ')
                               if word not in string.punctuation 
                               and label != 0)
    
    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]

    words = list(set(pos.samples()).union(set(neg.samples())))
    
    return get_max_diff(words, pos, neg)[:n]

def feature_unigram_probdiff(sent, max_diff_words):
    features = {}
    #for word in word_tokenize(sent):
    #    if word in max_diff_words:
    #        features['contains({})'.format(word)] = True
    sent_words = set(word_tokenize(sent.lower()))
    for word in max_diff_words:
        features['contains(%s)' % word] = word in sent_words

    return features

def patterns_maximizing_prob_diff(tagged_sents, n):
    ## extracts n pattenrs that maximize class probablity diff from:
    patterns = ['JJ NN|NNS','RB|RBR|RBS JJ', 'JJ JJ', 'NN|NNS JJ', 'RB|RBR|RBS VB|VBD|VBN|VBG']
    
    def posPatternFinder(tagged_sent, pattern):
        valid_pos = ['ADJ','ADV','CNJ','DET','EX','FW','MOD','N','NP','NUM','PRO' \
                     ,'P','TO','UH','V','VD','VG','VN','WH', 'NN', 'JJ', 'NNS', 'RB', 'RBR', 'RBS', \
                     'VB', 'VBD', 'VBN', 'VBG' ]
    
        matches = []
        pattern = pattern.split(' ')
        n = len(pattern)
        for ng in nltk.ngrams(tagged_sent,n):
            match = 0
            for i, (word, tag) in enumerate(ng):
                if '!' in pattern[i][0]:
                    if '|' in pattern[i]:
                        multi = pattern[i][1:].split('|')
                        if tag not in multi:
                            match += 1
                    else:
                        if pattern[i][1:] != tag:
                            match += 1
                
                elif '|' in pattern[i]:
                    multi = pattern[i].split('|')
                    if tag in multi:
                        match += 1
                elif pattern[i] in valid_pos and pattern[i] == tag:
                    match += 1
                elif pattern[i].lower() == word.lower():
                    match += 1
                elif pattern[i] == '.*':
                    match += 1
                    
            if match == n:
                matches.append(ng)
                
        if len(matches) > 0:
            if len(matches) > 1:
                pass
                #print matches
            return matches
    
    def extract_bigram_patterns(tagged_sents, patterns):
    
        extracted_patterns = []
        for label, sent in islice(tagged_sents,None):
            matches = []
            for p in patterns:
                m = posPatternFinder(sent, p)
                if m:
                    for ng in m:
                        words = [ word.lower() for word, pos in ng ]
                        matches.append((label, tuple(words)))
            
            extracted_patterns.extend(matches)
        
        return extracted_patterns
    
    def get_max_diff(patterns, pos, neg):
        prob_diff = []
        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()
        
        for pattern in patterns:
            p = pos.prob(pattern)
            n = neg.prob(pattern)
            
            if pos_fd[pattern] >= 2 or neg_fd[pattern] >= 2:
                prob_diff.append((abs(p - n), pattern))

        return sorted(prob_diff, reverse=True)

    input_patterns = extract_bigram_patterns(tagged_sents, patterns)
    
    cfd = nltk.ConditionalFreqDist(
                                   (label, pattern)
                                   for label, pattern in input_patterns )
    
    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]
    
    patterns = list(set(pos.samples()).union(set(neg.samples())))
    
    pattenrs_max_diff = get_max_diff(patterns, pos, neg)[:n]
    return set([ pattern for diff, pattern in pattenrs_max_diff ])

def feature_patterns(sent, max_prob_diff_patterns):
    features = {}

    sent_bigrams = set(nltk.bigrams(sent.lower().split(' ')))
    for bigram in max_prob_diff_patterns:
        if bigram in sent_bigrams:
            features['contains({0},{1})'.format(bigram[0], bigram[1])] = True
        else:
            features['contains({0},{1})'.format(bigram[0], bigram[1])] = False

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

    stopwords = get_stopwords()

    max_prob_diff_words = set([ word for diff, word in words_maximizing_prob_diff(sents, 500, stopwords)])
    max_prob_diff_patterns = patterns_maximizing_prob_diff(tagged_sents, 500)

    # Extract features.
    data = []
    for tag, sent in islice(sents,None):
        feat1 = feature_adjectives(sent, adjectives)
        feat2 = feature_adjectives_curated(sent, pos_words, neg_words)
        feat3 = feature_unigram_probdiff(sent, max_prob_diff_words)
        feat4 = feature_patterns(sent, max_prob_diff_patterns)
        feat1.update(feat2)
        feat1.update(feat3)
        feat1.update(feat4)
        data.append((feat1, tag))

    print 'Naive Bayes:\t%s' % evaluate(nltk.NaiveBayesClassifier, data, 10)
    #print 'Decision Tree:\t%s' % evaluate(nltk.DecisionTreeClassifier, data, 10)


if __name__ == '__main__':
    main()
