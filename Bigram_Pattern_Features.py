
__author__ = 'AJ Renold & Siddharth Agrawal'

import nltk
from itertools import islice


def patterns_maximizing_prob_diff(tagged_sents, n):
    # extracts n pattenrs that maximize class probablity diff from:
    patterns = ['JJ NN|NNS', 'RB|RBR|RBS JJ',
                'JJ JJ', 'NN|NNS JJ', 'RB|RBR|RBS VB|VBD|VBN|VBG']

    def posPatternFinder(tagged_sent, pattern):
        valid_pos = [
            'ADJ', 'ADV', 'CNJ', 'DET', 'EX', 'FW', 'MOD', 'N', 'NP', 'NUM', 'PRO', 'P', 'TO', 'UH', 'V', 'VD', 'VG', 'VN', 'WH', 'NN', 'JJ', 'NNS', 'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBN', 'VBG']

        matches = []
        pattern = pattern.split(' ')
        n = len(pattern)
        for ng in nltk.ngrams(tagged_sent, n):
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
                # print matches
            return matches

    def extract_bigram_patterns(tagged_sents, patterns):

        extracted_patterns = []
        for label, sent in islice(tagged_sents, None):
            matches = []
            for p in patterns:
                m = posPatternFinder(sent, p)
                if m:
                    for ng in m:
                        words = [word.lower() for word, pos in ng]
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
        for label, pattern in input_patterns)

    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]

    patterns = list(set(pos.samples()).union(set(neg.samples())))

    patterns = get_max_diff(patterns, pos, neg)
    patterns = patterns[:n]
    return set([pattern for diff, pattern in patterns])


def pos_and_neg_patterns_maximizing_prob_diff(tagged_sents, n):
    # extracts n pattenrs that maximize class probablity diff from:
    patterns = ['JJ NN|NNS', 'RB|RBR|RBS JJ',
                'JJ JJ', 'NN|NNS JJ', 'RB|RBR|RBS VB|VBD|VBN|VBG']

    def posPatternFinder(tagged_sent, pattern):
        valid_pos = [
            'ADJ', 'ADV', 'CNJ', 'DET', 'EX', 'FW', 'MOD', 'N', 'NP', 'NUM', 'PRO', 'P', 'TO', 'UH', 'V', 'VD', 'VG', 'VN', 'WH', 'NN', 'JJ', 'NNS', 'RB', 'RBR', 'RBS',
            'VB', 'VBD', 'VBN', 'VBG']

        matches = []
        pattern = pattern.split(' ')
        n = len(pattern)
        for ng in nltk.ngrams(tagged_sent, n):
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
                # print matches
            return matches

    def extract_bigram_patterns(tagged_sents, patterns):

        extracted_patterns = []
        for label, sent in islice(tagged_sents, None):
            matches = []
            for p in patterns:
                m = posPatternFinder(sent, p)
                if m:
                    for ng in m:
                        words = [word.lower() for word, pos in ng]
                        matches.append((label, tuple(words)))

            extracted_patterns.extend(matches)

        return extracted_patterns

    def get_max_diff(patterns, pos, neg):
        prob_diff_pos = []
        prob_diff_neg = []

        pos_fd = pos.freqdist()
        neg_fd = neg.freqdist()

        for pattern in patterns:
            p = pos.prob(pattern)
            n = neg.prob(pattern)

            if pos_fd[pattern] >= 2 or neg_fd[pattern] >= 2:
                prob_diff_pos.append(((p - n), pattern))
                prob_diff_neg.append(((n - p), pattern))

        return sorted(prob_diff_pos, reverse=True), sorted(prob_diff_neg, reverse=True)

    input_patterns = extract_bigram_patterns(tagged_sents, patterns)

    cfd = nltk.ConditionalFreqDist(
        (label, pattern)
        for label, pattern in input_patterns)

    cpdist = nltk.ConditionalProbDist(cfd, nltk.MLEProbDist)
    pos = cpdist[1]
    neg = cpdist[-1]

    patterns = list(set(pos.samples()).union(set(neg.samples())))

    pos_patterns, neg_patterns = get_max_diff(patterns, pos, neg)
    pos_patterns, neg_patterns = pos_patterns[:n], neg_patterns[:n]
    return set([pattern for diff, pattern in pos_patterns]), set([pattern for diff, pattern in neg_patterns])


def feature_patterns(sent, max_prob_diff_patterns):
    features = {}

    sent_bigrams = set(nltk.bigrams(sent.lower().split(' ')))
    for bigram in max_prob_diff_patterns:
        if bigram in sent_bigrams:
            features['contains({0},{1})'.format(bigram[0], bigram[1])] = True
        else:
            features['contains({0},{1})'.format(bigram[0], bigram[1])] = False

    return features


def feature_patterns_count(sent, max_pattern_pos, max_pattern_neg):
    features = {'feature_pattern_pos': 0, 'feature_pattern_neg': 0}
    pos, neg = 0, 0
    sent_bigrams = nltk.bigrams(sent.lower().split(' '))
    for bigram in sent_bigrams:
        if bigram in max_pattern_pos:
            pos += 1
        elif bigram in max_pattern_neg:
            neg += 1
    if len(sent_bigrams) == 0:
        pos, neg = 0
    else:
        pos = pos * 1.0 / len(sent_bigrams)
        neg = neg * 1.0 / len(sent_bigrams)
    features['feature_pattern_pos'] = pos
    features['feature_pattern_neg'] = neg
    features['diff_feature_pattern'] = pos - neg
    features['sum_feature_pattern'] = pos + neg
    return features
