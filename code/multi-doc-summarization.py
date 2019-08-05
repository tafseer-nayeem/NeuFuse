import math
import collections
import nltk
import time
from glob import glob
import os
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from nltk.util import ngrams
from nltk.corpus import stopwords

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("english")
stopset = set(stopwords.words('english'))

import math
import pulp
import numpy as np

# Constants to be used by you when you fill the functions
START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
MINUS_INFINITY_SENTENCE_LOG_PROB = -1000
root_dir = "out_files"
DATA_PATH = 'data/'
OUTPUT_PATH = 'gen_summary/'
WORD_LIMIT = 100

def calc_probabilities(training_corpus):
    unigram_c = collections.defaultdict(int)
    bigram_c = collections.defaultdict(int)
    trigram_c = collections.defaultdict(int)


    for sentence in training_corpus:
        tokens0 = sentence.strip().split()
        tokens1 = tokens0 + [STOP_SYMBOL]
        tokens2 = [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        tokens3 = [START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]
        # unigrams
        for unigram in tokens1:
            unigram_c[unigram] += 1

        # bigrams
        for bigram in nltk.bigrams(tokens2):
            bigram_c[bigram] += 1

        # trigrams
        for trigram in nltk.trigrams(tokens3):
            trigram_c[trigram] += 1

    unigrams_len = sum(unigram_c.itervalues())
    unigram_p = {k: math.log(float(v) / unigrams_len, 2) for k, v in unigram_c.iteritems()}
    unigram_c[START_SYMBOL] = len(training_corpus)
    bigram_p = {k: math.log(float(v) / unigram_c[k[0]], 2) for k, v in bigram_c.iteritems()}

    bigram_c[(START_SYMBOL, START_SYMBOL)] = len(training_corpus)
    trigram_p = {k: math.log(float(v) / bigram_c[k[:2]], 2) for k, v in trigram_c.iteritems()}
    return unigram_p, bigram_p, trigram_p
def q1_output(unigrams, bigrams, trigrams, filename):
    # output probabilities
    outfile = open(filename, 'w')

    unigrams_keys = unigrams.keys()
    unigrams_keys.sort()
    for unigram in unigrams_keys:
        outfile.write('UNIGRAM ' + unigram[0] + ' ' + str(unigrams[unigram]) + '\n')

    bigrams_keys = bigrams.keys()
    bigrams_keys.sort()
    for bigram in bigrams_keys:
        outfile.write('BIGRAM ' + bigram[0] + ' ' + bigram[1]  + ' ' + str(bigrams[bigram]) + '\n')

    trigrams_keys = trigrams.keys()
    trigrams_keys.sort()    
    for trigram in trigrams_keys:
        outfile.write('TRIGRAM ' + trigram[0] + ' ' + trigram[1] + ' ' + trigram[2] + ' ' + str(trigrams[trigram]) + '\n')

    outfile.close()


def score(ngram_p, n, corpus):
    scores = []
    for sentence in corpus:
        sentence_score = 0
        tokens0 = sentence.strip().split()
        if n == 1:
            tokens = tokens0 + [STOP_SYMBOL]
        elif n == 2:
            tokens = nltk.bigrams([START_SYMBOL] + tokens0 + [STOP_SYMBOL])
        elif n == 3:
            tokens = nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL])
        else:
            raise ValueError('Parameter "n" has an invalid value %s' % n)
        for token in tokens:
            try:
                p = ngram_p[token]
            except KeyError:
                p = MINUS_INFINITY_SENTENCE_LOG_PROB
            sentence_score += p
        scores.append(sentence_score)
    return scores

def score_output(scores, filename, end):
    outfile = open(filename, 'w')
    for score in scores:
        outfile.write(str(score) + end)
    outfile.close()

def linearscore(unigrams, bigrams, trigrams, corpus):

    scores = []
    lambda_ = 1.0 / 3
    for sentence in corpus:
        interpolated_score = 0
        tokens0 = sentence.strip().split()
        for trigram in nltk.trigrams([START_SYMBOL] + [START_SYMBOL] + tokens0 + [STOP_SYMBOL]):
            try:
                p3 = trigrams[trigram]
            except KeyError:
                p3 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p2 = bigrams[trigram[1:3]]
            except KeyError:
                p2 = MINUS_INFINITY_SENTENCE_LOG_PROB
            try:
                p1 = unigrams[trigram[2]]
            except KeyError:
                p1 = MINUS_INFINITY_SENTENCE_LOG_PROB
            interpolated_score += math.log(lambda_ * (2 ** p3) + lambda_ * (2 ** p2) + lambda_ * (2 ** p1), 2)
        scores.append(interpolated_score)
    return scores



def get_ngrams(sentence, N):
    tokens = tokenizer.tokenize(sentence.lower())
    clean = [stemmer.stem(token) for token in tokens]
    return [gram for gram in ngrams(clean, N)]

def get_len(element):
    return len(tokenizer.tokenize(element))

def get_overlap(sentence_a, sentence_b, N):
    tokens_a = tokenizer.tokenize(sentence_a.lower())
    tokens_b = tokenizer.tokenize(sentence_b.lower())

    ngrams_a  = [gram for gram in ngrams(tokens_a, N)]
    ngrams_b  = [gram for gram in ngrams(tokens_b, N)]

    if N == 1:
        ngrams_a = [gram for gram in ngrams_a if not gram in stopset]
        ngrams_b = [gram for gram in ngrams_b if not gram in stopset]

    overlap = [gram for gram in ngrams_a if gram in ngrams_b]

    return overlap

def build_binary_overlap_matrix(scored_sentences, overlap_discount, N):
    sentences = [tup[0] for tup in scored_sentences]    
    size = len(sentences)

    overlap_matrix = [[-1 for x in xrange(size)] for x in xrange(size)]

    for i, elem_i in enumerate(sentences):
        for j in range(i, len(sentences)):
            elem_j = sentences[j]

            ## 
            ## Get an approximation for the pairwise intersection term from ROUGE.
            overlap = get_overlap(elem_i, elem_j, N)
            score = len(overlap) * overlap_discount

            overlap_matrix[i][j] = score
            overlap_matrix[j][i] = score

    return overlap_matrix

def solve(sentences, length_constraint, damping, overlap_discount, N):

    sentences_scores = [tup[1] for tup in sentences]
    sentences_lengths = [get_len(tup[0]) for tup in sentences]

    overlap_matrix = build_binary_overlap_matrix(sentences, overlap_discount, N)

    sentences_idx = [tup[0] for tup in enumerate(sentences)]
    pairwise_idx = []
    for i in sentences_idx:
        for j in sentences_idx[i+1:]:
            pairwise_idx.append((i, j))

    x = pulp.LpVariable.dicts('sentences', sentences_idx, lowBound=0, upBound=1, cat=pulp.LpInteger)
    alpha = pulp.LpVariable.dicts('pairwise_interactions', (sentences_idx, sentences_idx), lowBound=0, upBound=1, cat=pulp.LpInteger)
    prob = pulp.LpProblem("ILP-R", pulp.LpMaximize)

    prob += sum(x[i] * sentences_scores[i] for i in sentences_idx) - damping * sum(alpha[i][j] * overlap_matrix[i][j] for i,j in pairwise_idx)
    prob += sum(x[i] * sentences_lengths[i] for i in sentences_idx) <= length_constraint
    for i in sentences_idx:
        for j in sentences_idx:
            prob += alpha[i][j] - x[i] <= 0
            prob += alpha[i][j] - x[j] <= 0
            prob += x[i] + x[j] - alpha[i][j] <= 1

    prob.solve()

    summary = []
    total_len = 0
    for idx in sentences_idx:
        if x[idx].value() == 1.0:
            total_len += sentences_lengths[idx]
            summary.append(sentences[idx])

    return summary, total_len

def ILP_R_Optimizer(sentences, length_constraint, overlap_discount=(1./150.), damping=0.9, max_depth=50, N=2):
    sorted_sentences = sorted(sentences, key=lambda tup:tup[1], reverse=True)

    tmp = sorted_sentences
    if len(sorted_sentences) > max_depth:
        sorted_sentences = sorted_sentences[:max_depth]

    summary, total_len = solve(sentences=sorted_sentences, 
                               length_constraint=length_constraint, 
                               damping=damping, 
                               overlap_discount=overlap_discount,
                               N=N)
    for i in range(3):
        for e in tmp:
            if e in summary:
                continue
            l = get_len(e[0])
            if l <= length_constraint - total_len:
                summary.append(e)
                total_len += l
                break

    return summary


def main():
    # start timer
    time.clock()
    dirs = [os.path.join(root_dir,x) for x in os.listdir(root_dir)]
    for d in dirs:
        print "Processing :::: " + d
        # print root
        corpus = []
        docs_folder = [os.path.join(d,f) for f in os.listdir(d)]
        for docs in docs_folder:
            #print docs
            with open(docs) as m:
                # print file_len(docs)
                for line in m:
                    corpus.append(line)
        unigrams, bigrams, trigrams = calc_probabilities(corpus)
        biscores = score(bigrams, 2, corpus)
        sen = []
        for i in range(0,len(corpus)):
            sen.append((corpus[i], biscores[i]*biscores[i]))
            

        selected_sentences = ILP_R_Optimizer(sen, WORD_LIMIT)

        result_summary = []
        for i in selected_sentences:
            temp = i[0].split("\n")
            result_summary.append(temp[0])

        
        score_output(result_summary, OUTPUT_PATH+d.split('/')[1] + '.txt', ' ')

if __name__ == "__main__": main()
