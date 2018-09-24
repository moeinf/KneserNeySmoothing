# This Python file uses the following encoding: utf-8
## This python file contains Kneser-Ney smoothing model as well
# as the script for experiments on ptb data and a toy example
from collections import defaultdict
import numpy as np


class KneserNeyNGram():
    def __init__(self,
                 sents,
                 full_seq,
                 input_type='full_seq',
                 n=2,
                 discount=0.75):

        """
        sents: list of sentences
        full_seq: the input sequence
        input_type: whether the input is a list of sentences or a long sequence
        n: order of the model
        D: discount value
        """

        self.n = n
        self.discount = discount
        self.input_type = input_type


        self._N_dot_tokens_dict = N_dot_tokens = defaultdict(set) # N+(·w_<i+1>)
        self._N_tokens_dot_dict = N_tokens_dot = defaultdict(set) # N+(w^<n-1> ·)
        self._N_dot_tokens_dot_dict = N_dot_tokens_dot = defaultdict(set) # N+(· w_<i-n+1>^<i-1> ·)
        self.counts = defaultdict(int)
        vocabulary = []

        # padding each sentence to the right model order
        sents = list(map(lambda x: ['<s>']*(n-1) + x , sents))


        # if the input is a single sequence of words
        if self.input_type == 'full_seq':
            for j in range(n+1):
                # all k-grams for 0 <= k <= n
                for i in range(0, len(full_seq) - j + 1):
                    ngram = tuple(full_seq[i: i + j])
                    self.counts[ngram] += 1
                    if ngram:
                        if len(ngram) == 1:
                            vocabulary.append(ngram[0])
                        else:
                            right_token, left_token, right_kgram, left_kgram, middle_kgram =\
                                ngram[-1:], ngram[:1], ngram[1:], ngram[:-1], ngram[1:-1]
                            N_dot_tokens[right_kgram].add(left_token)
                            N_tokens_dot[left_kgram].add(right_token)
                            if middle_kgram:
                                N_dot_tokens_dot[middle_kgram].add((left_token,right_token))
                self.vocab = set(vocabulary)

        # if the inputs are a list of sentences
        elif self.input_type == 'sents':
            for sent in sents:
                for j in range(n+1):
                    # all k-grams for 0 <= k <= n
                    for i in range(n-j, len(sent) - j + 1):
                        ngram = tuple(sent[i: i + j])
                        self.counts[ngram] += 1
                        if ngram:
                            if len(ngram) == 1:
                                vocabulary.append(ngram[0])
                            else:
                                right_token, left_token, right_kgram, left_kgram, middle_kgram =\
                                    ngram[-1:], ngram[:1], ngram[1:], ngram[:-1], ngram[1:-1]
                                N_dot_tokens[right_kgram].add(left_token)
                                N_tokens_dot[left_kgram].add(right_token)
                                if middle_kgram:
                                    N_dot_tokens_dot[middle_kgram].add((left_token,right_token))
                if n-1:
                    self.counts[('<s>',)*(n-1)] = len(sents)
                self.vocab = set(vocabulary)

        temp = 0
        for w in self.vocab:
            temp += len(self._N_dot_tokens_dict[(w,)])
        self._N_dot_dot_attr = temp


    def count(self, tokens):

        """
        returns the count for an n-gram or (n-1)-gram.
        tokens be the n-gram or (n-1)-gram tuple.
        """
        return self.counts[tokens]

    def V(self):
        """
        returns vocabulary size.
        """
        return len(self.vocab)

    def N_dot_dot(self):
        """
        returns N+(..)
        """
        return self._N_dot_dot_attr

    def N_tokens_dot(self, tokens):
        """
        returns the set of words w for which C(tokens,w) > 0
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_tokens_dot_dict[tokens]

    def N_dot_tokens(self, tokens):
        """
        returns the set of words w for which C(w,tokens) > 0
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dict[tokens]

    def N_dot_tokens_dot(self, tokens):
        """
        returns the set of word pairs (w,w') for which C(w,tokens,w') > 0
        """
        if type(tokens) is not tuple:
            raise TypeError('`tokens` has to be a tuple of strings')
        return self._N_dot_tokens_dot_dict[tokens]


    def cond_prob(self, token, prev_tokens=tuple()):
        n = self.n

        # unigram case
        if not prev_tokens and n == 1:
            return (self.count((token,))+1) / (self.count(()) + self.V())


        # lowest ngram (n >1 and unigram back-off)
        if not prev_tokens and n > 1:
            temp1 = len(self.N_dot_tokens((token,)))
            temp2 = self.N_dot_dot()
            # return temp1 *1.0 / temp2
            return ((temp1 + 1) * 1.0) / (temp2 + self.V())

        # highest n-gram (no back-off)
        if len(prev_tokens) == n-1:
            c = self.count(prev_tokens)
            if c == 0:
                return self.cond_prob(token, prev_tokens[1:])
            term1 = max(self.count(prev_tokens + (token,)) - self.discount, 0) / c
            unassigned_mass = self.discount * len(self.N_tokens_dot(prev_tokens)) / c
            back_off = self.cond_prob(token, prev_tokens[1:])
            return term1 + unassigned_mass * back_off

        # lower ngram (back off to lower-order models except the unigram)
        else:
            temp = len(self.N_dot_tokens_dot(prev_tokens))
            if temp == 0:
                return self.cond_prob(token, prev_tokens[1:])
            term1 = max(len(self.N_dot_tokens(prev_tokens + (token,))) - self.discount, 0) / temp
            unassigned_mass = self.discount * len(self.N_tokens_dot(prev_tokens)) / temp
            back_off = self.cond_prob(token, prev_tokens[1:])
            return term1 + unassigned_mass * back_off

###toy example
# toy_sents = [['1','2','6','7'],['4','6','2','5','2'],['1','2','3','4','5']
# model = KneserNeyNGram(toy_sents,n=3)
# print model.cond_prob('3',('1','2'))
# print model.count(('1','2','3'))
# print model.count(('1','2'))
# print len(model.N_tokens_dot(('1','2',)))
# print len(model.N_dot_tokens_dot(('2',)))
# print model.count(('2',))
# print len(model.N_dot_tokens(('3',)))
# print model.N_dot_dot()
# print model.V()

##PTB experiments
train_file = '../data/ptb/ptb.train.txt'
test_file = '../data/ptb/ptb.test.txt'
f_read_train=open(train_file,'r')
f_read_test = open(test_file,'r')

train_sents = f_read_train.read().splitlines()
train_sents = [x.split() for x in train_sents]
train_sents = list(map(lambda x:  x + ['</s>'], train_sents))
full_train_seq = []
[full_train_seq.extend(el) for el in train_sents]

test_sents = f_read_test.read().splitlines()
test_sents = [x.split() for x in test_sents]
test_sents = list(map(lambda x:  x + ['</s>'], test_sents))
full_test_seq=[]
[full_test_seq.extend(el) for el in test_sents]

print "Number of train sentences: ", len(train_sents)
print "Number of test sentences: ", len(test_sents)
print "Length of train sequence: ", len(full_train_seq)
print "Length of test sequence: ", len(full_test_seq)

print "sample test sentence: ", test_sents[2]
print "sample test partial sequence", full_test_seq[10:50]

model = KneserNeyNGram([], full_train_seq, input_type='full_seq', n=2, discount=0.75)
print "Vocabulary size: ", model.V()

def check_sum(model, seq):
    """
    used to check the probabilities of a row sum to 1

    """
    prob = []
    for i in range(model.n-1,100+model.n-1):
        context = tuple(seq[i-model.n+1:i])
        sum_prob = 0
        for w in model.vocab:
            sum_prob += model.cond_prob(w, context)
        prob.append(sum_prob)
    return prob

def seq_perplexity(model, seq):
    """
    perplexity of a model when the input is a sequence.

    """
    prob =0
    iter = 0
    for i in range(model.n - 1, len(seq)):
        c_p = model.cond_prob(seq[i], tuple(seq[i - model.n + 1:i]))
        if not c_p:
            return float('-inf')
        prob = prob + np.log(c_p)
        iter = iter + 1
    return np.exp(-1.0*prob/iter)

def sent_log_prob(model,sent):

    """
    log-probability of a sentence.

    """
    prob = 0
    sent = ['<s>'] * (model.n - 1) + sent + ['</s>']

    for i in range(model.n - 1, len(sent)):
        c_p = model.cond_prob(sent[i], tuple(sent[i - model.n + 1:i]))
        if not c_p:
            return float('-inf')
        prob = prob + np.log(c_p)

    return prob


def perplexity(model,sents):
    """
    Perplexity of a model when the inputs are sentences

    """

    iter = 0
    for sent in sents:
        iter += len(sent)
    l = 0
    print 'number of test sentences: ', len(sents)
    for sent in sents:
        l += sent_log_prob(model,sent) / iter
    return np.exp(-l)

# print "perplexity = ", perplexity(model,test_sents)
print "perplixity = ", seq_perplexity(model, full_test_seq)
# print check_sum(model,full_test_seq)