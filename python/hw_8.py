
# coding: utf-8

# In[ ]:

import csv, re
import numpy as np
from random import shuffle
from collections import Counter

from sklearn import cross_validation
from sklearn import svm

ftrain = open('features.train')
ftest = open('features.test')

def read_file(f):
    return [map(float, re.split('\s+', r.strip())) for r in f]

all_train = read_file(ftrain)
all_test = read_file(ftest)


# In[ ]:

def run_svm(X, Y, C, K):
    N = len(Y)
    
    ssvm = svm.SVC(kernel='poly', C=10000000, gamma=1, degree=K, coef0=1)
    ssvm.fit(X, Y)

    return ssvm


# In[ ]:

def build_vs_all_set(data, which):
    """Prepare a which-vs-all dataset."""
    X = []
    Y = []

    for p in data:
        X.append(p[1:])

        if int(p[0]) == int(which):
            Y.append(1.0)
        else:
            Y.append(-1.0)
    return X, Y

def build_vs_vs_set(data, one, other):
    """Prepare a one-vs-other dataset"""
    X = []
    Y = []
    
    for p in data:
        if int(p[0]) == one:
            X.append(p[1:])
            Y.append(1.0)
        elif int(p[0]) == other:
            X.append(p[1:])
            Y.append(-1.0)
    
    return X, Y


# In[ ]:

def train_and_score(train_set, degree, C, test_set=False):

    ssvm = svm.SVC(kernel='poly',
                   C=C,
                   gamma=1,
                   degree=degree,
                   coef0=1)
    ssvm.fit(*train_set) # X, Y

    return [ssvm.score(*train_set),
            len(ssvm.support_vectors_),
            ssvm.score(*test_set) if test_set else None]

def train_and_score_rbf(train_set, C, test_set=False):

    ssvm = svm.SVC(kernel='rbf', gamma=1, C=C)
    ssvm.fit(*train_set) # X, Y

    return [ssvm.score(*train_set),
            len(ssvm.support_vectors_),
            ssvm.score(*test_set) if test_set else None]


# In[ ]:

def ex2():
    r = []
    for j in [0, 2, 4, 6, 8]:
        r.append([j] +
            train_and_score(
                train_set = build_vs_all_set(all_train, j),
                degree = 2,
                C = 0.01))
    return min(r, key=lambda p: p[1])

def ex3():
    r = []
    for j in [1, 3, 5, 7, 9]:
        r.append([j] +
            train_and_score(
                train_set = build_vs_all_set(all_train, j),
                degree = 2,
                C = 0.01))
    return max(r, key=lambda p: p[1])
    
result_2 = ex2()
result_3 = ex3()

print result_2
print result_3

def ex4():
    return abs(result_3[2] - result_2[2])

print ex4()


# In[ ]:

def ex5():
    train_set = build_vs_vs_set(all_train, 1, 5)
    test_set = build_vs_vs_set(all_test, 1, 5)

    for C in [0.001, 0.01, 0.1, 1]:
        print train_and_score(
            train_set = train_set,
            degree=2,
            C=C,
            test_set = test_set)
        print '**'
        
ex5()


# In[ ]:

def ex6():
    train_set = build_vs_vs_set(all_train, 1, 5)
    test_set = build_vs_vs_set(all_test, 1, 5)
    
    for C in [0.0001, 0.001, 0.01, 1]:
        print "** C={:f}".format(C)
        for Q in [2, 5]:
            print train_and_score(train_set=train_set,
                                  degree=Q,
                                  C=C,
                                  test_set=test_set)
        print ''
        
ex6()


# In[ ]:

def ex7_8():

    def do_run():
        """Run and cross-validate all models with a partition."""
        train_set = build_vs_vs_set(all_train, 1, 5)
        # zip, shuffle, and unzip (to keep the X-Y correspondence)
        train_set = zip(*train_set)
        shuffle(train_set)
        train_set = zip(*train_set)
        
        this_run_results = []
        # for each C in the question
        for C in [0.0001, 0.001, 0.01, 0.1, 1]:            
            ssvm = svm.SVC(kernel='poly',
                           C=C,
                           gamma=1,
                           degree=2,
                           coef0=1)

            scores = cross_validation.cross_val_score(
                        ssvm, train_set[0], train_set[1], cv=10)
            
            this_run_results.append([C, np.mean(scores)])
        
        # return a tuple like (C, E_cv) for the best C found in this run
        return max(this_run_results,
                   key=lambda p: p[1])
    
    all_runs = [do_run() for _ in range(100)]

    # count occurrences in the results and average out E_cv
    return Counter([x[0] for x in all_runs]),            1 - np.mean([x[1] for x in all_runs])

ex7_8()


# In[ ]:

def ex9_10():
    train_set = build_vs_vs_set(all_train, 1, 5)
    test_set = build_vs_vs_set(all_test, 1, 5)
    
    for C in [0.01, 1, 100, 10e4, 10e6]:
        print train_and_score_rbf(
                    train_set=train_set,
                    C=C,
                    test_set=test_set)
        
ex9_10()

