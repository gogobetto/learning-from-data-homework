
# coding: utf-8

# In[ ]:

import numpy as np
from random import uniform, choice

from cvxopt import matrix, solvers
import cvxopt
cvxopt.solvers.options['show_progress'] = False

from sklearn.svm import libsvm
from sklearn import svm

def rand_point():
    return np.array([uniform(-1, 1), uniform(-1, 1)])

def f(A, B, P):
    return np.sign(np.cross(B-A, P-A, axis=0))

def dataset_point(A, B):
    p = rand_point()
    return (
        np.append([1], p),
        f(A, B, p))

def do_hip(w, x):
    return np.sign(np.dot(w, x))


# In[ ]:

def fraction_misclassified(A, B, w):
    many = 1000
    testset = [dataset_point(A, B) for i in xrange(many)]
    
    cnt = 0
    for p in testset:
        if p[1] != do_hip(w, p[0]):
            cnt +=1
            
    return float(cnt)/many

def compare_pla_svm(N):
    A = rand_point()
    B = rand_point()
    #A=np.array([1,1])
    #B=np.array([-1,-1])

    trainset = np.array([dataset_point(A, B) for i in xrange(N)])

    # throw away datasets with unbalanced examples
    cutoff = max(1, int(.05) * N)
    ones = [x for x in trainset if x[1] == 1]
    if len(ones) < cutoff or len(ones) > (N-cutoff):
        return
    
    w_pla = run_pla(trainset)
    if w_pla is not None:
        pla_perf = fraction_misclassified(A, B, w_pla)
        
        #w_svm, s_svm = run_svm(trainset)
        #svm_perf = fraction_misclassified(A, B, w_svm)
        
        # High-level interface to libsvm
        w_libsvm_h, s_libsvm_h = run_libsvm_highlevel(trainset)
        libsvm_h_perf = fraction_misclassified(A, B, w_libsvm_h)
        
        #return 1 if pla_perf > svm_perf else 0
        return 1 if pla_perf > libsvm_h_perf else 0
    
def run_pla(trainset):
    """Run PLA. Return None if the max number of iterations is exceeded."""
    converged = False
    iter_count = 0
    
    w = np.zeros(3)

    while not converged or iter_count<500:
        misclass = [x for x in trainset if x[1] != do_hip(w, x[0])]
        if len(misclass) == 0:
            #return iter_count, w
            break
        else:
            pivot = choice(misclass)
            w = w + pivot[1] * pivot[0]
            iter_count += 1

    return w

def run_svm(trainset):
    """Run SVM using cvxopt for quadratic programming."""
    N = len(trainset)
    
    # N elements
    Y = [x[1] for x in trainset]
    # Remove the leading 1 for X points!
    X = [np.array(x[0][1:]) for x in trainset]
    
    # enforce a > 0
    G = matrix(-1.0 * np.eye(N))
    h = matrix(0.0, (N, 1))
    
    # enforce y* dot alpha = 0
    # see example:
    # http://cvxopt.org/userguide/coneprog.html#quadratic-programming
    A = matrix(Y, (1, N))
    b = matrix(0.0)
    
    # linear term
    q = matrix(-1.0, (N, 1))

    # quadratic matrix
    P = np.zeros([N, N])
    for i in range(N):
        for j in range(N):
            P[i][j] = Y[i] * Y[j] * np.dot(X[i], X[j])
    P = matrix(P)
    
    res = solvers.qp(P, q, G, h, A, b)
    
    w = np.zeros(2)
    
    for i in range(N):
        w += res['x'][i] * Y[i] * X[i]
        
    supports = [ x[0] for x in enumerate(res['x'])                         if abs(x[1]) > 1e-5]
    i_sup, sup = max(enumerate(res['x']), key=lambda p: p[1])
    
    if not len(supports):
        print 'no supports'
    
    # compute all b's for verification (should be 'similar')
    # bs = [ (1/Y[i]) - np.dot(w, X[i]) for i in supports ]
    b = (1/Y[i_sup]) - np.dot(w, X[i_sup])
    # b = np.mean([1/Y[i] - np.dot(w, X[i]) for i in supports or [i_sup] ])
    
    # return a vector 3 weight, just like PLA does
    return np.append(b, w), len(supports)

def run_libsvm_highlevel(trainset):
    """Use libsvm for the training."""
    N = len(trainset)
    
    # N elements
    Y = np.array([x[1] for x in trainset])
    # Remove the leading 1 for X points!
    X = np.array([x[0][1:] for x in trainset])
    
    """
    support, support_vectors, n_class_SV, \
    sv_coef, intercept, probA, probB, x = libsvm.fit(X, Y, 0, 'linear')
    """
    
    ssvm = svm.SVC(kernel='linear', C=10000000)
    ssvm.fit(X, Y)
    """This parameters can be accessed through the members dual_coef_ which holds the product y_i \alpha_i,
    support_vectors_ which holds the support vectors, and intercept_ which holds the independent term \rho
    """

    w = np.zeros(2)
    for i in range(len(ssvm.support_vectors_)):
        orig_i = ssvm.support_[i]
        w += ssvm.dual_coef_[0][i]  * X[orig_i]
    
    # return a vector 3 weight, just like PLA does
    return np.append([ssvm.intercept_], w), []


# In[ ]:

#
# Exercises 8 and 9

def ex8():
    results = np.array([compare_pla_svm(10) for i in range(1000)])
    results = results[results != np.array(None)]
    return np.mean(results)

def ex9():
    results = np.array([compare_pla_svm(100) for i in range(100)])
    results = results[results != np.array(None)]
    return np.mean(results)


# In[ ]:

#
# Exercise 10

def count_support_vectors():
    A = rand_point()
    B = rand_point()

    trainset = np.array([dataset_point(A, B) for i in xrange(100)])
    svm_perf, supports = run_svm(trainset)
    return supports

def ex10():
    return np.mean([count_support_vectors() for i in range(100)])


# In[ ]:

A = rand_point()
B = rand_point()
#A=np.array([1,1])
#B=np.array([-1,-1])

N = 10

trainset = np.array([dataset_point(A, B) for i in xrange(N)])

# throw away datasets with unbalanced examples
cutoff = max(1, int(.05) * N)
ones = [x for x in trainset if x[1] == 1]
if len(ones) < cutoff or len(ones) > (N-cutoff):
    raise Exception("Unbalanced")
    

    
#print run_pla(trainset)
#print run_svm_libsvm(trainset)

w_pla = run_pla(trainset)
pla_perf = fraction_misclassified(A, B, w_pla)
print 'PLA weights', w_pla

w_svm, supports = run_svm_libsvm(trainset)
svm_perf = fraction_misclassified(A, B, w_svm)
print 'SVM weights', w_svm

print "Performance", pla_perf, svm_perf

