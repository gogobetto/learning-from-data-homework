
# coding: utf-8

# In[ ]:

from random import uniform

import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')

import numpy as np

from sklearn import svm

from cvxopt import matrix, solvers
import cvxopt
# cvxopt.solvers.options['show_progress'] = False


# In[ ]:

def ex11():
    X = [[1,0], [0,1], [0,-1], [-1,0], [0,2],[0,-2],[-2,0]]
    Y = [-1,-1,-1,1,1,1,1]

    def transform(x):
        return [
            x[1]**2 - 2*x[0] - 1,
            x[0]**2 - 2*x[1] + 1
        ]

    plt.scatter(*zip(*map(transform, X)), c=['k' if x==1 else 'w' for x in Y])
    plt.grid()
    
# ex11()


# In[ ]:

def ex12():
    X = [[1.,0.], [0.,1.], [0.,-1.], [-1.,0.], [0.,2.],[0.,-2.],[-2.,0.]]
    Y = [-1.,-1.,-1.,1.,1.,1.,1.]

    ssvm = svm.SVC(kernel='poly',
                   C=1e10,
                   gamma=1,
                   degree=2,
                   coef0=1)
    ssvm.fit(X, Y)

    return len(ssvm.support_vectors_)

ex12()


# In[ ]:

def rand_point():
    return np.array([uniform(-1, 1), uniform(-1, 1)])

def f(P):
    return np.sign(P[1] - P[0] + 0.25 * np.sin(np.pi * P[0]))

def dataset_point():
    p = rand_point()
    return (p, f(p))
                   
def make_dataset(N):
    return [dataset_point() for _ in range(N)]
    
def ex13():
    def do_run():
        train_set = zip(*make_dataset(100))

        ssvm = svm.SVC(kernel='rbf', gamma=1.5)
        ssvm.fit(*train_set)

        return 1 if ssvm.score(*train_set)==1.0 else 0
        
    return np.mean([do_run() for _ in range(1000)])
    
ex13()


# In[ ]:



