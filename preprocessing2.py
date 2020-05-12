# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx

from scipy.sparse import *
import pickle
import math
import scipy

def load_network(g):
    '''
    A = nx.adjacency_matrix(g, nodelist=[str(v) for v in range(g.number_of_nodes())])
    A = A.todense()
    A = np.squeeze(np.asarray(A))

    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=1) == 0)
    '''
    A = nx.adjacency_matrix(g, nodelist=[str(v) for v in range(g.number_of_nodes())])
    
    A = A - diags(A.diagonal())
    A = A + diags(np.asarray(A.sum(axis=1) == 0, dtype=np.float).flatten())

    return A

############################################################


def RWR(A, walk_len=5, cont_prob=0.98, normalize=True):
    """Random Walk on graph"""
    '''
    if normalize is True:
        #A = (A - A.min()) / float( A.max() - A.min() ) ###############################
        A = _normalizeSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = np.eye(n, dtype=float)
    P = P0.copy()
    M = np.zeros((n, n), dtype=float)
    for i in range(0, walk_len):
        P = cont_prob*np.dot(P, A) + (1. - cont_prob)*P0
        M = M + P
    '''
    if normalize is True:
        A = (A - A.min()) / (A.max() - A.min())
        A = _normalizeSimMat(A)
    # Random surfing
    n = A.shape[0]
    P0 = identity(n, dtype=np.float)
    P = P0.copy()
    M = csr_matrix( (n,n), dtype=np.float )
    for i in range(0, walk_len):
        P = cont_prob * np.dot(P, A) + (1. - cont_prob) * P0
        M = M + P

    return M


def _normalizeSimMat(A):
    #A = A.todense()
    #print(A)
    #A = np.squeeze(np.asarray(A))
    #print(A)
    """Normalize rows of similarity matrix"""
    '''
    A = A - np.diag(np.diag(A))
    A = A + np.diag(A.sum(axis=0) == 0)
    col = A.sum(axis=0)
    A = A.astype(np.float)/col[:, None]
    '''

    A = A - diags(A.diagonal())
    A = A + diags(np.asarray(A.sum(axis=1) == 0, dtype=np.float).flatten())
    A = A.astype(np.float)
    col = 1.0 / A.sum(axis=0)
    A = A.multiply(col.T)
    #A = A.multiply(1.0 / A.sum(axis=1))
    return A


############################################################


def PPMI_matrix(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _normalizeSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=np.float).flatten()
    #col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=np.float).flatten()
    #row = row.reshape((n, 1))
    D = np.sum(col)
    
    np.seterr(all='ignore')
    PPMI = lil_matrix((n,n), dtype=np.float )

    M = M.tocsr()
    for i, j in zip(*M.nonzero()):
        value = np.log( D * M[i, j] / (row[i]*col[j]) )
        if value > 0 and math.isnan(value) is False:
            PPMI[i, j] = value

    #PPMI = np.log(np.divide(D*M.todense(), np.dot(row, col)))
    #PPMI[np.isnan(PPMI)] = 0
    #PPMI[PPMI < 0] = 0

    return PPMI

def PPMI_matrix_old(M):
    """ Compute Positive Pointwise Mutual Information Matrix"""
    M = _normalizeSimMat(M)
    n = M.shape[0]
    col = np.asarray(M.sum(axis=0), dtype=float)
    col = col.reshape((1, n))
    row = np.asarray(M.sum(axis=1), dtype=float)
    row = row.reshape((n, 1))
    D = np.sum(col)

    np.seterr(all='ignore')
    PPMI = np.log(np.divide(D*M, np.dot(row, col)))
    PPMI[np.isnan(PPMI)] = 0
    PPMI[PPMI < 0] = 0

    return PPMI

'''
def _net_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    if X.min() < 0:
        print ("### Negative entries in the matrix are not allowed!")
        X[X < 0] = 0
        print ("### Matrix converted to nonnegative matrix.")
        print
    if (X.T == X).all():
        pass
    else:
        print ("### Matrix not symmetric.")
        X = X + X.T - np.diag(np.diag(X))
        print ("### Matrix converted to symmetric.")

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_normalize(Net):
    """
    Normalize Nets or list of Nets.
    """
    if isinstance(Net, list):
        for i in range(len(Net)):
            Net[i] = _net_normalize(Net[i])
        print (A.shape)
    else:
        Net = _net_normalize(Net)

    return Net

'''




''' 
if __name__ == "__main__":
    file_path = "../datasets/karate.gml"
    #file_path = "../NodeSketch/graphs/wiki_renaissance.gml"
    g = nx.read_gml(file_path)

    # Load STRING networks
    A = load_network(g)
    #print(A[0, :])
    #print(A)
    # Compute RWR + PPMI

    #print ("### Computing PPMI for network")
    A = RWR(A)
    print(A[0, :])
    #A = A.toarray()
    #print(type(A))

    A = PPMI_matrix(A)
    print("---------")
    print(A[0, :].toarray())
 '''

