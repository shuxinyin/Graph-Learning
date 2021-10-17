from scipy.sparse import coo_matrix, csr_matrix
import numpy as np
import networkx as nx
import scipy as sp

def nx_test_to_scipy_sparse_matrix1():
    G = nx.Graph([(1, 1)])
    A = nx.to_scipy_sparse_matrix(G)
    print(A.todense())

    A.setdiag(A.diagonal() * 2)
    print(A.todense())

def nx_test_to_scipy_sparse_matrix2():
    G = nx.MultiDiGraph()
    G.add_edge(0, 1, weight=2)
    G.add_edge(1, 0)
    G.add_edge(2, 2, weight=3)
    G.add_edge(2, 2)
    S = nx.to_scipy_sparse_matrix(G, nodelist=[0, 1, 2])
    print(S.shape)
    print(S)
    print(S.todense())

def test_sp_csr_matrix():
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    csr = csr_matrix((data, indices, indptr), shape=(3, 3))
    print(csr)
    print(csr.toarray())


def test_sp_coo_matrix():
    row = np.array([0, 0, 1, 3, 1, 0, 0])
    col = np.array([0, 2, 1, 3, 1, 0, 0])
    data = np.array([1, 1, 1, 1, 1, 1, 1])
    coo = coo_matrix((data, (row, col)), shape=(4, 4))
    print(coo)
    print(coo.toarray())


test_sp_coo_matrix()