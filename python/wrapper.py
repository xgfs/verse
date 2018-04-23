import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import glob
import os
import multiprocessing


class VERSE:
    """
    Compute VERSE embedding using multiple cores (if avaialble).
    """
    def __init__(self, cpath=None):
        path = os.path.dirname(os.path.realpath(__file__)) if cpath is None else cpath
        try:
            sofile = (glob.glob(os.path.join(path, 'verse*.so')) +
                      glob.glob(os.path.join(path, '*verse*.dll')))[0]
            self.C = ctypes.cdll.LoadLibrary(os.path.join(path, sofile))
        except (IndexError, OSError):
            raise RuntimeError('Cannot find/open VERSE shared library')
        self.C.verse_ppr_train.restype = ctypes.c_int
        self.C.verse_ppr_train.argtypes = [
            ndpointer(ctypes.c_float),  # w0
            ndpointer(ctypes.c_int),  # offsets
            ndpointer(ctypes.c_int),  # edges
            ctypes.c_int,  # num_nodes
            ctypes.c_int,  # num_edges
            ctypes.c_int,  # n_hidden
            ctypes.c_int,  # steps
            ctypes.c_int,  # n_neg_samples
            ctypes.c_float,  # lr
            ctypes.c_float,  # alpha
            ctypes.c_int,  # rng_seed
            ctypes.c_int  # n_threads
            ]
        self.C.verse_neigh_train.restype = ctypes.c_int
        self.C.verse_neigh_train.argtypes = [
            ndpointer(ctypes.c_float),  # w0
            ndpointer(ctypes.c_int),  # offsets
            ndpointer(ctypes.c_int),  # edges
            ctypes.c_int,  # num_nodes
            ctypes.c_int,  # num_edges
            ctypes.c_int,  # n_hidden
            ctypes.c_int,  # steps
            ctypes.c_int,  # n_neg_samples
            ctypes.c_float,  # lr
            ctypes.c_int,  # rng_seed
            ctypes.c_int  # n_threads
            ]

    def verse_ppr(self, graph, w=None, n_hidden=128, alpha=0.85,
                  steps=100000, n_neg_samples=3, lr=0.0025, rng_seed=0,
                  n_threads=-1):
        """Train or partially update an embedding with VERSE, using
            personalized PageRank as similarity.

        Args:
            graph: scipy sparse CSR matrix.
            w: initialized embedding. if the argument is not provided,
               the embedding will be initialized as in the paper.
            n_hidden: embedding dimensionality.
            alpha: alpha parameter of PPR similarity.
            steps: number of update steps for the algorithm.
            n_neg_samples: number of negative samples to use for updates.
            lr: learning rate.
            rng_seed: random seed. If 0 (default), randomize from time.
            n_threads: number of threads to use.

        Returns:
            The embedding. Note that if w is supplied, it is updated inplace.

        """
        nv = graph.shape[0]
        ne = graph.nnz
        if w is None:
            w = np.random.rand(nv, n_hidden).astype(np.float32) - 0.5
        if n_threads < 0:
            n_threads = multiprocessing.cpu_count() + 1 + n_threads
        if n_threads == 0:
            raise RuntimeError('Number of threds can not be zero!')
        self.C.verse_ppr_train(w, graph.indptr, graph.indices, nv, ne,
                               n_hidden, steps, n_neg_samples, lr, alpha,
                               rng_seed, n_threads)
        return w

    def verse_neigh(self, graph, w=None, n_hidden=128, steps=100000,
                    n_neg_samples=3, lr=0.0025, rng_seed=0, n_threads=-1):
        """Train or partially update an embedding with VERSE, using
            adjacency as similarity.

        Args:
            graph: scipy sparse CSR matrix.
            w: initialized embedding. if the argument is not provided,
               the embedding will be initialized as in the paper.
            n_hidden: embedding dimensionality.
            steps: number of update steps for the algorithm.
            n_neg_samples: number of negative samples to use for updates.
            lr: learning rate.
            rng_seed: random seed. If 0 (default), randomize from time.
            n_threads: number of threads to use.

        Returns:
            The embedding. Note that if w is supplied, it is updated inplace.

        """
        nv = graph.shape[0]
        ne = graph.nnz
        if w is None:
            w = np.random.rand(nv, n_hidden).astype(np.float32) - 0.5
        if n_threads < 0:
            n_threads = multiprocessing.cpu_count() + 1 + n_threads
        if n_threads == 0:
            raise RuntimeError('Number of threds can not be zero!')
        self.C.verse_neigh_train(w, graph.indptr, graph.indices, nv, ne,
                                 n_hidden, steps, n_neg_samples, lr,
                                 rng_seed, n_threads)
        return w
