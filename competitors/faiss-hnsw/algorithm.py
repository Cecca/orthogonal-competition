"""
NNS Competition – Algorithm Template
======================================
Implement your nearest neighbor search algorithm by filling in the three
methods below.  Do not rename the class or the methods.

Interface contract
------------------
fit(train, **index_params)
    Receives all training vectors as a float32 NumPy array of shape
    (n_train, dim), plus any keyword arguments declared under
    index_params in your scenarios.yaml.
    Build your index here.  Return value is ignored.
    A fresh Algorithm() instance is created for every scenario, so
    state never leaks between scenarios.

query(query, k, **query_params)
    Receives a SINGLE query vector as a float32 NumPy array of shape
    (dim,) and the number of neighbors k to retrieve.
    Must return a 1-D integer array of length k with the indices
    (into the training set) of the k nearest neighbors, in any order.
    Called once per query; wall-clock time is measured individually.

get_n_distances() -> int
    Returns the CUMULATIVE number of distance computations performed
    since this instance was created.  The harness calls this once
    immediately after fit() and once after all queries have finished,
    storing both values independently.  You are responsible for
    incrementing an internal counter inside fit() and query().

Rules
-----
* No index construction is allowed inside query().
* No I/O, network access, or subprocess calls during either phase.
"""

import numpy as np
import faiss


class Algorithm:

    def __init__(self):
        self._n_distances = 0   # cumulative distance counter – update in fit() and query()

    def fit(self, train: np.ndarray, **index_params) -> None:
        """
        Build your index over the training vectors.

        Parameters
        ----------
        train         : np.ndarray, shape (n_train, dim), dtype float32
        **index_params: arbitrary kwargs from scenarios.yaml -> index_params
                        e.g. ef_construction=200, M=32
        """
        # ---- YOUR CODE HERE ----
        # Brute-force baseline: store the training set; no distances at build time.
        self._index = faiss.IndexHNSWFlat(train.shape[1], index_params["M"])
        self._index.hnsw.efConstruction = index_params["efConstruction"]
        self._index.add(train)
        faiss.omp_set_num_threads(1)

    def query(self, query: np.ndarray, k: int, **query_params) -> np.ndarray:
        """
        Return the k nearest neighbors for a single query vector.

        Parameters
        ----------
        query         : np.ndarray, shape (dim,), dtype float32
        k             : int -- number of neighbors to retrieve
        **query_params: arbitrary kwargs from scenarios.yaml -> query_params
                        e.g. ef_search=50

        Returns
        -------
        neighbors : np.ndarray, shape (k,), dtype int32
        """
        self._index.hnsw.efSearch = query_params["ef"]
        D, I = self._index.search(np.expand_dims(query, axis=0).astype(np.float32), k)
        return I[0]

    def get_n_distances(self) -> int:
        """
        Return the total number of distance computations performed so far.
        The harness calls this after fit() and again after all queries.
        """
        # ---- YOUR CODE HERE ----
        return faiss.cvar.hnsw_stats.ndis
