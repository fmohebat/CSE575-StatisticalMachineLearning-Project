import importlib
import numpy as np
import time
import platform
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

from embedding import embedding_utils

glove = None
Glove = None
Corpus = None
# noinspection PyBroadException
try:
    glove = importlib.import_module('glove')
    Glove = importlib.import_module('glove.Glove')
    Corpus = importlib.import_module('glove.Corpus')
except Exception:
    print('Failed to import Glove - system info: ' + platform.platform())


class GloveEmbedding:

    def __init__(self, d, **kwargs):
        """
        The initializer of the Node2VecEmbedding class
        :param kwargs: a dict contains:
            d: dimension of the embedding
            window_size: context size for optimization
            max_iter: max number of iterations
            n_workers: number of parallel workers
        """
        self._method_name = 'GloVe-Embedding'
        self.d = d
        self.max_iter = kwargs['max_iter']
        self.walks = kwargs['walks']
        self.num_walks = len(self.walks)
        self.walk_len = len(self.walks[0])
        self.window_size = kwargs['window_size']
        self.n_workers = kwargs['n_workers']
        self.learning_rate = kwargs['learning_rate']
        self.num_threads = kwargs['num_threads']
        self.embedding = None
        self._node_num = None
    
    def get_method_name(self):
        return self._method_name

    def get_method_summary(self):
        return '%s_%d' % (self._method_name, self.d)

    def get_embedding(self):
        return self.embedding

    def get_edge_weight(self, i, j):
        return np.dot(self.embedding[i, :], self.embedding[j, :])

    def learn_embedding(self, graph=None, edge_f=None, is_weighted=False, no_python=False):
        t1 = time.time()
        walks = self.walks

        walks = [list(map(str, walk)) for walk in walks]
        self.embedding = embedding_utils.gensim_model_to_embedding(model, walks)
        self._node_num = self.embedding.shape[0]
        t2 = time.time()

        return self.embedding, t2-t1

    def get_reconstructed_adj(self, X=None, node_l=None):
        if X is not None:
            node_num = X.shape[0]
            self.embedding = X
        else:
            node_num = self._node_num
        adj_mtx_r = np.zeros((node_num, node_num))
        for v_i in range(node_num):
            for v_j in range(node_num):
                if v_i == v_j:
                    continue
                adj_mtx_r[v_i, v_j] = self.get_edge_weight(v_i, v_j)
        return adj_mtx_r


if __name__ == '__main__':
    print(str(glove))
