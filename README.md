# Statistical Machine Learning

## Problem Statement
In the multi-label classification for graph nodes problem, we are given a graph G(V,E,W) with a subset of nodes V<sub>l</sub> ⊂ V labeled, where V is the set of nodes in the graph (possibly augmented with other features), and V<sub>u</sub> = V/V<sub>l</sub> is the set of unlabeled nodes. Here W is the weight matrix, and E is the set of edges. Let Y be the set of m possible labels, and Y<sub>l</sub> = {y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>l</sub>} be the initial labels on nodes in the set V<sub>l</sub>. The task is to infer labels Y on all nodes V of the graph.

## Solution Idea
By utilizing the random sampling method established by node2vec and modifying the choice of embedd in gand classiﬁcation models, we expect to optimize the DLRW methods for the case of multi-label node classiﬁcation on large graphs. The improvement will be determined by improved node-classiﬁcation accuracy as measured by Macro/Micro-F1 scores on datasets across a variety of ﬁelds while maintaining the tractability of the algorithm for large graph datasets.

## The Pipeline

- **To generate word embedding**
    - File: **run\_experiment.py**
    - In the **main** function, set the dataset (or the walks files) and other options
    - In the **run\_experiment** function, choose the sampling strategy and the word embedding models to be used

- **To evaluate the word embedding**
    - File: **run\_evaluation.py**
    - Results: the results will be saved in the evaluation\_results.csv file
    - In the **run\_evaluation** function, set the embedding files to be used
    - In the **run\_classification\_experiment** function, set the multi-label classifiers to be used
    
- **Notes**
    - All the modules are compatible with the GEM library
    - The pipeline is inherited from the GEM library, which means we can add evaluation modules in GEM
        - Example from GEM: https://github.com/palash1992/GEM/blob/master/examples/run_karate.py
    
## Sampling Strategies

- **Biased Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/biased_walk.py
    - **citation**: Nguyen, Duong, and Fragkiskos D. Malliaros. "BiasedWalk: Biased Sampling for Representation Learning on Graphs." 2018 IEEE International Conference on Big Data (Big Data). IEEE, 2018.
    - **note**: the implementation is based on the author's [open-source code](https://github.com/duong18/BiasedWalk/tree/master/source)

- **Node2Vec Biased Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/node2vec_random_walk_sampling.py
    - **citation**: Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. ACM, 2016.
    - **note**: the implementation is based on the author's [open-source code](https://github.com/aditya-grover/node2vec)

- **Uniform Simple Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/simple_random_walk_sampling.py
    - **note**: the simple random walk which uniformly chooses the neighbor to visit

- **Approximated BFS Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/bfs_walk_sampling.py
    - **note**: the approximated BFS random walk by setting p = 0.25 and q = 4 in node2vec

- **Approximated DFS Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/temperature_random_walk.py
    - **note**: use logistic function to make the random walk gradually switch from BFS to DFS

- **The Combined Random Walk**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/sampling/dfs_walk_sampling.py
    - **note**: the approximated BFS random walk by setting p = 4 and q = 0.25 in node2vec


## Word Embedding Methods

- **CBOW**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/cbow_embedding.py
    - **note**: use the word2vec embedding model in gensim
    - **citation**: Rehurek, Radim, and Petr Sojka. "Software framework for topic modelling with large corpora." In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010.

- **SkipGram**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/node2vec_embedding.py
    - **note**: the SkipGram word embedding used in node2vec
    - **citation**: Rehurek, Radim, and Petr Sojka. "Software framework for topic modelling with large corpora." In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010.

- **Fast-Text**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/fast_text_embedding.py
    - **note**: the Fast-Text word embedding in gensim
    - **citation**: Rehurek, Radim, and Petr Sojka. "Software framework for topic modelling with large corpora." In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. 2010.

- **Glove**
    - **implementation**: https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/embedding/glove_embedding.py
    - **note**: the Glove word embedding in glove_python
    - **source**: https://github.com/maciejkula/glove-python
    - **citation**: Pennington, Jeffrey, Richard Socher, and Christopher Manning. "Glove: Global vectors for word representation." Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014.

## Dataset
All the dataset in the data directory have been parsed into edgelist format, which are compatible with the library [GEM: Graph Embedding Methods](https://github.com/palash1992/GEM) and the [node2vec (python implementation)](https://github.com/aditya-grover/node2vec).

### Due to file size limitation on Github, you might need to parse and generate some of the large dataset by yourself from the original raw dataset. The corresponding preprocessor/parser functions can be found under the [data_preprocessor directory](https://github.com/GuanSuns/Graph-Embedding-Algorithms/tree/master/data_preprocessor) 

- **The Flickr dataset**
    - **raw data**: http://leitang.net/social_dimension.html
    - **content**: 80513 nodes, 5899882 links, and 195 categories.
    - **preprocessor/parser**: [flickr_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/flickr_deepwalk_preprocessor.py)
    - **flickr-deepwalk.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **flickr-deepwalk-labels.txt**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
    - **citation**: L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM.
    
- **The BlogCatalog dataset in DeepWalk**
    - **raw data**: http://leitang.net/social_dimension.html
    - **content**: 10312 nodes, 333983 links, and 39 categories.
    - **preprocessor/parser**: [blog_catalog_deepwalk_preprocessor.py](https://github.com/GuanSuns/Graph-Embedding-Algorithms/blob/master/data_preprocessor/blog_catalog_deepwalk_preprocessor.py)
    - **blog-catalog.edgelist**: each line represents an edge in the graph; the format is (node_from, node_to, weight)
    - **blog-catalog-labels.txt**: each line represents a node and the labels it has; the format is (node_id, [list of labels]).
    - **citation**: L. Tang and H. Liu. Relational learning via latent social dimensions. In Proceedings of the 15th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’09, pages 817–826, New York, NY, USA, 2009. ACM.    

