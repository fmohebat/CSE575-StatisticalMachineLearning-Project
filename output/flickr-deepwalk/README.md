# Embeddings

## Setting 1
- Sampled walks: node2vec-random-walk-1573955197.082777.txt
    - Number of dimensions. Default is 128 (-d:)=128
    - Length of walk per source. Default is 80 (-l:)=80
    - Number of walks per source. Default is 10 (-r:)=10
    - Context size for optimization. Default is 10 (-k:)=10
    - Number of epochs in SGD. Default is 1 (-e:)=1
    - Return hyper-parameter. Default is 1 (-p:)=1
    - Inout hyper-parameter. Default is 1 (-q:)=1
- CBOW
    - embedding: flickr-deepwalk_CBOW-Embedding_1574110071.6025774.emb
        - d = 128
        - kwargs[\'max\_iter\'] = 1
        - kwargs[\'walks\'] = walks
        - kwargs[\'window\_size\'] = 10
        - kwargs[\'n\_workers\'] = 5
- FastText
    - embedding: flickr-deepwalk_FastText-Embedding_1574109675.4526346.emb
        - d = 128
        - kwargs[\'max\_iter\'] = 1
        - kwargs[\'walks\'] = walks
        - kwargs[\'window\_size\'] = 10
        - kwargs[\'n\_workers\'] = 5
- Node2Vec
    - embedding: flickr-deepwalk_Node2Vec-Embedding_1574109430.8421443.emb
        - d = 128
        - kwargs[\'max\_iter\'] = 1
        - kwargs[\'walks\'] = walks
        - kwargs[\'window\_size\'] = 10
        - kwargs[\'n\_workers\'] = 5
- LocallyLinearEmbedding
    - embedding: flickr-deepwalk_lle_svd_1574110394.0791392.emb
        - LocallyLinearEmbedding(d=128)
    
