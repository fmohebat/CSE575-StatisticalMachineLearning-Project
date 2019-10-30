# Statistical Machine Learning

## Problem Statement
In the multi-label classification for graph nodes problem, we are given a graph G(V,E,W) with a subset of nodes V<sub>l</sub> ⊂ V labeled, where V is the set of nodes in the graph (possibly augmented with other features), and V<sub>u</sub> = V/V<sub>l</sub> is the set of unlabeled nodes. Here W is the weight matrix, and E is the set of edges. Let Y be the set of m possible labels, and Y<sub>l</sub> = {y<sub>1</sub>, y<sub>2</sub>, ..., y<sub>l</sub>} be the initial labels on nodes in the set V<sub>l</sub>. The task is to infer labels Y on all nodes V of the graph.

## Solution Idea
By utilizing the random sampling method established by node2vec and modifying the choice of embedd in gand classiﬁcation models, we expect to optimize the DLRW methods for the case of multi-label node classiﬁcation on large graphs. The improvement will be determined by improved node-classiﬁcation accuracy as measured by Macro/Micro-F1 scores on datasets across a variety of ﬁelds while maintaining the tractability of the algorithm for large graph datasets.

## Used Libraries

## Implementation Pipeline

## References
