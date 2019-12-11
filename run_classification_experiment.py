import sys, csv
import argparse
from evaluation import evaluation_utils
# import pandas as pd

from gem.evaluation.evaluate_node_classification import TopKRanker

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def run_classification_experiment(emb_file, patterns_sorted, encoded_sorted, t_size=0.2):
    seed = 0

    # Creating train test split 80% 20%
    X_train, X_test, y_train, y_test = train_test_split(patterns_sorted.values.T, encoded_sorted.values.T, 
                                                    test_size=t_size,random_state=seed)
    
    clf_list = [
        DecisionTreeClassifier(random_state=seed),
        KNeighborsClassifier(n_neighbors=3),
        MLPClassifier(random_state=seed, max_iter=500),
        RandomForestClassifier(random_state=seed),
        # TopKRanker(LogisticRegression(solver='liblinear')), # OneVsRest
        # OneVsRestClassifier(LogisticRegression(solver='lbfgs')), # Terrible OneVsRest
        TopKRanker(LogisticRegression(solver='lbfgs')) # OneVsRest
    ] 
    
    l = []
    for clf in clf_list:
        print(clf.__class__.__name__, "\n")
        (macro_f1, micro_f1) = evaluate_clf(clf, X_train, X_test, y_train, y_test)
        l.append((emb_file.split('/')[-1],clf.__class__.__name__,t_size,macro_f1, micro_f1))
        print("\n\n")
    return l
    
    
    # Writing micro/macro F1 results in a tabular form to a csv output
    
def evaluate_clf(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    prediction = None
    if type(clf) is TopKRanker: # OneVsRest
        top_k_list = list(y_test.sum(axis=1))
        prediction = clf.predict(X_test, top_k_list)
    else:
        prediction = clf.predict(X_test)
    if prediction is None:
        print("run_classification_experiment Error - prediction was not run.")
        return
    macro_f1 = f1_score(y_test, prediction, average='macro')
    micro_f1 = f1_score(y_test, prediction, average='micro')
    print("Macro f1: ", macro_f1)
    print("Micro f1: ", micro_f1)
    return (macro_f1, micro_f1)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("emb_file", help="path to emb file to use for classification, ex. output/karate/karate_node2vec.emb")
    # parser.add_argument("label_file", help="path to labels corresponding to embedding file, ex. labels/blog3.labels")

    # run_classification_experiment(**vars(parser.parse_args()))

    # emb_files = [
    # "blog-catalog_FastText-Embedding.emb",
    # "blog-catalog_Node2Vec-Embedding.emb",
    # "blog-catalog_lap_eigmap_svd.emb",
    # "blog-catalog_GloVe-Embedding.emb",
    # "blog-catalog_hope_gsvd.emb"]

    emb_files = [
    "flickr-deepwalk_CBOW-Embedding_1574113172.4532826.emb",
    "flickr-deepwalk_FastText-Embedding_1574109675.4526346.emb",
    "flickr-deepwalk_GloVe-Embedding_1574115923.87.emb",
    "flickr-deepwalk_Node2Vec-Embedding_1574109430.8421443.emb",
    "flickr-deepwalk_lle_svd_1574113282.1872377.emb"
    ]

    emb_file_dir = "./output/flickr-deepwalk/"
    label_file = "./labels/flickr-deepwalk-labels.txt"

    for emb_file in emb_files:
        file = open('flickr' + emb_file + '_output.csv','w')
        csv_file = csv.writer(file)

        print(emb_file.split('/')[-1], label_file.split('/')[-1])

        # Pre-processing of emb_file and label_file
        patterns_sorted = evaluation_utils.emb_file_to_df(emb_file_dir+emb_file)
        encoded_sorted = evaluation_utils.k_encode_label_file(label_file)
        # if 'FastText' in emb_file:
        #     for i in range(6,10): # if FastText Start at 6
        #         l = run_classification_experiment(emb_file_dir+emb_file, patterns_sorted, encoded_sorted, i/10)
        #         for row in l:
        #             csv_file.writerow(row)
        # else:
        for i in range(1,10):
            l = run_classification_experiment(emb_file_dir+emb_file, patterns_sorted, encoded_sorted, i/10)
            for row in l:
                csv_file.writerow(row)
        file.close()