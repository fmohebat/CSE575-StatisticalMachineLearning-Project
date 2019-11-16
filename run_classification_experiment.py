import sys
import argparse
from evaluation import evaluation_utils
import pandas as pd

from gem.evaluation.evaluate_node_classification import TopKRanker

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

def run_classification_experiment(emb_file,label_file):
    seed = 0
    print(emb_file, label_file)

    # Pre-processing of emb_file and label_file
    patterns_sorted = evaluation_utils.emb_file_to_df(emb_file)
    encoded_sorted = evaluation_utils.k_encode_label_file(label_file)

    # Creating train test split 80% 20%
    X_train, X_test, y_train, y_test = train_test_split(patterns_sorted.values.T, encoded_sorted.values.T, 
                                                    test_size=.20,random_state=seed)
    
    clf_list = [
        DecisionTreeClassifier(random_state=seed),
        KNeighborsClassifier(n_neighbors=3),
        MLPClassifier(random_state=seed, max_iter=500),
        RandomForestClassifier(random_state=seed),
        # TopKRanker(LogisticRegression(solver='liblinear')), # OneVsRest
        # OneVsRestClassifier(LogisticRegression(solver='lbfgs')), # Terrible OneVsRest
        TopKRanker(LogisticRegression(solver='lbfgs')) # OneVsRest
        ] 
    
    for clf in clf_list:
        print(clf.__class__.__name__, "\n")
        evaluate_clf(clf, X_train, X_test, y_train, y_test)
        print("\n\n")
    
    
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
    print("Macro f1: ", f1_score(y_test, prediction, average='macro'))
    print("Micro f1: ", f1_score(y_test, prediction, average='micro'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("emb_file", help="path to emb file to use for classification, ex. output/karate/karate_node2vec.emb")
    parser.add_argument("label_file", help="path to labels corresponding to embedding file, ex. labels/blog3.labels")
    
    run_classification_experiment(**vars(parser.parse_args()))