import numpy as np
import sys
import networkx as nx

# noinspection PyBroadException
try:
    # noinspection PyPep8Naming
    import cPickle as pickle
except Exception:
    import pickle


def load_sbm_from_pickle(data_path, label_path=None):
    # Load the data
    print('\nReading dataset ...')
    G = nx.read_gpickle(data_path)

    label_info = None
    if label_path is not None:
        try:
            label_info = pickle.load(open(label_path, 'rb')).toarray()
        except UnicodeDecodeError:
            label_info = pickle.load(open(label_path, 'rb'), encoding='latin1').toarray()

    node_info = {}
    n_nodes = G.number_of_nodes()
    print('Num of nodes: ', n_nodes)
    print('Num of edges: ', G.number_of_edges())

    node_labels = [[] for _ in range(0, n_nodes)]
    if label_info is not None:
        n_labels = label_info.shape[1]
        print('Num of labels: ', n_labels)

        cnt_node_with_label = 0
        for i_node in range(0, n_nodes):
            # show the processing progress
            sys.stdout.write('\r')
            sys.stdout.write('Processing node %d' % (i_node, ))
            sys.stdout.flush()

            has_label = False
            for l in range(0, n_labels):
                if int(label_info[i_node][l]) == 1:
                    node_labels[i_node].append(l)
                    has_label = True

            if has_label:
                cnt_node_with_label += 1

        print('\nProportion of nodes with label info: %.2f%%' % (float(cnt_node_with_label)/n_nodes*100.0, ))

    node_info['labels'] = node_labels
    return G, node_info


def save_graph_to_edge_list(graph, node_labels, fname='sbm'):
    # save the graph, format: node_from, node_to, weight
    print('\nWriting graph into edgelist file ...')
    edge_list_fname = fname + '.edgelist'
    with open(edge_list_fname, 'w') as f:
        for i, j, w in graph.edges(data='weight', default=1.0):
            f.write('%d %d %f\n' % (i, j, float(w)))

    # save the node-labels to txt, format: node_id labels
    print('Writing node_labels into txt file ...')
    node_labels_fname = fname + '-labels.txt'
    with open(node_labels_fname, 'w') as f:

        for i_node in range(0, len(node_labels)):
            line = str(i_node)
            if len(node_labels[i_node]) == 0:
                continue

            for l in range(0, len(node_labels[i_node])):
                line += ' '
                line += str(node_labels[i_node][l])
            f.write('%s\n' % (line, ))


def main():
    data_sbm_file = '../data/sbm/sbm_graph.gpickle'
    label_file = '../data/sbm/sbm_node_labels.pickle'
    sbm_graph, sbm_node_info = load_sbm_from_pickle(data_sbm_file, label_file)
    save_graph_to_edge_list(sbm_graph, sbm_node_info['labels'])


if __name__ == '__main__':
    main()
