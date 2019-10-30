import data_preprocessor.snap_community_preprocessor as snap_preprocessor


def main():
    graph_file = '../raw_data/amazon-snap/com-amazon.ungraph.txt'
    community_file = '../raw_data/amazon-snap/com-amazon.all.dedup.cmty.txt'
    top5000_community_file = '../raw_data/amazon-snap/com-amazon.top5000.cmty.txt'
    graph, node_info = snap_preprocessor.load_graph_from_txt(graph_file, community_file, top5000_community_file)
    # save the graph into edgelist
    # save the top community_info to txt, format: node_id [list of community it belongs to]
    snap_preprocessor.save_graph_to_edge_list(graph, node_info, fname='amazon-snap')


if __name__ == '__main__':
    main()

