# GOAL OF THE PROGRAM
# Prepare the datasets to be used for ranking the users. 
# The dataset A.txt is split into A_train.txt (the dataset that will be 
# used to create the user ranking) and A_test.txt. The split is 80/20, 
# taken randomly but maintaining the proportion of true and false news.
# The data is divided into multiple files, so during the process of creating 
# the two datasets, it is necessary to consistently modify other files(e.g. graph_labels.npy) as well.

import networkx as nx
import numpy as np
import random

def split_graph_dataset(input_file, train_file, test_file, labels_file, node_ids_file, news_list_file, train_labels_file, test_labels_file, train_node_ids_file, test_node_ids_file, train_news_list_file, test_news_list_file, split_ratio=0.8):

    # Read the dataset
    edges = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Ignore empty lines
                u, v = map(int, line.split(', '))
                edges.append((u, v))
    
    # Create a graph using NetworkX
    G = nx.Graph()
    G.add_edges_from(edges)
    
    # Get connected components (i.e., the trees)
    components = list(nx.connected_components(G))
    
    # Load labels, node IDs, and news list
    labels = np.load(labels_file)
    node_ids = np.load(node_ids_file)
    with open(news_list_file, 'r') as f:
        news_list = [line.strip() for line in f]
    
    # Create a list of (component, label, node_id, news)
    data = [(comp, labels[idx], node_ids[idx], news_list[idx]) for idx, comp in enumerate(components)]
    
    # Split components into "true" (label 0) and "false" (label 1) news
    true_news = [item for item in data if item[1] == 0]
    false_news = [item for item in data if item[1] == 1]
    
    # Shuffle the true and false news separately to ensure random selection
    random.shuffle(true_news)
    random.shuffle(false_news)
    
    # Calculate the number of train/test samples for each label
    train_true_count = int(len(true_news) * split_ratio)
    train_false_count = int(len(false_news) * split_ratio)

    # Split the components into train and test
    train_true = true_news[:train_true_count]
    test_true = true_news[train_true_count:]
    
    train_false = false_news[:train_false_count]
    test_false = false_news[train_false_count:]

    # Combine train and test sets
    train_data = train_true + train_false
    test_data = test_true + test_false

    # Helper function to extract edges for components and format them
    def format_edges(components):
        formatted_edges = []
        for comp in components:
            subgraph = nx.subgraph(G, comp[0])
            for u, v in subgraph.edges():
                formatted_edges.append(f"{u}, {v}")
        return formatted_edges

    # Format edges for train and test sets
    train_edges = format_edges(train_data)
    test_edges = format_edges(test_data)
    
    # Write training edges to the train file
    with open(train_file, 'w') as f:
        f.write("\n".join(train_edges)) 
    
    # Write testing edges to the test file
    with open(test_file, 'w') as f:
        f.write("\n".join(test_edges))

    # Extract labels, node IDs, and news list for train and test sets
    train_labels = np.array([item[1] for item in train_data])
    test_labels = np.array([item[1] for item in test_data])
    
    train_node_ids = np.array([item[2] for item in train_data])
    test_node_ids = np.array([item[2] for item in test_data])
    
    train_news = [item[3] for item in train_data]
    test_news = [item[3] for item in test_data]

    # Save train and test labels
    np.save(train_labels_file, train_labels)
    np.save(test_labels_file, test_labels)

    # Save train and test node IDs
    np.save(train_node_ids_file, train_node_ids)

    # Save train and test news list
    with open(train_news_list_file, 'w') as f:
        f.write("\n".join(train_news))
    
    with open(test_news_list_file, 'w') as f:
        f.write("\n".join(test_news))

# Paths
file_type = "politifact"
input_file = f"./{file_type}/A.txt"
train_file = f"./{file_type}/A_train.txt"
test_file = f"./{file_type}/A_test.txt"
labels_file = f"./{file_type}/graph_labels.npy"
node_ids_file = f"./{file_type}/node_graph_id.npy"
news_list_file = f"./{file_type}/{file_type[:3]}_news_list.txt"
train_labels_file = f"./{file_type}/graph_labels_train.npy"
test_labels_file = f"./{file_type}/graph_labels_test.npy"
train_node_ids_file = f"./{file_type}/node_graph_id_train.npy"
test_node_ids_file = f"./{file_type}/node_graph_id_test.npy"
train_news_list_file = f"./{file_type}/{file_type[:3]}_news_list_train.txt"
test_news_list_file = f"./{file_type}/{file_type[:3]}_news_list_test.txt"

split_graph_dataset(input_file, train_file, test_file, labels_file, node_ids_file, news_list_file, train_labels_file, test_labels_file, train_node_ids_file, test_node_ids_file, train_news_list_file, test_news_list_file)