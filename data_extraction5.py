from typing import List
import numpy as np
import networkx as nx
import pickle

# Variable to define the file type
file_type = "politifact"


# Function to save data to a CSV file
def dump_data(data: List[List], file_name):
    with open(file_name, "w") as file:
        for graph in data:
            s = ""
            for value in graph:
                s += str(value) + ", "
            s = s[:-2] + "\n"
            file.write(s)

def create_final_dataset(file_name, dataset_name, labels):
    all_data = []
    # Load the timestamp mappings
    with open(f"./{file_type}/{file_type[:3]}_id_time_mapping.pkl", 'rb') as f:
        maps_timestamps = pickle.load(f)

    # Class to represent subgraphs
    class SubGraph:
        def __init__(self, graph: nx.Graph, info: int):
            self.graph = graph
            self.info = info

    # Create the main graph
    G = nx.Graph()

    # Read graph data from the file
    with open(f"./{file_type}/{dataset_name}") as f:
        for line in f:
            line = line.strip().split(', ')
            G.add_edge(int(line[0]), int(line[1]))

    # Load graph labels
    graphs_labels = np.load(f"./{file_type}/{labels}")
    
    subgraphs = []

    # Create connected subgraphs
    for index, cc in enumerate(nx.connected_components(G)):
        subgraphs.append(SubGraph(G.subgraph(cc), int(graphs_labels[index])))


    # Initialize variables for calculating averages
    i = 0
    total_r = 0
    max_degree_r = 0
    std_r = 0
    degree_centrality_r = 0
    closeness_centrality_r = 0
    pagerank_r = 0

    j = 0
    total_f = 0
    max_degree_f = 0
    std_f = 0
    degree_centrality_f = 0
    closeness_centrality_f = 0
    pagerank_f = 0

    # Analyze subgraphs
    for s in subgraphs:
        print("\rAnalyzing graph: " + str(i + j), end="")
        
        # Calculate the standard deviation of timestamps
        timestamps = []
        for node in s.graph.nodes:
            if node not in maps_timestamps or maps_timestamps[node] == "":
                continue
            timestamps.append(maps_timestamps[node])
        std = np.std(np.array(timestamps))
        
        d = nx.diameter(s.graph)  # Diameter
        _, neighbors = max(s.graph.degree, key=lambda x: x[1])  # Maximum degree
        dc = np.mean(list(nx.degree_centrality(s.graph).values()))  # Degree centrality
        cc = np.mean(list(nx.closeness_centrality(s.graph).values()))  # Closeness centrality
        pr = np.mean(list(nx.pagerank(s.graph).values()))  # PageRank
        
        # Save data
        all_data.append([d, neighbors, std, dc, cc, pr, s.info])
        
        # Aggregate metrics based on graph category
        if s.info == 0:  # Real graphs
            total_r += d
            max_degree_r += neighbors
            std_r += std
            degree_centrality_r += dc
            closeness_centrality_r += cc
            pagerank_r += pr
            i += 1
        else:  # Fake graphs
            total_f += d
            max_degree_f += neighbors
            std_f += std
            degree_centrality_f += dc
            closeness_centrality_f += cc
            pagerank_f += pr
            j += 1

    # Print statistics
    print("\nAverage diameter of real graphs: " + str(total_r / i))
    print("Average diameter of fake graphs: " + str(total_f / j))

    print("\nAverage max degree of real graphs: " + str(max_degree_r / i))
    print("Average max degree of fake graphs: " + str(max_degree_f / j))

    print("\nAverage std of timestamps of real graphs: " + str(std_r / i))
    print("Average std of timestamps of fake graphs: " + str(std_f / j))

    print("\nAverage degree centrality of real graphs: " + str(degree_centrality_r / i))
    print("Average degree centrality of fake graphs: " + str(degree_centrality_f / j))

    print("\nAverage closeness centrality of real graphs: " + str(closeness_centrality_r / i))
    print("Average closeness centrality of fake graphs: " + str(closeness_centrality_f / j))

    print("\nAverage pagerank of real graphs: " + str(pagerank_r / i))
    print("Average pagerank of fake graphs: " + str(pagerank_f / j))

    # Save all data to a file

    dump_data(all_data, file_name)


file_name = f"./{file_type}/{file_type}_resized.csv"
dataset_name = "A_train.txt"
labels = "graph_labels_train.npy"

create_final_dataset(file_name, dataset_name,labels)

file_name = f"./{file_type}/{file_type}_resized_test.csv"
dataset_name = "A_test.txt"
labels = "graph_labels_test.npy"

create_final_dataset(file_name, dataset_name,labels)