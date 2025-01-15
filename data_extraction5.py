from typing import List
import numpy as np
import networkx as nx
import pickle
from subgraph import *
import os

# Function to save data to a CSV file
def dump_data(data, file_name):
    with open(file_name, "w") as file:
        for graph in data:
            s = ""
            for value in graph:
                s += str(value) + ", "
            s = s[:-2] + "\n"
            file.write(s)

def create_final_dataset(file_type: str, train=True):
    if train:
        file_name = f"./out/{file_type}_resized.csv"
        dataset_name = "train.txt"
        labels = "graph_labels_train.npy"
    else:
        file_name = f"./out/{file_type}_resized_test.csv"
        dataset_name = "test.txt"
        labels = "graph_labels_test.npy"

    all_data = []
    # Load the timestamp mappings
    with open(f"./{file_type}/{file_type[:3]}_id_time_mapping.pkl", 'rb') as f:
        maps_timestamps = pickle.load(f)
        
    # Create the main graph
    G = nx.Graph()

    # Read graph data from the file
    with open(f"./out/{file_type}_{dataset_name}") as f:
        for line in f:
            line = line.strip().split(', ')
            G.add_edge(int(line[0]), int(line[1]))

    # Load graph labels
    graphs_labels = np.load(f"./out/{file_type}_{labels}")
    
    subgraphs = []

    # Create connected subgraphs
    for index, cc in enumerate(nx.connected_components(G)):
        subgraphs.append(SubGraph(G.subgraph(cc), int(graphs_labels[index])))

    # Analyze subgraphs
    i = 1
    for s in subgraphs:
        
        print(f"\rAnalyzing graph: {str(i)}/{len(subgraphs)}", end="")
        i += 1
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

    dump_data(all_data, file_name)
    
    if train:
        print(f"\nINFO: Final train dataset created for '{file_type}'")
    else:
        print(f"\nINFO: Final test dataset created for '{file_type}'")


if __name__ == "__main__":
    if not os.path.exists("./out"):
        os.makedirs("./out")
        
    if not os.path.isfile("./.config") or os.path.getsize("./.config") <= 0:
        print("INFO: Empty '.config' file")
        file_type = input("Which dataset would you like to analyse (gossipcop / politifact): ")
        if file_type != "gossipcop" and file_type != "politifact":
            print("ERROR: Incorrect input")
            exit(1)
        with open("./.config", "w") as f:
            f.write(file_type)
    else:
        with open("./.config", "r") as f:
            file_type = f.read()
        if file_type != "gossipcop" and file_type != "politifact":
            print("ERROR: Incorrect syntax of file '.config'")
            exit(1)
    
    create_final_dataset(file_type, True)
    create_final_dataset(file_type, False)
