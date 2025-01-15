# GOAL OF THIS PROGRAM: 
# Create a csv file (news_user_mapping.csv) with the ID news followed by the
# user IDs that retweeted that news

import numpy as np
import networkx as nx
import pickle
import csv
from subgraph import *

file_type = 'gossipcop'

G = nx.Graph()

with open(f"./{file_type}/A_train.txt") as f:
    for line in f:
        line = line.strip().split(', ')
        G.add_edge(int(line[0]), int(line[1]))


graphs_labels = np.load(f"./out/{file_type}_graph_labels_train.npy")


with open(f"./{file_type}/{file_type[:3]}_id_twitter_mapping.pkl", "rb") as f:
    user_mapping = pickle.load(f)


#news_mapping = np.load(f"./{file_type}/node_graph_id_train.npy")

subgraphs = []


for index, cc in enumerate(nx.connected_components(G)):
    subgraphs.append(SubGraph(G.subgraph(cc), int(graphs_labels[index])))


with open(f"./{file_type}/news_user_mapping.csv", mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    #csv_writer.writerow(["News_ID", "User_IDs"])

    for sg in subgraphs:
        # Get the nodes of the subgraph
        nodes = list(sg.graph.nodes())
        news_id = None
        user_ids = []

        # Find the root (assuming it is the node with a mapping containing `file_type`)
        for node in nodes:
            if file_type in user_mapping[node]:
                news_id = user_mapping[node]  # Identify the root's news ID
                root = node  # Identify the root node
                break

        if news_id is not None:  # If the root was found
            # Add all other connected nodes except the root
            for node in nodes:
                if node != root:  # Exclude the root
                    user_ids.append(user_mapping[node])

        # First field is the NEWS ID, the secondi is the User ID
        #if news_id is not None:
        csv_writer.writerow([news_id, ", ".join(map(str, user_ids))])

print("CSV file successfully saved with news and user mappings.")
