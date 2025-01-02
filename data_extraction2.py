import csv
import numpy as np
import networkx as nx
import pickle
from collections import defaultdict

file_type = "politifact"
news_user_graph_file = f"./{file_type}/A_train.txt"
id_mapping_file = f"./{file_type}/pol_id_twitter_mapping.pkl"
news_user_mapping_file = f"./{file_type}/news_user_mapping.csv"
graph_labels_file = f"./{file_type}/graph_labels_train.npy"
news_list_file = f"./{file_type}/pol_news_list_train.txt"
output_file = f"./{file_type}/user_news_statistics_with_centrality.csv"

# Load ID mapping (local to real ID)
with open(id_mapping_file, 'rb') as f:
    id_mapping = pickle.load(f)

# Load labels from graph_labels.npy
news_labels = np.load(graph_labels_file)  # 0 = true, 1 = false

# Create a mapping of news_id to index using pol_news_list_train.txt
news_id_to_index = {}
with open(news_list_file, 'r') as news_list:
    for idx, line in enumerate(news_list):
        news_id_to_index[line.strip()] = idx

# Initialize user statistics
def default_stats():
    return {
        'true': 0, 'false': 0, 'total': 0, 'pagerank': [], 'degree': [], 'closeness': [], 'components': 0
    }
user_stats = defaultdict(default_stats)

# Create the graph G from A_train.txt with local IDs
G = nx.Graph()
with open(news_user_graph_file, 'r') as f:
    for line in f:
        local_id1, local_id2 = map(int, line.strip().split(', '))
        G.add_edge(local_id1, local_id2)

# Read news_user_mapping.csv and update user statistics
with open(news_user_mapping_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        news_id = row[0]
        user_ids = row[1].split(', ')

        # Find the index of the news_id using the mapping
        if news_id in news_id_to_index:
            news_index = news_id_to_index[news_id]
        else:
            print(f"Warning: news_id {news_id} not found in {news_list_file}")
            continue

        # Determine if the news is true or false
        news_label = news_labels[news_index]

        if news_label == 0:
            label_key = "true"
        else:
            label_key = "false"

        # Update statistics for each user
        for user_id in user_ids:
            user_stats[user_id][label_key] += 1
            user_stats[user_id]['total'] += 1

# Identify connected components and calculate centralities
connected_components = list(nx.connected_components(G))

for i, component in enumerate(connected_components):
    subgraph = G.subgraph(component)

    print(f"\rProcessing component {i+1}/{len(connected_components)}", end="")

    # Calculate centralities for the subgraph
    pagerank = nx.pagerank(subgraph)
    degree_centrality = nx.degree_centrality(subgraph)
    closeness_centrality = nx.closeness_centrality(subgraph)

    # Assign centralities to users in this component using local IDs
    for node in subgraph.nodes:
        real_id = id_mapping.get(node, None)  # Map local ID to real ID if exists
        if real_id:  # If the node corresponds to a real user
            user_stats[real_id]['components'] += 1
            user_stats[real_id]['pagerank'].append(pagerank.get(node, 0))
            user_stats[real_id]['degree'].append(degree_centrality.get(node, 0))
            user_stats[real_id]['closeness'].append(closeness_centrality.get(node, 0))

# Calculate the average centrality for each user
for user_id, stats in user_stats.items():
    num_components = stats['components']
    if num_components > 0:
        stats['pagerank'] = np.mean(stats['pagerank'])
        stats['degree'] = np.mean(stats['degree'])
        stats['closeness'] = np.mean(stats['closeness'])

# Normalize the centralities using min-max normalization
max_pagerank = max(stats['pagerank'] for stats in user_stats.values())
min_pagerank = min(stats['pagerank'] for stats in user_stats.values())

max_degree = max(stats['degree'] for stats in user_stats.values())
min_degree = min(stats['degree'] for stats in user_stats.values())

max_closeness = max(stats['closeness'] for stats in user_stats.values())
min_closeness = min(stats['closeness'] for stats in user_stats.values())

for user_id, stats in user_stats.items():
    stats['pagerank'] = (stats['pagerank'] - min_pagerank) / (max_pagerank - min_pagerank) if max_pagerank != min_pagerank else 0
    stats['degree'] = (stats['degree'] - min_degree) / (max_degree - min_degree) if max_degree != min_degree else 0
    stats['closeness'] = (stats['closeness'] - min_closeness) / (max_closeness - min_closeness) if max_closeness != min_closeness else 0

# Write the results to a CSV file
with open(output_file, 'w', newline='') as csv_output:
    writer = csv.writer(csv_output)
    writer.writerow(['User ID', 'True News', 'False News', 'Total News', 'PageRank Centrality', 'Degree Centrality', 'Closeness Centrality'])
    
    for user_id, stats in user_stats.items():
        # Check if user_id contains the string of file_type (indicating it's a news node)
        if file_type not in str(user_id):
            writer.writerow([user_id, stats['true'], stats['false'], stats['total'], stats['pagerank'], stats['degree'], stats['closeness']])

print(f"\nUser statistics with centrality measures have been written to {output_file}")

