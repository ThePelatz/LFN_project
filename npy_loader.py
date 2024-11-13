import numpy as np
import pprint as pp
import networkx as nx
import pickle

file_type = 'gossicop'

with open(f"./{file_type}/{file_type[:3]}_id_time_mapping.pkl", 'rb') as f:
    maps_timestamps = pickle.load(f)

class SubGraph:
    def __init__(self, graph: nx.Graph, info: int):
        self.graph = graph
        self.info = info

G = nx.Graph()

with open(f"./{file_type}/A.txt") as f:
    for line in f:
        line = line.strip().split(', ')
        G.add_edge(int(line[0]), int(line[1]))

graphs_labels = np.load(f"./{file_type}/graph_labels.npy")
subgraphs = []
#print(nx.number_connected_components(G))
for index, cc in enumerate(nx.connected_components(G)):
    subgraphs.append(SubGraph(G.subgraph(cc), int(graphs_labels[index])))
    #print(cc)

#print(len(subgraphs))

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

for s in subgraphs:
    print("Analyzing graph: " + str(i+j))
    #calculating the std of timestamps
    timestamps = []
    for node in s.graph.nodes:
        if node not in maps_timestamps or maps_timestamps[node] == "":
            continue
        timestamps.append(maps_timestamps[node])
    std = np.std(np.array(timestamps))
    
    #calculating the diameter
    d = nx.diameter(s.graph)
    #calculating the max degree
    _, neighbors = max(s.graph.degree, key=lambda x: x[1])
    #calculating degree centrality
    dc = np.mean(list(nx.degree_centrality(s.graph).values()))
    #calculating closeness centrality
    cc = np.mean(list(nx.closeness_centrality(s.graph).values()))
    #calculating pagerank
    pr = np.mean(list(nx.pagerank(s.graph).values()))
    if s.info == 0:
        total_r += d
        max_degree_r += neighbors
        std_r += std
        degree_centrality_r += dc
        closeness_centrality_r += cc
        pagerank_r += pr
        i += 1
    else:
        total_f += d
        max_degree_f += neighbors
        std_f += std
        degree_centrality_f += dc
        closeness_centrality_f += cc
        pagerank_f += pr
        j += 1

print("Average diameter of real graphs: " + str(total_r / i))
print("Average diameter of fake graphs: " + str(total_f / j))

print()

print("Average max degree of real graphs: " + str(max_degree_r / i))
print("Average max degree of fake graphs: " + str(max_degree_f / j))

print()

print("Average std of timestamps of real graphs: " + str(std_r / i))
print("Average std of timestamps of fake graphs: " + str(std_f / j))

print()

print("Average degree centrality of real graphs: " + str(degree_centrality_r / i))
print("Average degree centrality of fake graphs: " + str(degree_centrality_f / j))

print()

print("Average closeness centrality of real graphs: " + str(closeness_centrality_r / i))
print("Average closeness centrality of fake graphs: " + str(closeness_centrality_f / j))

print()

print("Average pagerank of real graphs: " + str(pagerank_r / i))
print("Average pagerank of fake graphs: " + str(pagerank_f / j))