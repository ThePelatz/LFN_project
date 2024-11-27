import numpy as np
import pprint as pp
import networkx as nx
import matplotlib.pyplot as plt
import pickle
import time


class SubGraph:
    def __init__(self, graph: nx.Graph, info: int):
        self.graph = graph
        self.info = info

"""
Function : This function give a score between -2 and 2 about the truthfulness of the metric 
           with a confidence factor.
Input    : value : new value to analyze
           avg_r : average of the real sub-graph
           avg_f : average of the fake sub-graph
Output   : score : the confidence score 
                    -2 in the third close to fake           (Strong guess)
                    -1 in the middle third, closer to fake  (Weak guess)
                     0 in the middle                        (No guess)
                     1 in the middle third, closer to real  (Weak guess)
                     2 in the third close to real           (Strong guess)
"""
def confidence_scoring(value, avg_r, avg_f):

    # Compute the difference
    diff_r = abs(avg_r - value)    #Difference between avg of real sub-graph and the new value
    diff_f = abs(avg_f - value)    #Difference between avg of fake sub-graph and the new value
    inside = (value<avg_r and value>avg_f) or (value<avg_f and value>avg_r) #The value is between the 2 avg

    #Closest one
    closer = 0              # 0 = Equal
    if diff_r < diff_f:
        closer = 1          # 1 = Real
    elif diff_r > diff_f:
        closer = 2          # 2 = Fake

    #Confidence score
    score = 0                                       #No guess
    if closer == 1:                             #Real
        if 2*diff_r < diff_f or not inside:
            score = 2                               #Strong guess
        else:
            score = 1                               #Weak guess
    elif closer == 2:                           #Fake
        if 2*diff_f < diff_r or not inside:
            score = -2                              #Strong guess
        else:
            score = -1                              #Weak guess

    return score
# End of confidence_scoring


# ----- PARAMETERS
file_type = 'politifact'

# ----- INIT
start_time = time.time()

with open(f"./{file_type}/{file_type[:3]}_id_time_mapping.pkl", 'rb') as f:
    maps_timestamps = pickle.load(f)

G = nx.Graph()

with open(f"./{file_type}/A.txt") as f:
    for line in f:
        line = line.strip().split(', ')
        G.add_edge(int(line[0]), int(line[1]))

graphs_labels = np.load(f"./{file_type}/graph_labels.npy")
subgraphs = [SubGraph(G.subgraph(cc), int(graphs_labels[index])) for index, cc in enumerate(nx.connected_components(G))]


num_graphs = len(subgraphs)                 # Number of graph
results_matrix = np.zeros((num_graphs, 7))  # Matrix containing the metrics of all graphs 

# ----- ANALYSE ALL THE GRAPHS
for idx, s in enumerate(subgraphs):
    print(f"Analyzing graph: {idx} ({s.info})")

    timestamps = [maps_timestamps[node] for node in s.graph.nodes if node in maps_timestamps and maps_timestamps[node] != ""]
    std = np.std(timestamps) if timestamps else 0
    
    d = nx.diameter(s.graph)
    _, neighbors = max(s.graph.degree, key=lambda x: x[1])
    dc = np.mean(list(nx.degree_centrality(s.graph).values()))
    cc = np.mean(list(nx.closeness_centrality(s.graph).values()))
    pr = np.mean(list(nx.pagerank(s.graph).values()))

    results_matrix[idx] = [d, neighbors, std, dc, cc, pr, s.info]

# ----- COMPUTE THE AVERAGES
real_metrics = np.zeros(6)
fake_metrics = np.zeros(6)
num_real = 0
num_fake = 0

for i in range(num_graphs):
    if results_matrix[i][6] == 0:  # Real graph
        real_metrics += results_matrix[i][:6]
        num_real += 1
    else:  # Fake graph
        fake_metrics += results_matrix[i][:6]
        num_fake += 1

real_metrics /= num_real
fake_metrics /= num_fake

# ----- PREDICT ON ALL THE GRAPHS
good_predi = 0
bad_predi = 0
no_predi = 0

confidence_good = np.zeros(12)
confidence_bad  = np.zeros(12)

for i in range(num_graphs):
    # Compute the score of the graph
    scores = [
        confidence_scoring(results_matrix[i][j], real_metrics[j], fake_metrics[j])
        for j in range(6)
    ]
    final_score = sum(scores)


    if final_score > 0: #Guess -> Real
        if results_matrix[i][6] == 0:
            good_predi += 1
            confidence_good[final_score-1] += 1
        else:
            bad_predi += 1
            confidence_bad[final_score-1] += 1
  
    elif final_score < 0: #Guess -> Fake
        if results_matrix[i][6] == 1:
            good_predi += 1
            confidence_good[abs(final_score)-1] += 1
        else:
            bad_predi += 1
            confidence_bad[abs(final_score)-1] += 1

    else: #Can't guess
        no_predi += 1
        

# Affichage des résultats
print("\n\n---------- Results ----------")
print("File            : " + file_type)
print("Number of graph : " + str(num_graphs))
print(f"Execution time  : {(time.time() - start_time):.4f}s")

print("\n> All results (prediction and no prediction)")
print(f"Good prediction : {100 * good_predi / num_graphs:.1f}% ({good_predi})")
print(f"Bad prediction  : {100 * bad_predi / num_graphs:.1f}% ({bad_predi})")
print(f"No prediction   : {100 * no_predi / num_graphs:.1f}% ({no_predi})")

print("\n> Prediction results (prediction only)")
print(f"Good prediction : {100 * good_predi / (good_predi + bad_predi):.1f}% ({good_predi})")
print(f"Bad prediction  : {100 * bad_predi / (good_predi + bad_predi):.1f}% ({bad_predi})")




confidence_percentages = confidence_good / (confidence_good + confidence_bad)
average = np.full(12, 0.5)
x = range(len(confidence_percentages))

plt.plot(x, confidence_percentages, label = "% of good guess") 
plt.plot(x, average, label = "50%", linestyle="--")
plt.ylim(0, 1)
plt.show()


















