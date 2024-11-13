import pickle
import numpy as np

file_type = 'gossicop'

with open(f"./{file_type}/{file_type[:3]}_id_time_mapping.pkl", 'rb') as f:
    maps_timestamps = pickle.load(f)

graphs_labels = np.load(f"./{file_type}/graph_labels.npy")

for key, value in maps_timestamps.items():
    if key > 100:
        break
    if value == "":
        continue
    print(key, value)

print(graphs_labels[0])