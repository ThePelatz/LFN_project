import os

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
        

from dataset_loader import *

#dataset, timestamps = load_dataset(file_type)
#dump_data(analyse_dataset(dataset, timestamps), file_type)

from probalistic_pred import *

#prob_predictor(file_type)

from data_extraction0 import *

split_graph_dataset(file_type)