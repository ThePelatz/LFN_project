import os

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
        

from dataset_loader import *

load_dataset(file_type)

from data_extraction0 import *

split_graph_dataset(file_type)

from data_extraction1 import *

user_mapping(file_type)

from data_extraction2 import *

user_news_stats(file_type)

from data_extraction3 import *

user_ranking(file_type)

from data_extraction4 import *

create_ranking_metrics(file_type, True)
create_ranking_metrics(file_type, False)

from data_extraction5 import *

create_final_dataset(file_type, True)
create_final_dataset(file_type, False)

from probalistic_pred import *

print("\nINFO: Running the probabilistic Predictor\n")
prob_predictor(file_type)