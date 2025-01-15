# GOAL OF THIS PROGRAM
# Create a file "ranking_metrics.csv" that will contain the extra features
# computed taking the data from the user ranking:
# AvgRanking,ReliableUsersPercentage,MinUserScore,MaxuserScore,WeightedScore

import pandas as pd
import networkx as nx
import pickle
import os

# Reading files

def create_ranking_metrics(file_type, train=True):
    ranking_file = f"./out/{file_type}_USER_RANKING.csv"
    mapping_file = f"./{file_type}/{file_type[:3]}_id_twitter_mapping.pkl"
    
    if train:
        graph_file = f"./out/{file_type}_train.txt"
        output_file = f"./out/{file_type}_ranking_metrics.csv"
    else:
        graph_file = f"./out/{file_type}_test.txt"
        output_file = f"./out/{file_type}_ranking_metrics_test.csv"

    # Load the user ranking file
    user_ranking = pd.read_csv(ranking_file)

    # Load the mapping between local and real IDs
    with open(mapping_file, 'rb') as f:
        id_mapping = pickle.load(f)

    local_to_real_id = id_mapping

    # Build the graph
    graph = nx.read_edgelist(graph_file, delimiter=",", nodetype=int)

    # Compute metrics for each connected component
    results = []
    for component in nx.connected_components(graph):
        subgraph = graph.subgraph(component)

        # Map local IDs in the component to real IDs
        user_ids = []
        root = ""

        for node in component: 
            if node in local_to_real_id:
                if file_type not in local_to_real_id[node]:
                    user_ids.append(int(local_to_real_id[node]))
                else:
                    root = local_to_real_id[node]
        
        # Filter users that are in the filtered ranking
        valid_users = user_ranking[user_ranking['UserID'].isin(user_ids)]

        # Extract centrality values from the user ranking data
        pagerank_values = valid_users['PageRankCentrality'].values
        degree_values = valid_users['DegreeCentrality'].values
        closeness_values = valid_users['ClosenessCentrality'].values
        
        # Calculate the average centrality values for the connected component
        avg_pagerank = pagerank_values.mean() if len(pagerank_values) > 0 else 0
        avg_degree = degree_values.mean() if len(degree_values) > 0 else 0
        avg_closeness = closeness_values.mean() if len(closeness_values) > 0 else 0
        
        # Calculate other metrics (considering reliable users as users with score >1.5)
        if valid_users.empty:
            avg_ranking = user_ranking['UserScore'].mean()
            reliable_users_percentage = len(user_ranking[user_ranking['UserScore'] >= 1.5 ]) / len(user_ranking)
            min_user_score = user_ranking['UserScore'].min()
            max_user_score = user_ranking['UserScore'].max()
            weighted_score = (user_ranking['UserScore'] / user_ranking['TotalNews']).sum()
        else:
            avg_ranking = valid_users['UserScore'].mean()
            reliable_users_percentage = len(valid_users[valid_users['UserScore'] >= 1.5 ]) / len(valid_users)
            min_user_score = valid_users['UserScore'].min()
            max_user_score = valid_users['UserScore'].max()
            weighted_score = (valid_users['UserScore'] / valid_users['TotalNews']).sum()

        # Append the results for the current component
        results.append({
            "Root": root,
            "AvgRanking": avg_ranking,
            "ReliableUsersPercentage": reliable_users_percentage,
            "MinUserScore": min_user_score,
            "MaxuserScore": max_user_score,
            "WeightedScore": weighted_score,
            "AvgPageRank": avg_pagerank,
            "AvgDegreeCentrality": avg_degree,
            "AvgClosenessCentrality": avg_closeness
        })

    # Save the results to a CSV file
    pd.DataFrame(results).to_csv(output_file, index=False, header=True)

    if train:
        print(f"INFO: Train metrics correctly saved for '{file_type}'")
    else:
        print(f"INFO: Test metrics correctly saved for '{file_type}'")

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
    
    create_ranking_metrics(file_type, True)
    create_ranking_metrics(file_type, False)
