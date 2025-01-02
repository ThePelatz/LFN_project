# GOAL OF THIS PROGRAM: 
# By using the statistics in user_news_statistics_with_centrality.csv
# create a csv file (USER_RANKING.csv) where each user is ranked 
# by reliability calculated using "UserScore" (see the formula below)

import csv

file_type = "gossipcop"
input_file = f"./{file_type}/user_news_statistics_with_centrality.csv"
output_file = f"./{file_type}/USER_RANKING.csv"

alpha = 0.2  # Penalization factor for low total tweets
single_news_true_score = 1.0  # Score for users with a single true news
single_news_false_score = 0.0  # Score for users with a single false news

# Read user data from the input CSV file
users_data = []
with open(input_file, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        user = {
            "UserID": int(row["User ID"]),
            "TrueNews": int(row["True News"]),
            "FalseNews": int(row["False News"]),
            "TotalNews": int(row["Total News"]),
            "PageRankCentrality": float(row["PageRank Centrality"]),
            "DegreeCentrality": float(row["Degree Centrality"]),
            "ClosenessCentrality": float(row["Closeness Centrality"])
        }
        users_data.append(user)

# Normalize values
page_ranks = [user["PageRankCentrality"] for user in users_data]
degree_centralities = [user["DegreeCentrality"] for user in users_data]
closeness_centralities = [user["ClosenessCentrality"] for user in users_data]

min_pr = min(page_ranks)
max_pr = max(page_ranks)

min_dc = min(degree_centralities)
max_dc = max(degree_centralities)

min_cc = min(closeness_centralities)
max_cc = max(closeness_centralities)

# Calculate the score for each user
for user in users_data:
    
    # Handle cases with a single news
    if user["TotalNews"] == 1:
        if user["TrueNews"] == 1:
            user["UserScore"] = single_news_true_score
        elif user["FalseNews"] == 1:
            user["UserScore"] = single_news_false_score
        continue

    # Calculate FR (False Retweets)
    FR = user["FalseNews"] / user["TotalNews"] if user["TotalNews"] > 0 else 0
    
    # Calculate TFR (True to False Ratio)
    TFR = user["TrueNews"] / user["FalseNews"] if user["FalseNews"] > 0 else user["TrueNews"]
    
    # Calculate NP (PageRank)
    NP = (user["PageRankCentrality"] - min_pr) / (max_pr - min_pr) if max_pr != min_pr else 0
    
    # Calculate DC (Degree Centrality)
    DC = (user["DegreeCentrality"] - min_dc) / (max_dc - min_dc) if max_dc != min_dc else 0
    
    # Calculate CC (Closeness Centrality)
    CC = (user["ClosenessCentrality"] - min_cc) / (max_cc - min_cc) if max_cc != min_cc else 0
    
    # Penalize users with low TotalNews
    penalty = alpha / user["TotalNews"] if user["TotalNews"] > 0 else 0
    
    # Calculate UserScore
    user["UserScore"] = 0.4 * FR + 0.3 * TFR + 0.1 * NP + 0.1 * DC + 0.1 * CC

# Sort users based on their score (from most reliable to least reliable)
sorted_users = sorted(users_data, key=lambda x: x["UserScore"], reverse=True)

# Write the sorted results to the output file
with open(output_file, 'w', newline='') as csvfile:
    fieldnames = ["UserID", "TrueNews", "FalseNews", "TotalNews", "PageRankCentrality", "DegreeCentrality", "ClosenessCentrality", "UserScore"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for user in sorted_users:
        writer.writerow(user)

print("The CSV file with sorted user scores has been created.")
