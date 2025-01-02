import csv
import numpy as np
from collections import defaultdict


file_type = "politifact"
# File paths
news_user_mapping_file = f"./{file_type}/news_user_mapping.csv"
graph_labels_file = f"./{file_type}/graph_labels.npy"
output_file = f"./{file_type}/user_news_statistics.csv"

# Load labels from graph_labels.npy
news_labels = np.load(graph_labels_file)  # 0 = true, 1 = false

# Dictionary to store user statistics
user_stats = defaultdict(lambda: {'true': 0, 'false': 0, 'total': 0})

# Read news_user_mapping.csv and update user statistics
with open(news_user_mapping_file, 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        news_id = int(row[0])
        user_ids = row[1].split(', ')
        
        # Determine if the news is true or false
        news_label = news_labels[news_id]
        label_key = 'true' if news_label == 0 else 'false'
        
        # Update statistics for each user
        for user_id in user_ids:
            user_stats[user_id][label_key] += 1
            user_stats[user_id]['total'] += 1

# Write the user statistics to a new CSV file
with open(output_file, 'w', newline='') as csv_output:
    writer = csv.writer(csv_output)
    writer.writerow(['User ID', 'True News', 'False News', 'Total News'])
    
    for user_id, stats in user_stats.items():
        writer.writerow([user_id, stats['true'], stats['false'], stats['total']])

print(f"User statistics have been written to {output_file}")
