import numpy as np
import pandas as pd
import pprint as pp
import networkx as nx
import matplotlib.pyplot as plt
import pickle


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

    # Compute the difference between the point and the 2 averages
    diff_r = abs(avg_r - value)    #Difference between avg of real sub-graph and the new value
    diff_f = abs(avg_f - value)    #Difference between avg of fake sub-graph and the new value
    inside = (value<avg_r and value>avg_f) or (value<avg_f and value>avg_r) #Is the value is between the 2 avg ?

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
file_type = 'gossipcop'

# ----- LOADING THE DATA
data = pd.read_csv(f"{file_type}.csv", sep=',', header=None).values
print(data)


# ----- COMPUTE THE AVERAGES
real_metrics = np.zeros(6)
fake_metrics = np.zeros(6)
num_real = 0
num_fake = 0

for i in range(len(data)):
    if data[i][6] == 0:  # Real graph
        real_metrics += data[i][:6]
        num_real += 1
    else:  # Fake graph
        fake_metrics += data[i][:6]
        num_fake += 1

real_metrics /= num_real
fake_metrics /= num_fake


# ----- PREDICT ON ALL THE GRAPHS
good_predi = 0
bad_predi = 0
no_predi = 0

confidence_good = np.zeros(12)
confidence_bad  = np.zeros(12)

for i in range(len(data)):
    # Compute the score of the graph
    scores = [
        confidence_scoring(data[i][j], real_metrics[j], fake_metrics[j])
        for j in range(6)
    ]
    final_score = sum(scores)


    if final_score > 0: #Guess -> Real
        if data[i][6] == 0:
            good_predi += 1
            confidence_good[final_score-1] += 1
        else:
            bad_predi += 1
            confidence_bad[final_score-1] += 1
  
    elif final_score < 0: #Guess -> Fake
        if data[i][6] == 1:
            good_predi += 1
            confidence_good[abs(final_score)-1] += 1
        else:
            bad_predi += 1
            confidence_bad[abs(final_score)-1] += 1

    else: #Can't guess
        no_predi += 1
        

# ----- SHOW THE RESULTS
print("\n\n---------- Results ----------")
print("File            : " + file_type)
print("Number of graph : " + str(len(data)))

print("\n> All results (prediction and no prediction)")
print(f"Good prediction : {100 * good_predi / len(data):.1f}% ({good_predi})")
print(f"Bad prediction  : {100 * bad_predi / len(data):.1f}% ({bad_predi})")
print(f"No prediction   : {100 * no_predi / len(data):.1f}% ({no_predi})")

print("\n> Prediction results (prediction only)")
print(f"Good prediction : {100 * good_predi / (good_predi + bad_predi):.1f}% ({good_predi})")
print(f"Bad prediction  : {100 * bad_predi / (good_predi + bad_predi):.1f}% ({bad_predi})")


# Results for every confidence score
confidence_percentages = confidence_good / (confidence_good + confidence_bad)

print("\n Number by confidence")
print("\nConfidence score|    Good prediction    | Bad prediction")
print("----------------------------------------------------------------")
for i in range(len(confidence_percentages)):
    print(f" \t{i+1} \t| \t{int(confidence_good[i])} \t\t|\t {int(confidence_bad[i])}")


# Graph of the % of good prediction according to the confidence
average = np.full(12, 0.5)
x = range(len(confidence_percentages))

plt.plot(x, confidence_percentages, label="% of good guess", color="orange") 
plt.scatter(x, confidence_percentages, label="% of good guess", color="orange")
plt.plot(x, average, label="50%", linestyle="--")
plt.xlabel("Confidence score")
plt.xlim(1, 12)
plt.ylabel("% of good prediction")
plt.ylim(0, 1)
plt.show()


















