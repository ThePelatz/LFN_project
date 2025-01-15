# LEARNING FROM NETWORKS PROJECT
# Predicting news truthfulness through graph-based retweet patterns
Brocheton Damien 2133034, Martinez Zoren 2123873, Baggio Davide 2122547

[Full Project Report](https://github.com/davidebaggio/LFN_proj/blob/master/report/final_report.pdf)  

## Instructions for Execution

### 1. Obtain Metrics with Graph-Based Features
To run the models using only graph-based features, use the following scripts:

- **Feed Forward Neural Networks (FFNN)**: 
  ```bash
  python ffnn_pred.py
  ```

- **Random Forest (RF)**: 
  ```bash
  python rf_pred.py
  ```

- **Support Vector Machine (SVM)**: 
  ```bash
  python svm_pred.py
  ```

### 2. Obtain Metrics with Aggregated Features and User Reliability Ranking
To use aggregated features with "User Reliability Ranking" data, follow these steps:

#### a) Prepare the Features
Run the following scripts in order to generate the necessary data:

```bash
python data_extraction0.py
python data_extraction1.py
python data_extraction2.py
python data_extraction3.py
python data_extraction4.py
python data_extraction5.py
```

#### b) Train and Test the Model
After preparing the features, execute the following script to train and test the model:

```bash
python svm_pred_with_ranking.py
```
