# LEARNING FROM NETWORKS PROJECT
# Predicting news truthfulness through graph-based retweet patterns
Brocheton Damien 2133034, Martinez Zoren 2123873, Baggio Davide 2122547

[Full Project Report](https://github.com/davidebaggio/LFN_proj/blob/master/report/final_report.pdf)  

### 1. Install required libraries

```bash
  pip install -r requirements.txt
```
or manually install the libraries that are needed
```bash
  pip install matplotlib networkx numpy pandas scikit_learn tensorflow
```

### 2. Folder structure

The folder is composed as follows:
- Root folder contains the scripts needed for data analysis and predictors.
- gossipcop and politifact folders contains the dataset.
- report folder contains the reports of the project
- out folder will contain all the features files as well as the processed dataset (rankings etc.)

### 3. USAGE

First of all we need to load the dataset and then extract all the needed features. At the end we can run the probabilistic predictor as well as the ML models to infer the labels of the news.

#### 3.1 Automated script (data_analysis.py)

To load the dataset and compute the data extraction (features) run:
```bash
  python3 data_analysis.py
```
The code will generate a **.config** file. This will be called in the entire code base and contains the name of the dataset that is being used. You can modify the file whenever you want as long as it contains the name of one of the datasets (gossipcop/politifact). Once you change the **.config** file make sure that all the processed files are present for the current dataset in the *out* folder. Otherwise re-run the **data_analysis.py** script.

This script extracts all the needed features as well as computing the probabilistic predictor on the current data.

Once it has terminated you can run the ML models for training and prediction of real/fake news.

#### 3.2 Manual scripts (data_extraction.py)
If you want to have more control over the code base instead of running the **data_analysis.py** script you can always run manually different scripts and obtaining the same result. In order, run:

```bash
  python3 dataset_loader.py

  python3 data_extraction0.py
  python3 data_extraction1.py
  python3 data_extraction2.py
  python3 data_extraction3.py
  python3 data_extraction4.py
  python3 data_extraction5.py

  python3 probalistic_pred.py
```

#### 3.3 ML models

<<<<<<< HEAD
##### Models using only graph-based features
=======
To run the ML models on the retrieved data, simply run these commands:
>>>>>>> 5a4d243a9a5cb7deba25799cc19ba31479e74b77

- **Feed Forward Neural Networks (FFNN)**: 
  ```bash
  python3 ffnn_pred.py
  ```

- **Random Forest (RF)**: 
  ```bash
  python3 rf_pred.py
  ```

- **Support Vector Machine (SVM) without ranking**: 
  ```bash
  python3 svm_pred_without_ranking.py 
  ```

<<<<<<< HEAD
##### Models using aggregated features with "User Reliability Ranking" data

- **Support Vector Machine (SVM) with ranking**: 
  ```bash
  python3 svm_pred_with_ranking.py 
  ```
=======
  ```bash
  python3 svm_pred_with_ranking.py 
  ```


>>>>>>> 5a4d243a9a5cb7deba25799cc19ba31479e74b77
