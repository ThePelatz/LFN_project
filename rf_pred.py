import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV

file_type = "politifact"

# Load the data
df = pd.read_csv(f"{file_type}.csv", sep=',', header=None)
Data = df.values
m = Data.shape[0]
Y = Data[:, -1] 
X = Data[:, :-1]

print("Total number of samples:", m)

# Encode labels if they are categorical
if not np.issubdtype(Y.dtype, np.number):
    le = LabelEncoder()
    Y = le.fit_transform(Y)

# Split data into training, validation, and testing (60%, 20%, 20%)
Xtrain_val, Xtest, Ytrain_val, Ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain_val, Ytrain_val, test_size=0.25, random_state=42)

print("Training size: ", Xtrain.shape[0])
print("Validation size: ", Xval.shape[0])
print("Training and validation size", Xtrain_val.shape[0])
print("Test size", Xtest.shape[0])

# Standardize the data
scaler = StandardScaler()
Xtrain_val_scaled = scaler.fit_transform(Xtrain_val)
Xtest_scaled = scaler.transform(Xtest)
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xval_scaled = scaler.transform(Xval)

# Define hyperparameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize Random Forest
rf = RandomForestClassifier(random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
print("\nPerforming hyperparameter search...")
grid_search.fit(Xtrain_scaled, Ytrain)

# Best hyperparameters
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

print("\nBest hyperparameters:", best_params)

# Evaluate on validation data
val_predictions = best_rf.predict(Xval_scaled)
val_accuracy = accuracy_score(Yval, val_predictions)
print("Validation accuracy:", val_accuracy)

# Train final model on train+val
print("\nTRAINING FINAL MODEL")
final_rf = RandomForestClassifier(**best_params, random_state=42)
final_rf.fit(Xtrain_val_scaled, Ytrain_val)

# Evaluate on test data
test_predictions = final_rf.predict(Xtest_scaled)
test_accuracy = accuracy_score(Ytest, test_predictions)
print("\nTest accuracy:", test_accuracy)

# Classification report
print("\nClassification Report:")
report = classification_report(Ytest, test_predictions)
print(report)
with open(f'./report/rf_{file_type}_report.txt', 'w') as f:
    f.write("RF Classification Report:\n")
    f.write(f"Best hyperparameters: {best_params}\n")
    f.write(report)
