import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report

file_type = "gossipcop"

# Load the data
df = pd.read_csv(f"{file_type}.csv", sep=',', header=None)
Data = df.values
m = Data.shape[0]
Y = Data[:, -1]  # Last column contains labels
X = Data[:, :-1]  # All columns except the last one

print("Total number of samples:", m)

# Split data into training, validation, and testing (60%, 20%, 20%)
Xtrain_val, Xtest, Ytrain_val, Ytest = train_test_split(X, Y, test_size=0.2,random_state=1)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain_val, Ytrain_val, test_size=0.25,random_state=1)

print("Training size: ", Xtrain.shape[0])
print("Validation size: ", Xval.shape[0])
print("Training and validation size", Xtrain_val.shape[0])
print("Test size", Xtest.shape[0])

# Standardize the data
scaler = preprocessing.StandardScaler()
Xtrain_val_scaled = scaler.fit_transform(Xtrain_val)
Xtest_scaled = scaler.transform(Xtest)
Xtrain_scaled = scaler.fit_transform(Xtrain)
Xval_scaled = scaler.transform(Xval)

# Linear SVM
print("\nLinear SVM")
model_svm = SVC(kernel="linear")
hyper_params = {"C": [0.1, 1, 10, 100, 1000]}
GS = GridSearchCV(estimator=model_svm, param_grid=hyper_params, cv=5)
GS.fit(Xtrain_scaled, Ytrain)
print("Best value for hyperparameters: ", GS.best_params_)

best_model_linear = SVC(kernel="linear", C=GS.best_params_["C"])
best_model_linear.fit(Xtrain_scaled, Ytrain)
training_score_linear = best_model_linear.score(Xtrain_scaled, Ytrain)
print("Training score: ", training_score_linear)

# Poly SVM
print("\nPoly SVM")
model_svm = SVC(kernel="poly")
hyper_params = {
    "C": [0.1, 1, 10, 100, 1000],
    "degree": [2, 3, 4]
}
GS = GridSearchCV(estimator=model_svm, param_grid=hyper_params, cv=5)
GS.fit(Xtrain_scaled, Ytrain)
print("Best value for hyperparameters: ", GS.best_params_)

best_model_poly = SVC(kernel="poly", C=GS.best_params_["C"], degree=GS.best_params_["degree"])
best_model_poly.fit(Xtrain_scaled, Ytrain)
training_score_poly = best_model_poly.score(Xtrain_scaled, Ytrain)
print("Training score: ", training_score_poly)

# RBF SVM
print("\nRBF SVM")
model_svm = SVC(kernel="rbf")
hyper_params = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [0.01]
}
GS = GridSearchCV(estimator=model_svm, param_grid=hyper_params, cv=5)
GS.fit(Xtrain_scaled, Ytrain)
print("Best value for hyperparameters: ", GS.best_params_)

best_model_rbf = SVC(kernel="rbf", C=GS.best_params_["C"], gamma=GS.best_params_["gamma"])
best_model_rbf.fit(Xtrain_scaled, Ytrain)
training_score_rbf = best_model_rbf.score(Xtrain_scaled, Ytrain)
print("Training score: ", training_score_rbf)

# Sigmoid SVM
print("\nSigmoid SVM")
model_svm = SVC(kernel="sigmoid")
hyper_params = {
    "C": [0.1, 1, 10, 100, 1000],
    "gamma": [0.01],
    "coef0": [0, 1]
}
GS = GridSearchCV(estimator=model_svm, param_grid=hyper_params, cv=5)
GS.fit(Xtrain_scaled, Ytrain)
print("Best value for hyperparameters: ", GS.best_params_)

best_model_sigm = SVC(kernel="sigmoid", 
                     C=GS.best_params_["C"],
                     gamma=GS.best_params_["gamma"],
                     coef0=GS.best_params_["coef0"])
best_model_sigm.fit(Xtrain_scaled, Ytrain)
training_score_sigm = best_model_sigm.score(Xtrain_scaled, Ytrain)
print("Training score: ", training_score_sigm)

# Validation to choose best kernel
print("\nVALIDATION TO CHOOSE SVM KERNEL")
linear_val_score = best_model_linear.score(Xval_scaled, Yval)
poly_val_score = best_model_poly.score(Xval_scaled, Yval)
rbf_val_score = best_model_rbf.score(Xval_scaled, Yval)
sigm_val_score = best_model_sigm.score(Xval_scaled, Yval)

print("Linear validation score:", linear_val_score)
print("Polynomial validation score:", poly_val_score)
print("Rbf validation score:", rbf_val_score)
print("Sigmoid validation score:", sigm_val_score)

scores = {
    "linear": linear_val_score,
    "polynomial": poly_val_score,
    "rbf": rbf_val_score,
    "sigmoid": sigm_val_score
}
best_kernel = max(scores.items(), key=lambda x: x[1])[0]
best_score = scores[best_kernel]

print("Best kernel: ", best_kernel)
print("Validation score of best kernel: ", best_score)

# Train final model
print("\nTRAINING SCORE BEST MODEL")
models = {
    "linear": best_model_linear,
    "polynomial": best_model_poly,
    "rbf": best_model_rbf,
    "sigmoid": best_model_sigm
}

best_model = models[best_kernel]
best_model.fit(Xtrain_val_scaled, Ytrain_val)
train_score_best_model = best_model.score(Xtrain_val_scaled, Ytrain_val)
print("Score of the best model on the data used to learn it: ", train_score_best_model)

# Estimate generalization error
print("\nGENERALIZATION SCORE BEST MODEL")
generalization_score_best_model = best_model.score(Xtest_scaled, Ytest)
print("Estimate of the generalization score for best SVM model: ", generalization_score_best_model)

# Print classification report for best model
y_pred = best_model.predict(Xtest_scaled)
print("\nClassification Report:")
report = classification_report(Ytest, y_pred)
print(report)
with open(f'./report/svm_{file_type}_report.txt', 'w') as f:
    f.write("SVM Classification Report:\n")
    f.write(f"Best kernel: {best_kernel}\n")
    f.write(report)
