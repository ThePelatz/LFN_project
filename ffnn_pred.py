import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report

file_type = "politifact"

# Load the data
df = pd.read_csv(f"{file_type}.csv", sep=',', header=None)
Data = df.values
m = Data.shape[0]
Y = Data[:, -1]  # Last column contains labels
X = Data[:, :-1]  # All columns except the last one

print("Total number of samples:", m)

# Encode labels if they are categorical
if not np.issubdtype(Y.dtype, np.number):
    le = LabelEncoder()
    Y = le.fit_transform(Y)

# Split data into training, validation, and testing (60%, 20%, 20%)
Xtrain_val, Xtest, Ytrain_val, Ytest = train_test_split(X, Y, test_size=0.2, random_state=1)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain_val, Ytrain_val, test_size=0.25, random_state=1)

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

# Define a function to create a model
def create_ffnn(input_dim, hidden_units=64, dropout_rate=0.2, learning_rate=0.001):
    model = Sequential([
        Dense(hidden_units, activation='relu', input_dim=input_dim),
        Dropout(dropout_rate),
        Dense(hidden_units, activation='relu'),
        Dropout(dropout_rate),
        Dense(len(np.unique(Y)), activation='softmax')  # Output layer with softmax for classification
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hyperparameter search
hidden_units_list = [32, 64, 128]
dropout_rates = [0.1, 0.2, 0.3]
learning_rates = [0.001, 0.01]

best_model = None
best_val_acc = 0
best_params = {}

for hidden_units in hidden_units_list:
    for dropout_rate in dropout_rates:
        for learning_rate in learning_rates:
            print(f"Training model with hidden_units={hidden_units}, dropout_rate={dropout_rate}, learning_rate={learning_rate}")
            model = create_ffnn(Xtrain_scaled.shape[1], hidden_units, dropout_rate, learning_rate)
            
            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            
            # Train the model
            history = model.fit(Xtrain_scaled, Ytrain, 
                                validation_data=(Xval_scaled, Yval),
                                epochs=50, 
                                batch_size=32, 
                                callbacks=[early_stopping],
                                verbose=0)
            
            # Evaluate on validation data
            val_loss, val_acc = model.evaluate(Xval_scaled, Yval, verbose=0)
            print(f"Validation accuracy: {val_acc}")
            
            if val_acc > best_val_acc:
                best_model = model
                best_val_acc = val_acc
                best_params = {
                    "hidden_units": hidden_units,
                    "dropout_rate": dropout_rate,
                    "learning_rate": learning_rate
                }

print("\nBest hyperparameters:", best_params)
print("Best validation accuracy:", best_val_acc)

# Final training on train+val
print("\nTRAINING FINAL MODEL")
final_model = create_ffnn(Xtrain_val_scaled.shape[1], 
                          hidden_units=best_params["hidden_units"], 
                          dropout_rate=best_params["dropout_rate"], 
                          learning_rate=best_params["learning_rate"])

final_model.fit(Xtrain_val_scaled, Ytrain_val, 
                epochs=50, 
                batch_size=32, 
                verbose=1)

# Evaluate on test data
print("\nGENERALIZATION SCORE BEST MODEL")
test_loss, test_acc = final_model.evaluate(Xtest_scaled, Ytest, verbose=1)
print("Test accuracy:", test_acc)

# Classification report
y_pred = np.argmax(final_model.predict(Xtest_scaled), axis=-1)
print("\nClassification Report:")
report = classification_report(Ytest, y_pred)
print(report)
with open(f'./report/ffnn_{file_type}_report.txt', 'w') as f:
    f.write("FFNN Classification Report:\n")
    f.write(f"Best hyperparameters: {best_params}\n")
    f.write(report)
