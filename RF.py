import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib  # for saving the model
import multiprocessing  # To determine the number of CPU cores

# Load the data
data = pd.read_csv(r"D:\AR\n.csv")

# Separate input & outputs
X = data.iloc[:, :5].values  # Assuming the first 4 columns are features
y = data.iloc[:, 5:7].values  # Select columns 5 to end for the target (multiple columns)

# Check the shapes to ensure consistency
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Standardize the data
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)  # Standardize multiple output columns

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
num_workers = multiprocessing.cpu_count()  # Get the number of CPU cores

# Use DataLoader for batch-wise processing (optional for Random Forest, but kept for consistency with original code)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

# Define the parameter grid for RandomizedSearchCV
params = {
    'n_estimators': [100, 200, 500, 1000, 1500],  # More n_estimators for large data
    'max_depth': [10, 20, 30, None],  # Higher depths for large data
    'min_samples_split': [2, 5, 10],  # Minimum samples to split nodes
    'min_samples_leaf': [1, 2, 4],  # Minimum samples in leaf nodes
    'max_features': ['sqrt', 'log2', None],  # Consider all or a subset of features
    'bootstrap': [True, False]  # Whether to use bootstrap sampling
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, random_state=42)

# Set up RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=params,
    cv=3,
    scoring='neg_mean_squared_error',
    n_iter=50,  # Number of random parameter combinations to try
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the model with RandomizedSearchCV
random_search.fit(X_train, y_train)

# Get the best parameters and best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_
print("Best Parameters:", best_params)

# Save the best model to a file
joblib.dump(best_model, 'best_rf_model.pkl')

# Predict on the test set using the best model
y_pred_scaled = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Inverse transform predictions
y_test_actual = scaler_y.inverse_transform(y_test)  # Inverse transform test data

# Evaluate the model
mse = mean_squared_error(y_test_actual, y_pred)
aard = mean_absolute_percentage_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

# Print evaluation metrics
print(f"Test MSE: {mse}")
print(f"Test AARD: {aard}")
print(f"Test R2: {r2}")
