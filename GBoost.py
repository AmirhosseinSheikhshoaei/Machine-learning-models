from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import multiprocessing  # To determine the number of CPU cores
from torch.utils.data import DataLoader, TensorDataset
import torch

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
y = scaler_y.fit_transform(y)  # Standardize multi-column target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the data into PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
num_workers = multiprocessing.cpu_count()  # Get the number of CPU cores

# Use DataLoader for batch-wise processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)

# Initialize the GradientBoostingRegressor with hyperparameters
gboost = GradientBoostingRegressor(loss='squared_error')

# Wrap it with MultiOutputRegressor to handle multiple outputs
multi_target_gboost = MultiOutputRegressor(gboost)

# Define the parameter grid for RandomizedSearchCV
params = {
    'estimator__n_estimators': [100, 200, 500, 1000],  # Number of boosting stages
    'estimator__max_depth': [3, 4, 6, 8],  # Maximum depth of the tree
    'estimator__learning_rate': [0.1, 0.05, 0.01],  # Step size shrinking to prevent overfitting
    'estimator__subsample': [0.7, 0.8, 0.9],  # Fraction of samples to use for fitting each base model
    'estimator__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'estimator__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'estimator__max_features': [None, 'sqrt', 'log2']  # Number of features to consider when looking for the best split
}

# Set up RandomizedSearchCV for faster hyperparameter tuning
random_search = RandomizedSearchCV(
    estimator=multi_target_gboost,
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
joblib.dump(best_model, 'best_gboost_model.pkl')

# Predict on the test set using the best model
y_pred_scaled = best_model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)  # Inverse transform predictions
y_test_actual = scaler_y.inverse_transform(y_test)  # Inverse transform test data

# Evaluate the model
mse = mean_squared_error(y_test_actual, y_pred)
aard = mean_absolute_percentage_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)

print(f"Test MSE: {mse}")
print(f"Test AARD: {aard}")
print(f"Test R2: {r2}")
