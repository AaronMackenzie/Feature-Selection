import os
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Load CSV file
data_folder = 'PSO-main/data/'
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Encode target variable if it's categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform Ridge Regression for feature selection
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# Get absolute values of coefficients and determine threshold
ridge_coefficients = np.abs(ridge.coef_)
threshold = np.mean(ridge_coefficients)

# Select features with coefficients above the threshold
selected_features_ridge = X.columns[ridge_coefficients > threshold]

print("Ridge Regression Selected Features:\n", selected_features_ridge)

# Create a new dataframe with the selected features
X_selected = X[selected_features_ridge]

# Combine the selected features with the target variable
updated_data = pd.concat([X_selected, y], axis=1)

# Save the updated dataset to a new CSV file
output_file_path = os.path.join(data_folder+'RidgeRegression/', 'RR_' + input_data)
updated_data.to_csv(output_file_path, index=False)

print(f"Updated dataset saved to: {output_file_path}")
