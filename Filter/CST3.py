
import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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

# Standardize features (optional but recommended for chi-squared)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform chi-squared feature selection
chi2_selector = SelectKBest(chi2)
chi2_selector.fit(X_scaled, y)

# Get chi-squared scores and determine threshold
chi2_scores = chi2_selector.scores_
threshold = np.mean(chi2_scores)

# Select features with chi-squared scores above the threshold
selected_features_chi2 = X.columns[chi2_scores > threshold]

print("Chi-Squared Selected Features:\n", selected_features_chi2)

# Create a new dataframe with the selected features
X_selected = X[selected_features_chi2]

# Combine the selected features with the target variable
updated_data = pd.concat([X_selected, y], axis=1)

# Save the updated dataset to a new CSV file
output_file_path = os.path.join(data_folder,'ChiSquaredTesting/', 'CST_' + input_data)
updated_data.to_csv(output_file_path, index=False)

print(f"Updated dataset saved to: {output_file_path}")
