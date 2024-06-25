import os
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# Load CSV file
data_folder = 'PSO-main/data'
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Scale features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Apply Lasso for feature selection on scaled data
lasso = Lasso(alpha=0.01)
lasso.fit(X_scaled, y)

# Get the coefficients
lasso_coef = lasso.coef_

# Select features with non-zero coefficients
selected_features = X.columns[(lasso_coef != 0)]

print("Selected features using Lasso:\n", list(selected_features))

# Create a new DataFrame with the selected features from the original data
X_selected = X[selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

# Generate new file name
output_file_name = "LassoReg" + os.path.splitext(input_data)[0] + ".csv"
output_file_path = os.path.join(data_folder, output_file_name)
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)