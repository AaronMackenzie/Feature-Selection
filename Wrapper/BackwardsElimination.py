import os
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Define the data folder
data_folder = 'PSO-main/data/'

# Prompt user for the name of the CSV file
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Perform backward sequential feature selection
selector = SequentialFeatureSelector(logreg, n_features_to_select=5, direction='backward', scoring='accuracy')
selector.fit(X_scaled, y)

# Get the selected features
selected_features = selector.get_support()
print("Selected features: ", list(X.columns[selected_features]))

# Create a new DataFrame with the selected features
X_selected = X.loc[:, selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

# Ensure the output directory exists
output_directory = os.path.join(data_folder, 'BackwardsElimination')
os.makedirs(output_directory, exist_ok=True)

# Save the selected features to a new CSV file
output_file_path = os.path.join(output_directory, 'BE_' + input_data)
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)
