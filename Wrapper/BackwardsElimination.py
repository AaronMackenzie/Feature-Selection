import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def resolve_input():
    """CSV name (looked up in data/) or path from argv[1], else prompt. Last column = target."""
    name = sys.argv[1] if len(sys.argv) > 1 else input("Provide the name of the csv file (in data/): ")
    return name, (name if os.path.isabs(name) else os.path.join(DATA_DIR, name))



import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

input_data, file_path = resolve_input()
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Backward sequential selection down to 5 features (5 is hard-coded in the original script)
selector = SequentialFeatureSelector(logreg, n_features_to_select=5, direction='backward', scoring='accuracy')
selector.fit(X_scaled, y)

selected_features = selector.get_support()
print("Selected features: ", list(X.columns[selected_features]))

X_selected = X.loc[:, selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

output_file_path = os.path.join(OUT_DIR, 'BE_' + os.path.basename(input_data))
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)
