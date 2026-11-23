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
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

input_data, file_path = resolve_input()
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

# Select features with non-zero coefficients
lasso_coef = lasso.coef_
selected_features = X.columns[(lasso_coef != 0)]

print("Selected features using Lasso:\n", list(selected_features))

# Create a new DataFrame with the selected features from the original data
X_selected = X[selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

output_file_path = os.path.join(OUT_DIR, "LassoReg_" + os.path.basename(input_data))
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)
