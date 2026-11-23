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


import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

input_data, file_path = resolve_input()
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Encode target variable if it's categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=data.columns[-1])

# chi-squared needs non-negative inputs, so scale to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform chi-squared scoring
chi2_selector = SelectKBest(chi2)
chi2_selector.fit(X_scaled, y)

# Select features with chi-squared scores above the mean score
chi2_scores = chi2_selector.scores_
threshold = np.mean(chi2_scores)
selected_features_chi2 = X.columns[chi2_scores > threshold]

print("Chi-Squared Selected Features:\n", list(selected_features_chi2))

X_selected = X[selected_features_chi2]
updated_data = pd.concat([X_selected, y], axis=1)

output_file_path = os.path.join(OUT_DIR, 'CST_' + os.path.basename(input_data))
updated_data.to_csv(output_file_path, index=False)

print(f"Updated dataset saved to: {output_file_path}")
