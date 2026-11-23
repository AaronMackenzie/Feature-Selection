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
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

input_data, file_path = resolve_input()
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Scale features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

logreg = LogisticRegression()

# Define cross-validation strategy (unshuffled folds, as in the original script)
cv = StratifiedKFold(5)


# Function to evaluate model with a given number of features
def evaluate_model(n_features):
    selector = SequentialFeatureSelector(logreg, n_features_to_select=n_features, direction='forward', scoring='accuracy', cv=cv)
    selector.fit(X_scaled, y)
    selected_features = selector.get_support()
    scores = cross_val_score(logreg, X_scaled[:, selected_features], y, cv=cv, scoring='accuracy')
    return scores.mean()


# Determine the optimal number of features
n_features_options = range(1, X.shape[1])
mean_scores = [evaluate_model(n) for n in n_features_options]

optimal_n_features = n_features_options[np.argmax(mean_scores)]
print("Optimal number of features: ", optimal_n_features)

# Perform forward selection with the optimal number of features
selector = SequentialFeatureSelector(logreg, n_features_to_select=optimal_n_features, direction='forward', scoring='accuracy', cv=cv)
selector.fit(X_scaled, y)

final_selected_features = X.columns[selector.get_support()]
print("Final Selected Features: ", list(final_selected_features))

X_selected = X[final_selected_features]
selected_data = pd.concat([X_selected, y], axis=1)

output_file_path = os.path.join(OUT_DIR, "ForSel_" + os.path.basename(input_data))
selected_data.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)
