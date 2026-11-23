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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(42)

input_data, file_path = resolve_input()
data = pd.read_csv(file_path)

# Split the dataset into features and target
X = data.iloc[:, :-1]  # All columns except the last one (features)
y = data.iloc[:, -1]   # The last column (target)

# Encode target variable if it's categorical
if y.dtype == 'object':
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=data.columns[-1])

# Scale features to [0, 1] range
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Perform chi-squared feature selection on scaled data
chi2_selector = SelectKBest(chi2)
chi2_selector.fit(X_scaled, y)

# Select features with chi-squared scores above the mean score
chi2_scores = chi2_selector.scores_
threshold = np.mean(chi2_scores)
selected_features_chi2 = X.columns[chi2_scores > threshold]
print("Chi-Squared Selected Features:\n", list(selected_features_chi2))

# Refit the scaler on the selected features
X_selected = X[selected_features_chi2]
scaler_selected = MinMaxScaler()
X_selected_scaled = scaler_selected.fit_transform(X_selected)


# Define the simulated annealing algorithm
class SimulatedAnnealingFeatureSelection:
    def __init__(self, estimator, X, y, initial_temp=100, final_temp=1, alpha=0.9, max_iter=1000):
        self.estimator = estimator
        self.X = X
        self.y = y
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha
        self.max_iter = max_iter
        self.n_features = X.shape[1]

    def objective_function(self, selected_features):
        if not np.any(selected_features):  # an empty subset cannot be fitted
            return 0.0
        X_subset = self.X[:, selected_features]
        # NOTE: the subset is scored on this hold-out split, so the reported accuracy is optimistic
        X_train, X_test, y_train, y_test = train_test_split(X_subset, self.y, test_size=0.3, random_state=42)
        self.estimator.fit(X_train, y_train)
        y_pred = self.estimator.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def random_neighbor(self, current_features):
        neighbor = current_features.copy()
        idx = np.random.randint(0, self.n_features)
        neighbor[idx] = not neighbor[idx]
        return neighbor

    def acceptance_probability(self, old_cost, new_cost, temperature):
        if new_cost > old_cost:
            return 1.0
        return np.exp((new_cost - old_cost) / temperature)

    def fit(self):
        current_features = np.random.choice([True, False], size=self.n_features)
        current_cost = self.objective_function(current_features)
        best_features = current_features.copy()
        best_cost = current_cost

        temperature = self.initial_temp
        for iteration in range(self.max_iter):
            if temperature <= self.final_temp:
                break
            new_features = self.random_neighbor(current_features)
            new_cost = self.objective_function(new_features)
            if self.acceptance_probability(current_cost, new_cost, temperature) > np.random.rand():
                current_features = new_features
                current_cost = new_cost
                if new_cost > best_cost:
                    best_features = new_features
                    best_cost = new_cost
            temperature *= self.alpha

        self.best_cost = best_cost
        return best_features


# Apply simulated annealing on the subset of features selected by chi-squared
sa_selector = SimulatedAnnealingFeatureSelection(LogisticRegression(max_iter=1000), X_selected_scaled, y)
selected_features_sa = sa_selector.fit()

final_selected_features = selected_features_chi2[selected_features_sa]
print("Final Selected Features After Simulated Annealing:\n", list(final_selected_features))
print("Best hold-out accuracy seen during annealing: %.4f" % sa_selector.best_cost)

X_selected_final = X[final_selected_features]
selected_data_final = pd.concat([X_selected_final, y], axis=1)

output_file_path = os.path.join(OUT_DIR, "ChiSimAnneal_" + os.path.basename(input_data))
selected_data_final.to_csv(output_file_path, index=False)

print("\nSelected features have been saved to:", output_file_path)
