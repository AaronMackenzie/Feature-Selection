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



import random

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
np.random.seed(42)


class AntColonyFeatureSelector:
    def __init__(self, file_path, num_ants=10, num_iterations=100, evaporation_rate=0.5, alpha=1, beta=2):
        self.file_path = file_path
        self.data = pd.read_csv(file_path)
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.pheromone = np.ones((self.data.shape[1] - 1, self.data.shape[1] - 1))
        self.target = self.data.columns[-1]  # Assuming the target variable is the last column

        # Min-Max scaling of the features
        self.scale_features()

    def scale_features(self):
        scaler = MinMaxScaler()
        feature_columns = self.data.columns[:-1]  # All columns except the last one
        self.data[feature_columns] = scaler.fit_transform(self.data[feature_columns])

    def evaluate_path(self, path):
        """Evaluate the quality of the selected path."""
        selected_features = [index for index, value in enumerate(path) if value == 1]

        if not selected_features:  # Avoid empty feature set
            return -np.inf

        X = self.data.iloc[:, selected_features]
        y = self.data[self.target]

        # Use LogisticRegression and cross-validation to evaluate feature set
        classifier = LogisticRegression(max_iter=1000)
        scores = cross_val_score(classifier, X, y, cv=5, scoring='accuracy')
        return scores.mean()

    def select_features(self):
        best_path = None
        best_score = -np.inf

        for iteration in range(self.num_iterations):
            paths = []
            scores = []

            for ant in range(self.num_ants):
                path = self.generate_path()
                score = self.evaluate_path(path)
                paths.append(path)
                scores.append(score)

                if score > best_score:
                    best_score = score
                    best_path = path

            self.update_pheromone(paths, scores)

        self.best_score = best_score
        selected_features = [i for i in range(len(best_path)) if best_path[i] == 1]
        return selected_features

    def generate_path(self):
        # NOTE: every feature is switched on with probability 0.5; the pheromone matrix is
        # updated below but never read here, so this behaves as a random search.
        path = np.zeros(self.data.shape[1] - 1)
        for i in range(len(path)):
            if random.random() < 0.5:
                path[i] = 1
        return path

    def update_pheromone(self, paths, scores):
        self.pheromone *= (1 - self.evaporation_rate)

        for path, score in zip(paths, scores):
            for i in range(len(path)):
                if path[i] == 1:
                    self.pheromone[i] += score

    def run(self):
        selected_features = self.select_features()
        modified_data = self.data.iloc[:, selected_features].copy()
        # Include the target class in the output file
        modified_data[self.target] = self.data[self.target]
        print("Selected features:\n", list(modified_data.columns[:-1]))
        print("Best cross-validated accuracy during search: %.4f" % self.best_score)
        modified_data.to_csv(os.path.join(OUT_DIR, 'Ant_FS.csv'), index=False)
        return modified_data


if __name__ == "__main__":
    name, file_path = resolve_input()
    ant_colony_fs = AntColonyFeatureSelector(file_path)
    selected_data = ant_colony_fs.run()
    print("Selected features dataset saved to", os.path.join(OUT_DIR, 'Ant_FS.csv'))
