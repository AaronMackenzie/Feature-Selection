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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms  # pip install deap

random.seed(42)
np.random.seed(42)

# Load dataset
input_data, file_path = resolve_input()
data = pd.read_csv(file_path)
target_name = data.columns[-1]

# Split dataset into features and target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Genetic Algorithm for Feature Selection


# Define the evaluation function
def evaluate(individual):
    # Convert the binary mask to a list of selected features
    selected_features = [index for index, bit in enumerate(individual) if bit == 1]
    if not selected_features:
        return 0,
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]

    # Train a classifier
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train_selected, y_train)
    predictions = clf.predict(X_test_selected)

    # NOTE: fitness is accuracy on the same hold-out split, so the reported value is optimistic
    accuracy = accuracy_score(y_test, predictions)
    return accuracy,


# Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(X.columns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

population = toolbox.population(n=50)
ngen = 20
cxpb = 0.5
mutpb = 0.2

# Genetic Algorithm Execution
algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

# Extract the best individual
best_individual = tools.selBest(population, k=1)[0]
selected_features = [index for index, bit in enumerate(best_individual) if bit == 1]
print("Best fitness (hold-out accuracy): %.4f" % best_individual.fitness.values[0])

# Create a new dataset with only the selected features (target column name taken from the data)
selected_features_names = X.columns[selected_features]
new_data = data[list(selected_features_names)].copy()
new_data[target_name] = y

output_file_path = os.path.join(OUT_DIR, 'GA_' + os.path.basename(input_data))
new_data.to_csv(output_file_path, index=False)

print(f"Selected features: {list(selected_features_names)}")
print(f"New dataset saved to {output_file_path}")
