import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from deap import base, creator, tools, algorithms
import random
import numpy as np

# Load dataset
data_folder = 'PSO-main/data/GeneticAlgorithm/'
input_data = input("Provide the name of the csv file: ")
file_path = os.path.join(data_folder, input_data)

data = pd.read_csv(file_path)

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
    
    # Calculate accuracy
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

# Create a new dataset with only the selected features
selected_features_names = X.columns[selected_features]
new_data = data[selected_features_names]
new_data['Class'] = y

# Save the updated dataset to a new CSV file
output_file_path = os.path.join(data_folder,'GA_' + input_data)
new_data.to_csv(output_file_path, index=False)


print(f"Selected features: {selected_features_names}")
print(f"New dataset saved to {output_file_path}")
