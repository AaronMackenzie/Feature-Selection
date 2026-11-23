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
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_and_prepare_data(csv_file_path):
    df = pd.read_csv(csv_file_path)

    # Convert non-numeric columns to numeric where possible
    for col in df.columns[:-1]:  # All columns except the last one
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN values (non-numeric entries that could not be converted)
    df.dropna(inplace=True)

    return df


def select_features(X, y):
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=5000)

    # Determine the optimal number of features by 5-fold cross-validation
    best_score = 0
    best_num_features = 1
    for num_features in range(1, X.shape[1] + 1):
        rfe = RFE(model, n_features_to_select=num_features)
        rfe.fit(X_scaled, y)
        score = np.mean(cross_val_score(rfe, X_scaled, y, cv=5, scoring='accuracy'))
        if score > best_score:
            best_score = score
            best_num_features = num_features

    # Perform RFE with the optimal number of features
    rfe = RFE(model, n_features_to_select=best_num_features)
    rfe.fit(X_scaled, y)

    return X.columns[rfe.support_]


def save_selected_features(df, selected_features, output_file_path):
    df_selected = df[selected_features.tolist() + [df.columns[-1]]]  # Add the target column back
    df_selected.to_csv(output_file_path, index=False)
    print(f"Data with selected features has been saved to {output_file_path}")


def main():
    input_data, file_path = resolve_input()
    df = load_and_prepare_data(file_path)

    # Separate features and target
    X = df.iloc[:, :-1]  # All columns except the last one
    y = df.iloc[:, -1]   # The last column

    # Min-max scale but keep the column names (a bare array has no .columns)
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns, index=X.index)

    selected_features = select_features(X, y)
    print("Selected features:", list(selected_features))

    save_selected_features(df, selected_features, os.path.join(OUT_DIR, 'RFE_FS.csv'))


if __name__ == "__main__":
    main()
