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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler


def feature_selection():
    input_data, file_path = resolve_input()
    output_csv = os.path.join(OUT_DIR, 'Tree_FS.csv')

    data = pd.read_csv(file_path)

    # Check if the last column is numeric or categorical (target)
    if pd.api.types.is_numeric_dtype(data.iloc[:, -1]):
        print("Detected numeric target column.")
    else:
        print("Detected categorical target column.")

    # Separate features and target
    X = data.iloc[:, :-1]  # All columns except the last one
    y = data.iloc[:, -1]   # The last column

    # Tree-based feature importances (random_state fixed so results repeat)
    model = ExtraTreesClassifier(random_state=42)

    # Keep the column names: fit_transform returns a bare array
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    model.fit(X_scaled, y)

    importances = model.feature_importances_
    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Keep features whose importance is above the threshold
    threshold = 0.01
    selected_features = feature_importances[feature_importances['Importance'] > threshold]['Feature']

    # Fall back to the top N features if nothing meets the threshold
    if selected_features.empty:
        print("No features met the threshold. Selecting top N features instead.")
        top_n = 5
        selected_features = feature_importances.head(top_n)['Feature']

    print("Selected features:\n", list(selected_features))

    selected_data = data[selected_features.tolist() + [data.columns[-1]]]
    selected_data.to_csv(output_csv, index=False)
    print(f"Selected features data has been saved to {output_csv}")


if __name__ == "__main__":
    feature_selection()
