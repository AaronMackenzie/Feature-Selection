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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_classif, SelectKBest


def load_and_prepare_data(df):
    # Separate features and target
    X = df.iloc[:, :-1].copy()  # All columns except the last one
    y = df.iloc[:, -1]          # The last column

    # Convert categorical data to numerical values (before imputing: the mean imputer needs numbers)
    for column in X.select_dtypes(include=['object']).columns:
        X[column] = LabelEncoder().fit_transform(X[column].astype(str))

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Scale the features
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    return X, y


def perform_anova_feature_selection(X, y, k=10):
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    return X.columns[selector.get_support()]


def save_selected_features(df, selected_features, output_path):
    selected_data = df[selected_features.tolist() + [df.columns[-1]]]
    selected_data.to_csv(output_path, index=False)
    print(f"Selected features have been saved to {output_path}")


def main():
    input_data, file_path = resolve_input()
    df = pd.read_csv(file_path)

    X, y = load_and_prepare_data(df)

    # k = 10 is hard-coded in the original script; it is kept
    selected_features = perform_anova_feature_selection(X, y)
    print("Selected features:\n", list(selected_features))

    save_selected_features(df, selected_features, os.path.join(OUT_DIR, 'ANOVA_FS.csv'))


if __name__ == "__main__":
    main()
