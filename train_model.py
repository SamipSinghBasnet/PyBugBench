import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

DATA_PATH = "data/python_bug_data.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["loc", "avg_complexity", "maintainability"]
    X = df[feature_cols]
    y = df["buggy"]
    return X, y

def train_and_evaluate():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=42,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Model Results ===")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()
