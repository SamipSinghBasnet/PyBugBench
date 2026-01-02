import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

DATA_PATH = Path("data/python_bug_data.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.dropna()
    X = df[["loc", "avg_complexity", "maintainability"]]
    y = df["buggy"]
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc = None

    res = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc,
        "report": classification_report(y_test, y_pred)
    }

    print(f"\n==== {name} ====")
    print(res["report"])

    # Save report
    with open(RESULTS_DIR / f"{name}_report.txt", "w") as f:
        f.write(res["report"])

    return res


def main():
    X_train, X_test, y_train, y_test = load_data()
    results = []

    # ------------------------ LIGHTGBM ------------------------
    lgb = LGBMClassifier(
        n_estimators=300,
        num_leaves=64,
        class_weight="balanced",
        learning_rate=0.05,
        random_state=42
    )
    lgb.fit(X_train, y_train)
    results.append(evaluate_model("LightGBM", lgb, X_test, y_test))

    # ------------------------ CATBOOST ------------------------
    cat = CatBoostClassifier(
        iterations=300,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        verbose=False,
        class_weights=[1, 2]  # balance classes
    )
    cat.fit(X_train, y_train)
    results.append(evaluate_model("CatBoost", cat, X_test, y_test))

    # ------------------------ MLP (Neural Network) ------------------------
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=20,
            activation="relu",
            solver="adam",
            random_state=42
        ))
    ])
    mlp.fit(X_train, y_train)
    results.append(evaluate_model("MLP_NeuralNet", mlp, X_test, y_test))

    # ------------------------ SAVE SUMMARY ------------------------
    df_summary = pd.DataFrame(results)
    df_summary.to_csv(RESULTS_DIR / "phase2_model_summary.csv", index=False)
    print("\nSaved → results/phase2_model_summary.csv")


if __name__ == "__main__":
    main()
