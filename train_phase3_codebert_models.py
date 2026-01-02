import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


NPZ_PATH = Path("data/codebert_embeddings.npz")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def load_data():
    """
    Load simulated CodeBERT embeddings + metrics from NPZ file.

    Expected keys in NPZ:
        - X_code:      [N, 768]  CodeBERT embeddings
        - X_metrics:   [N, 1]    Simple metrics (e.g., LOC)
        - y:           [N]       Labels (0/1)
    """
    if not NPZ_PATH.exists():
        raise FileNotFoundError(f"NPZ file not found: {NPZ_PATH}")

    data = np.load(NPZ_PATH)
    # <<< IMPORTANT: these names match extract_codebert_embeddings.py >>>
    X_code = data["X_code"]
    X_metrics = data["X_metrics"]
    y = data["y"]

    # Just in case something is slightly off, align lengths
    n = min(len(X_code), len(X_metrics), len(y))
    X_code = X_code[:n]
    X_metrics = X_metrics[:n]
    y = y[:n]

    # Hybrid feature vector = [CodeBERT | metrics]
    X_hybrid = np.hstack([X_code, X_metrics])

    print(f"Loaded NPZ from {NPZ_PATH}")
    print(f"  X_code shape:   {X_code.shape}")
    print(f"  X_metrics shape:{X_metrics.shape}")
    print(f"  X_hybrid shape: {X_hybrid.shape}")
    print(f"  y shape:        {y.shape}")

    return X_code, X_metrics, X_hybrid, y


def eval_model(name, model, X_train, X_test, y_train, y_test, results_list):
    """
    Fit model, print classification report, compute metrics,
    and append one row to results_list.
    """
    print(f"\n================= {name} =================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Some models have predict_proba, others only decision_function
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
    else:
        # Fallback: use predictions as scores (not ideal, but ok for now)
        y_prob = y_pred

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        roc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc = float("nan")

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc:.4f}")

    results_list.append(
        {
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": roc,
        }
    )


def main():
    # 1) Load data
    X_code, X_metrics, X_hybrid, y = load_data()

    # 2) Train/test split on the HYBRID features (CodeBERT + metrics)
    X_train, X_test, y_train, y_test = train_test_split(
        X_hybrid,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    results = []

    # ---------------- Logistic Regression (hybrid) ----------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
    )
    eval_model(
        "Logistic Regression (CodeBERT + metrics)",
        log_reg,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        results,
    )

    # ---------------- Random Forest (hybrid) ----------------
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )
    eval_model(
        "Random Forest (CodeBERT + metrics)",
        rf,
        X_train,
        X_test,
        y_train,
        y_test,
        results,
    )

    # ---------------- XGBoost (hybrid) ----------------
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",  # good default for CPU
        n_jobs=4,
        random_state=42,
    )
    eval_model(
        "XGBoost (CodeBERT + metrics)",
        xgb,
        X_train,
        X_test,
        y_train,
        y_test,
        results,
    )

    # 3) Save summary
    df_results = pd.DataFrame(results)
    out_path = RESULTS_DIR / "phase3_codebert_results.csv"
    df_results.to_csv(out_path, index=False)
    print("\n================= SUMMARY (Phase 3 – CodeBERT Hybrid) =================")
    print(df_results)
    print(f"\n[OK] Saved Phase 3 CodeBERT results → {out_path}")


if __name__ == "__main__":
    main()
