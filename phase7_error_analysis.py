from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
)

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


DATA_CSV = Path("data/python_bug_data.csv")
OUT_FP_CSV = Path("results/phase7_false_positives.csv")
OUT_FN_CSV = Path("results/phase7_false_negatives.csv")
OUT_SUMMARY = Path("results/phase7_error_summary.txt")


def load_data():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    # We’ll keep repo, file, commit for interpretation
    keep_cols = ["repo", "commit", "file", "loc", "buggy"]
    df = df[keep_cols].dropna()

    df["buggy"] = df["buggy"].astype(int)
    df["loc"] = df["loc"].astype(float)

    return df


def train_global_lgbm(X_train, y_train):
    """Train a reasonably strong global model on LOC with class balancing."""
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.1,
        max_depth=-1,
        objective="binary",
        random_state=42,
        class_weight="balanced",  # important for imbalance
    )
    model.fit(X_train, y_train)
    return model


def compute_metrics(y_true, y_pred, y_proba):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = (y_true == y_pred).mean()
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


def main():
    df = load_data()
    print("=== PHASE 7: Error analysis on global model ===")
    print(f"Total rows: {len(df)}")

    X = df[["loc"]].values
    y = df["buggy"].values

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print("Train class counts:", np.bincount(y_train))
    print("Test class counts:", np.bincount(y_test))

    # ---- Train model ----
    model = train_global_lgbm(X_train, y_train)

    # ---- Predict on test ----
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test >= 0.5).astype(int)  # default threshold for error analysis

    metrics = compute_metrics(y_test, y_pred, proba_test)
    print("\n=== Global LightGBM (default thr=0.5) on TEST ===")
    print(metrics)

    # ---- Attach predictions to df_test ----
    df_test = df_test.copy()
    df_test["y_true"] = y_test
    df_test["y_pred"] = y_pred
    df_test["proba_buggy"] = proba_test

    # False positives: predicted 1, actually 0
    fp = df_test[(df_test["y_true"] == 0) & (df_test["y_pred"] == 1)].copy()
    # Sort by highest probability of being buggy (most confident but wrong)
    fp = fp.sort_values(by="proba_buggy", ascending=False)

    # False negatives: predicted 0, actually 1
    fn = df_test[(df_test["y_true"] == 1) & (df_test["y_pred"] == 0)].copy()
    # Sort by lowest probability of being buggy (most “confidently clean” but wrong)
    fn = fn.sort_values(by="proba_buggy", ascending=True)

    # Save top-k (e.g., 200) for manual inspection
    TOP_K = 200
    fp_head = fp.head(TOP_K)
    fn_head = fn.head(TOP_K)

    OUT_FP_CSV.parent.mkdir(parents=True, exist_ok=True)
    fp_head.to_csv(OUT_FP_CSV, index=False)
    fn_head.to_csv(OUT_FN_CSV, index=False)

    print(f"\nSaved top {len(fp_head)} false positives → {OUT_FP_CSV}")
    print(f"Saved top {len(fn_head)} false negatives → {OUT_FN_CSV}")

    # Also dump a tiny text summary
    with OUT_SUMMARY.open("w") as f:
        f.write("PHASE 7 – Error analysis summary\n")
        f.write("================================\n\n")
        f.write("Global LightGBM (class_weight='balanced', thr=0.5)\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
        f.write("\n")
        f.write(f"False positives (count): {len(fp)}\n")
        f.write(f"False negatives (count): {len(fn)}\n")

    print("\n✅ Wrote error summary →", OUT_SUMMARY)


if __name__ == "__main__":
    main()
