from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    classification_report,
)

from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


DATA_CSV = Path("data/python_bug_data.csv")
RESULTS_CSV = Path("results/phase6_imbalance_results.csv")


# -------------------- Utility functions -------------------- #

def load_full_data():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    # Only metric we really have is LOC (complexity + MI are zero)
    df = df[["repo", "loc", "buggy"]].dropna()
    df["buggy"] = df["buggy"].astype(int)
    df["loc"] = df["loc"].astype(float)
    return df


def evaluate_model(y_true, y_pred, y_proba):
    """Compute metrics and return as dict."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    accuracy = (y_true == y_pred).mean()

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc_auc = np.nan

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else np.nan,  # handle NaN
    }


def tune_threshold(y_val, proba_val, thresholds=None):
    """Search over thresholds to maximize F1 on the validation set."""
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best_thr = 0.5
    best_f1 = -1.0

    for thr in thresholds:
        y_pred_thr = (proba_val >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_val, y_pred_thr, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    return best_thr, best_f1


def train_rf(X_train, y_train, class_weight=None):
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight=class_weight,
    )
    rf.fit(X_train, y_train)
    return rf


def train_lgbm(X_train, y_train, class_weight=None):
    # LGBMClassifier accepts class_weight='balanced' or a dict
    lgbm = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,
        objective="binary",
        random_state=42,
        class_weight=class_weight,
    )
    lgbm.fit(X_train, y_train)
    return lgbm


# -------------------- Core experiment logic -------------------- #

def run_experiment(scope_name, df, results):
    """
    scope_name: 'global', 'django', 'scikit-learn', etc.
    df: DataFrame with columns ['loc', 'buggy'] (and 'repo' if needed).
    results: list to append metrics dicts into.
    """

    print("\n" + "=" * 80)
    print(f"PHASE 6 – Imbalance experiments for: {scope_name}")
    print("=" * 80)

    X = df[["loc"]].values
    y = df["buggy"].values

    print(f"Samples: {len(df)} | Positive (buggy=1): {y.sum()} "
          f"({y.sum() / len(y):.3f} ratio)")

    # First, split into train+temp and test so test is untouched
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Then, split train_val into train and validation for threshold tuning
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )
    # Now: 60% train, 20% val, 20% test

    print(f"Train size: {len(y_train)}, Val size: {len(y_val)}, Test size: {len(y_test)}")

    # ----- MODELS TO RUN -----
    setups = [
        ("RandomForest", "baseline", {"class_weight": None}),
        ("RandomForest", "balanced", {"class_weight": "balanced"}),
        ("LightGBM", "baseline", {"class_weight": None}),
        ("LightGBM", "balanced", {"class_weight": "balanced"}),
    ]

    for model_name, variant, params in setups:
        print("\n" + "-" * 40)
        print(f"{scope_name} / {model_name} / {variant}")
        print("-" * 40)

        # Train on TRAIN split
        if model_name == "RandomForest":
            model = train_rf(X_train, y_train, class_weight=params["class_weight"])
        elif model_name == "LightGBM":
            model = train_lgbm(X_train, y_train, class_weight=params["class_weight"])
        else:
            continue

        # ---- 1) Evaluate with default threshold 0.5 on TEST ----
        proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test >= 0.5).astype(int)
        base_metrics = evaluate_model(y_test, y_pred_test, proba_test)

        print(f"[Default thr=0.5] Accuracy={base_metrics['accuracy']:.4f}, "
              f"Prec={base_metrics['precision']:.4f}, "
              f"Rec={base_metrics['recall']:.4f}, "
              f"F1={base_metrics['f1']:.4f}, "
              f"ROC-AUC={base_metrics['roc_auc']:.4f}")

        results.append({
            "scope": scope_name,
            "model": model_name,
            "variant": f"{variant}_default0.5",
            "threshold": 0.5,
            **base_metrics,
        })

        # ---- 2) Threshold tuning on VALIDATION ----
        proba_val = model.predict_proba(X_val)[:, 1]
        best_thr, best_f1_val = tune_threshold(y_val, proba_val)

        print(f"[Tuning] Best threshold on val={best_thr:.3f} "
              f"(val F1={best_f1_val:.4f})")

        # Re-evaluate on TEST using chosen threshold
        y_pred_test_thr = (proba_test >= best_thr).astype(int)
        tuned_metrics = evaluate_model(y_test, y_pred_test_thr, proba_test)

        print(f"[Test @ thr={best_thr:.3f}] Accuracy={tuned_metrics['accuracy']:.4f}, "
              f"Prec={tuned_metrics['precision']:.4f}, "
              f"Rec={tuned_metrics['recall']:.4f}, "
              f"F1={tuned_metrics['f1']:.4f}, "
              f"ROC-AUC={tuned_metrics['roc_auc']:.4f}")

        results.append({
            "scope": scope_name,
            "model": model_name,
            "variant": f"{variant}_tuned",
            "threshold": float(best_thr),
            **tuned_metrics,
        })


def main():
    df = load_full_data()

    all_results = []

    # 1) Global experiment on all repos combined
    run_experiment("global", df, all_results)

    # 2) Per-repo: django (high buggy ratio)
    df_django = df[df["repo"] == "django"].copy()
    run_experiment("django", df_django, all_results)

    # 3) Per-repo: scikit-learn (lower buggy ratio)
    df_sklearn = df[df["repo"] == "scikit-learn"].copy()
    run_experiment("scikit-learn", df_sklearn, all_results)

    # Save results
    results_df = pd.DataFrame(all_results)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n✅ Saved Phase 6 imbalance experiments →", RESULTS_CSV)
    print("\nTop rows:")
    print(results_df.sort_values(
        by=["scope", "f1"], ascending=[True, False]
    ).groupby("scope").head(5))


if __name__ == "__main__":
    main()
