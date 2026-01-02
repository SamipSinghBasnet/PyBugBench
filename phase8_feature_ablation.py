# phase8_feature_ablation.py

import os
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier

DATA_CSV = Path("data/python_bug_data.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUT_CSV = RESULTS_DIR / "phase8_feature_ablation_results.csv"


def load_data():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)

    # Basic sanity check
    required_cols = {"repo", "file", "buggy", "loc"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Create file extension feature
    def get_ext(path):
        # Take last segment after ".", if exists
        if isinstance(path, str) and "." in path:
            return path.rsplit(".", 1)[-1]
        return "<no_ext>"

    df["ext"] = df["file"].apply(get_ext)

    # Ensure types
    df["buggy"] = df["buggy"].astype(int)
    df["loc"] = df["loc"].astype(float)

    return df


def build_feature_matrix(df, feature_set):
    """
    feature_set is a list like ["loc"] or ["loc", "repo", "ext"].
    We one-hot encode categorical features (repo, ext).
    """
    use_cols = feature_set.copy()
    X = df[use_cols].copy()

    cat_cols = [c for c in use_cols if c in ["repo", "ext"]]
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

    return X.values, X.columns.tolist()


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    try:
        roc = roc_auc_score(y_test, probs)
    except ValueError:
        roc = np.nan

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
    }


def main():
    print("=== PHASE 8: Feature ablation & robustness ===")

    df = load_data()
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    X_full_loc = df[["loc"]].values
    y = df["buggy"].values

    # Single global split so all experiments are comparable
    X_train_base, X_test_base, y_train_base, y_test = train_test_split(
        X_full_loc, y, test_size=0.2, stratify=y, random_state=42
    )
    # We won't actually use X_train_base directly, just its indices / alignment
    # So re-split using indices on the full df to rebuild X for each feature set
    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Define feature sets
    feature_sets = {
        "loc_only": ["loc"],
        "loc_repo": ["loc", "repo"],
        "loc_ext": ["loc", "ext"],
        "loc_repo_ext": ["loc", "repo", "ext"],
    }

    # Models to test
    model_specs = {
        "RandomForest": lambda seed: RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            n_jobs=-1,
        ),
        "LightGBM": lambda seed: LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary",
            random_state=seed,
            n_jobs=-1,
        ),
    }

    # Robustness configs
    seeds = [42, 7]
    train_fracs = [1.0, 0.3]

    results = []

    for fs_name, fs_cols in feature_sets.items():
        print("\n" + "=" * 80)
        print(f"Feature set: {fs_name} -> {fs_cols}")
        print("=" * 80)

        # Build design matrix for this feature set
        X_all, col_names = build_feature_matrix(df, fs_cols)
        X_train_all = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train_all = y[train_idx]
        y_test_local = y_test  # same as y[test_idx]

        for model_name, model_builder in model_specs.items():
            for seed in seeds:
                for frac in train_fracs:
                    print("-" * 40)
                    print(
                        f"{fs_name} / {model_name} / seed={seed} / train_frac={frac}"
                    )

                    # Subsample training data if frac < 1
                    if frac < 1.0:
                        X_tr, _, y_tr, _ = train_test_split(
                            X_train_all,
                            y_train_all,
                            train_size=frac,
                            stratify=y_train_all,
                            random_state=seed,
                        )
                    else:
                        X_tr, y_tr = X_train_all, y_train_all

                    model = model_builder(seed)
                    metrics = evaluate_model(model, X_tr, y_tr, X_test, y_test_local)

                    row = {
                        "feature_set": fs_name,
                        "features": ",".join(fs_cols),
                        "model": model_name,
                        "seed": seed,
                        "train_fraction": frac,
                        "n_train": len(y_tr),
                        "n_test": len(y_test_local),
                    }
                    row.update(metrics)
                    results.append(row)

                    print(
                        f"Acc={metrics['accuracy']:.4f}, "
                        f"Prec={metrics['precision']:.4f}, "
                        f"Rec={metrics['recall']:.4f}, "
                        f"F1={metrics['f1']:.4f}, "
                        f"ROC-AUC={metrics['roc_auc']:.4f}"
                    )

    df_res = pd.DataFrame(results)
    df_res.to_csv(OUT_CSV, index=False)
    print("\n✅ Saved Phase 8 feature ablation results →", OUT_CSV)

    # Also print best rows by F1
    print("\nTop 15 rows by F1:")
    print(df_res.sort_values("f1", ascending=False).head(15))


if __name__ == "__main__":
    main()
