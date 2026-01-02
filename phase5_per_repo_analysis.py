import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ----------------- PATHS -----------------
DATA_CSV = Path("data/python_bug_data.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# ----------------- UTILS -----------------
def compute_metrics(y_true, y_pred, y_proba):
    """Return accuracy, precision, recall, f1, roc_auc."""
    acc = (y_true == y_pred).mean()

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label=1,
        zero_division=0,   # avoids undefined warnings
    )

    try:
        roc = roc_auc_score(y_true, y_proba)
    except Exception:
        roc = np.nan

    return acc, prec, rec, f1, roc


def plot_confusion(repo, model_name, cm):
    """Save confusion matrix as PNG."""
    import itertools

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{repo} – {model_name}\nConfusion Matrix")
    plt.colorbar(im, ax=ax)

    classes = ["Non-buggy (0)", "Buggy (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], "d"),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    plt.tight_layout()

    safe_repo = str(repo).replace("/", "_")
    safe_model = model_name.replace(" ", "_")
    out_path = RESULTS_DIR / f"phase5_confusion_{safe_repo}_{safe_model}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Saved confusion matrix → {out_path}")


# ----------------- MAIN PER-REPO ANALYSIS -----------------
def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    df = pd.read_csv(DATA_CSV)
    print("=== PHASE 5: Per-repository analysis ===")
    print(f"Total rows: {len(df)}")
    print(f"Repos: {sorted(df['repo'].unique())}")

    all_results = []

    repos = sorted(df["repo"].unique())

    for repo in repos:
        print("\n" + "=" * 80)
        print(f"Processing repo: {repo}")
        print("=" * 80)

        df_r = df[df["repo"] == repo].copy()
        n_rows = len(df_r)
        print(f"Rows in this repo: {n_rows}")

        # Need both classes for classification
        if df_r["buggy"].nunique() < 2:
            print(f"[SKIP] Repo {repo} has only one class label. Skipping.")
            continue

        X = df_r[["loc"]].astype(float).values  # metrics-only for now
        y = df_r["buggy"].astype(int).values

        # Stratified train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )

        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train class counts: {np.bincount(y_train)}")
        print(f"Test class counts:  {np.bincount(y_test)}")

        # Scale for linear model (trees don't need it, but scaling is cheap)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------- MODELS ----------
        models = {}

        # Logistic Regression (metrics)
        models["Logistic Regression"] = (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",  # handle imbalance
                n_jobs=-1,
            ),
            X_train_scaled,
            X_test_scaled,
        )

        # Random Forest
        models["Random Forest"] = (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=42,
            ),
            X_train,
            X_test,
        )

        # XGBoost
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                eval_metric="logloss",
                tree_method="hist",
                random_state=42,
            ),
            X_train,
            X_test,
        )

        # LightGBM
        models["LightGBM"] = (
            LGBMClassifier(
                n_estimators=200,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            X_train,
            X_test,
        )

        # CatBoost
        models["CatBoost"] = (
            CatBoostClassifier(
                iterations=200,
                learning_rate=0.1,
                depth=6,
                loss_function="Logloss",
                verbose=False,
                random_state=42,
            ),
            X_train,
            X_test,
        )

        # Store confusion matrices so we can later pick best-by-F1
        repo_confusions = {}

        for model_name, (model, Xtr, Xte) in models.items():
            print(f"\n--- {repo} / {model_name} ---")
            try:
                model.fit(Xtr, y_train)
                y_pred = model.predict(Xte)

                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(Xte)[:, 1]
                else:
                    # Fallback: decision_function or hard predictions
                    if hasattr(model, "decision_function"):
                        scores = model.decision_function(Xte)
                        # min-max scale scores to [0,1] for ROC
                        s_min, s_max = scores.min(), scores.max()
                        if s_max > s_min:
                            y_proba = (scores - s_min) / (s_max - s_min)
                        else:
                            y_proba = np.zeros_like(scores)
                    else:
                        y_proba = y_pred.astype(float)

                acc, prec, rec, f1, roc = compute_metrics(y_test, y_pred, y_proba)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

                print(f"Accuracy:  {acc:.4f}")
                print(f"Precision: {prec:.4f}")
                print(f"Recall:    {rec:.4f}")
                print(f"F1-score:  {f1:.4f}")
                print(f"ROC-AUC:   {roc:.4f}")
                print("Confusion matrix:")
                print(cm)

                all_results.append(
                    {
                        "repo": repo,
                        "model": model_name,
                        "accuracy": acc,
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "roc_auc": roc,
                        "tn": int(cm[0, 0]),
                        "fp": int(cm[0, 1]),
                        "fn": int(cm[1, 0]),
                        "tp": int(cm[1, 1]),
                        "n_samples": int(len(y_test)),
                        "n_positive": int((y_test == 1).sum()),
                        "n_negative": int((y_test == 0).sum()),
                    }
                )

                repo_confusions[model_name] = cm

            except Exception as e:
                print(f"[ERROR] {repo} / {model_name} failed with: {e}")
                continue

        # After all models for this repo, plot confusion for best F1
        if repo_confusions:
            df_repo = pd.DataFrame(
                [r for r in all_results if r["repo"] == repo]
            )
            if not df_repo.empty:
                best_idx = df_repo["f1"].idxmax()
                best_row = df_repo.loc[best_idx]
                best_model = best_row["model"]
                print(
                    f"\n[BEST] For repo {repo}, best by F1: "
                    f"{best_model} (F1={best_row['f1']:.4f})"
                )
                cm_best = repo_confusions.get(best_model)
                if cm_best is not None:
                    plot_confusion(repo, best_model, cm_best)

    # Save all results as CSV
    if all_results:
        df_all = pd.DataFrame(all_results)
        out_csv = RESULTS_DIR / "phase5_per_repo_results.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"\n✅ Saved per-repo results → {out_csv}")
        print("\nTop 15 rows:")
        print(df_all.sort_values(by=['repo', 'f1'], ascending=[True, False]).head(15))
    else:
        print("No results collected – something went wrong.")


if __name__ == "__main__":
    main()
