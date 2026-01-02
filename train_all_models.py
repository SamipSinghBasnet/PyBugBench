import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Try to import XGBoost if installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    print("[WARN] XGBoost is not usable, will skip XGBoost model.")
    print("       Reason:", e)


DATA_PATH = Path("data/python_bug_data.csv")
RESULTS_DIR = Path("results")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run build_dataset.py first.")

    df = pd.read_csv(DATA_PATH)

    expected_cols = {"repo", "commit", "file", "buggy", "loc", "avg_complexity", "maintainability"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.dropna(subset=["loc", "avg_complexity", "maintainability", "buggy"]).copy()

    X = df[["loc", "avg_complexity", "maintainability"]]
    y = df["buggy"].astype(int)

    print(f"Total samples: {len(df)}")
    print("Class distribution:")
    print(y.value_counts())

    return X, y


def train_and_evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n================= {name} =================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Predict probabilities if possible
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        y_proba = None
        roc_auc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    report = classification_report(y_test, y_pred, zero_division=0)

    print("Classification report:")
    print(report)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc_auc,
        "report": report,
    }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []

    # 1) Logistic Regression
    logreg_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    res_logreg = train_and_evaluate_model(
        "Logistic Regression (metrics only)",
        logreg_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append(res_logreg)

    # 2) Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )

    res_rf = train_and_evaluate_model(
        "Random Forest (metrics only)",
        rf_model,
        X_train,
        X_test,
        y_train,
        y_test,
    )
    results.append(res_rf)

    # 3) XGBoost (if available)
    if HAS_XGB:
        xgb_model = XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            scale_pos_weight=float((y_train == 0).sum()) / max((y_train == 1).sum(), 1),
            random_state=42,
        )

        res_xgb = train_and_evaluate_model(
            "XGBoost (metrics only)",
            xgb_model,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        results.append(res_xgb)
    else:
        print("\n[INFO] Skipping XGBoost because it is not usable.")

    # Save per-model text reports
    for r in results:
        safe_name = r["model"].lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        report_path = RESULTS_DIR / f"{safe_name}_report.txt"
        with report_path.open("w") as f:
            f.write(r["model"] + "\n\n")
            f.write(r["report"])
            f.write("\n")
        print(f"[OK] Saved report → {report_path}")

    # Save summary CSV
    summary_rows = []
    for r in results:
        summary_rows.append({
            "model": r["model"],
            "accuracy": r["accuracy"],
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1"],
            "roc_auc": r["roc_auc"],
        })

    df_summary = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / "model_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\n[OK] Saved model summary → {summary_path}")

    # Print summary to console
    print("\n================= SUMMARY =================")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
