from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_PATH = Path("data/python_bug_data.csv")
OUT_PATH = Path("results/phase3_feature_importance.csv")


def load_xy():
    df = pd.read_csv(DATA_PATH)
    feature_cols = ["loc", "avg_complexity", "maintainability"]
    X = df[feature_cols].values
    y = df["buggy"].astype(int).values
    return X, y, feature_cols


def train_rf(X_train, y_train, X_test, y_test, feature_names):
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    print("\n=== Random Forest (metrics) ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    importances = rf.feature_importances_
    return importances


def train_xgb(X_train, y_train, X_test, y_test, feature_names):
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
        scale_pos_weight=1.0,
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_proba = xgb.predict_proba(X_test)[:, 1]
    print("\n=== XGBoost (metrics) ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_proba))

    importances = xgb.feature_importances_
    return importances


def main():
    X, y, feature_names = load_xy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rf_imp = train_rf(X_train, y_train, X_test, y_test, feature_names)
    xgb_imp = train_xgb(X_train, y_train, X_test, y_test, feature_names)

    df_imp = pd.DataFrame(
        {
            "feature": feature_names,
            "rf_importance": rf_imp,
            "xgb_importance": xgb_imp,
        }
    ).sort_values("xgb_importance", ascending=False)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(OUT_PATH, index=False)
    print(f"\nSaved feature importance → {OUT_PATH}")


if __name__ == "__main__":
    main()
