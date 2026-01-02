import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("data/python_bug_data.csv")
OUT_DIR = Path("results/phase3_eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def basic_stats(df: pd.DataFrame):
    print("=== BASIC INFO ===")
    print(df.head())
    print("\nShape:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\n=== BUGGY DISTRIBUTION ===")
    print(df["buggy"].value_counts())
    print("\nBuggy ratio:", df["buggy"].mean())


def repo_stats(df: pd.DataFrame):
    print("\n=== SAMPLES PER REPO ===")
    print(df["repo"].value_counts())

    print("\n=== BUGGY RATIO PER REPO ===")
    buggy_rate = df.groupby("repo")["buggy"].mean().sort_values(ascending=False)
    print(buggy_rate)

    buggy_rate.to_csv(OUT_DIR / "buggy_rate_per_repo.csv")


def numeric_stats(df: pd.DataFrame):
    cols = ["loc", "avg_complexity", "maintainability"]
    print("\n=== NUMERIC STATS ===")
    print(df[cols].describe())

    corr = df[cols + ["buggy"]].corr()
    print("\n=== CORRELATION (metrics vs buggy) ===")
    print(corr)

    corr.to_csv(OUT_DIR / "correlation_metrics_buggy.csv")


def plot_hist(df: pd.DataFrame, col: str, bins: int = 50):
    plt.figure(figsize=(6, 4))
    plt.hist(df[col], bins=bins, alpha=0.7)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{col}_hist.png")
    plt.close()


def plot_hist_by_buggy(df: pd.DataFrame, col: str, bins: int = 50):
    plt.figure(figsize=(6, 4))
    plt.hist(df[df["buggy"] == 0][col], bins=bins, alpha=0.6, label="clean")
    plt.hist(df[df["buggy"] == 1][col], bins=bins, alpha=0.6, label="buggy")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"{col} by buggy vs clean")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{col}_by_buggy.png")
    plt.close()


def main():
    df = load_data()
    basic_stats(df)
    repo_stats(df)
    numeric_stats(df)

    # Overall distributions
    for col in ["loc", "avg_complexity", "maintainability"]:
        plot_hist(df, col)
        plot_hist_by_buggy(df, col)


if __name__ == "__main__":
    main()
