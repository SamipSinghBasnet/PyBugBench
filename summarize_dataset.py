import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/python_bug_data.csv")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run build_dataset.py first.")

    df = pd.read_csv(DATA_PATH)

    print("=== BASIC INFO ===")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    print("\n=== BUGGY DISTRIBUTION ===")
    if "buggy" in df.columns:
        print(df["buggy"].value_counts())
        print("\nBuggy ratio:", df["buggy"].mean())
    else:
        print("No 'buggy' column found.")

    if "repo" in df.columns:
        print("\n=== SAMPLES PER REPO ===")
        print(df["repo"].value_counts())

    print("\n=== LOC STATS ===")
    if "loc" in df.columns:
        print(df["loc"].describe())

    print("\n=== AVG COMPLEXITY STATS ===")
    if "avg_complexity" in df.columns:
        print(df["avg_complexity"].describe())

    print("\n=== MAINTAINABILITY STATS ===")
    if "maintainability" in df.columns:
        print(df["maintainability"].describe())


if __name__ == "__main__":
    main()
