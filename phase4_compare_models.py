import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


RESULTS_DIR = Path("results")

PHASE1_CSV = RESULTS_DIR / "model_summary.csv"
PHASE2_CSV = RESULTS_DIR / "phase2_model_summary.csv"
PHASE3_CSV = RESULTS_DIR / "phase3_codebert_results.csv"
OUT_COMBINED_CSV = RESULTS_DIR / "phase4_all_models_comparison.csv"


def load_results(path: Path, label: str) -> pd.DataFrame:
    """Load a summary CSV if it exists, otherwise return empty DataFrame."""
    if not path.exists():
        print(f"[WARN] {label} results file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[OK] Loaded {label} from {path} (rows={len(df)})")
    return df


def make_plots(df_all: pd.DataFrame) -> None:
    """Generate comparison plots for each metric."""
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Sort once by F1 so bars look nice
    df_plot = df_all.sort_values("f1", ascending=False).reset_index(drop=True)

    for metric in metrics:
        if metric not in df_plot.columns:
            print(f"[WARN] Metric {metric} not in DataFrame, skipping plot.")
            continue

        plt.figure(figsize=(10, 6))
        plt.barh(df_plot["model"], df_plot[metric])
        plt.xlabel(metric.upper())
        plt.title(f"Model Comparison – {metric.upper()}")
        plt.gca().invert_yaxis()  # best at top
        plt.tight_layout()

        out_path = RESULTS_DIR / f"phase4_{metric}_comparison.png"
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[OK] Saved plot → {out_path}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- Load summaries from previous phases ----
    df1 = load_results(PHASE1_CSV, "Phase 1 (baseline)")
    df2 = load_results(PHASE2_CSV, "Phase 2 (advanced metrics-only)")
    df3 = load_results(PHASE3_CSV, "Phase 3 (CodeBERT hybrid)")

    # Concatenate all available results
    dfs = [d for d in [df1, df2, df3] if not d.empty]
    if not dfs:
        print("[ERROR] No result CSVs found. Run earlier phases first.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    # Optional: add a simple phase label if you want
    # You can infer from model names, or you can manually tag;
    # for now we just keep the 'model' column as-is.

    # Save combined table
    df_all.to_csv(OUT_COMBINED_CSV, index=False)
    print(f"\n[OK] Saved combined model comparison → {OUT_COMBINED_CSV}\n")

    # Print a nice ranking by F1
    if "f1" in df_all.columns:
        print("=== MODELS RANKED BY F1 (DESCENDING) ===")
        print(
            df_all.sort_values("f1", ascending=False)[
                ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
            ].to_string(index=False)
        )
    else:
        print("[WARN] No 'f1' column found, skipping ranking output.")

    # Make bar plots for each metric
    make_plots(df_all)


if __name__ == "__main__":
    main()
