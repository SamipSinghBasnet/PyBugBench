#!/usr/bin/env python3

"""
PHASE 9 – Build tables for academic report (Markdown + LaTeX)

Generates:
- phase9_table_overall_models.(md|tex)
- phase9_table_per_repo_best.(md|tex)
- phase9_table_imbalance_tuned.(md|tex)
- phase9_table_feature_ablation.(md|tex)
"""

from pathlib import Path
import pandas as pd

# -----------------------------------------------------------
# File paths
# -----------------------------------------------------------

RESULTS_DIR = Path("results")

PHASE4_COMPARISON = RESULTS_DIR / "phase4_all_models_comparison.csv"
PHASE5_PER_REPO = RESULTS_DIR / "phase5_per_repo_results.csv"
PHASE6_IMBALANCE = RESULTS_DIR / "phase6_imbalance_results.csv"
PHASE8_RESULTS = RESULTS_DIR / "phase8_feature_ablation_results.csv"   # FIXED

# -----------------------------------------------------------
# Helper – load CSV safely
# -----------------------------------------------------------

def safe_read_csv(path, desc):
    if not path.exists():
        print(f"[WARN] Missing {desc}: {path}")
        return None
    print(f"[OK] Loaded {desc} from {path}")
    return pd.read_csv(path)

# -----------------------------------------------------------
# Helper – save Markdown + LaTeX tables
# -----------------------------------------------------------

def save_markdown_and_latex(df, name, index=False):
    md_path = RESULTS_DIR / f"{name}.md"
    tex_path = RESULTS_DIR / f"{name}.tex"

    # Markdown
    md = df.to_markdown(index=index)
    md_path.write_text(md)
    print(f"[OK] Saved Markdown table → {md_path}")

    # LaTeX
    tex = df.to_latex(index=index, float_format="%.4f")
    tex_path.write_text(tex)
    print(f"[OK] Saved LaTeX table   → {tex_path}")

# -----------------------------------------------------------
# Table 1: Overall model comparison (Phase 4)
# -----------------------------------------------------------

def build_overall_model_table():
    df = safe_read_csv(PHASE4_COMPARISON, "Phase 4 model comparison")
    if df is None:
        return
    save_markdown_and_latex(df, "phase9_table_overall_models", index=False)

# -----------------------------------------------------------
# Table 2: Per-repository BEST models (Phase 5)
# -----------------------------------------------------------

def build_per_repo_table():
    df = safe_read_csv(PHASE5_PER_REPO, "Phase 5 per-repo results")
    if df is None:
        return

    # Keep only best F1 per repository
    df_best = df.sort_values(["repo", "f1"], ascending=[True, False]) \
                .groupby("repo").head(1)

    df_best = df_best[["repo", "model", "accuracy", "precision", "recall", "f1", "roc_auc"]]
    save_markdown_and_latex(df_best, "phase9_table_per_repo_best", index=False)

# -----------------------------------------------------------
# Table 3: Imbalance tuned models (Phase 6)
# -----------------------------------------------------------

def build_imbalance_table():
    df = safe_read_csv(PHASE6_IMBALANCE, "Phase 6 imbalance results")
    if df is None:
        return

    # Keep only the tuned variants
    df_tuned = df[df["variant"].str.contains("tuned")]
    save_markdown_and_latex(df_tuned, "phase9_table_imbalance_tuned", index=False)

# -----------------------------------------------------------
# Table 4: Feature ablation (Phase 8)
# -----------------------------------------------------------

def build_feature_ablation_table():
    df = safe_read_csv(PHASE8_RESULTS, "Phase 8 feature ablation")
    if df is None:
        return

    # Expect following columns: feature_set, model, f1, train_frac, seed, features
    # Some older versions might not have train_frac: handle gracefully
    if "train_frac" not in df.columns:
        df["train_frac"] = 1.0   # Default value

    # Sort: best F1, then small training fraction, then seed
    df_sorted = df.sort_values(["f1", "train_frac", "seed"], ascending=[False, True, True])

    save_markdown_and_latex(df_sorted, "phase9_table_feature_ablation", index=False)

# -----------------------------------------------------------
# Main
# -----------------------------------------------------------

def main():
    print("\n=== PHASE 9: Building tables for the paper/report ===\n")

    build_overall_model_table()
    build_per_repo_table()
    build_imbalance_table()
    build_feature_ablation_table()

    print("\n=== DONE (Phase 9 complete) ===\n")

if __name__ == "__main__":
    main()
