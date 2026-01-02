import numpy as np
import pandas as pd
from pathlib import Path

# Paths
DATA_CSV = Path("data/python_bug_data.csv")
OUT_NPZ = Path("data/codebert_embeddings.npz")

# How many samples to use for the deep model part
N_SAMPLES = 10000          # keep small so everything is fast
EMBED_DIM = 768            # CodeBERT hidden size (so shapes match later)


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")

    # Load full dataset
    df = pd.read_csv(DATA_CSV)

    # Sample subset for the “deep” model part
    if len(df) > N_SAMPLES:
        df = df.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

    print(f"Using {len(df)} rows for *simulated* CodeBERT embeddings.")

    # Labels
    y = df["buggy"].astype(int).values

    # Simple metrics (same as earlier phases: just LOC for now)
    X_metrics = df[["loc"]].astype(float).values

    # === IMPORTANT PART ===
    # Instead of running real CodeBERT (which is hanging on this Mac),
    # we simulate embeddings with random vectors that have the same shape
    # as real CodeBERT outputs: [N, 768].
    rng = np.random.default_rng(42)
    X_code = rng.standard_normal(size=(len(df), EMBED_DIM)).astype(np.float32)

    print(f"Simulated code embedding shape: {X_code.shape}")
    print(f"Metrics shape: {X_metrics.shape}")
    print(f"Labels shape:   {y.shape}")

    # Save in the format expected by train_phase3_codebert_models.py
    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, X_code=X_code, X_metrics=X_metrics, y=y)

    print(f"\n✅ Saved *simulated* CodeBERT embeddings → {OUT_NPZ}")


if __name__ == "__main__":
    main()
