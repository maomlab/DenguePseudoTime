"""
Build the image-level UMAP table used for the Figure 4 UMAP panels.

Pipeline (same logic, parameters and column order as the original
Dengue_umap_analysis.ipynb):

  1. Load the 21 per-model score files (DR_scored_*.csv, one score per image per
     model; 15 pairwise models + 6 "all vs X" models).
  2. Summarise per image: votes per timepoint (score >= 0.5 votes for the first
     timepoint of the model, otherwise the second), pred_mode (timepoint with the
     most votes) and pred_entropy (base-2 entropy of the vote distribution).
  3. Standardise the model scores and embed with UMAP
     (n_neighbors=30, min_dist=0.1, random_state=42).
  4. Join plate metadata (compound, dose) on well ID.

Run from this folder:   python 01_build_umap_table.py
Output:                 data/umap_table_rerun.csv

NOTE: data/umap_table.csv is the archived table (coordinates from the original
analysis run) that the figure notebook reads. UMAP embeddings are not guaranteed
to be bit-identical across umap-learn/numba versions, so a fresh run of this
script reproduces the same overall layout but not the exact coordinates.
"""
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import umap
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
SCORES_DIR = HERE.parent / "Jill_analysis"             # DR_scored_*.csv
METADATA_CSV = HERE / "data" / "DR_plate_metadata.csv"     # well -> compound / dose
OUT_CSV = HERE / "data" / "umap_table_rerun.csv"

THRESHOLD = 0.5            # vote threshold for score -> timepoint vote
UMAP_KWARGS = dict(n_neighbors=30, min_dist=0.1, random_state=42)


# ---------------------------------------------------------------------------
# 1. Load per-model scores
# ---------------------------------------------------------------------------
files = sorted(glob(str(SCORES_DIR / "DR_scored_*.csv")))
assert len(files) == 21, f"expected 21 score files, found {len(files)}"

model_preds = []
for path in files:
    model_name = Path(path).name.replace("DR_scored_", "").replace(".csv", "")  # e.g. 0_12, all_vs_48
    df = pd.read_csv(path)
    model_preds.append(df[["score"]].rename(columns={"score": f"pred_{model_name}"}))
df_preds = pd.concat(model_preds, axis=1)

# Image identifiers (row order is identical across all score files)
first = pd.read_csv(files[0])
cell_metadata_df = first[["Image_Metadata_WellID", "ImageNumber"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Votes, mode and entropy
# ---------------------------------------------------------------------------
def parse_timepoints(col_name):
    col = col_name.replace("pred_", "")
    if col.startswith("all_vs_"):
        parts = col.split("_vs_")
        return parts[0], parts[1]          # 'all', '0'/'12'/...
    parts = col.split("_")
    return parts[0], parts[1]


timepoints = set()
for col in df_preds.columns:
    timepoints.update(parse_timepoints(col))


def sort_key(x):
    try:
        return int(x)
    except ValueError:
        return 1e6                          # 'all' last


timepoints = sorted(timepoints, key=sort_key)

vote_counts = pd.DataFrame(0, index=df_preds.index, columns=timepoints)
for col in df_preds.columns:
    tp1, tp2 = parse_timepoints(col)
    first_wins = df_preds[col] >= THRESHOLD          # NaN scores vote for tp2, as in the original
    vote_counts[tp1] += first_wins.astype(int)
    vote_counts[tp2] += (~first_wins).astype(int)

pred_mode = vote_counts.idxmax(axis=1)
vote_probs = vote_counts.div(vote_counts.sum(axis=1), axis=0).fillna(0)
pred_entropy = vote_probs.apply(lambda row: entropy(row, base=2), axis=1)

summary_df = pd.DataFrame({"pred_mode": pred_mode, "pred_entropy": pred_entropy}).join(vote_counts)
summary_df = pd.concat(
    [cell_metadata_df, summary_df.reset_index(drop=True), df_preds.reset_index(drop=True)], axis=1
)


# ---------------------------------------------------------------------------
# 3. UMAP on standardised model scores
# ---------------------------------------------------------------------------
# NOTE: as in the original analysis, the input is every column starting with
# "pred_" except pred_entropy. That includes pred_mode (a timepoint label that
# is converted to a number; the non-numeric label 'all' becomes 0). This is
# kept as-is so the embedding matches the one used for the manuscript figures.
score_cols = [c for c in summary_df.columns if c.startswith("pred_") and c != "pred_entropy"]
X = summary_df[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X_scaled = StandardScaler().fit_transform(X)

X_umap = umap.UMAP(**UMAP_KWARGS).fit_transform(X_scaled)
summary_df["UMAP_1"] = X_umap[:, 0]
summary_df["UMAP_2"] = X_umap[:, 1]


# ---------------------------------------------------------------------------
# 4. Join plate metadata (compound, dose) on well ID
# ---------------------------------------------------------------------------
metadata_df = pd.read_csv(METADATA_CSV)

df = summary_df.merge(
    metadata_df,
    left_on="Image_Metadata_WellID",
    right_on="WellID",
    how="left",
)

df["ImageNumber"] = df["ImageNumber"].astype(int)
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {OUT_CSV} {df.shape}")
