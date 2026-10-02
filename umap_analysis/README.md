# umap_analysis

Image-level UMAP panels for Figure 4: images colored by predicted timepoint, LY411575 dose, and LY411575 vs pralatrexate.

## Contents

| File | Purpose |
|---|---|
| `DENV_TimeSeries_DR_UMAP_Analysis.ipynb` | Generates the three UMAP panels (SVG, written to `figures/`) |
| `01_build_umap_table.py` | Rebuilds the image-level UMAP table from the per-model scores in `../Jill_analysis/` |
| `data/umap_table.csv` | Archived image-level table used for the figures: UMAP coordinates, compound, dose (2,285 images) |
| `data/DR_plate_metadata.csv` | Well-to-compound/dose mapping used by `01_build_umap_table.py` |
| `figures/` | Output SVGs |
| `requirements.txt` | Python dependencies |

The notebook also reads `../product/expected_timepoint_all20260306.tsv` (predicted timepoint per compound and dose from the likelihood model, `../DENV_TimeOfArrest_v2.qmd`).

## Running

```
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute DENV_TimeSeries_DR_UMAP_Analysis.ipynb
```

Run from this folder; all paths are relative.

## How the UMAP table is built

`01_build_umap_table.py` loads the 21 per-model score files, counts votes per timepoint (score >= 0.5 votes for the first timepoint of a model, otherwise the second), computes the mode and base-2 entropy of the votes, standardizes the model scores, and embeds them with UMAP (`n_neighbors=30`, `min_dist=0.1`, `random_state=42`). As in the original analysis, the UMAP input is every column starting with `pred_` except `pred_entropy`, which includes `pred_mode`.

UMAP layouts are not guaranteed to be identical across umap-learn/numba versions. A re-run reproduces the same overall layout as `data/umap_table.csv` (Procrustes disparity 0.025; about 74% of each image's 15 nearest neighbors are shared) but not the exact coordinates, so the figures read the archived table.

## Notes

- The likelihood-model file spells three compounds differently from the plate metadata (`IMD 0354`, `remdesivir`, `SB 271046` vs `IMD-0354`, `Remdesivir`, `SB-271046`). The notebook harmonizes these names so every image is colored in the timepoint panel (the notebook asserts that no image is left without a timepoint).
