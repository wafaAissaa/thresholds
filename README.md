# Thresholds

Small research workspace to annotate a French corpus with linguistic features and compute difficulty
thresholds per level (N1-N4) using different statistical strategies (IQR, logistic regression, percentiles).

## Overview

- `annotate_corpus.py`: Sends a single hardcoded text to a local phenomena server and saves the JSON response.
- `annotate_corpus_API.py`: Batch-annotates texts from `Qualtrics_Annotations_B.csv` via the UCLouvain API and
  writes one JSON per text in `outputs/`.
- `compute_thresholds_IQR.py`: Builds feature groups, loads annotation outputs, computes per-level thresholds
  using IQR bounds, saves `results/thresholds_IQR.json` and a CSV table, and optionally distributions.
- `compute_thresholds_LogReg.py`: Fits a logistic regression per feature to derive a decision threshold
  (0.5) between levels; N4 uses IQR-style bounds; outputs to `results_depricated/`.
- `compute_thresholds_Percentile.py`: Computes thresholds using symmetric percentiles for q = 5..95 and
  writes one JSON/CSV per q in `results/`.

## Inputs

- `Qualtrics_Annotations_B.csv`: Source dataset (tab-delimited).
- `outputs/*.json`: Per-text feature annotations (generated).

## Outputs

- `results/thresholds_IQR.json` and `results/thresholds_IQR.csv`
- `results/distributions.json`
- `results/thresholds_Percentile_*.json` and `results/thresholds_Percentile_*.csv`
- Deprecated artifacts in `outputs_depricated*` and `results_depricated*`

## Typical workflow

1) Generate annotations

```bash
python annotate_corpus_API.py
```

2) Compute thresholds (IQR)

```bash
python compute_thresholds_IQR.py
```

3) (Optional) Compute thresholds with logistic regression or percentiles

```bash
python compute_thresholds_LogReg.py
python compute_thresholds_Percentile.py
```

## Notes

- `annotate_corpus.py` is a small local test client and expects a phenomena server running at the configured IP.
- The UCLouvain annotator endpoint is hardcoded in `annotate_corpus_API.py`.
