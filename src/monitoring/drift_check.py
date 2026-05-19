"""
Data drift detection using Evidently AI.

Run periodically (cron / Airflow) to compare a window of recent inference inputs
to the training data distribution. Flags drift via the Wasserstein distance and
per-feature statistical tests.

Usage:
    python src/monitoring/drift_check.py \
        --reference data/processed/train.npz \
        --current data/inference_logs/last_7_days.npz \
        --output reports/drift_report.html
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def beats_to_dataframe(beats: np.ndarray, n_features: int = 20) -> pd.DataFrame:
    """Reduce each 360-sample beat to summary stats so Evidently can analyze it.

    Why this matters: drift detection on 360 raw timestep features is noisy.
    Summary statistics (mean, std, peaks) are more interpretable signals of
    real distribution change.
    """
    summary = pd.DataFrame({
        "mean": beats.mean(axis=1),
        "std": beats.std(axis=1),
        "min": beats.min(axis=1),
        "max": beats.max(axis=1),
        "range": beats.max(axis=1) - beats.min(axis=1),
        "abs_mean": np.abs(beats).mean(axis=1),
        # Sample evenly-spaced timesteps as features (preserves morphology signal)
    })
    sample_idxs = np.linspace(0, beats.shape[1] - 1, 10).astype(int)
    for i, idx in enumerate(sample_idxs):
        summary[f"sample_{idx}"] = beats[:, idx]
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="Path to reference .npz")
    parser.add_argument("--current", required=True, help="Path to current .npz")
    parser.add_argument("--output", default="reports/drift_report.html")
    args = parser.parse_args()

    ref = np.load(args.reference)["X"]
    cur = np.load(args.current)["X"]

    ref_df = beats_to_dataframe(ref)
    cur_df = beats_to_dataframe(cur)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref_df, current_data=cur_df)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report.save_html(args.output)
    print(f"✅ Drift report saved to {args.output}")


if __name__ == "__main__":
    main()
