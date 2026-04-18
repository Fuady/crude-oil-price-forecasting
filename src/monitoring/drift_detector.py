"""
src/monitoring/drift_detector.py
----------------------------------
Feature and prediction drift monitoring for the oil price forecasting pipeline.

In production O&G trading environments:
  - Market regimes change (OPEC+ policy shifts, geopolitical events)
  - Feature distributions drift (macro regime changes, new data sources)
  - Model predictions drift (model staleness, regime change)

Run weekly:
  python src/monitoring/drift_detector.py

Or schedule with cron:
  0 7 * * 1 cd /path/to/project && python src/monitoring/drift_detector.py
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MONITORING_DIR = Path("monitoring/reports")
MONITORING_DIR.mkdir(parents=True, exist_ok=True)

# Key features to monitor (most informative for oil forecasting)
KEY_FEATURES_TO_MONITOR = [
    "rsi_14", "macd_hist", "bb_pct_b", "hvol_20d",
    "dxy", "dxy_roc_5d", "yield_10y", "yield_curve_slope",
    "sentiment_score", "sentiment_ma5",
    "eia_weekly_change_mb", "eia_crude_inventory_mb",
    "roc_1d", "roc_5d", "atr_pct",
    "log_return_1d", "price_vs_sma20",
]


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index. PSI > 0.25 → retrain needed."""
    expected = expected[np.isfinite(expected)]
    actual   = actual[np.isfinite(actual)]
    if len(expected) < 10 or len(actual) < 10:
        return 0.0

    breakpoints = np.unique(np.percentile(expected, np.linspace(0, 100, bins + 1)))
    if len(breakpoints) < 2:
        return 0.0

    exp_c = np.histogram(expected, bins=breakpoints)[0]
    act_c = np.histogram(actual,   bins=breakpoints)[0]

    exp_p = (exp_c / len(expected)).clip(min=1e-6)
    act_p = (act_c / max(len(actual), 1)).clip(min=1e-6)

    return float(np.sum((act_p - exp_p) * np.log(act_p / exp_p)))


def run_drift_analysis(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list,
    report_name: str = "drift",
) -> dict:
    """Compute PSI for each monitored feature."""
    results = {
        "report_name": report_name,
        "timestamp":   datetime.now().isoformat(),
        "n_reference": len(reference_df),
        "n_current":   len(current_df),
        "feature_drift": {},
        "drifted":     [],
        "critical":    [],
    }

    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            continue
        ref_vals = reference_df[feat].dropna().values
        cur_vals = current_df[feat].dropna().values
        psi = compute_psi(ref_vals, cur_vals)

        level = "critical" if psi > 0.25 else "moderate" if psi > 0.10 else "stable"
        results["feature_drift"][feat] = {
            "psi":        round(psi, 4),
            "level":      level,
            "ref_mean":   round(float(np.nanmean(ref_vals)), 4),
            "cur_mean":   round(float(np.nanmean(cur_vals)), 4),
            "mean_shift": round(float(abs(np.nanmean(cur_vals) - np.nanmean(ref_vals))), 4),
        }
        if psi > 0.10:
            results["drifted"].append(feat)
        if psi > 0.25:
            results["critical"].append(feat)

    all_psi = [v["psi"] for v in results["feature_drift"].values()]
    results["overall_psi_mean"] = round(np.mean(all_psi) if all_psi else 0.0, 4)
    results["n_drifted"]  = len(results["drifted"])
    results["n_critical"] = len(results["critical"])

    if results["n_critical"] > 0:
        results["recommendation"] = (
            f"RETRAIN REQUIRED — {results['n_critical']} critical features drifted: "
            f"{results['critical'][:3]}"
        )
    elif results["n_drifted"] > 3:
        results["recommendation"] = (
            "INVESTIGATE — Multiple features drifting. "
            "Check for macro regime change or OPEC policy shift."
        )
    else:
        results["recommendation"] = "No action needed. Model distribution stable."

    return results


def run_monitoring(instrument: str = "brent") -> dict:
    """Run full monitoring pipeline."""
    from src.features.feature_pipeline import load_features

    print(f"\n{'='*55}")
    print(f"  Drift Monitoring — {instrument.upper()}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    data = load_features(instrument)

    reference_df = data["train_df"]
    # Simulate "current" data as recent test period
    current_df   = data["test_df"].tail(60)

    avail_features = [f for f in KEY_FEATURES_TO_MONITOR
                      if f in reference_df.columns and f in current_df.columns]

    print(f"\n  Reference samples : {len(reference_df):,}")
    print(f"  Current samples   : {len(current_df):,}")
    print(f"  Features monitored: {len(avail_features)}")

    results = run_drift_analysis(
        reference_df, current_df, avail_features,
        report_name=f"{instrument}_{datetime.now().strftime('%Y%m%d')}"
    )

    print(f"\n  Overall PSI mean  : {results['overall_psi_mean']:.4f}")
    print(f"  Drifted features  : {results['n_drifted']}")
    print(f"  Critical features : {results['n_critical']}")
    print(f"\n  Recommendation: {results['recommendation']}")

    if results["drifted"]:
        print(f"\n  Drifted features:")
        for feat in results["drifted"][:8]:
            d = results["feature_drift"][feat]
            print(f"    {feat:<35} PSI={d['psi']:.4f} "
                  f"({d['ref_mean']:.3f} → {d['cur_mean']:.3f}) [{d['level'].upper()}]")

    # Save JSON report
    ts       = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = MONITORING_DIR / f"drift_{instrument}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Report saved: {out_path}")

    # Try Evidently HTML report
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=reference_df[avail_features].head(500),
            current_data=current_df[avail_features],
        )
        html_path = MONITORING_DIR / f"drift_{instrument}_{ts}.html"
        report.save_html(str(html_path))
        print(f"  Evidently HTML report: {html_path}")
    except Exception as e:
        print(f"  Evidently HTML skipped ({e}). JSON report saved above.")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="brent", choices=["brent", "wti"])
    args = parser.parse_args()
    run_monitoring(args.instrument)
    print("\n✓ Monitoring complete.")
