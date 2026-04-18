"""
src/models/train.py
--------------------
Training orchestrator — runs all models and logs to MLflow.

Usage:
    python src/models/train.py
    python src/models/train.py --model xgboost
    python src/models/train.py --model lstm
    python src/models/train.py --model all --instrument brent

View results:
    mlflow ui --port 5000
    Open http://localhost:5000
"""

import sys
import argparse
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.features.feature_pipeline import load_features
from src.models.baseline import NaiveForecaster, ARIMAForecaster
from src.models.xgboost_model import XGBoostOilForecaster
from src.models.lstm_model import LSTMTrainer, OilPriceDataset
from src.models.ensemble import EnsembleForecaster

MODELS_DIR  = Path("models")
PLOTS_DIR   = Path("models/plots")
MODELS_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

MLFLOW_EXPERIMENT = "Oil_Price_Forecasting"


# ---------------------------------------------------------------------------
# Helper plots
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(
    dates, actuals, preds, title: str, path: Path
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    axes[0].plot(dates, actuals, label="Actual", color="#378ADD", linewidth=1.2)
    axes[0].plot(dates, preds,   label="Predicted", color="#E24B4A",
                 linewidth=1.0, alpha=0.8, linestyle="--")
    axes[0].set_title(f"{title} — Price Forecast vs Actual")
    axes[0].legend()
    axes[0].set_ylabel("Brent Price ($/bbl)")

    errors = np.array(actuals) - np.array(preds)
    axes[1].bar(dates, errors, color=["#E24B4A" if e < 0 else "#1D9E75" for e in errors],
                alpha=0.6, width=1)
    axes[1].axhline(0, color="#888", linewidth=0.8)
    axes[1].set_title("Forecast Error")
    axes[1].set_ylabel("Error ($/bbl)")

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_training_curve(history: dict, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["train_loss"], label="Train", color="#378ADD")
    if history.get("val_loss"):
        ax.plot(history["val_loss"], label="Validation", color="#D85A30")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Huber Loss")
    ax.set_title(f"{title} — Training Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, n_top: int = 20, path: Path = None) -> None:
    importances = pd.Series(
        model.regressor.feature_importances_, index=model.feature_cols
    ).nlargest(n_top)
    fig, ax = plt.subplots(figsize=(10, 7))
    importances.sort_values().plot(kind="barh", ax=ax, color="#378ADD")
    ax.set_title(f"Top {n_top} Feature Importances (XGBoost Regressor)")
    ax.set_xlabel("Importance score")
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Train functions
# ---------------------------------------------------------------------------

def train_baselines(data: dict, instrument: str) -> dict:
    print("\n" + "─" * 55)
    print("  Training: Baseline Models")
    print("─" * 55)

    prices = data["train_df"]["close"]
    test_prices = data["test_df"]["close"]

    naive = NaiveForecaster()
    naive.fit(prices)

    # Quick walk-forward on test set
    actuals, naive_preds = [], []
    for i in range(1, min(len(test_prices), 200)):
        naive_preds.append(float(test_prices.iloc[i - 1]))
        actuals.append(float(test_prices.iloc[i]))

    naive_mae  = float(np.mean(np.abs(np.array(actuals) - np.array(naive_preds))))
    naive_rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(naive_preds)) ** 2)))
    naive_dir  = float(np.mean(
        np.sign(np.diff(actuals)) == np.sign(np.diff(naive_preds))
    ))

    print(f"  Naïve — MAE: ${naive_mae:.2f}/bbl | Dir: {naive_dir:.2%}")

    metrics = {
        "naive_price_mae_usd":  round(naive_mae, 4),
        "naive_price_rmse_usd": round(naive_rmse, 4),
        "naive_dir_accuracy":   round(naive_dir, 4),
    }

    # ARIMA (fast version on last 2 years)
    try:
        recent = pd.concat([data["train_df"]["close"].iloc[-504:],
                            data["val_df"]["close"]])
        arima = ARIMAForecaster(order=(2, 1, 2))
        arima_metrics = arima.evaluate_walk_forward(recent, horizon=1, min_train=252, step=10)
        print(f"  ARIMA — MAE: {arima_metrics['mae']:.6f}  Dir: {arima_metrics['dir_acc']:.2%}")
        metrics["arima_return_mae"]  = arima_metrics["mae"]
        metrics["arima_dir_accuracy"] = arima_metrics["dir_acc"]
    except Exception as e:
        print(f"  ARIMA skipped: {e}")

    with mlflow.start_run(run_name=f"baselines_{instrument}"):
        mlflow.log_params({"model_type": "baselines", "instrument": instrument})
        mlflow.log_metrics(metrics)

    joblib.dump(naive, MODELS_DIR / f"naive_{instrument}.joblib")
    return metrics


def train_xgboost(data: dict, instrument: str) -> dict:
    print("\n" + "─" * 55)
    print("  Training: XGBoost Multi-Factor Forecaster")
    print("─" * 55)

    feature_cols = data["feature_cols"]
    train_df     = data["train_df"]
    val_df       = data["val_df"]
    test_df      = data["test_df"]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {len(train_df):,} rows | Val: {len(val_df):,} | Test: {len(test_df):,}")

    model = XGBoostOilForecaster()
    model.fit(train_df, val_df, feature_cols)

    metrics = model.evaluate(test_df, test_df["close"])
    print(f"\n  Test results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<30} {v:.4f}")

    with mlflow.start_run(run_name=f"xgboost_{instrument}") as run:
        mlflow.log_params({
            "model_type":     "xgboost",
            "instrument":     instrument,
            "n_features":     len(feature_cols),
            "n_estimators":   model.reg_params["n_estimators"],
            "max_depth":      model.reg_params["max_depth"],
            "learning_rate":  model.reg_params["learning_rate"],
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        # Feature importance plot
        fi_path = PLOTS_DIR / f"xgb_feature_importance_{instrument}.png"
        plot_feature_importance(model, n_top=20, path=fi_path)
        mlflow.log_artifact(str(fi_path))

        # SHAP
        try:
            import shap
            shap_df = model.explain(test_df, max_samples=200)
            top15   = shap_df.abs().mean().nlargest(15).index.tolist()
            fig, ax = plt.subplots(figsize=(10, 7))
            shap_df[top15].abs().mean().sort_values().plot(
                kind="barh", ax=ax, color="#EF9F27"
            )
            ax.set_title("SHAP Feature Importance (Mean |Value|)")
            shap_path = PLOTS_DIR / f"xgb_shap_{instrument}.png"
            plt.tight_layout()
            plt.savefig(shap_path, dpi=120, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(shap_path))
        except Exception:
            pass

        # Forecast vs actual plot
        xgb_ret  = model.predict_return(test_df)
        pred_px  = test_df["close"].values * (1 + xgb_ret)
        true_px  = test_df["close"].shift(-1).fillna(method="ffill").values

        fc_path = PLOTS_DIR / f"xgb_forecast_{instrument}.png"
        plot_forecast_vs_actual(
            test_df.index[:200], true_px[:200], pred_px[:200],
            "XGBoost", fc_path
        )
        mlflow.log_artifact(str(fc_path))

        mlflow.sklearn.log_model(
            model.regressor, "xgboost_regressor",
            registered_model_name=f"oil_xgboost_{instrument}"
        )
        run_id = run.info.run_id

    model.save(MODELS_DIR / f"xgboost_{instrument}.joblib")
    print(f"  Saved: models/xgboost_{instrument}.joblib")
    print(f"  MLflow run: {run_id}")
    return metrics


def train_lstm(data: dict, instrument: str) -> dict:
    print("\n" + "─" * 55)
    print("  Training: LSTM with Temporal Attention")
    print("─" * 55)

    feature_cols = data["feature_cols"]
    train_df     = data["train_df"]
    val_df       = data["val_df"]
    test_df      = data["test_df"]

    LOOKBACK = 60
    HORIZON  = 1

    # Use a subset of features for LSTM (top 30 by variance)
    feat_var = train_df[feature_cols].var().nlargest(30).index.tolist()
    print(f"  LSTM features: {len(feat_var)} (top 30 by variance)")

    train_ds = OilPriceDataset(train_df, feat_var, lookback=LOOKBACK, horizon=HORIZON)
    val_ds   = OilPriceDataset(val_df,   feat_var, lookback=LOOKBACK, horizon=HORIZON)
    test_ds  = OilPriceDataset(test_df,  feat_var, lookback=LOOKBACK, horizon=HORIZON)

    print(f"  Train windows: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

    trainer = LSTMTrainer(n_features=len(feat_var), lookback=LOOKBACK)
    trainer.feature_cols = feat_var

    print("\n  Training (may take several minutes)...")
    trainer.fit(train_ds, val_ds, epochs=60, lr=5e-4, batch_size=64,
                patience=12, verbose=True)

    metrics = trainer.evaluate(test_ds)
    print(f"\n  Test results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<30} {v:.4f}")

    with mlflow.start_run(run_name=f"lstm_{instrument}"):
        mlflow.log_params({
            "model_type":   "lstm",
            "instrument":   instrument,
            "n_features":   len(feat_var),
            "lookback":     LOOKBACK,
            "hidden_size":  128,
            "epochs":       len(trainer.history["train_loss"]),
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

        curve_path = PLOTS_DIR / f"lstm_training_curve_{instrument}.png"
        plot_training_curve(trainer.history, "LSTM", curve_path)
        mlflow.log_artifact(str(curve_path))

    trainer.save(MODELS_DIR / f"lstm_{instrument}.pt")
    print(f"  Saved: models/lstm_{instrument}.pt")
    return metrics


def train_ensemble(data: dict, instrument: str,
                   xgb_metrics: dict = None, lstm_metrics: dict = None) -> dict:
    print("\n" + "─" * 55)
    print("  Training: Weighted Ensemble")
    print("─" * 55)

    xgb_path  = MODELS_DIR / f"xgboost_{instrument}.joblib"
    lstm_path = MODELS_DIR / f"lstm_{instrument}.pt"

    ensemble = EnsembleForecaster()

    xgb_model = None
    if xgb_path.exists():
        xgb_model = XGBoostOilForecaster.load(xgb_path)
        print(f"  Loaded XGBoost from {xgb_path}")
    else:
        print("  XGBoost model not found — skipping ensemble")
        return {}

    lstm_trainer = None
    if lstm_path.exists():
        lstm_trainer = LSTMTrainer.load(lstm_path)
        print(f"  Loaded LSTM from {lstm_path}")

    ensemble.set_models(xgb_model, lstm_trainer)

    # Optimize weights on validation set
    val_df     = data["val_df"]
    feature_cols = xgb_model.feature_cols
    xgb_val_preds = xgb_model.predict_return(val_df)

    if lstm_trainer is not None:
        from src.models.lstm_model import OilPriceDataset
        val_ds = OilPriceDataset(
            val_df, lstm_trainer.feature_cols, lookback=60
        )
        from torch.utils.data import DataLoader
        import torch
        loader = DataLoader(val_ds, batch_size=128, shuffle=False)
        lstm_trainer.model.eval()
        lstm_val_preds = []
        with torch.no_grad():
            for Xb, _ in loader:
                pred, _ = lstm_trainer.model(Xb.to(lstm_trainer.device))
                lstm_val_preds.extend(pred.cpu().numpy()[:, 0])
        lstm_val_preds = np.array(lstm_val_preds)
        n = min(len(xgb_val_preds), len(lstm_val_preds))
        ensemble.optimize_weights(val_df.iloc[:n], xgb_val_preds[:n], lstm_val_preds[:n])
    else:
        ensemble.xgb_weight  = 1.0
        ensemble.lstm_weight = 0.0

    metrics = ensemble.evaluate(data["test_df"], data["test_df"]["close"])
    print(f"\n  Ensemble test results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<30} {v:.4f}")

    with mlflow.start_run(run_name=f"ensemble_{instrument}"):
        mlflow.log_params({
            "model_type":  "ensemble",
            "instrument":  instrument,
            "xgb_weight":  ensemble.xgb_weight,
            "lstm_weight": ensemble.lstm_weight,
        })
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})

    ensemble.save(MODELS_DIR / f"ensemble_weights_{instrument}.joblib")
    print(f"  Saved: models/ensemble_weights_{instrument}.joblib")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instrument", default="brent", choices=["brent", "wti"])
    parser.add_argument("--model", default="all",
                        choices=["all", "baseline", "xgboost", "lstm", "ensemble"])
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Oil Price Forecasting — Model Training")
    print(f"  Instrument: {args.instrument.upper()} | Models: {args.model}")
    print(f"{'='*55}")

    print("\nLoading feature matrices...")
    data = load_features(args.instrument)
    print(f"  Train: {len(data['train_df']):,} | Val: {len(data['val_df']):,} | Test: {len(data['test_df']):,}")
    print(f"  Features: {len(data['feature_cols'])}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    results = {}

    if args.model in ("all", "baseline"):
        results["baseline"] = train_baselines(data, args.instrument)
    if args.model in ("all", "xgboost"):
        results["xgboost"] = train_xgboost(data, args.instrument)
    if args.model in ("all", "lstm"):
        results["lstm"] = train_lstm(data, args.instrument)
    if args.model in ("all", "ensemble"):
        results["ensemble"] = train_ensemble(data, args.instrument)

    # Summary
    print(f"\n{'='*55}")
    print("  RESULTS SUMMARY")
    print(f"{'='*55}")
    for model_name, metrics in results.items():
        print(f"\n  {model_name.upper()}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"    {k:<35} {v}")

    summary_path = MODELS_DIR / f"training_summary_{args.instrument}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Training complete.")
    print(f"  View in MLflow: mlflow ui --port 5000")
    print(f"  Launch API: uvicorn src.serving.api:app --reload --port 8000")
    print(f"  Launch Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
