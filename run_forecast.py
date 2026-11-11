"""Fit on earlier observations and export one day's 24 hourly forecasts."""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import TARGET, TREE_INPUTS, fit_model, predict, prepare_data

HERE = Path(__file__).resolve().parent


def backtest_2025(history, errors, fallback, output, data_dir):
    """Refit before each month, then score that month's available prices."""
    predictions = []
    for cutoff in pd.date_range("2025-01-01", "2025-12-01", freq="MS"):
        rows = history[
            history.date.dt.to_period("M").eq(cutoff.to_period("M"))
            & history[TARGET].notna()
        ].copy()
        actual = rows[TARGET].copy()
        rows[TARGET] = np.nan
        model = fit_model(history, errors, cutoff, fallback)
        result = predict(model, rows)
        result["actual"] = actual
        predictions.append(result)
        print(f"Finished {cutoff:%Y-%m}: {len(result)} scored hours.", flush=True)
    result = pd.concat(predictions, ignore_index=True)
    difference = result.prediction - result.actual
    metrics = {
        "hours": len(result), "MAE": float(difference.abs().mean()),
        "RMSE": float(np.sqrt(np.mean(difference ** 2))),
        "bias": float(difference.mean()),
    }
    reference = pd.read_csv(data_dir / "reference_2025.csv.gz")
    check = result[["delivery_utc", "prediction"]].merge(reference, on="delivery_utc", validate="one_to_one")
    if len(check) != len(reference) or not np.allclose(check.prediction, check.reference_prediction, atol=1e-6, rtol=0):
        raise ValueError("The 2025 backtest differs from the preserved results.")
    output.mkdir(parents=True)
    result[["delivery_utc", "opr_date", "hour_beginning", "actual", "regression", "correction", "prediction", "route"]].to_csv(output / "backtest.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("All historical predictions match the preserved 2025 results.")
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-10-27")
    parser.add_argument("--data-dir", type=Path, default=HERE / "data", help="Prepared input bundle; see docs/data.md")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--backtest-2025", action="store_true")
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    cutoff = pd.Timestamp(args.date)
    if cutoff < pd.Timestamp("2025-01-01"):
        raise ValueError("Use a date from 2025 onward; the frozen core fallback was fitted on 2023–2024.")
    output = args.output or HERE / "results" / ("backtest_2025" if args.backtest_2025 else args.date)
    if output.exists():
        raise FileExistsError(f"Choose a new output folder: {output}")

    manifest = json.loads((data_dir / "manifest.json").read_text())
    for filename, expected in manifest["files_sha256"].items():
        actual = hashlib.sha256((data_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"The supplied reference data changed: {filename}")
    history = prepare_data(pd.read_csv(data_dir / "history.csv.gz"))
    errors = pd.read_csv(data_dir / "historical_errors.csv.gz")
    if not errors.delivery_utc.is_unique:
        raise ValueError("The historical error file contains duplicate hours.")
    if not np.allclose(errors[TARGET] - errors.base, errors.target_residual):
        raise ValueError("Historical errors must equal actual minus held-out regression.")
    if not pd.to_datetime(errors.base_fit_max_date).lt(pd.to_datetime(errors.opr_date)).all():
        raise ValueError("Historical regression fits must end before their prediction dates.")
    if not np.isfinite(errors[TREE_INPUTS + ["target_residual"]].to_numpy()).all():
        raise ValueError("Historical correction inputs contain missing values.")
    fallback = json.loads((data_dir / "fallback.json").read_text())
    if args.backtest_2025:
        backtest_2025(history, errors, fallback, output, data_dir)
        return

    inputs = pd.read_csv(args.inputs or data_dir / "oct27_inputs.csv")
    inputs = inputs[pd.to_datetime(inputs.opr_date).eq(cutoff)].copy()
    if len(inputs) != 24 or sorted(inputs.hour_beginning) != list(range(24)):
        raise ValueError("This runner expects one ordinary 24-hour delivery day.")
    inputs[TARGET] = np.nan
    inputs = prepare_data(inputs)
    utc = pd.to_datetime(inputs.delivery_utc, utc=True)
    local = utc.dt.tz_convert("America/New_York")
    if not local.dt.hour.eq(inputs.hour_beginning).all() or not local.dt.strftime("%Y-%m-%d").eq(args.date).all():
        raise ValueError("The delivery timestamps do not match the requested Eastern day.")

    print(f"Fitting the retained model using available labels before {args.date}...")
    model = fit_model(history, errors, cutoff, fallback)
    result = predict(model, inputs)
    result["HE"] = result.hour_beginning + 1
    forecast = result[["opr_date", "HE", "prediction"]].rename(
        columns={"prediction": "price_dollars_per_MWh"}
    )
    peak = result.loc[result.HE.between(8, 23), "prediction"].mean()
    summary = {
        "delivery_date": args.date,
        "regression_fit_through": model["fit_max_date"],
        "correction_fit_through": model["tree_fit_max_date"],
        "full_regression_training_hours": model["fit_rows"],
        "historical_correction_hours": model["residual_rows"],
        "daily_average": float(result.prediction.mean()),
        "peak_HE8_23_average": float(peak),
        "routes": result.route.value_counts().to_dict(),
    }
    output.mkdir(parents=True)
    forecast.to_csv(output / "forecast.csv", index=False)
    result[["opr_date", "HE", "regression", "correction", "prediction", "route"]].to_csv(
        output / "hourly_breakdown.csv", index=False
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    reference = data_dir / f"reference_{cutoff:%Y%m%d}.csv"
    if reference.exists():
        saved = pd.read_csv(reference)
        difference = np.max(np.abs(forecast.price_dollars_per_MWh - saved.price_dollars_per_MWh))
        if difference > 1e-6:
            raise ValueError(f"The recreated forecast differs from the saved forecast by {difference}.")
        print("All 24 predictions match the saved forecast within $0.000001/MWh.")
    print(json.dumps(summary, indent=2))
    print(f"Forecast saved to {output / 'forecast.csv'}")


if __name__ == "__main__":
    main()
