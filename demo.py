"""Run the actual model on synthetic inputs; these are not market results."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from model import GAS, OUTAGES, REGIONS, TARGET, TREE_INPUTS, fit_model, predict, prepare_data


def synthetic_history():
    rng = np.random.default_rng(42)
    utc = pd.date_range('2023-01-01', '2025-01-02', freq='h', inclusive='left', tz='America/New_York').tz_convert('UTC')
    local = utc.tz_convert('America/New_York')
    n = len(utc)
    annual = 2 * np.pi * local.dayofyear.to_numpy() / 365.2425
    daily = 2 * np.pi * local.hour.to_numpy() / 24
    frame = pd.DataFrame(dict(delivery_utc=utc.astype(str), opr_date=local.strftime('%Y-%m-%d'),
                              hour_beginning=local.hour, delivery_day_of_week=local.dayofweek,
                              is_weekend=local.dayofweek >= 5, modeling_day_allowed=True))
    frame['load_value'] = 90000 + 16000 * np.cos(annual) - 10000 * np.cos(daily) + rng.normal(0, 3000, n)
    frame['wind_value'] = 6000 + 2000 * np.sin(annual) + rng.uniform(0, 2500, n)
    frame['solar_value_provisional'] = 7000 * np.maximum(0, -np.cos(daily))
    for i, name in enumerate(GAS): frame[name] = 2.0 + i * .15 + .5 * np.cos(annual) + rng.uniform(0, .2, n)
    for i, name in enumerate(OUTAGES): frame[name] = 9000 + 2500 * np.sin(annual + i) + rng.uniform(0, 1500, n)
    for i, region in enumerate(REGIONS): frame[f'region_{region}'] = frame.load_value * (.05 + .015 * i) + rng.normal(0, 500, n)
    frame[TARGET] = 20 + .0005 * (frame.load_value - frame.wind_value - frame.solar_value_provisional - 60000) + 3 * frame[GAS[0]] + rng.normal(0, 2, n)
    return prepare_data(frame)


def synthetic_errors(history):
    rows = history[history.date.ge('2024-01-01') & history.date.lt('2025-01-01')].copy()
    rng = np.random.default_rng(7)
    errors = rows[['opr_date', 'delivery_utc']].copy()
    for name in TREE_INPUTS: errors[name] = rng.normal(size=len(rows))
    errors['target_residual'] = rng.normal(0, 1, len(rows))
    return errors


def synthetic_fallback(history):
    past = history[history.date.lt('2025-01-01')]
    calendar = past.groupby(['hour_beginning', 'delivery_day_of_week'])[TARGET].mean().rename('mean').reset_index()
    return dict(calendar_baseline=calendar.to_dict('records'), training_mean=[0, 0, 0],
                training_scale=[1, 1, 1], coefficients=[30.0] + [0.0] * 43)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('demo_output'))
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Choose a new output directory.')
    history = synthetic_history()
    model = fit_model(history, synthetic_errors(history), '2025-01-01', synthetic_fallback(history))
    inputs = history[history.opr_date.eq('2025-01-01')].copy()
    inputs[TARGET] = np.nan
    result = predict(model, inputs)
    assert len(result) == 24 and np.isfinite(result.prediction).all()
    args.output.mkdir(parents=True)
    result[['delivery_utc', 'opr_date', 'hour_beginning', 'regression', 'correction', 'prediction', 'route']].to_csv(args.output / 'synthetic_forecast.csv', index=False)
    summary = dict(data_type='synthetic demonstration only; not market performance', hours=len(result),
                   regression_fit_through=model['fit_max_date'], correction_fit_through=model['tree_fit_max_date'],
                   routes=result.route.value_counts().to_dict())
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
