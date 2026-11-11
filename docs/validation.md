# Validation and results

## Replayed research results

The packaged runner was executed on November 11, 2026 with the original prepared data, using its pinned model environment. The full monthly 2025 replay matched all 8,735 preserved predictions within $0.000001/MWh. Independently recalculated metrics were MAE 7.223890692527511, RMSE 13.594886060791266, and signed bias −2.289121946716256 dollars per MWh.

The same replay refits each monthly regression and correction tree. Historical held-out residuals are retained intermediate inputs. This establishes reproducibility of the final fitting/prediction stage, not regeneration of the complete upstream research history.

The October 27, 2026 forecast also matched all 24 reference prices within $0.000001/MWh. Its daily forecast average was 57.00913497093425 and HE8–23 average 60.916051718573875. Twenty-two hours used the hybrid and two used geographic regression.

## Compared populations

`results/metrics_2025.csv` preserves the original independently audited metrics:

- All-policy regression: 8,735 hours, MAE 7.673425312333071, RMSE 14.45258375614207.
- All-policy final hybrid: 8,735 hours, MAE 7.223890692527499, RMSE 13.59488606079124.
- Complete-input final hybrid: 7,699 hours, MAE 7.2116309965978225, RMSE 13.201285077468564.

The first two share the same timestamps. The complete-input line must not be compared as if it used all 8,735 hours. Tiny final decimal differences between the independent saved-model audit and the new fit are floating-point effects.

`results/metrics_20261027.csv` preserves the separate one-day evaluation, including the overnight, morning, midday, evening, late, and peak blocks. The morning HE7–10 block contributed about 74.3% of that day's squared error. A near-correct daily or peak average can conceal large intraday errors.

## Existing independent audit

The retained `original_audit.json` records twelve independent ridge solution checks, twelve regression replays, twelve depth-six tree replays, chronology checks, signed residual identity, fallback routing, and unchanged source hashes. It reports no tested numerical implementation or metric-calculation defect. `original_runner_verification.json` records the original compact runner verification. These are preserved project audit records, distinct from the new packaging checks.

## Public checks

`python -m unittest discover -s tests -v` checks target-independent inference, finite forecasts and all missing-input routes, fit-date boundaries, same-day ramp adjacency across missing hours, and linear natural-spline tails. `python demo.py` tests fitting/prediction on synthetic history without source data. GitHub Actions runs both with Python 3.12.

## Interpretation

These are monthly chronological development-backtest results. Architecture and feature choices were informed by 2025 diagnostics, so they are not an unbiased estimate from an untouched holdout. Hourly observations are dependent, and large spikes can dominate RMSE. The October 27 example is a separate single-day forecast, not an annual validation substitute. No revenue or trading-performance claims are made.
