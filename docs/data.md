# Data sources and prepared input contract

## Public distribution

Bulk source datasets, raw API responses, local source documents, and prepared historical rows are not included. Public accessibility of a feed does not establish redistribution permission. Original code, aggregate diagnostics, and figures are included; synthetic demo data are generated locally. No API key is needed by the compact model or demo.

PJM source discovery starts at [Data Miner 2](https://dataminer2.pjm.com/) and the [API portal](https://api.pjm.com/). Relevant feed families are day-ahead hourly LMPs (`da_hrl_lmps`), forecast load (`load_frcstd_7_day`), hourly wind and solar forecasts, and forecast generation outages (`frcstd_gen_outages`). Raw API responses must be converted to the prepared contract below; the runner is not a raw-feed downloader. Gas indexes require a separately licensed or otherwise permitted source.

## Files expected by the research runner

`--data-dir` points to a directory containing:

- `history.csv.gz`: prepared hourly features and available historical labels.
- `historical_errors.csv.gz`: earlier held-out regression errors and tree inputs.
- `fallback.json`: frozen calendar/core fallback parameters.
- `oct27_inputs.csv`: prepared features for the default October 27, 2026 run.
- `reference_2025.csv.gz`: reference hourly backtest predictions.
- `reference_20261026.csv` and `reference_20261027.csv`: reference daily forecasts, when available.
- `manifest.json`: a `files_sha256` mapping from each included filename to its SHA-256 checksum. The runner verifies this mapping before fitting.

Reference files are used to check predictions after fitting, not as model features. Historical-error rows are model-training observations and are distinct from the prediction references.

## Hourly feature columns

Required identifiers and calendar flags are `delivery_utc`, `opr_date`, `hour_beginning`, `delivery_day_of_week`, `is_weekend`, and `modeling_day_allowed`. The target is `system_energy_price_da`; inference rows can leave it missing. Weekdays use Monday 0 through Sunday 6.

Core inputs are `load_value`, `wind_value`, and `solar_value_provisional`. Six gas inputs are `gas_IGBDC21`, `gas_IGBDZ21`, `gas_IGBDX21`, `gas_IGBEL21`, `gas_IGBDE21`, and `gas_IGBEK21`. Outages are `outage_other_total_outages` and `outage_western_total_outages`. Regional columns are `region_DOMINION`, `region_COMED`, `region_ATSI`, `region_PPL`, and `region_PSE&G`. Optional inputs can be missing, but columns must exist for route selection.

Load, remaining demand, and outage quantities are represented in MW. Renewable hourly forecast quantities retain the adopted one-hour interval convention before subtraction. Gas prices retain the source dollar-per-MMBtu convention. Prices and all error metrics are dollars per MWh.

Historical-error rows require `delivery_utc`, `opr_date`, `base_fit_max_date`, `system_energy_price_da`, `base`, and `target_residual`, plus the eight `TREE_INPUTS` named in `model.py`. `target_residual` must equal actual minus `base`; `base_fit_max_date` must precede the predicted local date. All tree input values must be finite.

## Retained timestamp conventions

UTC is the physical-hour key. `America/New_York` defines delivery dates and Eastern prevailing local hours. For ordinary days, hour beginning 0 corresponds to HE1; the reported peak block HE8–23 contains 16 hours. The forecast runner checks that local timestamps match the requested date and hour.

Spring 23-hour days remain in the historical calendar. Entire autumn 25-hour reversal days are excluded under the retained source-label policy; the 2025 annual score therefore covers 8,735 rather than 8,760 hours. The daily export runner requires exactly 24 hours and rejects nonordinary forecast days.

Load and wind use the adopted hour-ending-minus-one convention. Solar retains the project's `hour_ending % 24` working mapping. Gas uses power-delivery day minus one flow date and the original report-date filter. Outage and renewable availability conventions are preserved from the prepared history. Latest API revisions should not be represented as independently reconstructed historical forecast vintages.

The October 27 inputs carried Chicago at 2.295 and Transco Zone 6 non-NY at 1.125 because those two quotes were missing from that input set. This is a documented input fallback, not a newly observed quote.

## Reproducibility boundary

The public synthetic demonstration runs independently. Reproducing the reported research metrics requires the original permitted prepared bundle. The compact release refits the final regression and tree, but starts after raw cleaning and residual-history generation. It does not claim that downloading current PJM responses alone reconstructs that historical bundle.
