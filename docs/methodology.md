# Model specification

## Target and forecast clock

The target is hourly PJM day-ahead system energy price, in dollars per MWh. The underlying energy component is common across the observed pricing locations; using one target per physical hour avoids duplicating the same outcome across nodes. Congestion and marginal losses are separate components of nodal LMP and are not the prediction target.

The research backtest refits before each calendar month of 2025, using earlier labels. The separate October 27, 2026 run fits through October 26. Prepared history contains 2023–2025 price labels plus September–October 26, 2026 labels; January–August 2026 feature rows have no price labels and do not enter supervised fitting.

## Seasonal regression backbone

Remaining demand equals load minus wind minus solar. Each season has its own natural cubic spline with boundary knots at its training minimum/maximum and interior knots at training P30, P50, P80, P95, and P98. The tails are linear. Winter is December–February, spring March–May, summer June–August, and autumn September–November.

The 89 predictors comprise 40 calendar columns, 24 spline/linear remaining-demand terms, six gas prices, two outage terms, five regional levels, ten regional hinges, and two ramps. The intercept is additional. Calendar columns encode hour, weekday, and month with reference categories. Numerical terms are standardized with training-history moments.

Coefficients minimize weighted squared error plus ridge penalties. Observations lose half their weight every 365.2425 days; weights are normalized to mean one. The intercept is unpenalized. Ordinary standardized terms use penalty 10; the four additional regional levels beyond Dominion use 100,000; the ten regional hinge terms use 1,000,000. The implementation cross-checks its normal-equation solution against an augmented least-squares solve.

Six additive gas indexes are retained: Eastern Gas South (IGBDC21), MichCon (IGBDZ21), Chicago city-gates (IGBDX21), Transco Zone 6 non-NY (IGBEL21), Columbia Appalachia (IGBDE21), and TETCO M3 (IGBEK21). These are correlated fuel-cost proxies, not plant-level delivered costs. Outages use Western and Other totals; the redundant RTO sum is excluded.

## Geography and trajectory

For Dominion, ComEd, ATSI, PPL, and PSE&G, regional excess demand equals regional forecast load minus the historical expected regional share multiplied by system forecast load. Expected shares are grouped by season, hour, and weekend status. Five levels and two positive-part hinges per region enter the regression. Hinges use training P80 and P95.

Incoming ramp is current remaining demand minus the previous forecast hour. Ahead ramp is next forecast hour minus current remaining demand. Neighbors must be exactly one UTC hour apart and belong to the same delivery date. A missing hour is never bridged. HE1 and HE24 lack one same-day neighbor and normally use geographic regression without the ramp model.

## Residual correction

The correction target is actual price minus an earlier held-out regression prediction. `historical_errors.csv.gz` preserves the earlier residual observations and their feature normalizations. The compact runner selects residual rows before its cutoff, fits the tree again, and does not regenerate the original regression residual history.

Tree inputs are remaining-demand pressure, mean gas level, annual sine/cosine phase, local hour, outage ratio, combined remaining-demand/outage pressure, and gas stress. Pressure variables use training seasonal reference values; they do not measure physical available capacity.

The final scikit-learn GradientBoostingRegressor uses squared-error loss, 150 trees, learning rate 0.03, maximum depth 6, minimum leaf size 24, and random seed 42. It uses the same annual weight half-life as the regression. Its signed prediction is added to the full regression route without a fitted mixing coefficient.

## Missing inputs

The routing hierarchy is full hybrid, geographic regression, Dominion regression, remaining-demand/gas/outage regression, core calendar/load/wind/solar regression, and calendar mean. Only the full route receives the tree correction. Core/calendar parameters remain frozen from the historical 2023–2024 fit; the more informed regressions are refitted.

In 2025 there are 7,699 full-route hours, 700 geographic hours, 183 remaining-demand hours, 63 Dominion hours, 24 core hours, and 66 calendar hours. All 8,735 scored hours receive finite forecasts. Fallback hours account for 16.89% of final squared error.

## Research decisions

Earlier work compared calendar baselines, linear demand, seasonal hinges, natural splines, recency weighting, regional composition, and forecast trajectory features. A fixed seasonal exponential response extrapolated poorly. Additional mapped regional gas interactions and daily peak/position features were not retained. Earlier shallow correction experiments did not improve their selection periods; the final depth-six correction reflects later development comparisons. This release presents the retained model rather than treating every exploratory idea as an implemented final feature.
