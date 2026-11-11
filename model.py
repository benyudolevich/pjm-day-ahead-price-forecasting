"""Fit the retained price model and use simpler models when inputs are missing."""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

GAS = [f"gas_{code}" for code in (
    "IGBDC21", "IGBDZ21", "IGBDX21", "IGBEL21", "IGBDE21", "IGBEK21"
)]
CORE = ["load_value", "wind_value", "solar_value_provisional"]
OUTAGES = ["outage_other_total_outages", "outage_western_total_outages"]
REGIONS = ["DOMINION", "COMED", "ATSI", "PPL", "PSE&G"]
REGIONAL_LOAD = [f"region_{region}" for region in REGIONS]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
GROUPS = ["season", "hour_beginning", "is_weekend"]
RAMPS = ["ramp_in", "ramp_ahead"]
TREE_INPUTS = [
    "pressure", "gas_level", "annual_sin", "annual_cos", "hour",
    "outage_ratio", "system_pressure", "gas_stress",
]
TARGET = "system_energy_price_da"


def prepare_data(data):
    """Calculate remaining demand, calendar labels and same-day hourly changes."""
    data = data.copy().sort_values("delivery_utc").reset_index(drop=True)
    data["date"] = pd.to_datetime(data.opr_date)
    data["delivery_month"] = data.date.dt.month
    season_by_month = {
        12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
        5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn",
        10: "Autumn", 11: "Autumn",
    }
    data["season"] = data.delivery_month.map(season_by_month)
    data["remaining_demand"] = (
        data.load_value - data.wind_value - data.solar_value_provisional
    )
    data["gas_basket"] = data[GAS].mean(axis=1, skipna=False)
    data["outages"] = data[OUTAGES].sum(axis=1, skipna=False)
    data["stress"] = data.remaining_demand + data.outages
    utc = pd.to_datetime(data.delivery_utc, utc=True)
    previous = utc.diff().eq(pd.Timedelta(hours=1)) & data.opr_date.eq(data.opr_date.shift())
    following = utc.shift(-1).sub(utc).eq(pd.Timedelta(hours=1)) & data.opr_date.eq(data.opr_date.shift(-1))
    data["ramp_in"] = data.remaining_demand.diff().where(previous)
    data["ramp_ahead"] = (data.remaining_demand.shift(-1) - data.remaining_demand).where(following)
    return data


def complete_inputs(data):
    return data.modeling_day_allowed & data[CORE + GAS + OUTAGES].notna().all(axis=1)


def recency_weights(data, cutoff):
    """An observation loses half its weight every 365.2425 days."""
    age = (pd.Timestamp(cutoff) - data.date).dt.total_seconds().to_numpy() / 86400
    weights = 2 ** (-age / 365.2425)
    return weights / weights.mean()


def fit_coefficients(matrix, data, cutoff, penalties):
    weights = recency_weights(data, cutoff)
    target = data[TARGET].to_numpy()
    penalty = np.diag(penalties)
    coefficients = np.linalg.solve(
        matrix.T @ (weights[:, None] * matrix) + penalty,
        matrix.T @ (weights * target),
    )
    check = np.linalg.lstsq(
        np.vstack([np.sqrt(weights)[:, None] * matrix, np.sqrt(penalty)]),
        np.r_[np.sqrt(weights) * target, np.zeros(len(coefficients))],
        rcond=None,
    )[0]
    if not np.allclose(coefficients, check, atol=1e-7):
        raise ValueError("The two regression calculations disagree.")
    return coefficients.tolist()


def penalties_for(matrix, base_columns=None):
    penalties = np.full(matrix.shape[1], 10.0)
    penalties[0] = 0
    if base_columns is not None:
        penalties[base_columns + 1:base_columns + 5] = 100_000
        penalties[base_columns + 5:base_columns + 15] = 1_000_000
    return penalties


def natural_spline(values, knots):
    """Create smooth cubic terms with straight-line tails."""
    low, high = knots[0], knots[-1]
    x = (np.asarray(values, dtype=float) - low) / (high - low)
    t = (np.asarray(knots) - low) / (high - low)
    terms = [
        (np.maximum(0, x - knot) ** 3 - np.maximum(0, x - t[-1]) ** 3)
        / (t[-1] - knot)
        for knot in t[:-1]
    ]
    return np.column_stack([term - terms[-1] for term in terms[:-1]])


def calendar_matrix(data):
    return np.column_stack(
        [np.ones(len(data))]
        + [(data.hour_beginning.to_numpy() == hour).astype(float) for hour in range(1, 24)]
        + [(data.delivery_day_of_week.to_numpy() == day).astype(float) for day in range(1, 7)]
        + [(data.delivery_month.to_numpy() == month).astype(float) for month in range(2, 13)]
    )


def numeric_features(data, parameters):
    features = data[GAS + OUTAGES].copy()
    for season in SEASONS:
        active = data.season.eq(season).to_numpy()
        demand = data.remaining_demand.to_numpy()
        features[f"net_{season}_linear"] = (demand - parameters["net_anchor"]) * active
        curves = natural_spline(demand, parameters["knots"][season])
        for index in range(curves.shape[1]):
            features[f"net_{season}_spline_{index}"] = curves[:, index] * active
    return features


def regression_matrix(data, parameters):
    features = numeric_features(data, parameters)[parameters["columns"]].to_numpy()
    scaled = (features - parameters["mean"]) / parameters["scale"]
    return np.column_stack([calendar_matrix(data), scaled])


def expected_share(data, profile):
    lookup = pd.DataFrame(profile).set_index(GROUPS).expected_share
    return lookup.reindex(pd.MultiIndex.from_frame(data[GROUPS])).to_numpy()


def regional_excess(data, parameters):
    return np.column_stack([
        data[f"region_{region}"].to_numpy()
        - expected_share(data, parameters["profiles"][region]) * data.load_value.to_numpy()
        for region in REGIONS
    ])


def geographic_matrix(data, parameters):
    excess = regional_excess(data, parameters)
    levels = (excess - parameters["mean"]) / parameters["scale"]
    hinges = np.column_stack([
        np.maximum(0, excess[:, index, None] - parameters["knots"][index])
        for index in range(len(REGIONS))
    ])
    hinges = (hinges - parameters["hinge_mean"]) / parameters["hinge_scale"]
    return np.column_stack([regression_matrix(data, parameters["backbone"]), levels, hinges])


def full_matrix(data, parameters):
    ramps = (data[RAMPS].to_numpy() - parameters["ramp_mean"]) / parameters["ramp_scale"]
    return np.column_stack([geographic_matrix(data, parameters["geography"]), ramps])


def fit_model(data, historical_errors, cutoff, fallback):
    """Fit using labels and saved historical errors strictly before cutoff."""
    cutoff = pd.Timestamp(cutoff)
    train = data[data.date.lt(cutoff) & complete_inputs(data) & data[TARGET].notna()]
    errors = historical_errors[pd.to_datetime(historical_errors.opr_date).lt(cutoff)].copy()
    errors["date"] = pd.to_datetime(errors.opr_date)
    if train.empty or errors.empty:
        raise ValueError("There is not enough history before the requested date.")
    geography_train = train[train[REGIONAL_LOAD].notna().all(axis=1)]
    full_train = geography_train[geography_train[RAMPS].notna().all(axis=1)]

    knots = {
        season: [float(group.remaining_demand.min())]
        + group.remaining_demand.quantile([.30, .50, .80, .95, .98]).tolist()
        + [float(group.remaining_demand.max())]
        for season, group in train.groupby("season")
    }
    if any(np.diff(values).min() <= 0 for values in knots.values()):
        raise ValueError("The seasonal spline knots must be distinct.")
    backbone = {"knots": knots, "net_anchor": float(train.remaining_demand.median())}
    features = numeric_features(train, backbone)
    backbone.update(columns=list(features), mean=features.mean().tolist(), scale=features.std(ddof=0).tolist())
    matrix = regression_matrix(train, backbone)
    backbone["coefficients"] = fit_coefficients(matrix, train, cutoff, penalties_for(matrix))

    dominion_train = train[train.region_DOMINION.notna() & train.load_value.gt(0)]
    profile = dominion_train.assign(share=dominion_train.region_DOMINION / dominion_train.load_value)
    profile = profile.groupby(GROUPS).share.mean().rename("expected_share").reset_index().to_dict("records")
    excess = dominion_train.region_DOMINION.to_numpy() - expected_share(dominion_train, profile) * dominion_train.load_value.to_numpy()
    dominion = {"profile": profile, "mean": float(excess.mean()), "scale": float(excess.std())}
    matrix = np.column_stack([regression_matrix(dominion_train, backbone), (excess - dominion["mean"]) / dominion["scale"]])
    dominion["coef"] = fit_coefficients(matrix, dominion_train, cutoff, penalties_for(matrix))

    profiles = {}
    for region in REGIONS:
        rows = train[train[f"region_{region}"].notna()].copy()
        rows["share"] = rows[f"region_{region}"] / rows.load_value
        profiles[region] = rows.groupby(GROUPS).share.agg(expected_share="mean", profile_hours="size").reset_index().to_dict("records")
    geography = {"backbone": backbone, "profiles": profiles}
    excess = regional_excess(geography_train, geography)
    geography.update(mean=excess.mean(axis=0).tolist(), scale=excess.std(axis=0).tolist(), knots=np.quantile(excess, [.8, .95], axis=0).T.tolist())
    hinges = np.column_stack([np.maximum(0, excess[:, index, None] - geography["knots"][index]) for index in range(len(REGIONS))])
    geography.update(hinge_mean=hinges.mean(axis=0).tolist(), hinge_scale=hinges.std(axis=0).tolist())
    base_columns = 41 + len(backbone["columns"])
    matrix = geographic_matrix(geography_train, geography)
    geography["coefficients"] = fit_coefficients(matrix, geography_train, cutoff, penalties_for(matrix, base_columns))

    full = {"geography": geography, "ramp_mean": full_train[RAMPS].mean().tolist(), "ramp_scale": full_train[RAMPS].std(ddof=0).tolist()}
    matrix = full_matrix(full_train, full)
    full["coefficients"] = fit_coefficients(matrix, full_train, cutoff, penalties_for(matrix, base_columns))
    references = train.groupby("season").agg(
        N95=("remaining_demand", lambda values: values.quantile(.95)),
        stress95=("stress", lambda values: values.quantile(.95)),
        outage_median=("outages", "median"),
    )
    tree = GradientBoostingRegressor(
        loss="squared_error", n_estimators=150, learning_rate=.03,
        max_depth=6, min_samples_leaf=24, random_state=42,
    ).fit(errors[TREE_INPUTS], errors.target_residual, sample_weight=recency_weights(errors, cutoff))
    return {
        "cutoff": str(cutoff.date()), "fit_max_date": train.opr_date.max(),
        "tree_fit_max_date": errors.opr_date.max(), "fit_rows": len(full_train),
        "residual_rows": len(errors), "backbone": backbone, "dominion": dominion,
        "geography": geography, "trajectory": full, "refs": references,
        "tree": tree, "fallback": fallback,
    }


def predict(model, data):
    """Start with calendar prices, then use the most informed available model."""
    data = data.copy()
    fallback = model["fallback"]
    calendar = pd.DataFrame(fallback["calendar_baseline"]).set_index(["hour_beginning", "delivery_day_of_week"])["mean"]
    index = pd.MultiIndex.from_frame(data[["hour_beginning", "delivery_day_of_week"]])
    data["regression"] = calendar.reindex(index).to_numpy()
    data["route"] = "calendar"
    core = data.modeling_day_allowed & data[CORE].notna().all(axis=1)
    complete = complete_inputs(data)
    dominion_ok = complete & data.region_DOMINION.notna()
    geography_ok = complete & data[REGIONAL_LOAD].notna().all(axis=1)
    full_ok = geography_ok & data[RAMPS].notna().all(axis=1)

    if core.any():
        scaled = (data.loc[core, CORE].to_numpy() - fallback["training_mean"]) / fallback["training_scale"]
        matrix = np.column_stack([calendar_matrix(data[core]), scaled])
        data.loc[core, "regression"] = matrix @ fallback["coefficients"]
        data.loc[core, "route"] = "month_core"
    if complete.any():
        data.loc[complete, "regression"] = regression_matrix(data[complete], model["backbone"]) @ model["backbone"]["coefficients"]
        data.loc[complete, "route"] = "net_demand"
    if dominion_ok.any():
        rows = data[dominion_ok]
        parameters = model["dominion"]
        excess = rows.region_DOMINION.to_numpy() - expected_share(rows, parameters["profile"]) * rows.load_value.to_numpy()
        matrix = np.column_stack([regression_matrix(rows, model["backbone"]), (excess - parameters["mean"]) / parameters["scale"]])
        data.loc[dominion_ok, "regression"] = matrix @ parameters["coef"]
        data.loc[dominion_ok, "route"] = "dominion"
    if geography_ok.any():
        data.loc[geography_ok, "regression"] = geographic_matrix(data[geography_ok], model["geography"]) @ model["geography"]["coefficients"]
        data.loc[geography_ok, "route"] = "geographic"
    if full_ok.any():
        data.loc[full_ok, "regression"] = full_matrix(data[full_ok], model["trajectory"]) @ model["trajectory"]["coefficients"]
        data.loc[full_ok, "route"] = "full_hybrid"

    references = model["refs"]
    data["pressure"] = data.remaining_demand / data.season.map(references.N95)
    data["gas_level"] = data.gas_basket
    data["outage_ratio"] = data.outages / data.season.map(references.outage_median)
    data["system_pressure"] = data.stress / data.season.map(references.stress95)
    phase = 2 * np.pi * (data.date.dt.dayofyear - 1) / np.where(data.date.dt.is_leap_year, 366, 365)
    data["annual_sin"], data["annual_cos"] = np.sin(phase), np.cos(phase)
    data["hour"] = data.hour_beginning
    data["gas_stress"] = data.gas_level * np.maximum(0, data.system_pressure - 1)
    data["correction"] = 0.0
    if full_ok.any():
        data.loc[full_ok, "correction"] = model["tree"].predict(data.loc[full_ok, TREE_INPUTS])
    data["prediction"] = data.regression + data.correction
    if not np.isfinite(data.prediction).all():
        raise ValueError("One or more hours have no usable forecast.")
    return data
