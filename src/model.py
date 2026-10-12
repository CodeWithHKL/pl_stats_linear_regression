from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pl_matches.csv"  # built by src.fetch_data
TARGET = "GOALS"

# In-match model: explains goals from match stats (only known after the match).
STATS_FEATURES = ["SHOTS", "SHOTS ON TARGET", "CORNERS", "HOME"]
# Pre-match model: team attack strength vs opponent defence, plus home advantage.
FIXTURE_FEATURES = ["TEAM", "OPPONENT", "HOME"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def build_model(kind, alpha=0.01):
    """'stats' or 'fixture'. Both are Poisson regressions (goals are counts, never negative)."""
    if kind == "stats":
        return PoissonRegressor(alpha=1e-4, max_iter=1000)
    if kind == "fixture":
        encode = ColumnTransformer(
            [("teams", OneHotEncoder(handle_unknown="ignore"), ["TEAM", "OPPONENT"])],
            remainder="passthrough",
        )
        return make_pipeline(encode, PoissonRegressor(alpha=alpha, max_iter=1000))
    raise ValueError(f"unknown model kind: {kind}")


def features_for(kind):
    return STATS_FEATURES if kind == "stats" else FIXTURE_FEATURES


def fit(data, kind, **kwargs):
    return build_model(kind, **kwargs).fit(data[features_for(kind)], data[TARGET])


def cross_validate(data, kind, cv=5, **kwargs):
    """Mean cross-validated MSE for the model and for a predict-the-mean baseline.

    Rows are ordered by season, so unshuffled folds test on later periods than they train on.
    """
    X, y = data[features_for(kind)], data[TARGET]
    score = lambda m: -cross_val_score(m, X, y, cv=cv, scoring="neg_mean_squared_error").mean()
    return {"model": score(build_model(kind, **kwargs)), "baseline": score(DummyRegressor())}
