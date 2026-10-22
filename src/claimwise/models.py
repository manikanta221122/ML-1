"""Model factories: severity regressor, fraud classifier, anomaly detector."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from . import config as C


def _preprocessor(cat_cols, num_cols):
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    return ColumnTransformer(
        [("cat", enc, cat_cols), ("num", "passthrough", num_cols)],
        verbose_feature_names_out=False,
    )


def build_severity_model(cat_cols, num_cols, max_iter=400):
    """Gradient boosting on log1p(loss) -> handles the heavy right skew."""
    reg = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.06, max_iter=max_iter, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
        random_state=C.RANDOM_STATE,
    )
    pipe = Pipeline([("prep", _preprocessor(cat_cols, num_cols)), ("model", reg)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def build_baseline_model(cat_cols, num_cols):
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import OneHotEncoder
    prep = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols), ("num", "passthrough", num_cols)])
    return TransformedTargetRegressor(
        regressor=Pipeline([("prep", prep), ("model", Ridge(alpha=3.0))]),
        func=np.log1p, inverse_func=np.expm1)


def build_fraud_model():
    return HistGradientBoostingClassifier(
        learning_rate=0.06, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
        class_weight="balanced", early_stopping=True, validation_fraction=0.15,
        random_state=C.RANDOM_STATE,
    )


def build_anomaly_model():
    return IsolationForest(n_estimators=200, contamination="auto",
                           random_state=C.RANDOM_STATE, n_jobs=-1)
