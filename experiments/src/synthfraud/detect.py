from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from synthfraud.schema import Schema


def encode_fit_detector(train, schema: Schema, *, seed: int = 0, kind: str = "hgb"):
    cat = list(schema.categorical)
    num = list(schema.numeric)
    if kind == "hgb":
        clf = HistGradientBoostingClassifier(random_state=seed, max_depth=4, max_iter=80)
    else:
        clf = LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)
    transformers = [("num", StandardScaler(), num)]
    if cat:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)
        )
    model = Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])
    model.fit(train[schema.feature_cols], train[schema.label].astype(int))
    return model


def predict_proba(model, frame, schema: Schema):
    return model.predict_proba(frame[schema.feature_cols])[:, 1]
