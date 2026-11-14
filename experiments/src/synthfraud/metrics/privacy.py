"""Risk A metrics. mean_nn_distance is logged, not a privacy claim (ganev2025inadequacy)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from synthfraud.schema import Schema


def privacy(
    train: pd.DataFrame, ref: pd.DataFrame, synth: pd.DataFrame, schema: Schema
) -> dict:
    train_x, fitted = _matrix(train, schema)
    ref_x, _ = _matrix(ref, schema, fitted=fitted)
    synth_x, _ = _matrix(synth, schema, fitted=fitted)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(synth_x)
    d_mem = nn.kneighbors(train_x, return_distance=True)[0].ravel()
    d_non = nn.kneighbors(ref_x, return_distance=True)[0].ravel()
    y = np.concatenate([np.ones(len(d_mem)), np.zeros(len(d_non))])
    s = -np.concatenate([d_mem, d_non])
    try:
        mia_auc = float(roc_auc_score(y, s))
    except ValueError:
        mia_auc = float("nan")
    return {
        "mia_distance_auc": mia_auc,
        "mean_nn_distance_members": float(d_mem.mean()),
        "mean_nn_distance_nonmembers": float(d_non.mean()),
        "note": "mean_nn_distance is logged, not a privacy claim",
    }


RECON_THRESHOLDS = (0.0, 1e-3, 0.01, 0.05, 0.1)


def table_reconstruction(
    train: pd.DataFrame,
    ref: pd.DataFrame,
    synth: pd.DataFrame,
    schema: Schema,
    *,
    thresholds: tuple[float, ...] = RECON_THRESHOLDS,
    numeric_decimals: int = 4,
) -> dict:
    """T1 table-as-reconstruction. The adversary publishes $D_{syn}$ as the
    reconstructed set and does not use ``ref``. ``ref`` is an evaluation
    control only (specificity). This is not ReconSyn (T2: many draws from $G$).
    """
    train_x, fitted = _matrix(train, schema)
    ref_x, _ = _matrix(ref, schema, fitted=fitted)
    synth_x, _ = _matrix(synth, schema, fitted=fitted)
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(synth_x)
    d_mem = nn.kneighbors(train_x, return_distance=True)[0].ravel()
    d_non = nn.kneighbors(ref_x, return_distance=True)[0].ravel()
    fraud_mem = train[schema.label].to_numpy() == 1
    fraud_non = ref[schema.label].to_numpy() == 1
    exact_mem = _exact_match_mask(train, synth, schema, numeric_decimals)
    exact_non = _exact_match_mask(ref, synth, schema, numeric_decimals)
    out = {
        "n_train": int(len(train)),
        "n_ref": int(len(ref)),
        "n_synth": int(len(synth)),
        "n_train_fraud": int(fraud_mem.sum()),
        "n_ref_fraud": int(fraud_non.sum()),
        "numeric_decimals": int(numeric_decimals),
        "exact_match_rate_train": float(exact_mem.mean()),
        "exact_match_rate_ref": float(exact_non.mean()),
        "exact_match_advantage": float(exact_mem.mean() - exact_non.mean()),
        "nn_recall": {},
        "note": (
            "Attack uses only the released table. Ref scores specificity, "
            "not an auxiliary attack dataset. Not ReconSyn."
        ),
    }
    if fraud_mem.any():
        out["exact_match_rate_train_fraud"] = float(exact_mem[fraud_mem].mean())
    else:
        out["exact_match_rate_train_fraud"] = None
    if fraud_non.any():
        out["exact_match_rate_ref_fraud"] = float(exact_non[fraud_non].mean())
    else:
        out["exact_match_rate_ref_fraud"] = None
    if out["exact_match_rate_train_fraud"] is not None and out["exact_match_rate_ref_fraud"] is not None:
        out["exact_match_advantage_fraud"] = (
            out["exact_match_rate_train_fraud"] - out["exact_match_rate_ref_fraud"]
        )
    else:
        out["exact_match_advantage_fraud"] = None
    for t in thresholds:
        key = f"eps_{t}"
        rec_tr = float((d_mem <= t).mean())
        rec_rf = float((d_non <= t).mean())
        cell = {
            "recall_train": rec_tr,
            "recall_ref": rec_rf,
            "advantage": rec_tr - rec_rf,
        }
        if fraud_mem.any():
            cell["recall_train_fraud"] = float((d_mem[fraud_mem] <= t).mean())
        if fraud_non.any():
            cell["recall_ref_fraud"] = float((d_non[fraud_non] <= t).mean())
        if "recall_train_fraud" in cell and "recall_ref_fraud" in cell:
            cell["advantage_fraud"] = cell["recall_train_fraud"] - cell["recall_ref_fraud"]
        out["nn_recall"][key] = cell
    return out


DEFAULT_AIA_TARGETS = ("card6", "card4", "DeviceType")


def attribute_inference(
    train: pd.DataFrame,
    ref: pd.DataFrame,
    synth: pd.DataFrame,
    schema: Schema,
    *,
    targets: tuple[str, ...] = DEFAULT_AIA_TARGETS,
    seed: int = 0,
    eval_n: int = 20_000,
    fit_cap: int = 20_000,
) -> dict:
    """T1 attribute inference (Stadler-style). Train a predictor on $D_{syn}$
    for a released column; score accuracy on members vs ref.

    ``advantage`` is membership-shaped (train minus ref). ``lift_vs_majority``
    is population-shaped (ref minus synth majority). Neither is the banned
    synth-versus-ref HGB membership classifier.
    Identity $n_{train}$ may be subsampled to ``fit_cap`` for the fit only.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

    rng = np.random.default_rng(seed)
    known = set(schema.all_cols)
    out = {
        "eval_n": int(eval_n),
        "fit_cap": int(fit_cap),
        "targets": {},
        "note": (
            "Predictor is fit on the released table only. Ref is evaluation. "
            "Not synth-vs-ref MIA. Not SearchAttack."
        ),
    }
    train_idx = rng.choice(len(train), size=min(eval_n, len(train)), replace=False)
    ref_idx = rng.choice(len(ref), size=min(eval_n, len(ref)), replace=False)
    train_eval = train.iloc[train_idx]
    ref_eval = ref.iloc[ref_idx]
    for target in targets:
        if target not in known or target not in synth.columns:
            continue
        feat_num = [c for c in schema.numeric if c != target]
        feat_cat = [c for c in schema.categorical if c != target]
        if target != schema.label:
            feat_num = feat_num + [schema.label]
        if not feat_num and not feat_cat:
            continue
        y_s = synth[target].astype(str)
        n_fit = min(fit_cap, len(synth))
        if n_fit < len(synth):
            fit_idx = rng.choice(len(synth), size=n_fit, replace=False)
            synth_fit = synth.iloc[fit_idx]
            y_fit = y_s.iloc[fit_idx]
        else:
            synth_fit = synth
            y_fit = y_s
        classes = y_fit.unique()
        if len(classes) < 2:
            continue
        counts = y_fit.value_counts()
        keep = set(counts[counts >= 2].index.astype(str))
        if len(keep) < 2:
            continue
        mask = y_fit.astype(str).isin(keep)
        synth_fit = synth_fit.loc[mask]
        y_fit = y_fit.loc[mask]
        majority = str(y_fit.mode().iloc[0])
        enc_y = LabelEncoder()
        y_fit_i = enc_y.fit_transform(y_fit)
        transformers = []
        if feat_num:
            transformers.append(("num", StandardScaler(), feat_num))
        if feat_cat:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feat_cat)
            )
        clf = HistGradientBoostingClassifier(
            random_state=seed, max_depth=4, max_iter=80, early_stopping=False
        )
        model = Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])
        model.fit(synth_fit[feat_num + feat_cat], y_fit_i)
        pred_tr = enc_y.inverse_transform(model.predict(train_eval[feat_num + feat_cat]))
        pred_rf = enc_y.inverse_transform(model.predict(ref_eval[feat_num + feat_cat]))
        y_tr = train_eval[target].astype(str).to_numpy()
        y_rf = ref_eval[target].astype(str).to_numpy()
        acc_tr = float((pred_tr == y_tr).mean())
        acc_rf = float((pred_rf == y_rf).mean())
        maj_rf = float((y_rf == majority).mean())
        out["targets"][target] = {
            "n_fit": int(len(synth_fit)),
            "n_classes": int(len(enc_y.classes_)),
            "majority": majority,
            "acc_train": acc_tr,
            "acc_ref": acc_rf,
            "advantage": acc_tr - acc_rf,
            "majority_acc_ref": maj_rf,
            "lift_vs_majority": acc_rf - maj_rf,
        }
    return out


def _exact_match_mask(
    real: pd.DataFrame, synth: pd.DataFrame, schema: Schema, decimals: int
) -> np.ndarray:
    synth_keys = set(_row_keys(synth, schema, decimals))
    return np.fromiter(
        (k in synth_keys for k in _row_keys(real, schema, decimals)),
        dtype=bool,
        count=len(real),
    )


def _row_keys(frame: pd.DataFrame, schema: Schema, decimals: int):
    nums = [np.round(frame[c].to_numpy(dtype=float), decimals) for c in schema.numeric]
    cats = [frame[c].astype(str).to_numpy() for c in schema.categorical]
    label = frame[schema.label].to_numpy()
    for i in range(len(frame)):
        yield (
            tuple(float(a[i]) for a in nums)
            + tuple(str(a[i]) for a in cats)
            + (int(label[i]),)
        )


def density_ratio_mia(
    train: pd.DataFrame,
    ref: pd.DataFrame,
    synth: pd.DataFrame,
    schema: Schema,
    *,
    seed: int = 0,
    n_neighbors: int = 5,
) -> dict:
    """DOMIAS-style scores on a T1 table (van Breugel et al., AISTATS 2023).

    ``mia_knn_ratio_auc`` approximates $\\hat p_G(x)/\\hat p_X(x)$ with $k$-NN
    densities (the C3 attack under test). ``mia_clf_ratio_auc`` is a global
    HGB density-ratio diagnostic; it failed the identity n_train control
    and must not be cited.
    ``p_X`` is fit on half of ``ref``; the other half are the non-member
    evaluation rows. Distance MIA in ``privacy()`` remains the harness default.
    This function does not mark C3 verified by itself.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(seed)
    train_x, fitted = _matrix(train, schema)
    ref_x, _ = _matrix(ref, schema, fitted=fitted)
    synth_x, _ = _matrix(synth, schema, fitted=fitted)
    y_ref = ref[schema.label].to_numpy()
    strat = y_ref if len(np.unique(y_ref)) > 1 else None
    idx = np.arange(len(ref))
    fit_idx, eval_idx = train_test_split(
        idx, test_size=0.5, stratify=strat, random_state=seed
    )
    ref_fit_x = ref_x[fit_idx]
    ref_eval_x = ref_x[eval_idx]
    knn_scores = _log_knn_ratio(synth_x, ref_fit_x, train_x, ref_eval_x, n_neighbors)
    clf_scores = _clf_ratio(synth_x, ref_fit_x, train_x, ref_eval_x, rng)
    out = {
        "n_ref_fit": int(len(fit_idx)),
        "n_ref_eval": int(len(eval_idx)),
        "n_neighbors": int(n_neighbors),
        "mia_knn_ratio_auc": _auc(knn_scores["y"], knn_scores["s"]),
        "mia_clf_ratio_auc": _auc(clf_scores["y"], clf_scores["s"]),
        "note": (
            "Density-ratio MIA uses a reference split (same as the harness ref). "
            "It is not a no-auxiliary-data attack."
        ),
    }
    fraud_mask_train = train[schema.label].to_numpy() == 1
    fraud_mask_eval = ref.iloc[eval_idx][schema.label].to_numpy() == 1
    n_f_mem = int(fraud_mask_train.sum())
    n_f_non = int(fraud_mask_eval.sum())
    out["n_fraud_members"] = n_f_mem
    out["n_fraud_nonmembers_eval"] = n_f_non
    if n_f_mem >= 20 and n_f_non >= 20:
        y_f = np.concatenate([np.ones(n_f_mem), np.zeros(n_f_non)])
        out["mia_knn_ratio_auc_fraud"] = _auc(
            y_f,
            np.concatenate(
                [knn_scores["s_mem"][fraud_mask_train], knn_scores["s_non"][fraud_mask_eval]]
            ),
        )
        out["mia_clf_ratio_auc_fraud"] = _auc(
            y_f,
            np.concatenate(
                [clf_scores["s_mem"][fraud_mask_train], clf_scores["s_non"][fraud_mask_eval]]
            ),
        )
    else:
        out["mia_knn_ratio_auc_fraud"] = None
        out["mia_clf_ratio_auc_fraud"] = None
    return out


def _auc(y: np.ndarray, s: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, s))
    except ValueError:
        return float("nan")


def _log_knn_ratio(
    synth_x: np.ndarray,
    ref_fit_x: np.ndarray,
    train_x: np.ndarray,
    ref_eval_x: np.ndarray,
    k: int,
) -> dict:
    k_g = min(k, len(synth_x))
    k_x = min(k, len(ref_fit_x))
    nn_g = NearestNeighbors(n_neighbors=k_g, metric="euclidean").fit(synth_x)
    nn_x = NearestNeighbors(n_neighbors=k_x, metric="euclidean").fit(ref_fit_x)

    def log_ratio(query: np.ndarray) -> np.ndarray:
        d_g = np.maximum(nn_g.kneighbors(query, return_distance=True)[0], 1e-12)
        d_x = np.maximum(nn_x.kneighbors(query, return_distance=True)[0], 1e-12)
        return -np.log(d_g.mean(axis=1)) + np.log(d_x.mean(axis=1))

    s_mem = log_ratio(train_x)
    s_non = log_ratio(ref_eval_x)
    y = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
    s = np.concatenate([s_mem, s_non])
    return {"y": y, "s": s, "s_mem": s_mem, "s_non": s_non}


def _clf_ratio(
    synth_x: np.ndarray,
    ref_fit_x: np.ndarray,
    train_x: np.ndarray,
    ref_eval_x: np.ndarray,
    rng: np.random.Generator,
    cap: int = 20_000,
) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    n = min(len(synth_x), len(ref_fit_x), cap)
    pos = rng.choice(len(synth_x), size=n, replace=False)
    neg = rng.choice(len(ref_fit_x), size=n, replace=False)
    x = np.vstack([synth_x[pos], ref_fit_x[neg]])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    clf = HistGradientBoostingClassifier(max_depth=6, max_iter=80, random_state=0)
    clf.fit(x, y)
    s_mem = clf.predict_proba(train_x)[:, 1]
    s_non = clf.predict_proba(ref_eval_x)[:, 1]
    y_eval = np.concatenate([np.ones(len(s_mem)), np.zeros(len(s_non))])
    s = np.concatenate([s_mem, s_non])
    return {"y": y_eval, "s": s, "s_mem": s_mem, "s_non": s_non}


def _matrix(frame: pd.DataFrame, schema: Schema, fitted=None):
    num = frame[list(schema.numeric)].to_numpy(dtype=float)
    if fitted is None:
        scaler = StandardScaler()
        num_x = scaler.fit_transform(num)
        enc = None
        cat_x = np.zeros((len(frame), 0))
        if schema.categorical:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_x = enc.fit_transform(frame[list(schema.categorical)])
        return np.hstack([num_x, cat_x]), (enc, scaler)
    enc, scaler = fitted
    num_x = scaler.transform(num)
    cat_x = (
        enc.transform(frame[list(schema.categorical)])
        if enc is not None
        else np.zeros((len(frame), 0))
    )
    return np.hstack([num_x, cat_x]), fitted
