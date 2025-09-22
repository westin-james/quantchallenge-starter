from __future__ import annotations
import time, warnings
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

CORE = ["G","J","H","C","M","E"]
MAX_Y1_FEATS = 25
ALWAYS_KEEP = {"A","D","O","P"}

LAG_FEATURES = [1, 2, 6, 7]

ROLLING_WINDOWS = [3, 5, 10, 15]
ROLLING_FEATURES = ["G","J","H"]

TIME_BASE_PERIODS = (7, 14, 28)
ALWAYS_KEEP_TIME = {"sin_7","cos_7"}

HOLDOUT_FRAC = 0.20
PENALTY = 2.5

# ---------------------- FEATURE HELPERS ---------------------- #
def add_lag_features(df: pd.DataFrame, target_col="Y1", lags=LAG_FEATURES) -> pd.DataFrame:
    if target_col not in df.columns:
        return df.copy()
    out = df.copy()
    lag_cols = []
    for lag in lags:
        lag_col = f"{target_col}_lag{lag}"
        out[lag_col] = out[target_col].shift(lag)
        lag_cols.append(lag_col)
    for col in lag_cols:
        out[col] = out[col].bfill().fillna(0.0)
    return out

def add_rolling_features(df: pd.DataFrame, cols=ROLLING_FEATURES, windows=ROLLING_WINDOWS) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            for w in windows:
                out[f"{col}_roll_mean_{w}"] = out[col].rolling(w, min_periods=1).mean()
                out[f"{col}_roll_std_{w}"] = out[col].rolling(w, min_periods=1).std().fillna(0.0)
    return out

def add_g_interactions(df: pd.DataFrame, base_col="G") -> pd.DataFrame:
    if base_col not in df.columns:
        return df.copy()
    out = df.copy()
    for col in ["J","H","C","M","E"]:
        if col in out.columns:
            out[f"{base_col}x{col}"] = out[base_col] * out[col]
    out[f"{base_col}_squared"] = out[base_col] ** 2
    return out

def add_time_features(df: pd.DataFrame, periods=TIME_BASE_PERIODS) -> pd.DataFrame:
    if "time" not in df.columns:
        return df.copy()
    out = df.copy()
    t = out["time"].to_numpy().astype(float)
    for P in periods:
        w = 2.0 * np.pi / float(P)
        out[f"sin_{P}"] = np.sin(w * t)
        out[f"cos_{P}"] = np.cos(w * t)
    return out

def _featurize_pair(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tr = add_time_features(train_df)
    te = add_time_features(test_df)

    tr = add_rolling_features(tr)
    te = add_rolling_features(te)

    tr = add_g_interactions(tr)
    te = add_g_interactions(te)

    tr = add_lag_features(tr, "Y1", LAG_FEATURES)
    return tr, te

# ---------------------- UTILS ---------------------- #
def holdout_split(n, frac=HOLDOUT_FRAC):
    cut = int(np.floor(n*(1-frac)))
    return np.arange(cut), np.arange(cut, n)

def r2(y, p):
    return float(r2_score(y, p))

def recency_weights(n, lam=2.0):
    if n<=1:
        return np.ones(n, dtype=np.float32)
    age  = np.arange(n, dtype=np.float32)/(n-1)
    return np.exp(-lam*(1.0-age)).astype(np.float32)

def numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    X = df[cols].copy()
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X.fillna(0.0)

def both_present(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str) -> bool:
    return (col in train_df.columns) and (col in test_df.columns)

def build_y1_candidate_list(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    drop = {"time","id","Y1","Y2"}
    common = [c for c in train_df.columns if c in test_df.columns and c not in drop]
    core_first = [c for c in CORE if c in common]
    interactions = [c for c in common if "x" in c or "_squared" in c]
    rolling_feats = [c for c in common if "_roll_" in c]
    time_bases = [c for c in common if c.startswith("sin_") or c.startswith("cos_")]
    lag_feats = [c for c in common if "_lag" in c]
    other_extras = [c for c in common if c not in core_first and c not in interactions
                    and c not in rolling_feats and c not in time_bases and c not in lag_feats]
    return core_first + interactions + lag_feats + rolling_feats + other_extras + time_bases

def preselect_features(train: pd.DataFrame, test: pd.DataFrame, candidate_feats: List[str], idx_tr, max_feats=MAX_Y1_FEATS):
    y = train["Y1"].iloc[idx_tr].to_numpy()
    Xcand = numeric_df(train, candidate_feats).iloc[idx_tr]
    corrs = {}
    for c in candidate_feats:
        x = Xcand[c].to_numpy()
        if np.std(x) < 1e-12:
            corrs[c] = 0.0
        else:
            cx = np.corrcoef(x, y)[0,1]
            corrs[c] = 0.0 if not np.isfinite(cx) else abs(cx)
        
    core_present = [c for c in CORE if c in candidate_feats]
    interactions = [c for c in candidate_feats if "x" in c or "_squared" in c]
    lag_feats = [c for c in candidate_feats if "_lag" in c]
    other_extras = [c for c in candidate_feats if c not in core_present
                    and c not in interactions and c not in lag_feats]
    
    interactions_sorted = sorted(interactions, key=lambda c: corrs.get(c,0.0), reverse=True)
    lag_feats_sorted = sorted(lag_feats, key=lambda c: corrs.get(c,0.0), reverse=True)
    extras_sorted = sorted(other_extras, key=lambda c: corrs.get(c,0.0), reverse=True)

    keep = core_present[:]
    for c in interactions_sorted[:6]:
        if len(keep) >= max_feats: break
        keep.append(c)
    for c in lag_feats_sorted[:4]:
        if len(keep) >= max_feats: break
        keep.append(c)
    for c in extras_sorted:
        if len(keep) >= max_feats: break
        keep.append(c)

    available_pins = [c for c in ALWAYS_KEEP if both_present(train, test, c)]
    for p in sorted(ALWAYS_KEEP_TIME):
        if both_present(train, test, p):
            available_pins.append(p)
    for p in available_pins:
        if p not in keep:
            keep.append(p)
    
    if len(keep) > max_feats:
        protected = set(CORE) | set(available_pins) | set(interactions_sorted[:3]) | set(lag_feats_sorted[:2])
        removable = [c for c in keep if c not in protected]
        removable_sorted = sorted(removable, key=lambda c: corrs.get(c,0.0))
        to_remove = len(keep) - max_feats
        for c in removable_sorted[:to_remove]:
            keep.remove(c)
    return keep, corrs

# ---------------------- MODELS ---------------------- #
def adaptive_two_window_ridge(train, test, idx_tr, idx_ho, feats,
                              cuts=None, alphas=None, late_lams=None):
    if cuts is None:
        cuts = np.round(np.arange(0.82, 0.94, 0.005), 3)
    if alphas is None:
        alphas = [50.0, 100.0, 150.0, 200.0, 300.0]
    if late_lams is None:
        late_lams = [1.5, 2.0, 2.5, 3.0, 4.0]

    X = numeric_df(train, feats)
    y = train["Y1"].reset_index(drop=True)
    best = None

    for cut in cuts:
        cut_idx = int(np.floor(len(idx_tr)*cut))
        cut_idx = max(1, min(cut_idx, len(idx_tr)-1))
        early_abs = idx_tr[:cut_idx]
        late_abs = idx_tr[cut_idx:]
        if len(early_abs)==0 or len(late_abs)==0:
            continue

        scE = RobustScaler().fit(X.iloc[early_abs])
        scL = RobustScaler().fit(X.iloc[late_abs])
        Xe = scE.transform(X.iloc[early_abs]).astype(np.float32)
        Xl = scL.transform(X.iloc[late_abs]).astype(np.float32)
        ye = y.iloc[early_abs].to_numpy()
        yl = y.iloc[late_abs].to_numpy()

        boundary_time = train["time"].iloc[idx_tr[cut_idx]]
        is_late = (train["time"].iloc[idx_ho].to_numpy() >= boundary_time)

        for aE in alphas:
            mE = Ridge(alpha=aE).fit(Xe,ye)
            for aL in alphas:
                for lam in late_lams:
                    sw = recency_weights(len(late_abs), lam)
                    mL = Ridge(alpha=aL).fit(Xl, yl, sample_weight=sw)
                    p = np.empty(len(idx_ho), dtype=np.float32)
                    if (~is_late).sum()>0:
                        p[~is_late]=mE.predict(scE.transform(X.iloc[idx_ho[~is_late]]))
                    if (is_late).sum()>0:
                        p[is_late]=mL.predict(scL.transform(X.iloc[idx_ho[is_late]]))
                    score = r2(y.iloc[idx_ho], p)
                    if (best is None) or (score > best[0]):
                        best = (score, cut, aE, aL, lam)
    
    if best is None:
        return None
    
    score, cut, aE, aL, lam = best
    name = f"AdaptiveTwoWindowCut{cut:.3f}_aE{aE}_aL{aL}_lam{lam}"

    cut_idx = int(np.floor(len(idx_tr)*cut))
    cut_idx = max(1, min(cut_idx, len(idx_tr)-1))
    early_abs = idx_tr[:cut_idx]
    late_abs = idx_tr[cut_idx:]

    scE = RobustScaler().fit(X.iloc[early_abs])
    scL = RobustScaler().fit(X.iloc[late_abs])
    mE = Ridge(alpha=aE).fit(scE.transform(X.iloc[early_abs]), y.iloc[early_abs])
    sw = recency_weights(len(late_abs), lam)
    mL = Ridge(alpha=aL).fit(scL.transform(X.iloc[late_abs]), y.iloc[late_abs], sample_weight=sw)

    boundary_time = train["time"].iloc[idx_tr[cut_idx]]
    is_late = (train["time"].iloc[idx_ho].to_numpy() >= boundary_time)
    p = np.empty(len(idx_ho), dtype=np.float32)
    if (~is_late).sum()>0:
        p[~is_late]=mE.predict(scE.transform(X.iloc[idx_ho[~is_late]]))
    if (is_late).sum()>0:
        p[is_late]=mL.predict(scL.transform(X.iloc[idx_ho[is_late]]))

    return dict(
        name=name, hold=r2(y.iloc[idx_ho], p), std=0.008, y_ho=p,
        builder=("adaptive_two_window", cut, aE, aL, lam, feats)
    )
    
def huber_ensemble(train, idx_tr, idx_ho, feats, epsilons=[1.2, 1.5, 2.0], alpha=1e-4):
    X = numeric_df(train, feats)
    y = train["Y1"].reset_index(drop=True)
    sc = RobustScaler().fit(X.iloc[idx_tr])
    Xs = sc.transform(X).astype(np.float32)

    predictions, scores = [], []
    for eps in epsilons:
        m = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=3000)
        m.fit(Xs[idx_tr], y.iloc[idx_tr])
        p = m.predict(Xs[idx_ho])
        s = r2(y.iloc[idx_ho], p)
        predictions.append(p); scores.append(s)

    w = np.array(scores); w = w / (w.sum() if w.sum()>0 else 1.0)
    ensemble_pred = np.average(predictions,axis=0, weights=w)
    ensemble_score = r2(y.iloc[idx_ho], ensemble_pred)
    return dict(name=f"HuberEnsemble_eps{epsilons}", hold=ensemble_score, std=0.010,
                y_ho=ensemble_pred, builder=("huber_ensemble", epsilons, alpha, feats))

def random_forest_temporal(train, idx_tr, idx_ho, feats, n_estimators=400, max_depth=12):
    X = numeric_df(train, feats)
    y = train["Y1"].reset_index(drop=True)
    rf = RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        min_samples_split=10, min_samples_leaf=5, max_features='sqrt',
        n_jobs=-1, random_state=SEED
    )
    rf.fit(X.iloc[idx_tr], y.iloc[idx_tr])
    predictions = rf.predict(X.iloc[idx_ho])
    score = r2(y.iloc[idx_ho], predictions)
    return dict(name=f"RandomForest_n{n_estimators}_d{max_depth}", hold=score, std=0.012,
                y_ho=predictions, builder=("random_forest", n_estimators, max_depth, feats))

def weighted_average_ensemble(models: Dict[str, dict], y_true):
    if len(models) < 2:
        k = list(models.keys())[0]
        return models[k]
    scores = [m['hold'] for m in models.values()]
    predictions = [m['y_ho'] for m in models.values()]
    scores_arr = np.array(scores)
    scores_exp = np.exp(scores_arr * 10)
    weights = scores_exp / scores_exp.sum()
    ensemble_pred = np.average(predictions, axis=0, weights=weights)
    ensemble_score = r2(y_true, ensemble_pred)
    model_names = list(models.keys())
    return dict(
        name=f"WeightedEnsemble_{len(models)}models",
        hold=ensemble_score,
        std=0.006,
        y_ho=ensemble_pred,
        builder=("weighted_ensemble", model_names, weights.tolist(), [m['builder'] for m in models.values()])
    )

def final_predict(train, test, y, spec):
    kind = spec[0]
    if kind == "adaptive_two_window":
        cut, aE, aL, lam, feats = spec[1:]
        X = numeric_df(train, feats); Xte = numeric_df(test, feats)
        n = len(train)
        cut_idx = int(np.floor(n*cut)); cut_idx = max(1, min(cut_idx, n-1))
        early_abs = np.arange(cut_idx); late_abs = np.arange(cut_idx, n)
        scE = RobustScaler().fit(X.iloc[early_abs])
        scL = RobustScaler().fit(X.iloc[late_abs])
        mE = Ridge(alpha=aE).fit(scE.transform(X.iloc[early_abs]), y.to_numpy()[early_abs])
        sw = recency_weights(len(late_abs), lam)
        mL = Ridge(alpha=aL).fit(scL.transform(X.iloc[late_abs]), y.to_numpy()[late_abs], sample_weight=sw)
        boundary_time = train["time"].iloc[cut_idx]
        is_late = (test["time"].to_numpy() >= boundary_time)
        p = np.empty(len(test), dtype=np.float32)
        if (~is_late).sum()>0:
            p[~is_late]=mE.predict(scE.transform(Xte[~is_late]))
        if (is_late).sum()>0:
            p[is_late]=mL.predict(scL.transform(Xte[is_late]))
        return p
    
    elif kind == "huber_ensemble":
        epsilons, alpha, feats = spec[1:]
        X = numeric_df(train, feats); Xte = numeric_df(test, feats)
        sc = RobustScaler().fit(X)
        Xs_tr = sc.transform(X).astype(np.float32)
        Xs_te = sc.transform(Xte).astype(np.float32)
        predictions = []
        for eps in epsilons:
            m = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=3000)
            m.fit(Xs_tr, y.to_numpy())
            predictions.append(m.predict(Xs_te))
        return np.mean(predictions, axis=0)
    
    elif kind == "random_forest":
        n_estimators, max_depth, feats = spec[1:]
        X = numeric_df(train, feats); Xte = numeric_df(test, feats)
        rf = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=10, min_samples_leaf=5, max_features='sqrt',
            n_jobs=-1, random_state=SEED
        )
        rf.fit(X, y.to_numpy())
        return rf.predict(Xte)
    
    elif kind == "weighted_ensemble":
        model_names, weights, builders = spec[1:]
        predictions = [final_predict(train, test, y, b) for b in builders]
        return np.average(predictions, axis=0, weights=weights)
    
    return np.zeros(len(test), dtype=np.float32)

# ---------------------- PUBLIC API ---------------------- #
def evaluate_y1_advanced_cv(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    t0 = time.time()

    tr_fe, te_fe = _featurize_pair(train_df, test_df)

    n = len(tr_fe)
    idx_tr, idx_ho = holdout_split(n, HOLDOUT_FRAC)
    y = tr_fe["Y1"].reset_index(drop=True)

    candidates = build_y1_candidate_list(tr_fe, te_fe)
    selected, _corrs = preselect_features(tr_fe, te_fe, candidates, idx_tr, max_feats=MAX_Y1_FEATS)

    models = {}
    atw = adaptive_two_window_ridge(tr_fe, te_fe, idx_tr, idx_ho, selected)
    if atw: models["adaptive_two_window"] = atw

    he = huber_ensemble(tr_fe, idx_tr, idx_ho, selected)
    models["huber_ensemble"] = he

    rft = random_forest_temporal(tr_fe, idx_tr, idx_ho, selected)
    models["random_forest"] = rft

    we = weighted_average_ensemble(models, y.to_numpy()[idx_ho])
    models["weighted_ensemble"] = we

    best_key = max(models.keys(), key=lambda k: models[k]["hold"])
    chosen = models[best_key]
    spec = chosen["builder"]
    selected_name = chosen["name"]
    selected_hold = float(chosen["hold"])

    return dict(
        MeanR2=float(selected_hold),
        Details=dict(
            name=selected_name,
            selected_feats=list(selected),
            holdout_r2=float(selected_hold),
        ),
        CachedArtifacts=dict(
            spec=spec,
            selected_feats=list(selected),
            setup=dict(holdout_frac=HOLDOUT_FRAC, seed=SEED),
        ),
        RuntimeSec=float(time.time() - t0),
    )

class Y1AdvancedFitted:
    def __init__(self, train_df: pd.DataFrame, spec, selected_feats: List[str]):
        self._train_df = train_df.copy()
        self._spec = spec
        self._feats = list(selected_feats)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        train_df = self._train_df.copy()
        test_df = test_df.copy()
        
        tr_fe, te_fe = _featurize_pair(train_df, test_df)

        feats = [c for c in self._feats if (c in tr_fe.columns and c in te_fe.columns)]
        if not feats:
            feats = [c for c in CORE if c in tr_fe.columns and c in te_fe.columns]

        y = tr_fe["Y1"].reset_index(drop=True)
        preds = final_predict(tr_fe, te_fe, y, self._spec)

        lo = float(np.quantile(y, 0.001))
        hi = float(np.quantile(y, 0.999))
        preds = np.clip(preds, lo, hi)
        return preds