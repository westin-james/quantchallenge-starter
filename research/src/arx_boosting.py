import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
import lightgbm as lgb

def asinh_transform(y, c): return np.arcsinh(y / c)
def asinh_inverse(yt, c): return np.sinh(yt) * c
def time_decay_weights(n, strength=1.2): return np.exp(np.linspace(-strength, 0.0, n))

def fit_ar_coefs(y: pd.Series, p: int) -> np.ndarray:
    X = np.column_stack([y.shift(k) for k in range(1, p + 1)])
    X = pd.DataFrame(X, index=y.index)
    mask = ~np.isnan(X).any(axis=1)
    coef, *_ = np.linalg.lstsq(X[mask].values, y[mask].values, rcond=None)
    return coef

def ar_in_sample(y: pd.Series, coef: np.ndarray) -> pd.Series:
    X = np.column_stack([y.shift(k) for k in range(1, len(coef) + 1)])
    return pd.Series(X @ coef, index=y.index)

def arx_prepare(y2: pd.Series, tr_idx, ho_idx, p=8,
                use_asinh=True, c_mult=3.0,
                use_time_decay=True, decay=1.2):
    coef = fit_ar_coefs(y2, p)
    ar_in = ar_in_sample(y2, coef)
    resid = y2 - ar_in
    if use_asinh:
        c = c_mult * np.median(np.abs(resid.iloc[tr_idx].dropna()))
        y_tr_t = asinh_transform(resid.iloc[tr_idx], c)
        y_ho_t = asinh_transform(resid.iloc[ho_idx], c)
        inv_hold = lambda z: asinh_inverse(z, c)
    else:
        y_tr_t = resid.iloc[tr_idx]; y_ho_t = resid.iloc[ho_idx]
        inv_hold = lambda z: z
    w_tr = time_decay_weights(len(tr_idx), decay) if use_time_decay else None
    return dict(ar_in=ar_in, y_tr_t=y_tr_t, y_ho_t=y_ho_t,
                inv_hold=inv_hold, w_tr=w_tr, coef=coef)

def lgb_holdout_resid_score(params: dict,
                            X_tr_hold, X_ho_hold,
                            y_tr_t, y_ho_t, w_tr,
                            inv_hold, ar_in_ho, y2_ho,
                            early_rounds=60):
    mdl = lgb.LGBMRegressor(**params)
    mdl.fit(X_tr_hold, y_tr_t,
            sample_weight=w_tr,
            eval_set=[(X_ho_hold, y_ho_t)],
            callbacks=[lgb.early_stopping(early_rounds, verbose=False)])
    pred_t = mdl.predict(X_ho_hold)
    level_pred = inv_hold(pred_t) + ar_in_ho
    best_it = int(getattr(mdl, "best_iteration_", params.get("n_estimators", 800)))
    return float(r2_score(y2_ho, level_pred)), best_it

def oof_lgb_resid_level(params: dict, X: pd.DataFrame, y_level: pd.Series,
                        splits, ar_in: pd.Series,
                        use_asinh=True, c_scale=3.0,
                        use_time_decay=True, decay_strength=1.2,
                        early_rounds=60):
    n = len(y_level)
    oof_level = np.full(n, np.nan)
    rounds = []
    for tr_idx, va_idx in splits:
        resid = (y_level - ar_in)
        y_tr_raw = resid.iloc[tr_idx]; y_va_raw = resid.iloc[va_idx]
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]

        tr_mask = np.isfinite(y_tr_raw.values)
        va_mask = np.isfinite(y_va_raw.values)
        Xtr2, y_tr2 = Xtr.iloc[tr_mask], y_tr_raw.iloc[tr_mask]
        Xva2, y_va2 = Xva.iloc[va_mask], y_va_raw.iloc[va_mask]

        if use_asinh:
            c = c_scale * np.median(np.abs(y_tr2.dropna()))
            y_tr_t = asinh_transform(y_tr2, c)
            y_va_t = asinh_transform(y_va2, c)
            inv = lambda z: asinh_inverse(z, c)
        else:
            y_tr_t, y_va_t = y_tr2, y_va2
            inv = lambda z: z

        w = time_decay_weights(len(tr_idx), decay_strength)[tr_mask] if use_time_decay else None

        mdl = lgb.LGBMRegressor(**params)
        mdl.fit(Xtr2, y_tr_t, sample_weight=w,
                eval_set=[(Xva2, y_va_t)],
                callbacks=[lgb.early_stopping(early_rounds, verbose=False)])
        pred_t = mdl.predict(Xva2)
        lvl_pred = inv(pred_t) + ar_in.iloc[va_idx].values
        oof_level[va_idx] = lvl_pred
        rounds.append(int(getattr(mdl, "best_iteration_", params.get("n_estimators", 800))))
    return oof_level, rounds