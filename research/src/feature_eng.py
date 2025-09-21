import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Y2_FEATS_RIDGE = ["D", "K", "A"]

Y2_LAGS_STD = [1, 2, 5, 20]
Y2_ROLL_STD = [5, 20]
Y2_LAGS_A = [1, 2, 3, 5, 10, 20]
Y2_ROLL_A = [5, 10, 20]

def _y2_tree_base(train_df):
    base = ["A","K","B","D","F"]
    for c in ["O","P"]:
        if c in train_df.columns:
            base.append(c)
    return base

def make_enhanced_y2_features(train_df, test_df, y1_train=None, y1_test=None, include_time=True):
    base_cols = (["time"] if include_time else []) + _y2_tree_base(train_df)
    df_all = pd.concat([train_df[base_cols], test_df[base_cols]], axis=0, ignore_index=True)
    if include_time:
        df_all = df_all.sort_values("time").reset_index(drop=True)

    if (y1_train is not None) and (y1_test is not None):
        y1_all = pd.concat([pd.Series(y1_train), pd.Series(y1_test)], axis=0, ignore_index=True)
        df_all["Y1_pred"] = y1_all

    for c in _y2_tree_base(train_df):
        lags = Y2_LAGS_A if c == "A" else Y2_LAGS_STD
        rolls = Y2_ROLL_A if c == "A" else Y2_ROLL_STD
        for L in lags:
            df_all[f"{c}_lag{L}"] = df_all[c].shift(L)
        df_all[f"{c}_diff1"] = df_all[c].diff(1)
        df_all[f"{c}_diff2"] = df_all[c].diff(2)
        for W in rolls:
            m = df_all[c].rolling(W, min_periods=max(2, W//2))
            df_all[f"{c}_rmean{W}"] = m.mean()
            df_all[f"{c}_rstd{W}"] = m.std()
            df_all[f"{c}_rmin{W}"] = m.min()
            df_all[f"{c}_rmax{W}"] = m.max()
            returns = df_all[c].pct_change(1).replace([np.inf, -np.inf], 0.0)
            df_all[f"{c}_vol{W}"] = returns.rolling(W, min_periods=max(2, W//2)).std()
            rng = df_all[f"{c}_rmax{W}"] - df_all[f"{c}_rmin{W}"]
            df_all[f"{c}_relpos{W}"] = (df_all[c] - df_all[f"{c}_rmin{W}"]) / (rng + 1e-8)

    for c in ["A", "K"]:
        for p in [1, 2, 5, 10]:
            df_all[f"{c}_ret{p}"] = df_all[c].pct_change(p).replace([np.inf, -np.inf], 0.0)
        for span in [5, 10, 20, 30]:
            df_all[f"{c}_ewm{span}"] = df_all[c].ewm(span=span, adjust=False, min_periods=3).mean()
        for W in [5, 10, 20]:
            m = df_all[c].rolling(W, min_periods=max(2, W//2)).mean()
            s = df_all[c].rolling(W, min_periods=max(2, W//2)).std()
            df_all[f"{c}_z{W}"] = (df_all[c] - m) / (s.replace(0, np.nan))
        momentum_windows = Y2_ROLL_A if c == "A" else Y2_ROLL_STD
        for W in momentum_windows:
            if W >= 5:
                baseline = df_all[f"{c}_rmean{W}"]
                df_all[f"{c}_momentum{W}"] = (df_all[c] / (baseline + 1e-8)) - 1.0
    
    eps = 1e-6
    df_all["A_over_K"] = df_all["A"] / (df_all["K"].abs() + eps)
    df_all["D_over_A"] = df_all["D"] / (df_all["A"].abs() + eps)
    df_all["B_over_F"] = df_all["B"] / (df_all["F"].abs() + eps)
    df_all["K_over_D"] = df_all["K"] / (df_all["D"].abs() + eps)

    for c1 in ["A", "K"]:
        for c2 in ["B", "D", "F"]:
            if c1 != c2:
                df_all[f"{c1}_times_{c2}"] = df_all[c1] * df_all[c2]
                df_all[f"{c1}_minus_{c2}"] = df_all[c1] - df_all[c2]
                df_all[f"{c1}_plus_{c2}"] = df_all[c1] + df_all[c2]

    if "Y1_pred" in df_all.columns:
        for c in ["A", "K", "D"]:
            df_all[f"Y1_times_{c}"] = df_all["Y1_pred"] * df_all[c]
            df_all[f"Y1_over_{c}"] = df_all["Y1_pred"] / (df_all[c].abs() + eps)
    
    df_all = df_all.ffill().fillna(0.0)

    n_tr = len(train_df)
    X_all = df_all.copy()
    train_part = X_all.iloc[:n_tr]
    clip_lo = train_part.quantile(0.005, numeric_only=True)
    clip_hi = train_part.quantile(0.995, numeric_only=True)
    X_all = X_all.clip(lower=clip_lo, upper=clip_hi, axis=1)

    X_tr = X_all.iloc[:n_tr].copy()
    X_te = X_all.iloc[n_tr:].copy()

    variances = X_tr.var(numeric_only=True)
    keep_cols = variances[variances > 1e-10].index.tolist()
    if not include_time and "time" in keep_cols:
        keep_cols.remove("time")

    return X_tr[keep_cols], X_te[keep_cols], keep_cols

class Y2TinyInteractions(BaseEstimator, TransformerMixin):
    def __init__(self, colnames):
        self.colnames = colnames
        assert set(["D","K","A"]).issubset(set(colnames)), "D,K,A must be present"
    def fit(self, X, y=None): return self
    def transform(self, X):
        import numpy as np, pandas as pd
        df = X.copy()
        d = df["D"].values
        k = df["K"].values
        a = df["A"].values
        inter = np.vstack([d*k, a*k, a*d]).T
        return np.hstack([df.values, inter])