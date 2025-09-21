import numpy as np
import pandas as pd
import lightgbm as lgb
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score


from src.feature_eng import make_enhanced_y2_features, Y2TinyInteractions, Y2_FEATS_RIDGE
from src.splits import purged_splits, holdout_split
from src.config import RANDOM_STATE

class EnhancedConfig:
    N_SPLITS: int = 3
    GAP: int = 300
    MIN_TRAIN_FRAC: float = 0.05
    HOLDOUT_FRAC: float = 0.20
    SEED: int = RANDOM_STATE
    TOPK_Y2: int = 55
    SPEED_MODE: bool = True
    USE_META_LEARNING: bool = True
    USE_Y1_FOR_Y2: bool = True
    RIDGE_ALPHA_V2: float = 100.0
    USE_Y2_INTERACTIONS: bool = True

LGB_BASE = dict(
    objective="regression", metric="rmse", n_estimators=5000, num_leaves=15,
    learning_rate=0.015, reg_lambda=30.0, reg_alpha=0.5, min_data_in_leaf=256,
    subsample=0.7, feature_fraction=0.6, bagging_freq=1, verbosity=-1,
    random_state=RANDOM_STATE, n_jobs=-1,
)

def _oof_predictions(model, X, y, splits, is_lgb=False, early_rounds=150):
    n = len(y)
    oof = np.zeros(n)
    fold_r2 = []
    best_rounds = []
    for tr_idx, va_idx in splits:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        if is_lgb:
            mdl = lgb.LGBMRegressor(**model.get_params())
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(early_rounds, verbose=False)])
            pred = mdl.predict(Xva)
            best_it = getattr(mdl, "best_iteration_", getattr(mdl, "best_iteration", LGB_BASE["n_estimators"]))
            best_rounds.append(int(best_it))
        else:
            mdl = clone(model)
            mdl.fit(Xtr,ytr)
            pred = mdl.predict(Xva)
        oof[va_idx] = pred; fold_r2.append(r2_score(yva, pred))
    return oof, fold_r2, best_rounds

def enhanced_y1_selections(train_df, y1, splits):
    Y1_FEATS = ["G","J","H","C","M","E"]
    alphas = [6.0, 8.0, 10.0, 12.0, 15.0]
    enet_cfgs = [(10.0, 0.05), (12.0, 0.05)]
    cand, scores = {}, {}
    for a in alphas:
        model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=a))])
        oof, f, _ = _oof_predictions(model, train_df[Y1_FEATS], y1, splits, is_lgb=False)
        scores[f"ridge_{a:g}"] = dict(oof=r2_score(y1, oof), mean_fold=float(np.mean(f)))
        cand[f"ridge_{a:g}"] = model
    for a, l1 in enet_cfgs:
        m = Pipeline([("scaler", StandardScaler()), ("enet", ElasticNet(alpha=a, l1_ratio=l1, max_iter=2000))])
        oof, f, _ = _oof_predictions(m, train_df[Y1_FEATS], y1, splits, is_lgb=False)
        scores[f"enet_{a:g}"] = dict(oof=r2_score(y1, oof), mean_fold=float(np.mean(f)))
        cand[f"enet_{a:g}"] = m
    best = max(scores.keys(), key=lambda k: scores[k]["mean_fold"])
    return cand[best], best, scores


def evaluate_y2_enhanced_cv(train_df, test_df, y1, y2, cfg: EnhancedConfig):
    n = len(train_df)
    splits = list(purged_splits(n, cfg.N_SPLITS, cap=cfg.GAP, min_train_frac=cfg.MIN_TRAIN_FRAC))
    tr_idx, ho_idx = holdout_split(n,cfg.HOLDOUT_FRAC)

    y1_model, y1_name, y1_scores = enhanced_y1_selections(train_df, y1, splits)
    y1_oof, _, _ = _oof_predictions(y1_model, train_df[["G","J","H","C","M","E"]], y1, splits, is_lgb=False)
    y1_model.fit(train_df[["G","J","H","C","M","E"]], y1)
    y1_test_pred = y1_model.predict(test_df[["G","J","H","C","M","E"]])

    X_y2_tr, X_y2_te, cols = make_enhanced_y2_features(train_df, test_df, y1_oof, y1_test_pred, include_time=True)

    probe = dict(LGB_BASE)
    probe['n_estimators'] = 200
    lgb_probe = lgb.LGBMRegressor(**probe)
    lgb_probe.fit(X_y2_tr.iloc[tr_idx], y2.iloc[tr_idx])
    gain = lgb_probe.booster_.feature_importance(importance_type="gain")
    keep = list(pd.DataFrame({"f": X_y2_tr.columns, "g":gain}).sort_values("g", ascending=False).head(cfg.TOPK_Y2)["f"])
    X_y2_tr = X_y2_tr[keep]; X_y2_te = X_y2_te[keep]

    y2_ridge = Pipeline([("inter", Y2TinyInteractions(colnames=Y2_FEATS_RIDGE)),
                         ("scaler", StandardScaler()),
                         ("ridge", Ridge(alpha=cfg.RIDGE_ALPHA_V2))])
    y2_ridge_oof, _, _ = _oof_predictions(y2_ridge, train_df[Y2_FEATS_RIDGE], y2, splits, is_lgb=False)

    grid_lr = [0.015, 0.02] if cfg.SPEED_MODE else [0.01, 0.015, 0.02]
    grid_sub = [0.70] if cfg.SPEED_MODE else [0.65, 0.70, 0.75]
    grid_ff = [0.55, 0.60] if cfg.SPEED_MODE else [0.5, 0.55, 0.6, 0.65]
    grid_min = [256, 384] if cfg.SPEED_MODE else [128, 256, 384]
    grid_l2 = [45.0, 60.0] if cfg.SPEED_MODE else [30.0, 45.0, 60.0]
    grid_l1 = [0.5] if cfg.SPEED_MODE else [0.5, 1.0]
    grid_lea = [15] if cfg.SPEED_MODE else [15, 31]
    grid_it = [2500] if cfg.SPEED_MODE else [3000, 4000, 5000]

    X_tr_hold = X_y2_tr.iloc[tr_idx]; X_ho_hold = X_y2_tr.iloc[ho_idx]
    best_core, best_combo = -1e9, None
    for lr in grid_lr:
        for subs in grid_sub:
            for ff in grid_ff:
                for mleaf in grid_min:
                    for l2 in grid_l2:
                        for l1 in grid_l1:
                            for leaves in grid_lea:
                                for iters in grid_it:
                                    params = dict(LGB_BASE, learning_rate=lr, subsample=subs,
                                                  feature_fraction=ff, min_data_in_leaf=mleaf,
                                                  reg_lambda=l2, reg_alpha=l1, num_leaves=leaves,
                                                  n_estimators=iters, random_state=cfg.SEED)
                                    mdl = lgb.LGBMRegressor(**params)
                                    mdl.fit(X_tr_hold, y2.iloc[tr_idx],
                                            eval_set=[(X_ho_hold, y2.iloc[ho_idx])],
                                            callbacks=[lgb.early_stopping(100, verbose=False)])
                                    pred = mdl.predict(X_ho_hold)
                                    score = mdl.predict(X_ho_hold)
                                    score = r2_score(y2.iloc[ho_idx], pred)
                                    if score > best_score:
                                        best_score = score; best_combo = params.copy()
                                        best_combo["n_estimators"] = int(getattr(mdl, "best_iteration_", params["n_estimators"]))

    lgb_chosen = lgb.LGBMRegressor(**best_combo)
    lgb_oof, _, lgb_rounds = _oof_predictions(lgb_chosen, X_y2_tr, y2, splits, is_lgb=True)
    base_rounds = int(np.median(lgb_rounds)) if lgb_rounds else best_combo["n_estimators"]
    final_rounds = min(max(300, int(base_rounds * 1.05)), 2000)

    from scipy.optimize import minimize_scalar
    simple = minimize_scalar(lambda w: -r2_score(y2.values, w*lgb_oof + (1-w)*y2_ridge_oof),
                             bounds=(0.0, 1.0), method='bounded')
    simple_w = float(simple.x)
    simple_r2 = r2_score(y2.values,simple_w*lgb_oof + (1-simple_w)*y2_ridge_oof)

    use_meta = False; meta_model = None; meta_r2 = -1e9
    if cfg.USE_META_LEARNING:
        from sklearn.linear_model import Ridge as Ridge2
        X_meta = np.column_stack([lgb_oof, y2_ridge_oof, y1_oof, lgb_oof*y2_ridge_oof, lgb_oof*y1])
        meta_model = Ridge2(alpha=1.0).fit(X_meta, y2.values)
        meta_pred = meta_model.predict(X_meta)
        meta_r2 = r2_score(y2.values, meta_pred)
        use_meta = meta_r2 > simple_r2
    
    mean_r2 = max(meta_r2, simple_r2)
    return dict(
        MeanR2=float(mean_r2),
        Details=dict(
            best_params={k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_combo.items()},
            final_rounds=int(final_rounds),
            simple_w=float(simple_w),
            used_meta=bool(use_meta),
        ),
        CachedArtifacts=dict(
            X_y2_tr=X_y2_tr, X_y2_te=X_y2_te, keep_cols=keep,
            best_params=best_combo, final_rounds=final_rounds,
            y1_model=y1_model, y1_oof=y1_oof, y1_test_pred=y1_test_pred,
            y2_ridge=y2_ridge
        )
    )

class V2EnhancedFitted:
    def __init__(self,lgb_models, ridge_model, meta_model, simple_w, use_meta):
        self.lgb_models = lgb_models
        self.ridge_model = ridge_model
        self.meta_model = meta_model
        self.simple_w = simple_w
        self.use_meta = use_meta

    def predict(self, X_lgb, X_ridge, extra_meta=None):
        import numpy as np
        lgb_pred = np.mean([m.predict(X_lgb) for m in self.lgb_models], axis=0)
        ridge_pred = self.ridge_model.predict(X_ridge)
        if self.use_meta and self.meta_model is not None and extra_meta is not None:
            X_meta = np.column_stack([lgb_pred, ridge_pred, extra_meta, lgb_pred*ridge_pred, lgb_pred*extra_meta])
            return self.meta_model.predict(X_meta)
        return self.simple_w * lgb_pred + (1 - self.simple_w) * ridge_pred