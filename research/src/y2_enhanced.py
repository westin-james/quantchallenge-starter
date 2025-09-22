import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from tqdm import tqdm
from itertools import product
import math
import random

from src.feature_eng import make_enhanced_y2_features, Y2TinyInteractions
from src.config import RANDOM_STATE

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

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
    RIDGE_ALPHA_Y2: float = 100.0
    USE_Y2_INTERACTIONS: bool = True
    USE_OP_IN_Y2_RIDGE: str = "auto"

    #ASHA#
    ASHA_MODE: bool = True
    ASHA_BUDGETS = (900, 2000, 3500)
    ASHA_KEEP_FRAC: float = 1/3
    ASHA_BUCKET_BY_LR: bool = True
    ASHA_RANDOM_PROMOTE_FRAC: float = 0.05

LGB_BASE = dict(
    objective="regression", metric="rmse", n_estimators=5000, num_leaves=15,
    learning_rate=0.015, reg_lambda=30.0, reg_alpha=0.5, min_data_in_leaf=256,
    subsample=0.7, feature_fraction=0.6, bagging_freq=1, verbosity=-1,
    random_state=RANDOM_STATE, n_jobs=-1,
)

def _oof_predictions(model, X, y, splits, is_lgb=False, early_rounds=150):
    n = len(y); oof = np.zeros(n); fold_r2 = []; best_rounds = []
    for tr_idx, va_idx in splits:
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]; ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]
        if is_lgb:
            mdl = lgb.LGBMRegressor(**model.get_params())
            mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(early_rounds, verbose=False)])
            pred = mdl.predict(Xva)
            best_it = getattr(mdl, "best_iteration_", getattr(mdl, "best_iteration", LGB_BASE["n_estimators"]))
            best_rounds.append(int(best_it))
        else:
            mdl = clone(model); mdl.fit(Xtr,ytr); pred = mdl.predict(Xva)
        oof[va_idx] = pred; fold_r2.append(r2_score(yva, pred))
    return oof, fold_r2, best_rounds

def enhanced_y1_selection(train_df, y1, splits):
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
        scores[f"enet_{a:g}_l1{l1:g}"] = dict(oof=r2_score(y1, oof), mean_fold=float(np.mean(f)))
        cand[f"enet_{a:g}_l1{l1:g}"] = m
    best = max(scores.keys(), key=lambda k: scores[k]["mean_fold"])
    return cand[best], best, scores

def purged_splits(n, n_splits=3, gap=0, min_train_frac=0.05):
    idx = np.arange(n)
    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    min_train = max(1, int(min_train_frac * n))
    for i in range(1, len(boundaries)):
        val_start = boundaries[i - 1]
        val_end = boundaries[i]
        va_idx = idx[val_start:val_end]
        tr_end = max(min_train, val_start - gap)
        tr_idx = idx[:tr_end]
        if len(tr_idx) == 0:
            continue
        yield tr_idx, va_idx

def holdout_split(n, holdout_frac=0.2):
    cut = int(np.floor(n*(1.0-holdout_frac)))
    return np.arange(cut), np.arange(cut, n)

#ASHA_HELPER#
def _asha_search(
    X_tr, y_tr, X_ho, y_ho,
    base_params: dict,
    grids: dict,
    budgets=(900, 2000, 3500),
    keep_frac: float = 1/3,
    bucket_by_lr: bool = True,
    random_promote_frac: float = 0.05,
    seed: int = 42,
):

    rng = random.Random(seed)

    combos = list(product(grids["lr"], grids["sub"], grids["ff"], grids["min"],
                          grids["l2"], grids["l1"], grids["lea"]))
    
    rung0_scores = []
    with tqdm(total=len(combos), desc=f"ASHA rung 1/{len(budgets)} (T={budgets[0]})", ncols=100) as pbar:
        for (lr, subs, ff, mleaf, l2, l1, leaves) in combos:
            p = dict(base_params,
                     learning_rate=lr, subsample=subs, feature_fraction=ff,
                     min_data_in_leaf=mleaf, reg_lambda=l2, reg_alpha=l1, num_leaves=leaves,
                     n_estimators=budgets[0])
            mdl = lgb.LGBMRegressor(**p)
            mdl.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], callbacks=[lgb.early_stopping(60, verbose=False)])
            pred = mdl.predict(X_ho)
            score = r2_score(y_ho, pred)
            rung0_scores.append((score,p))
            pbar.set_postfix({'best_R2': f'{max(rung0_scores, key=lambda t: t[0])[0]:.4f}'})
            pbar.update(1)

    def promote(scores, kfrac, by_lr=False, random_frac=0.0):
        scores.sort(key=lambda t: t[0], reverse=True)
        promoted = []
        if by_lr:

            buckets = {}
            for s, p in scores:
                buckets.setdefault(p["learning_rate"], []).append((s, p))
            for lr_val, arr in buckets.items():
                arr.sort(key=lambda t: t[0], reverse=True)
                k = max(1, int(len(arr) * kfrac))
                promoted.extend(p for _, p in arr[:k])
        else:
            k = max(1, int(len(scores) * kfrac))
            promoted.extend(p for _, p in scores[:k])
        if random_frac > 0.0 and scores is rung0_scores:
            n_extra = max(0, int(len(scores) * random_frac))
            tail = [p for _, p in scores[max(0, len(scores)//2):]]
            rng.shuffle(tail)

            for p in tail:
                if p not in promoted and len(promoted) < k + n_extra:
                    promoted.append(p)
        return promoted
        
    cur = promote(rung0_scores, keep_frac, by_lr=bucket_by_lr, random_frac=random_promote_frac)

    for r_idx, B in enumerate(budgets[1:], start=2):
        nxt_scores = []
        desc = f"ASHA rung {r_idx}/{len(budgets)} (T={B})"
        with tqdm(total=len(cur), desc=desc, ncols=100) as pbar:
            for p in cur:
                p2 = dict(p)
                p2["n_estimators"] = B
                mdl = lgb.LGBMRegressor(**p)
                mdl.fit(X_tr, y_tr, eval_set=[(X_ho, y_ho)], callbacks=[lgb.early_stopping(90, verbose=False)])
                pred = mdl.predict(X_ho)
                score = r2_score(y_ho, pred)
                nxt_scores.append((score, p2))
                pbar.set_postfix({'best_R2': f'{max(nxt_scores, key=lambda t: t[0])[0]:.4f}'})
                pbar.update(1)
        cur = promote(nxt_scores, keep_frac, by_lr=bucket_by_lr, random_frac=0.0)

    final_best = max(nxt_scores if budgets[-1] != budgets[0] else rung0_scores, key=lambda t: t[0])
    best_score, best_params = final_best[0], final_best[1]
    return best_params, float(best_score)

def evaluate_y2_enhanced_cv(train_df, test_df, y1, y2, cfg: EnhancedConfig):
    n = len(train_df)
    splits = list(purged_splits(n, cfg.N_SPLITS, gap=cfg.GAP, min_train_frac=cfg.MIN_TRAIN_FRAC))
    tr_idx, ho_idx = holdout_split(n, cfg.HOLDOUT_FRAC)

    y1_model, y1_name, y1_scores = enhanced_y1_selection(train_df, y1, splits)
    y1_oof, _, _ = _oof_predictions(y1_model, train_df[["G","J","H","C","M","E"]], y1, splits, is_lgb=False)
    y1_model.fit(train_df[["G","J","H","C","M","E"]], y1)
    y1_test_pred = y1_model.predict(test_df[["G","J","H","C","M","E"]])

    X_y2_tr, X_y2_te, cols = make_enhanced_y2_features(train_df, test_df, y1_oof, y1_test_pred, include_time=True)

    probe = dict(LGB_BASE); probe['n_estimators'] = 200
    lgb_probe = lgb.LGBMRegressor(**probe)
    lgb_probe.fit(X_y2_tr.iloc[tr_idx], y2.iloc[tr_idx])
    gain = lgb_probe.booster_.feature_importance(importance_type="gain")
    keep = list(pd.DataFrame({"f": X_y2_tr.columns, "g": gain}).sort_values("g", ascending=False).head(cfg.TOPK_Y2)["f"])
    X_y2_tr = X_y2_tr[keep]; X_y2_te = X_y2_te[keep]

    base_cols = [c for c in ["D", "K", "A"] if c in train_df.columns]
    ext_cols = [c for c in ["D","K","A","O","P"] if c in train_df.columns]

    def build_ridge(cols):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("inter", Y2TinyInteractions(colnames=cols)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=cfg.RIDGE_ALPHA_Y2))
        ])

    candidates = []

    ridge_base = build_ridge(base_cols)
    oof_base, _, _ = _oof_predictions(ridge_base, train_df[base_cols], y2, splits, is_lgb=False)
    score_base = r2_score(y2.values, oof_base)
    candidates.append(("base", base_cols, ridge_base, oof_base, score_base))

    if (cfg.USE_OP_IN_Y2_RIDGE != "never") and (len(ext_cols) > len(base_cols)):
        ridge_ext = build_ridge(ext_cols)
        oof_ext, _, _ = _oof_predictions(ridge_ext, train_df[ext_cols], y2, splits, is_lgb=False)
        score_ext = r2_score(y2.values, oof_ext)
        candidates.append(("ext", ext_cols, ridge_ext, oof_ext, score_ext))
    
    if cfg.USE_OP_IN_Y2_RIDGE == "always" and len(ext_cols) > len(base_cols):
        only_ext = [c for c in candidates if c[0] == "ext"]
        choice = max(only_ext or candidates, key=lambda t: t[4])
    elif cfg.USE_OP_IN_Y2_RIDGE == "never":
        choice = [c for c in candidates if c[0] == "base"][0]
    else:
        choice = max(candidates, key=lambda t: t[4])
    
    _tag, y2_ridge_cols, y2_ridge, y2_ridge_oof, _score = choice

    grid_lr = [0.01, 0.0125, 0.015, 0.0175, 0.019, 0.02] if cfg.SPEED_MODE else [0.01, 0.015, 0.02]
    grid_sub = [0.65, 0.70, 0.75] if cfg.SPEED_MODE else [0.65, 0.70, 0.75]
    grid_ff = [0.50, 0.55, 0.60, 0.65] if cfg.SPEED_MODE else [0.5, 0.55, 0.6, 0.65]
    grid_min = [128, 160, 192, 224, 256, 320] if cfg.SPEED_MODE else [128, 256, 384]
    grid_l2 = [30.0, 45.0, 60.0] if cfg.SPEED_MODE else [30.0, 45.0, 60.0]
    grid_l1 = [0.5] if cfg.SPEED_MODE else [0.5, 1.0]
    grid_lea = [15, 31] if cfg.SPEED_MODE else [15, 31]
    grid_it = [3000] if cfg.SPEED_MODE else [3000, 4000, 5000]

    total_combinations = len(grid_lr) * len(grid_sub) * len(grid_ff) * len(grid_min) * len(grid_l2) * len(grid_l1) * len(grid_lea) * len(grid_it)

    print(f"\nGrid search mode: {'FAST' if cfg.SPEED_MODE else 'FULL'}")
    print(f"Total parameter combinations to test: {total_combinations}")

    X_tr_hold = X_y2_tr.iloc[tr_idx]; X_ho_hold = X_y2_tr.iloc[ho_idx]
    best_score, best_combo = -1e9, None

    if cfg.ASHA_MODE:
        grids = dict(lr=grid_lr, sub=grid_sub, ff=grid_ff, min=grid_min,
                        l2=grid_l2, l1=grid_l1, lea=grid_lea)
        best_combo, best_score = _asha_search(
            X_tr_hold, y2.iloc[tr_idx], X_ho_hold, y2.iloc[ho_idx],
            base_params=dict(LGB_BASE, random_state=cfg.SEED),
            grids=grids,
            budgets=tuple(cfg.ASHA_BUDGETS),
            keep_frac=float(cfg.ASHA_KEEP_FRAC),
            bucket_by_lr=bool(cfg.ASHA_BUCKET_BY_LR),
            random_promote_frac=float(cfg.ASHA_RANDOM_PROMOTE_FRAC),
            seed=int(cfg.SEED),
        )
        print(f"\n[ASHA] Best holdout R2: {best_score:.4f}")
    else:
        param_combinations = list(product(grid_lr, grid_sub, grid_ff, grid_min, grid_l2, grid_l1, grid_lea, grid_it))
        with tqdm(total=total_combinations, desc="Grid search progress", ncols=100) as pbar:
            for lr, subs, ff, mleaf, l2, l1, leaves, iters in param_combinations:
                params = dict(LGB_BASE, learning_rate=lr, subsample=subs,
                                feature_fraction=ff, min_data_in_leaf=mleaf,
                                reg_lambda=l2, reg_alpha=l1, num_leaves=leaves,
                                n_estimators=iters, random_state=cfg.SEED)
                
                mdl = lgb.LGBMRegressor(**params)
                mdl.fit(X_tr_hold, y2.iloc[tr_idx],
                        eval_set=[(X_ho_hold, y2.iloc[ho_idx])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
                
                pred = mdl.predict(X_ho_hold)
                score = r2_score(y2.iloc[ho_idx], pred)       
                if score > best_score:
                    best_score = score
                    best_combo = params.copy()
                    best_combo["n_estimators"] = int(getattr(mdl, "best_iteration_", params["n_estimators"]))
                    pbar.set_postfix({'best_R2': f'{best_score:.4f}'})
                pbar.update(1)

        print(f"\nBest holdout R^2 achieved: {best_score:.4f}")

    lgb_chosen = lgb.LGBMRegressor(**best_combo)
    lgb_oof, _, lgb_rounds = _oof_predictions(lgb_chosen, X_y2_tr, y2, splits, is_lgb=True)
    base_rounds = int(np.median(lgb_rounds)) if lgb_rounds else best_combo["n_estimators"]
    final_rounds = min(max(300, int(base_rounds * 1.05)), 2000)

    from scipy.optimize import minimize_scalar
    simple = minimize_scalar(lambda w: -r2_score(y2.values, w*lgb_oof + (1-w)*y2_ridge_oof),
                             bounds=(0.0, 1.0), method='bounded')
    simple_w = float(simple.x)
    simple_r2 = r2_score(y2.values, simple_w*lgb_oof + (1-simple_w)*y2_ridge_oof)

    use_meta = False; meta_model = None; meta_r2 = -1e9
    if cfg.USE_META_LEARNING:
        from sklearn.linear_model import Ridge as Ridge2
        X_meta = np.column_stack([lgb_oof, y2_ridge_oof, y1_oof, lgb_oof*y2_ridge_oof, lgb_oof*y1_oof])
        meta_model = Ridge2(alpha=1.0).fit(X_meta, y2.values)
        meta_pred = meta_model.predict(X_meta)
        meta_r2 = r2_score(y2.values, meta_pred)
        use_meta = meta_r2 > simple_r2
    
    mean_r2 = float(max(meta_r2, simple_r2))

    return dict(
        MeanR2=mean_r2,
        Details=dict(
            best_params={k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in best_combo.items()},
            final_rounds=int(final_rounds),
            simple_w=simple_w,
            used_meta=bool(use_meta),
        ),
        CachedArtifacts=dict(
            X_y2_tr=X_y2_tr, X_y2_te=X_y2_te, keep_cols=keep,
            best_params=best_combo, final_rounds=final_rounds,
            y1_model=y1_model, y1_oof=y1_oof, y1_test_pred=y1_test_pred,
            y2_ridge=y2_ridge, y2_ridge_cols=y2_ridge_cols,
        )
    )

def y2_ridge_cols_from_choice(choice_tuple):
    _, cols, _, _, _ = choice_tuple
    return cols

class Y2EnhancedFitted:
    def __init__(self, lgb_models, ridge_model, meta_model, simple_w, use_meta, X_lgb_test, ridge_feats):
        self.lgb_models = lgb_models
        self.ridge_model = ridge_model
        self.meta_model = meta_model
        self.simple_w = simple_w
        self.use_meta = use_meta
        self.X_lgb_test = X_lgb_test
        self.ridge_feats = ridge_feats

    def predict(self, X_lgb, X_ridge, extra_meta=None):
        import numpy as np
        lgb_pred = np.mean([m.predict(X_lgb) for m in self.lgb_models], axis=0)
        ridge_pred = self.ridge_model.predict(X_ridge)
        if self.use_meta and self.meta_model is not None and extra_meta is not None:
            X_meta = np.column_stack([lgb_pred, ridge_pred, extra_meta, 
                                      lgb_pred * ridge_pred, lgb_pred * extra_meta])
            return self.meta_model.predict(X_meta)
        return self.simple_w * lgb_pred + (1 - self.simple_w) * ridge_pred