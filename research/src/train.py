from typing import Dict
import pandas as pd

from .models import make_pipeline
from .config import FEATURE_COLS, RANDOM_STATE
from .y2_enhanced import (
    EnhancedConfig, evaluate_y2_enhanced_cv, Y2EnhancedFitted
)
import numpy as np

def train_best_models(X, y_by_target: Dict[str, pd.Series], selections: Dict[str, str], ctx: dict):
    """
    Train best model per target. If Y2 chooses 'lgb_y2enhanced' we run the full
    enhanced trainer using ctx['train_df'], ctx['test_df'] and cached artifacts
    recorded during evaluatio (if present).
    Returns dict of fitted models keyed by target. For simple models, they are sklearn pipelines.
    For Y2 enhanced, returns Y2EnhancedFitted.
    """
    fitted = {}
    train_df = ctx["train_df"]; test_df = ctx["test_df"]

    for tgt, mkey in selections.items():
        if mkey == "lgb_y2_enhanced" and tgt == "Y2":
            # Re-run evaluator to get artifacts (or you could cache from evaluate step)
            cfg = EnhancedConfig()
            res = evaluate_y2_enhanced_cv(train_df, test_df, y_by_target["Y1"], y_by_target["Y2"], cfg)
            art = res["CachedArtifacts"]
            best_params = dict(art["best_params"])
            best_params["n_estimators"] = res["Details"]["final_rounds"]

            # Train rifge on all train (tiny interactions already inside y2_ridge)
            ridge_full = art["y2_ridge"].fit(train_df[["D", "K", "A"]], y_by_target["Y2"])

            # train 2-seed LightGBM ensemble on full enhanced features
            seeds = [RANDOM_STATE, RANDOM_STATE+1]
            lgb_models = []
            for s in seeds:
                params = dict(best_params, random_state=s)
                mdl = __import__("lightgbm").LGBMRegressor(**params)
                mdl.fit(art["X_y2_tr"], y_by_target["Y2"])
                lgb_models.append(mdl)

            # Decide blend mode same as evaluation
            simple_w = float(res["Details"]["simple_w"])
            use_meta = bool(res["Details"]["used_meta"])
            meta_model = None
            if use_meta:
                from sklearn.linear_model import Ridge as Ridge2
                # Build meta on train using OOF arrays from evaluation already embedded there
                # for final fit, reuse OOF-based meta weights as-is (simple and robust)
                meta_model = Ridge2(alpha=1.0)
                X_meta = np.column_stack([
                    np.mean([m.predict(art["X_y2_tr"]) for m in lgb_models], axis=0),
                    ridge_full.predict(train_df[["D","K","A"]]),
                    art["y1_oof"],
                    ])
                # include interactions
                X_meta = np.column_stack([X_meta, X_meta[:,0]*X_meta[:,1], X_meta[:,0]*X_meta[:,2]])
                meta_model.fit(X_meta, y_by_target["Y2"])

            fitted[tgt] = Y2EnhancedFitted(
                lgb_models=lgb_models,
                ridge_model=ridge_full,
                meta_model=meta_model,
                simple_w=simple_w,
                use_meta=use_meta
            )
        else:
            pipe = make_pipeline(mkey, tgt)
            pipe.fit(X, y_by_target[tgt])
            fitted[tgt] = pipe
    
    return fitted