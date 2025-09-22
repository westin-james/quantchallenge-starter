from typing import Dict
import numpy as np
import pandas as pd

from .models import make_pipeline
from .config import RANDOM_STATE
from .y2_enhanced import (
    EnhancedConfig, evaluate_y2_enhanced_cv, Y2EnhancedFitted
)
from .y1_advanced_v11 import Y1AdvancedFitted, evaluate_y1_advanced_cv

def train_best_models(X, y_by_target: Dict[str, pd.Series], selections: Dict[str, str], ctx: dict):
    """
    Train best model per target. If Y2 chooses 'lgb_y2enhanced' we run the full
    enhanced trainer using ctx['train_df'], ctx['test_df'] and cached artifacts
    recorded during evaluatio (if present).
    Returns dict of fitted models keyed by target. For simple models, they are sklearn pipelines.
    For Y2 enhanced, returns Y2EnhancedFitted.
    """
    fitted = {}
    train_df = ctx["train_df"]
    test_df = ctx["test_df"]

    for tgt, mkey in selections.items():
        print(f"[TRAIN][START] target={tgt} model={mkey}")
        if mkey == "lgb_y2_enhanced":
            if tgt != "Y2":
                pipe = make_pipeline("ridge", tgt)
                pipe.fit(X, y_by_target[tgt])
                fitted[tgt] = pipe
                print(f"[TRAIN][END]   target={tgt} model={mkey} -> fallback ridge fitted")
                continue

            # Re-run evaluator to get artifacts (or you could cache from evaluate step)
            cfg = EnhancedConfig()
            res = evaluate_y2_enhanced_cv(train_df, test_df, y_by_target["Y1"], y_by_target["Y2"], cfg)
            art = res["CachedArtifacts"]
            best_params = dict(art["best_params"])
            best_params["n_estimators"] = res["Details"]["final_rounds"]

            # Train ridge on all train (tiny interactions already inside y2_ridge)
            ridge_cols = art.get("y2_ridge_cols", ["D","K","A"])
            ridge_full = art["y2_ridge"].fit(train_df[ridge_cols], y_by_target["Y2"])

            # train 2-seed LightGBM ensemble on full enhanced features
            try:
                import lightgbm as lgb
            except Exception as e:
                raise RuntimeError("LightGBM not available at train time") from e
            
            seeds = [RANDOM_STATE, RANDOM_STATE + 1]
            lgb_models = []

            for s in seeds:
                params = dict(best_params, random_state=s)
                mdl = lgb.LGBMRegressor(**params)
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
                lgb_oof_full = np.mean([m.predict(art["X_y2_tr"]) for m in lgb_models], axis=0)
                ridge_oof_full = ridge_full.predict(train_df[ridge_cols])
                y1_oof = art["y1_oof"]
                X_meta = np.column_stack([lgb_oof_full, ridge_oof_full, y1_oof,
                                          lgb_oof_full * ridge_oof_full, lgb_oof_full * y1_oof])
                # include interactions
                meta_model = Ridge2(alpha=1.0).fit(X_meta, y_by_target["Y2"])

            fitted["Y2"] = Y2EnhancedFitted(
                lgb_models=lgb_models,
                ridge_model=ridge_full,
                meta_model=meta_model,
                simple_w=simple_w,
                use_meta=use_meta,
                X_lgb_test=art["X_y2_te"],
                ridge_feats=ridge_cols,
            )
            print(f"[TRAIN][END]   target={tgt} model={mkey} (LGB ensemble + ridge{' + meta' if use_meta else ''})")
        elif mkey == "y1_advanced_v11":
            if tgt != "Y1":
                pipe = make_pipeline("ridge", tgt)
                pipe.fit(X, y_by_target[tgt])
                fitted[tgt] = pipe
                print(f"[TRAIN][END]   target={tgt} model={mkey} -> fallback ridge fitted")
                continue
            res = evaluate_y1_advanced_cv(train_df, test_df)
            art = res["CachedArtifacts"]
            spec = art["spec"]
            feats = art["selected_feats"]
            fitted["Y1"] = Y1AdvancedFitted(train_df=train_df, spec=spec, selected_feats=feats)
            print(f"[TRAIN][END]   target=Y1 model={mkey} (advanced pipeline)")
        else:
            pipe = make_pipeline(mkey, tgt)
            pipe.fit(X, y_by_target[tgt])
            fitted[tgt] = pipe
            print(f"[TRAIN][END]   target={tgt} model={mkey}")
    
    return fitted