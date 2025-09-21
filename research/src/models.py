from typing import Callable, Dict, Optional
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

from .config import RIDGE_ALPHA, RF_PARAMS, LGB_BASE

def make_pipeline(model_key: str, target_key: str) -> Pipeline:
    if model_key == "linreg":
        return Pipeline([("model", LinearRegression())])
    
    if model_key == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=RIDGE_ALPHA))])
    
    if model_key == "rf":
        return Pipeline([("model", RandomForestRegressor(**RF_PARAMS))])
    
    if model_key == "lgb":
        assert HAS_LGB, "LightGBM not available"
        return Pipeline([("model", lgb.LGBMRegressor(**LGB_BASE))])
    
    if model_key == "lgb_y2_enhanced":
        raise ValueError("lgb_y2_enhanced is a composite model; use custom evaluator/trainer.")
    
    raise ValueError(f"Unknown Model Key: {model_key}")

def get_model_display_name(model_key: str) -> str:
    return {
        "linreg": "Linear Regression",
        "ridge": "Ridge",
        "rf": "Random Forest",
        "lgb": "LightGBM",
        "lgb_y2_enhanced": "LightGBM Y2 (Enhanced + Blend)",
    }.get(model_key, model_key)

#def extract_feature_importances(fitted_pipeline, feature_names):
    #try:
        #model = fitted_pipeline.named_steps.get("model", None)
        #importances = getattr(model, "feature_importances_", None)
        #if importances is None:
            #return None
        #import pandas as pd
        #return (pd.DataFrame({"feature": feature_names, "importance": importances})
                #.sort_values("importance", ascending=False).reset_index(drop=True))
    #except Exception:
        #return None