from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .config import RF_PARAMS, RIDGE_ALPHA

def make_pipeline(model_key: str, target_key: str) -> Pipeline:
    if model_key == "linreg":
        return Pipeline([("model", LinearRegression())])
    
    if model_key == "ridge":
        return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=RIDGE_ALPHA))])
    
    if model_key == "rf":
        return Pipeline([("model", RandomForestRegressor(**RF_PARAMS))])
    
    raise ValueError(f"Unknown Model Key: {model_key}")

def get_model_display_name(model_key: str) -> str:
    return {
        "linreg": "Linear Regression",
        "ridge": "Ridge",
        "rf": "Random Forest",
    }.get(model_key, model_key)

def extract_feature_importances(fitted_pipeline, feature_names):
    try:
        model = fitted_pipeline.named_steps.get("model", None)
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None
        import pandas as pd
        return (pd.DataFrame({"feature": feature_names, "importance": importances})
                .sort_values("importance", ascending=False).reset_index(drop=True))
    except Exception:
        return None