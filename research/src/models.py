from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
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
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ])
    
    if model_key == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()), 
            ("model", Ridge(alpha=RIDGE_ALPHA)),
        ])
    
    if model_key == "rf":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(**RF_PARAMS)),
        ])
    
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