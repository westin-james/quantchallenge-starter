from typing import Dict
import pandas as pd
from .models import make_pipeline, extract_feature_importances

def train_best_models(X, y_by_target: Dict[str, pd.Series], selections: Dict[str, str]):
    fitted = {}
    importances = {}
    for target_key, model_key in selections.items():
        pipe = make_pipeline(model_key, target_key)
        pipe.fit(X, y_by_target[target_key])
        fitted[target_key] = pipe
    return fitted

def get_importances_for_fitted(fitted_by_target: Dict[str, object], feature_cols):
    out = {}
    for tgt, pipe in fitted_by_target.items():
        imp_df = extract_feature_importances(pipe, feature_cols)
        if imp_df is not None:
            out[tgt] = imp_df
    return out