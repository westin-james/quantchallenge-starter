from typing import Dict, List, Callable, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from .models import make_pipeline, get_model_display_name

def crossval_grid(X, y_by_target: Dict[str, pd.Series], cv, model_keys: List[str], 
                  scoring: str = "r2", custom_evaluators: Optional[Dict[str, Callable]]= None,
                  ctx: Optional[dict] = None,) -> pd.DataFrame:
    rows=[]
    custom_evaluators = custom_evaluators or {}

    for model_key in model_keys:
        for target_key, y in y_by_target.items():
            if model_key in custom_evaluators:
                result = custom_evaluators[model_key](target_key, X, y_by_target, ctx or {})
                if result is None:
                    continue
                rows.append({
                    "ModelKey": model_key,
                    "Model": get_model_display_name(model_key),
                    "Target": target_key,
                    "MeanR2": float(result["MeanR2"]),
                    "StdR2": np.nan,
                    "Scores": np.array([result["MeanR2"]]),
                    "CustomDetails": result,
                })
            else:
                pipe = make_pipeline(model_key, target_key)
                scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
                rows.append({
                    "ModelKey": model_key,
                    "Model": get_model_display_name(model_key),
                    "Target": target_key,
                    "MeanR2": float(np.mean(scores)),
                    "StdR2": float(np.std(scores, ddof=1)),
                    "Scores": scores,
                    "CustomDetails": None,
                })

    df = pd.DataFrame(rows).sort_values(["Target","MeanR2"], ascending=[True, False]).reset_index(drop=True)
    return df

def summarize_wide(cv_long_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (cv_long_df
             .pivot(index=["ModelKey","Model"], columns="Target", values="MeanR2")
             .reset_index()
             .rename_axis(None, axis=1))
    for col in ["Y1", "Y2"]:
        if col not in pivot: pivot[col] = np.nan
    pivot["Combined"] = (pivot["Y1"] + pivot["Y2"]) / 2.0
    pivot = pivot.sort_values("Combined", ascending=False).reset_index(drop=True)
    return pivot[["ModelKey","Model","Y1","Y2","Combined"]]

def pick_best_per_target(cv_long_df: pd.DataFrame) -> Dict[str, str]:
    best = {}
    for tgt, df_t in cv_long_df.groupby("Target"):
        idx = df_t["MeanR2"].idxmax()
        best[tgt] = df_t.loc[idx,"ModelKey"]
    return best