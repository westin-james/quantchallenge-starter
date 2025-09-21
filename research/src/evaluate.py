from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

from .models import make_pipeline, get_model_display_name

def crossval_grid(X, y_by_target: Dict[str, pd.Series], cv, model_keys: List[str], 
                  scoring: str = "r2") -> pd.DataFrame:
    rows=[]
    for model_key in model_keys:
        for target_key, y in y_by_target.items():
            pipe = make_pipeline(model_key, target_key)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring=scoring)
            rows.append({
                "ModelKey": model_key,
                "Model": get_model_display_name(model_key),
                "Target": target_key,
                "MeanR2": np.mean(scores),
                "StdR2": np.std(scores, ddof=1),
                "Scores": scores,
            })
    df = pd.DataFrame(rows).sort_values(["Target","MeanR2"], ascending=[True, False]).reset_index(drop=True)
    return df

def summarize_wide(cv_long_df: pd.DataFrame) -> pd.DataFrame:
    pivot = (cv_long_df
             .pivot(index=["ModelKey","Model"], columns="Target", values="MeanR2")
             .reset_index()
             .rename_axis(None, axis=1))
    if "Y1" not in pivot: pivot["Y1"] = np.nan
    if "Y2" not in pivot: pivot["Y2"] = np.nan
    pivot["Combined"] = (pivot["Y1"] ++ pivot["Y2"]) / 2.0
    pivot = pivot.sort_values("Combined", ascending=False).reset_index(drop=True)
    return pivot[["ModelKey","Model","Y1","Y2","Combined"]]

def pick_best_per_target(cv_long_df: pd.DataFrame) -> Dict[str, str]:
    best = {}
    for tgt, df_t in cv_long_df.groupby("Target"):
        idx = df_t["MeanR2"].idxmax()
        best[tgt] = cv_long_df.loc[idx,"ModelKey"]
    return best