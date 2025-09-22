import pandas as pd
from pathlib import Path
from typing import List, Tuple

from src.config import DATA_DIR, BASE_FEATURES

_BASE_FEATS = list("ABCDEFGHIJKLMN")

def _read_with_headers_or_fallback(path: Path, expected_cols: List[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
        if set(expected_cols).issubset(df.columns):
            return df
        raise ValueError("Header mismatch")
    except Exception:
        return pd.read_csv(path, skiprows=1, names=expected_cols, low_memory=False)
    
def _read_extra_op(path: Path, expected_len: int) -> pd.DataFrame:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        try:
            df = pd.read_csv(path, skiprows=1, names=["O", "P"], low_memory=False)
        except Exception:
            return None
        
    keep = [c for c in ["O", "P"] if c in df.columns]
    if not keep:
        return None
    df = df[keep].copy()

    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.reset_index(drop=True)
    if len(df) < expected_len:
        pad = pd.DataFrame({c: [pd.NA] * (expected_len - len(df)) for c in keep})
        df = pd.concat([df, pad], axis=0, ignore_index=True)
    elif len(df) > expected_len:
        df = df.iloc[:expected_len].reset_index(drop=True)
    
    return df

def load_train_test() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    
    train_expected = ['time'] + BASE_FEATURES + ['Y1','Y2']
    test_expected = ['id', 'time'] + BASE_FEATURES

    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    train_df = _read_with_headers_or_fallback(train_path, train_expected)
    test_df = _read_with_headers_or_fallback(test_path, test_expected)

    tr_new = _read_extra_op(DATA_DIR / "train_new.csv", len(train_df))
    te_new = _read_extra_op(DATA_DIR / "test_new.csv", len(test_df))

    if tr_new is not None:
        for c in tr_new.columns:
            train_df[c] = tr_new[c]
    if te_new is not None:
        for c in te_new.columns:
            test_df[c] = te_new[c]
        
    extras = [c for c in ["O", "P"] if c in train_df.columns]
    feature_cols = BASE_FEATURES + extras

    return train_df, test_df, feature_cols

def build_matrices(train_df : pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]):
    X_train = train_df[feature_cols]
    y_by_target = {"Y1": train_df["Y1"], "Y2": train_df["Y2"]}
    X_test = test_df[feature_cols]
    return X_train, y_by_target, X_test