import pandas as pd
from .config import DATA_DIR, COLUMN_NAMES, TEST_COLUMN_NAMES, FEATURE_COLS

def load_train_test():
    try:
        train_path = DATA_DIR / "train.csv"
        train_df = pd.read_csv(train_path, skiprows=1, names=COLUMN_NAMES)
    except Exception as e:
        print(f"Error loading train.csv\n{e}")
        exit()
    try:
        test_path = DATA_DIR / "test.csv"
        test_df = pd.read_csv(test_path, skiprows=1, names=TEST_COLUMN_NAMES)
    except Exception as e:
        print(f"Error loading train.csv\n{e}")
        exit()

    return train_df, test_df

def build_matrices(train_df, test_df):
    X_train = train_df[FEATURE_COLS]
    y1 = train_df['Y1']
    y2 = train_df['Y2']
    X_test = test_df[FEATURE_COLS]
    return X_train, {"Y1": y1, "Y2": y2}, X_test