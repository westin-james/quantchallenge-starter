from pathlib import Path

DATA_DIR = Path("./data")
SUBMISSION_PATH = Path("preds.csv")

BASE_FEATURES = list("ABCDEFGHIJKLMN")
# COLUMN_NAMES = ['time','A','B','C','D','E','F','G','H','I','J','K','L','M','N','Y1','Y2']
# TEST_COLUMN_NAMES = ['id','time','A','B','C','D','E','F','G','H','I','J','K','L','M','N']
# FEATURE_COLS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

N_SPLITS = 3
RANDOM_STATE = 42

RIDGE_ALPHA = 100.0

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

LGB_BASE = dict(
    objective="regression",
    metric="rmse",
    n_estimators=800,
    num_leaves=31,
    learning_rate=0.05,
    subsample=0.7,
    feature_fraction=0.7,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=-1
)

MODEL_KEYS = ["linreg", "ridge", "rf", "lgb", "y1_advanced_v11", "lgb_y2_enhanced"]
TARGET_KEYS = ["Y1", "Y2"]