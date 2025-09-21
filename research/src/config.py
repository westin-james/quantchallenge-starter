from pathlib import Path

DATA_DIR = Path("./data")
SUBMISSION_PATH = Path("preds.csv")

COLUMN_NAMES = ['time','A','B','C','D','E','F','G','H','I','J','K','L','M','N','Y1','Y2']
TEST_COLUMN_NAMES = ['id','time','A','B','C','D','E','F','G','H','I','J','K','L','M','N']
FEATURE_COLS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N']

N_SPLITS = 3
RIDGE_ALPHA = 100.0
RANDOM_STATE = 42
RF_PARAMS = dict(
    n_estimators=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

MODEL_KEYS = ["linreg", "ridge", "rf"]
TARGET_KEYS = ["Y1", "Y2"]