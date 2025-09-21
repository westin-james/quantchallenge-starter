from sklearn.model_selection import TimeSeriesSplit
from .config import N_SPLITS

def make_timeseries_cv():
    return TimeSeriesSplit(n_splits=N_SPLITS)