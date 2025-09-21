import numpy as np

def purged_splits(n, n_splits=3, gap=0, min_train_frac=0.05):
    idx = np.arange(n)
    boundaries = np.linspace(0, n, n_splits + 1, dtype=int)
    min_train = max(1, int(min_train_frac * n))
    for i in range(1, len(boundaries)):
        val_start = boundaries[i - 1]
        val_end = boundaries[i]
        va_idx = idx[val_start:val_end]
        tr_end = max(min_train, val_start - gap)
        tr_idx = idx[:tr_end]
        if len(tr_idx) == 0:
            continue
        yield tr_idx, va_idx

def holdout_split(n, holdout_frac=0.2):
    cut = int(np.floor(n*(1.0-holdout_frac)))
    return np.arange(cut), np.arange(cut, n)