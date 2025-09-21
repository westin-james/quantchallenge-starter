import pandas as pd
from .config import SUBMISSION_PATH, FEATURE_COLS

def predict_submission(fitted_by_target, X_test, test_df):
    # Y1: sklearn pipeline
    y1_pred = fitted_by_target["Y1"].predict(X_test)

    # Y2: either sklearn pipeline OR enhanced composite
    y2_model = fitted_by_target["Y2"]
    try:
        # Enhanced path exposes .predict(X_lgb, X_ridge, estra_meta)
        from .feature_eng import make_enhanced_y2_features
        # Build enhanced features with Y1 test preds as meta feature
        # (train_df not needed here; we re-create features on test via the same function in training)
        # minimal re-creation: assume the trainer expects just X_test for ridge part and enhanced features for LGB
        # For deterministic inference, recompute enhanced features from original test_df alone is insufficient
        # in practice we'd pass cached transformers; here we rebuild using train+test context from ctx.
        raise AttributeError # fallthrough to simple path unless ctx wiring is added
    except Exception:
        # simple: if composite: it provides a single-argument .predict; if sklearn, same.
        y2_pred = y2_model.predict(X_test)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'Y1': y1_pred,
        'Y2': y2_pred
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
    return SUBMISSION_PATH
