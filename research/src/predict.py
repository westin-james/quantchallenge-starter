import pandas as pd
from .config import SUBMISSION_PATH
from src.y2_enhanced import Y2EnhancedFitted
from src.y1_advanced_v11 import Y1AdvancedFitted

def predict_submission(fitted_by_target, X_test, test_df):
    # Y1 predictions
    y1_model = fitted_by_target["Y1"]
    if isinstance(y1_model, Y1AdvancedFitted):
        y1_pred = y1_model.predict(test_df)
    else:
        y1_pred = y1_model.predict(X_test)
    # Y2 predictions
    y2_model = fitted_by_target["Y2"]

    if isinstance(y2_model, Y2EnhancedFitted):
        X_lgb   = y2_model.X_lgb_test
        X_ridge = test_df[y2_model.ridge_feats]
        y2_pred = y2_model.predict(X_lgb, X_ridge, extra_meta=y1_pred)
    else:
        y2_pred = y2_model.predict(X_test)

    submission = pd.DataFrame({"id": test_df["id"], "Y1": y1_pred, "Y2": y2_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    return SUBMISSION_PATH
