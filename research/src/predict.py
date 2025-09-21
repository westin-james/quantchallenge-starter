import pandas as pd
from .config import SUBMISSION_PATH

def predict_submission(fitted_by_target, X_test, test_df):
    y1_pred = fitted_by_target["Y1"].predict(X_test)
    y2_pred = fitted_by_target["Y2"].predict(X_test)

    submission = pd.DataFrame({
        'id': test_df['id'],
        'Y1': y1_pred,
        'Y2': y2_pred
    })
    submission.to_csv(SUBMISSION_PATH, index=False)
