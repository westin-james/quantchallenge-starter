from src.data import load_train_test, build_matrices
from src.cv import make_timeseries_cv
from src.evaluate import crossval_grid, summarize_wide, pick_best_per_target
from src.train import train_best_models
from src.predict import predict_submission
from src.config import MODEL_KEYS

# Custom evaluator hook for lgb_y2_enhanced
from src.y2_enhanced import EnhancedConfig, evaluate_y2_enhanced_cv

from src.visualize import save_all_plots

def lgb_y2_enhanced_eval_adapter(target_key, X, y_by_target, ctx):
    # Only applies to Y2
    if target_key != "Y2":
        return None
    cfg = EnhancedConfig()
    train_df = ctx["train_df"]; test_df = ctx["test_df"]
    return evaluate_y2_enhanced_cv(train_df, test_df, y_by_target["Y1"], y_by_target["Y2"], cfg)

def main():
    ######################## 1. LOADING DATA ########################
    print("\n ####################### 1. LOADING DATA ########################################\n")
    train_df, test_df, feature_cols = load_train_test()
    X_train, y_by_target, X_test = build_matrices(train_df, test_df, feature_cols)

    ######################## 2. CROSS VALIDATION SETUP ########################
    print("\n######################## 2. CROSS VALIDATION SETUP ###############################\n")
    tscv = make_timeseries_cv()
    print("Using 3-fold time series crossval")

    ######################## 3. EVALUATE MODELS ########################
    print("\n######################## 3. EVALUATE ALL MODELS (Y1 & Y2) ########################\n")
    custom = {"lgb_y2_enhanced": lgb_y2_enhanced_eval_adapter}
    ctx = {"train_df": train_df, "test_df": test_df}
    cv_long = crossval_grid(X_train, y_by_target, tscv, MODEL_KEYS, scoring="r2", 
                            custom_evaluators=custom, ctx=ctx)
    
    summary = summarize_wide(cv_long)
    print(summary.to_string(index=False))

    ######################## 4. SELECT BEST MODELS ########################
    print("\n######################## 4. SELECT BEST PER TARGET ###############################\n")
    selections = pick_best_per_target(cv_long)
    print(f"Best model per target: {selections}")

    ######################## 5. TRAIN FINAL MODELS ########################
    print("\n######################## 5. TRAIN FINAL MODELS ###################################\n")
    fitted_by_target = train_best_models(X_train, y_by_target, selections, ctx)

    ######################## 6. PREDICTIONS ########################
    print("\n######################## 6. PREDICTIONS ##########################################\n")

    out_path = predict_submission(fitted_by_target, X_test, test_df)
    print(f"Predictions saved to: {out_path}")

    ######################## 7. VISUALIZATION ########################
    print("\n######################## 7. VISUALIZATIONS #######################################\n")
    res = save_all_plots(
        run_name="model_eval",
        cv_long_df=cv_long,
        summary_wide=summary,
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        fitted_by_target=fitted_by_target,
        base_dir="./output",
    )
    print(f"Saved plots to: {res.run_dir}")
    for key, path in res.files.items():
        print(f"  - {key}: {path}")

    print("\nANALYSIS COMPLETE")
    print(f"Best combined performance: {summary['Combined'].iloc[0]:.4f}")

if __name__ == "__main__":
    main()