from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def _safe_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def _slugify(name: str) -> str:
    s = "".join(c if c.isalnum() or c in ("-","_") else "-" for c in (name or "run")).strip("-")
    return s or "run"

def make_run_dir(run_name: str = "cv", base: str | Path = "./output") -> Path:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    run_dir = base / f"{_slugify(run_name)}_{_safe_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def _save(fig: plt.Figure, run_dir: Path, filename: str, dpi: int = 220) -> str:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / filename
    fig.savefig(path.as_posix(), bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return path.as_posix()

#######################################################################
################################ Plots ################################
#######################################################################

def plot_cv_summary(cv_long_df, summary_wide, importances_by_target=None):
    fig, axes = plt.subplots(1, 2, figsize=(15,6))
    fig.suptitle('Model Analysis', fontsize=16, fontweight='bold')

    axes = np.atleast_1d(axes).ravel()

    # (0,0) Wide summary bar chart per model 
    ax1 = axes[0]
    x = np.arange(len(summary_wide))
    width = 0.25
    ax1.bar(x - width,  summary_wide['Y1'], width, label='Y1', alpha=0.8, color='blue')
    ax1.bar(x,          summary_wide['Y2'], width, label='Y2', alpha=0.8, color='red')
    ax1.bar(x + width,  summary_wide['Combined'], width, label='Combined', alpha=0.8, color='green')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R^2 Score')
    ax1.set_title('Average CV R^2 by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_wide['Model'], rotation=0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (0,1) Boxplots per (model,target)
    ax2 = axes[1]
    labels = [f"{r.Model}-{r.Target}" for _, r in cv_long_df.iterrows()]
    ax2.boxplot(cv_long_df["Scores"].tolist(), labels=labels)
    ax2.tick_params(axis='x', rotation=75)
    ax2.set_title('CV Score Distributions')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig

def plot_cv_folds_lines(cv_long_df: pd.DataFrame) -> plt.Figure:
    n_folds = max(len(s) for s in cv_long_df["Scores"])
    x = np.arange(1, n_folds + 1)

    fig, ax = plt.subplots(figsize=(12,6))
    ax.set_title("Per-Fold R^2 by Model & Target", fontsize=14,fontweight="bold")
    for _, row in cv_long_df.iterrows():
        scores = np.asarray(row["Scores"]).ravel()
        if len(scores) < n_folds:
            scores = np.pad(scores, (0, n_folds - len(scores)), mode="edge")
        label = f"{row['Model']} - {row['Target']}"
        ax.plot(x, scores, marker="o", label=label, alpha=0.85)
    ax.set_xlabel("Fold")
    ax.set_ylabel("R^2")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig

def plot_cv_folds_heatmap(cv_long_df: pd.DataFrame) -> plt.Figure:
    rows, labels = [], []
    max_folds = max(len(s) for s in cv_long_df["Scores"])
    for _, r in cv_long_df.iterrows():
        s = np.asarray(r["Scores"]).ravel()
        if len(s) < max_folds:
            s = np.pad(s, (0, max_folds - len(s)), mode="edge")
        rows.append(s)
        labels.append(f"{r['Model']} - {r['Target']}")
    M = np.vstack(rows)

    fig, ax = plt.subplots(figsize=(1.8 * max_folds + 6, 0.4 * len(labels) + 3))
    im = ax.imshow(M, aspect="auto")
    ax.set_title("Per-fold R^2 Heatmap", fontsize = 14, fontweight="bold")
    ax.set_xticks(np.arange(max_folds))
    ax.set_xticklabels([f"F{c+1}" for c in range(max_folds)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="R^2")
    plt.tight_layout()
    return fig

def plot_missingness(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]) -> plt.Figure:
    tr_miss = train_df[feature_cols].isna().mean().values * 100.0
    te_miss = test_df[feature_cols].isna().mean().values * 100.0
    x = np.arange(len(feature_cols))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(10, 0.5 * len(feature_cols) + 6), 6))
    ax.bar(x - width / 2, tr_miss, width, label="Train %missing", alpha=0.05)
    ax.bar(x + width / 2, te_miss, width, label="Test %missing", alpha=0.05)
    ax.set_title("Feature Missingness (Train vs Test)", fontsize=14, fontweight="bold")
    ax.set_ylabel("% Missing")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_cols, rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_time_series_OP_vs_Y2(train_df: pd.DataFrame) -> Optional[plt.Figure]:
    if "time" not in train_df.columns:
        return None
    cols = [c for c in ["O", "P"] if c in train_df.columns]
    if not cols:
        return None
    
    t = train_df["time"].values
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.set_title("O/P vs Y2 over Time", fontsize=14, fontweight ="bold")
    for c in cols:
        ax1.plot(t, train_df[c].values, alpha=0.9, label=c)
    ax1.set_xlabel("time")
    ax1.set_ylabel("O/P")
    
    if "Y2" in train_df.columns:
        ax2 = ax1.twinx()
        ax2.plot(t, train_df["Y2"].values, linestyle="--", alpha=0.7, label="Y2")
        ax2.set_ylabel("Y2")
        lines, lbls = ax1.get_legend_handles_labels()
        lines2, lbls2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, lbls + lbls2, loc="best")
    else:
        ax1.legend(loc="best")
    
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def plot_lgb_importance_from_fitted(y2_enhanced_fitted, top_n: int = 20) -> Optional[plt.Figure]:
    try:
        models = y2_enhanced_fitted.lgb_models
        feat_names = getattr(y2_enhanced_fitted.X_lgb_test, "columns", None)
        if feat_names is None:
            return None
        feat_names = list(feat_names)
        gains = None
        for m in models:
            g = m.booster_.feature_importance(importance_type="gain")
            gains = g if gains is None else (gains + g)
        gains = gains / float(len(models))
    except Exception:
        return None
    
    order = np.argsort(gains)[::-1][:top_n]
    top_feats = [feat_names[i] for i in order]
    top_gain = gains[order]

    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(order) + 2)))
    ax.barh(y, top_gain, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(top_feats)
    ax.set_xlabel("Average Gain")
    ax.set_title(f"Y2 LightGBM Feature Importance (Top {len(order)})", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig

def plot_y2_blend_info(y2_enhanced_fitted) -> plt.Figure:
    simple_w = float(getattr(y2_enhanced_fitted, "simple_w", 0.5))
    use_meta = bool(getattr(y2_enhanced_fitted, "use_meta", False))

    fig, ax = plt.subplots(figsize=(6,4))
    ax.set_title("Y2 Blend Weights", fontsize=13, fontweight="bold")
    ax.bar([0], [simple_w], width=0.6, label="LGB weight", alpha=0.85)
    ax.bar([1], [1.0 - simple_w], width=0.6, label="Ridge weight", alpha=0.85)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["LGB", "Ridge"])
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    ax.text(0.5, 0.95, f"Meta-learning: {'ON' if use_meta else 'OFF'}",
            ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    return fig

def _short_params_label(p: Dict[str, Any]) -> str:
    return (f"lr={p['learning_rate']}, subs={p['subsample']}, ff={p['feature_fraction']}, "
            f"leaf={p['min_data_in_leaf']}, L2={p['reg_lambda']}, L1={p['reg_alpha']}, "
            f"leaves={p['num_leaves']}, it={p['n_estimators']}")

def plot_y2_enhanced_progress(best_history: List[Dict[str, Any]]) -> Optional[plt.Figure]:
    if not best_history:
        return None
    steps = [h["step"] for h in best_history]
    r2s = [h["r2"] for h in best_history]
    labs = [_short_params_label(h["params"]) for h in best_history]

    fig_w = max(10, int(1.2 * len(steps)))
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.plot(steps, r2s, marker="o")
    ax.set_xlabel("New-best timesteps")
    ax.set_ylabel("Best R2 on holdout")
    ax.set_title("Y2 Enhanced - Best R2 over Parameter Search", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(steps)
    ax.set_xticklabels(labs, rotation=45, ha="right")
    ax.annotate(f"Final best: {r2s[-1]:.4f}", xy=(steps[-1], r2s[-1]),
                xytest=(5,10), textcoords="offset points")
    plt.tight_layout()
    return fig

@dataclass(frozen=True)
class SaveResult:
    run_dir: str
    files: Dict[str, str]

def save_all_plots(
    *,
    run_name: str,
    cv_long_df: pd.DataFrame,
    summary_wide: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    fitted_by_target: Optional[Dict[str, Any]] = None,
    base_dir: str | Path = "./output",
) -> SaveResult:
    
    run_dir = make_run_dir(run_name=run_name, base=base_dir)
    saved: Dict[str, str] = {}

    # 1. Summary
    fig = plot_cv_summary(cv_long_df, summary_wide)
    saved["cv_summary"] = _save(fig, run_dir, "P1-cv_summary.png")

    # 2. Per-fold lines and heatmap
    fig = plot_cv_folds_lines(cv_long_df)
    saved["cv_folds_lines"] = _save(fig, run_dir, "P2-cv_folds_lines.png")

    fig = plot_cv_folds_heatmap(cv_long_df)
    saved["cv_folds_heatmap"] = _save(fig, run_dir, "P3-cv_folds_heatmap.png")

    # 3. Missingness
    fig = plot_missingness(train_df, test_df, feature_cols)
    saved["missingness"] = _save(fig, run_dir, "P4-missingness.png")

    # 4. OP vs Y2 over time
    fig = plot_time_series_OP_vs_Y2(train_df)
    if fig is not None:
        saved["OP_vs_Y2"] = _save(fig, run_dir, "P5-OP_vs_Y2.png")

    # 5. Enhanced Y2 extras
    try:
        from src.y2_enhanced import Y2EnhancedFitted
        y2_model = fitted_by_target.get("Y2") if fitted_by_target else None
        if y2_model is not None and isinstance(y2_model, Y2EnhancedFitted):
            fig = plot_lgb_importance_from_fitted(y2_model, top_n=25)
            if fig is not None:
                saved["y2_lgb_importance"] = _save(fig, run_dir, "P6-y2_lgb_importance.png")
            fig = plot_y2_blend_info(y2_model)
            saved["y2_blend"] = _save(fig, run_dir, "P7-y2_blend.png")
        try:
            row = (cv_long_df[(cv_long_df["ModelKey"] == "lgb_y2_enhanced") &
                                (cv_long_df["Target"] == "Y2")].iloc[0])
            details = row["CustomDetails"]
            if isinstance(details, dict):
                hist = (details.get("Details") or {}).get("best_history", [])
                fig2 = plot_y2_enhanced_progress(hist)
                if fig2 is not None:
                    saved["y2_enhanced_progress"] = _save(fig2, run_dir, "P8-y2_enhanved_progress.png")
        except Exception:
            pass
    except Exception:
        pass

    manifest = {
        "run_dir": str(run_dir),
        "files": saved,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
    } 
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return SaveResult(run_dir=str(run_dir), files=saved)