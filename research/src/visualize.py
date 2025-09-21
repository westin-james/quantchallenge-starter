import numpy as np
import matplotlib.pyplot as plt

def plot_cv_summary(cv_long_df, summary_wide, importances_by_target=None):
    fig, axes = plt.subplots(2, 2, figsize=(15,10))
    fig.suptitle('Model Analysis', fontsize=16, fontweight='bold')

    # (0,0) Wide summary bar chart per model 
    ax1 = axes[0,0]
    x = np.arange(len(summary_wide))
    width = 0.25
    ax1.bar(x - width, summary_wide['Y1'], width, label='Y1', alpha=0.8, color='blue')
    ax1.bar(x, summary_wide['Y2'], width, label='Y2', alpha=0.8, color='red')
    ax1.bar(x + width, summary_wide['Combined'], width, label='Combined', alpha=0.8, color='green')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R^2 Score')
    ax1.set_title('Average CV R^2 by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(summary_wide['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # (0,1) Boxplots per (model,target)
    ax2 = axes[0,1]
    labels = [f"{r.Model}-{r.Target}" for _, r in cv_long_df.iterrows()]
    ax2.boxplot(cv_long_df["Scores"].tolist(), labels=labels)
    ax2.tick_params(axis='x', rotation=75)
    ax2.set_title('CV Score Distributions')
    ax2.grid(True, alpha=0.3)

    # (1,0) Importances Y1
    ax3 = axes[1,0]
    if importances_by_target and "Y1" in importances_by_target:
        top = importances_by_target["Y1"].head(8)
        ax3.barh(range(len(top)), top["importance"])
        ax3.set_yticks(range(len(top)))
        ax3.set_yticklabels(top["feature"])
        ax3.set_title("Y1 Feature Importances (best model)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.axis('off')
        ax3.set_title("Y1 Feature Importances (N/A)")

    # (1,0) Importances Y2
    ax4 = axes[1,1]
    if importances_by_target and "Y2" in importances_by_target:
        top = importances_by_target["Y2"].head(8)
        ax4.barh(range(len(top)), top["importance"])
        ax4.set_yticks(range(len(top)))
        ax4.set_yticklabels(top["feature"])
        ax4.set_title("Y2 Feature Importances (best model)")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.axis('off')
        ax4.set_title("Y2 Feature Importances (N/A)")

    plt.tight_layout()
    plt.show()

    # ax2 = axes[1,0]
    # top_features_y1 = rf_y1_importance.head(8)
    # ax2.barh(range(len(top_features_y1)), top_features_y1['importance'], color='blue', alpha=0.7)
    # ax2.set_yticks(range(len(top_features_y1)))
    # ax2.set_yticklabels(top_features_y1['feature'])
    # ax2.set_xlabel("Importance")
    # ax2.set_title('Y1 Feature Importance (Random Forest)')
    # ax2.grid(True, alpha=0.3)

    # # Y2 feature importance
    # ax3 = axes[1,1]
    # top_features_y2 = rf_y2_importance.head(8)
    # ax3.barh(range(len(top_features_y2)), top_features_y2['importance'], color='red', alpha=0.7)
    # ax3.set_yticks(range(len(top_features_y2)))
    # ax3.set_yticklabels(top_features_y2['feature'])
    # ax3.set_xlabel("Importance")
    # ax3.set_title('Y2 Feature Importance (Random Forest)')
    # ax3.grid(True, alpha=0.3)

    # # CV scores distribution
    # ax4 = axes[0,1]
    # cv_data = [y1_linear_cv, y1_rf_cv, y2_linear_cv, y2_rf_cv]
    # cv_labels = ['Linear Y1', 'RF Y1', 'Linear Y2', 'RF Y2']
    # ax4.boxplot(cv_data, tick_labels=cv_labels)
    # ax4.set_ylabel("R^2 Score")
    # ax4.set_title('Cross Validation Score Distribution')
    # ax4.tick_params(axis='x', rotation=45)
    # ax4.grid(True, alpha=0.3)

