import numpy as np
import matplotlib.pyplot as plt

def plot_all(results, rf_y1_importance, rf_y2_importance, y1_linear_cv, y1_rf_cv, y2_linear_cv, y2_rf_cv):
    fig, axes = plt.subplots(2, 2, figsize=(15,10))
    fig.suptitle('Model Analysis', fontsize=16, fontweight='bold')

    # Model performance comparison 
    ax1 = axes[0,0]
    x = np.arange(len(results))
    width = 0.25
    ax1.bar(x - width, results['Y1_R^2'], width, label='Y1', alpha=0.8, color='blue')
    ax1.bar(x, results['Y2_R^2'], width, label='Y2', alpha=0.8, color='red')
    ax1.bar(x + width, results['Combined'], width, label='Combined', alpha=0.8, color='green')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('R^2 Score')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results['Model'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Y1 feature importance
    ax2 = axes[1,0]
    top_features_y1 = rf_y1_importance.head(8)
    ax2.barh(range(len(top_features_y1)), top_features_y1['importance'], color='blue', alpha=0.7)
    ax2.set_yticks(range(len(top_features_y1)))
    ax2.set_yticklabels(top_features_y1['feature'])
    ax2.set_xlabel("Importance")
    ax2.set_title('Y1 Feature Importance (Random Forest)')
    ax2.grid(True, alpha=0.3)

    # Y2 feature importance
    ax3 = axes[1,1]
    top_features_y2 = rf_y2_importance.head(8)
    ax3.barh(range(len(top_features_y2)), top_features_y2['importance'], color='red', alpha=0.7)
    ax3.set_yticks(range(len(top_features_y2)))
    ax3.set_yticklabels(top_features_y2['feature'])
    ax3.set_xlabel("Importance")
    ax3.set_title('Y2 Feature Importance (Random Forest)')
    ax3.grid(True, alpha=0.3)

    # CV scores distribution
    ax4 = axes[0,1]
    cv_data = [y1_linear_cv, y1_rf_cv, y2_linear_cv, y2_rf_cv]
    cv_labels = ['Linear Y1', 'RF Y1', 'Linear Y2', 'RF Y2']
    ax4.boxplot(cv_data, tick_labels=cv_labels)
    ax4.set_ylabel("R^2 Score")
    ax4.set_title('Cross Validation Score Distribution')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()