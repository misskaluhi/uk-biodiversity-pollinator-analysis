"""
main.py
=======
Complete pipeline for the UK Biodiversity & Pollinator Analysis.

Executes all phases sequentially:
  1. Data preprocessing – load, clean, merge five UK biodiversity datasets
  2. Genetic Algorithm – optimise feature selection, year range, interpolation
  3. Random Forest – train and evaluate the predictive model
  4. Visualisations – generate all required plots
  5. Output files – save CSV results, figures, and summary report
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# Add code directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_preprocessing import merge_datasets, interpolate_data
from genetic_algorithm import (
    run_ga, apply_best_solution, decode_chromosome,
    FEATURE_NAMES, TARGET_COL
)
from random_forest_model import train_and_evaluate

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')


def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Visualisation functions
# ---------------------------------------------------------------------------

def plot_pollinator_trend(master_df):
    """Plot the pollinator index time series (1980-2024)."""
    df = master_df.dropna(subset=['Pollinator_Index'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['Year'], df['Pollinator_Index'], 'b-o', markersize=4, linewidth=1.5,
            label='Pollinator Index')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Baseline (1980=100)')

    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Pollinator Index (1980 = 100)', fontsize=12)
    ax.set_title('Status of Pollinating Insects in the UK (1980-2024)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1978, 2026)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'pollinator_trend.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_correlation_heatmap(master_df):
    """Plot correlation heatmap of all features and target."""
    # Use only numeric columns, drop Year
    numeric_cols = [c for c in master_df.columns if c not in ['Year', 'Year_Numeric']]
    # Interpolate linearly for correlation purposes
    df_interp = master_df[numeric_cols].interpolate(method='linear').dropna()

    corr = df_interp.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=14)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'feature_correlation_heatmap.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ga_fitness(history_df):
    """Plot GA fitness evolution over generations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(history_df['Generation'], history_df['Best_Fitness'],
            'b-', linewidth=2, label='Generation Best R²')
    ax.plot(history_df['Generation'], history_df['Mean_Fitness'],
            'r--', linewidth=1.5, label='Generation Mean R²')
    ax.fill_between(
        history_df['Generation'],
        history_df['Mean_Fitness'] - history_df['Std_Fitness'],
        history_df['Mean_Fitness'] + history_df['Std_Fitness'],
        alpha=0.2, color='red', label='±1 Std Dev'
    )
    ax.plot(history_df['Generation'], history_df['Overall_Best'],
            'g-', linewidth=2, label='Overall Best R²')

    ax.set_xlabel('Generation', fontsize=12)
    ax.set_ylabel('Fitness (R² Score)', fontsize=12)
    ax.set_title('Genetic Algorithm Fitness Evolution', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'ga_fitness_evolution.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_feature_importance(fi_df):
    """Plot bar chart of Random Forest feature importances."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi_df)))
    bars = ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors)

    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Random Forest Feature Importances', fontsize=14)
    ax.invert_yaxis()

    # Add value labels
    for bar, val in zip(bars, fi_df['Importance']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'feature_importance_plot.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_predictions_vs_actual(predictions_df):
    """Plot scatter of predicted vs actual values with regression line."""
    fig, ax = plt.subplots(figsize=(8, 8))

    train = predictions_df[predictions_df['Split'] == 'Train']
    test = predictions_df[predictions_df['Split'] == 'Test']

    ax.scatter(train['Actual'], train['Predicted'], c='blue', alpha=0.7,
               s=60, label=f'Train (n={len(train)})', edgecolors='white', linewidth=0.5)
    ax.scatter(test['Actual'], test['Predicted'], c='red', alpha=0.7,
               s=60, label=f'Test (n={len(test)})', edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    all_vals = pd.concat([predictions_df['Actual'], predictions_df['Predicted']])
    mn, mx = all_vals.min(), all_vals.max()
    margin = (mx - mn) * 0.05
    ax.plot([mn - margin, mx + margin], [mn - margin, mx + margin],
            'k--', alpha=0.5, label='Perfect prediction')

    ax.set_xlabel('Actual Pollinator Index', fontsize=12)
    ax.set_ylabel('Predicted Pollinator Index', fontsize=12)
    ax.set_title('Predicted vs Actual Pollinator Index', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'predictions_vs_actual.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_residuals(predictions_df):
    """Plot residual analysis (residuals vs predicted and distribution)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Residuals vs Predicted
    ax1 = axes[0]
    train = predictions_df[predictions_df['Split'] == 'Train']
    test = predictions_df[predictions_df['Split'] == 'Test']

    ax1.scatter(train['Predicted'], train['Residual'], c='blue', alpha=0.7,
                s=60, label='Train', edgecolors='white', linewidth=0.5)
    ax1.scatter(test['Predicted'], test['Residual'], c='red', alpha=0.7,
                s=60, label='Test', edgecolors='white', linewidth=0.5)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Predicted Pollinator Index', fontsize=12)
    ax1.set_ylabel('Residual', fontsize=12)
    ax1.set_title('Residuals vs Predicted Values', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Residual distribution
    ax2 = axes[1]
    ax2.hist(predictions_df['Residual'], bins=15, color='steelblue',
             edgecolor='white', alpha=0.8)
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Residual', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Residual Distribution', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Residual Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'residuals_plot.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def generate_summary(config, metrics, fi_df, master_df, final_df):
    """Generate and save a text summary of all results."""
    top5 = fi_df.head(5)

    summary = f"""=====================================
RESULTS SUMMARY
=====================================

DATA PREPROCESSING:
- Total years in merged dataset: {len(master_df)}
- Year range in merged dataset: {master_df['Year'].min()} - {master_df['Year'].max()}
- All features available: {FEATURE_NAMES}
- Missing values handled: Interpolation ({config['interpolation_method']})

GENETIC ALGORITHM OPTIMISATION:
- Best year range: {config['year_start']} - {config['year_end']}
- Selected features: {config['selected_features']}
- Interpolation method: {config['interpolation_method']}
- Number of data points after filtering: {config['final_rows']}

RANDOM FOREST PERFORMANCE:
- Training R²: {metrics['train_r2']:.4f}
- Test R²: {metrics['test_r2']:.4f}
- Cross-validation R² (mean ± std): {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}
- Test RMSE: {metrics['test_rmse']:.4f}
- Test MAE: {metrics['test_mae']:.4f}

TOP {len(top5)} IMPORTANT FEATURES:
"""
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        summary += f"  {i}. {row['Feature']}: {row['Importance']:.4f}\n"

    summary += f"""
KEY FINDINGS:
- The pollinator index has declined from 100 (baseline in 1980) to approximately {master_df.loc[master_df['Year']==2024, 'Pollinator_Index'].values[0]:.1f} in 2024.
- The GA selected {len(config['selected_features'])} features and optimised the analysis window to {config['year_start']}-{config['year_end']}.
- {config['interpolation_method'].capitalize()} interpolation was used to handle missing values across different dataset time ranges.
- The Random Forest model achieved an R² of {metrics['test_r2']:.4f} on the test set.
- Cross-validation R² of {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f} indicates {'robust' if metrics['cv_std'] < 0.15 else 'moderate'} model stability.
=====================================
"""
    path = os.path.join(PROJECT_ROOT, 'results_summary.txt')
    with open(path, 'w') as f:
        f.write(summary)
    print(f"  Saved: {path}")

    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    """Execute the full analysis pipeline."""
    ensure_dirs()

    # -----------------------------------------------------------------------
    # Phase 1: Data Preprocessing
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("PHASE 1: DATA PREPROCESSING")
    print("=" * 60)

    master_df, missing_summary = merge_datasets()

    # Save the raw merged dataset
    master_df.to_csv(os.path.join(DATA_DIR, 'cleaned_merged_dataset.csv'), index=False)
    print(f"\n  Saved cleaned_merged_dataset.csv")

    # -----------------------------------------------------------------------
    # Phase 2: Genetic Algorithm Optimisation
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2: GENETIC ALGORITHM OPTIMISATION")
    print("=" * 60)
    print("\nRunning GA (pop=50, gen=100, cx=0.8, mut=0.1)...\n")

    best_chromosome, best_fitness, ga_history = run_ga(
        master_df,
        pop_size=50,
        n_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_k=3,
        random_state=42,
        verbose=True
    )

    # Save GA history
    ga_history.to_csv(os.path.join(DATA_DIR, 'ga_optimization_results.csv'), index=False)
    print(f"\n  Saved ga_optimization_results.csv")

    # -----------------------------------------------------------------------
    # Phase 3: Prepare Final Dataset and Train Random Forest
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 3: RANDOM FOREST MODEL")
    print("=" * 60)

    final_df, selected_features, config = apply_best_solution(master_df, best_chromosome)
    print(f"\n  Final dataset: {final_df.shape[0]} rows, {len(selected_features)} features")
    print(f"  Year range: {config['year_start']}-{config['year_end']}")
    print(f"  Features: {selected_features}")
    print(f"  Interpolation: {config['interpolation_method']}\n")

    results = train_and_evaluate(final_df, selected_features, random_state=42)

    # Save predictions
    results['predictions_df'].to_csv(
        os.path.join(DATA_DIR, 'model_predictions.csv'), index=False
    )
    print(f"\n  Saved model_predictions.csv")

    # Save feature importances
    results['feature_importance_df'].to_csv(
        os.path.join(DATA_DIR, 'feature_importance.csv'), index=False
    )
    print(f"  Saved feature_importance.csv")

    # -----------------------------------------------------------------------
    # Phase 4: Visualisations
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 4: GENERATING VISUALISATIONS")
    print("=" * 60)

    plot_pollinator_trend(master_df)
    plot_correlation_heatmap(master_df)
    plot_ga_fitness(ga_history)
    plot_feature_importance(results['feature_importance_df'])
    plot_predictions_vs_actual(results['predictions_df'])
    plot_residuals(results['predictions_df'])

    # -----------------------------------------------------------------------
    # Phase 5: Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 5: RESULTS SUMMARY")
    print("=" * 60)

    summary = generate_summary(
        config, results['metrics'],
        results['feature_importance_df'],
        master_df, final_df
    )
    print(summary)

    print("Pipeline complete! All outputs saved.")
    print(f"  Code:    {os.path.join(PROJECT_ROOT, 'code')}/")
    print(f"  Data:    {DATA_DIR}/")
    print(f"  Figures: {FIGURES_DIR}/")


if __name__ == '__main__':
    main()
