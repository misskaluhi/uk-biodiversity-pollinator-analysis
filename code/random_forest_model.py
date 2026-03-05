"""
random_forest_model.py
======================
Trains and evaluates a Random Forest regressor for predicting pollinator
decline, using the features and year range selected by the Genetic Algorithm.

Outputs include:
  - Model performance metrics (R², RMSE, MAE)
  - Cross-validation scores
  - Feature importance rankings
  - Predictions vs. actuals
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


TARGET_COL = 'Pollinator_Index'


def train_and_evaluate(final_df, selected_features, random_state=42):
    """
    Train a Random Forest model and evaluate its performance.

    Parameters
    ----------
    final_df : pd.DataFrame
        The prepared dataset with Year, target, and selected features.
    selected_features : list[str]
        Column names of features to use.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    results : dict
        Dictionary containing:
        - 'model': the fitted RandomForestRegressor
        - 'metrics': dict of evaluation metrics
        - 'predictions_df': DataFrame of actual vs predicted
        - 'feature_importance_df': DataFrame of feature importances
        - 'cv_scores': array of cross-validation R² scores
    """
    X = final_df[selected_features].values
    y = final_df[TARGET_COL].values

    # 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set:     {X_test.shape[0]} samples")
    print(f"  Features:     {len(selected_features)}")

    # Initialise and train model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state
    )
    rf.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Cross-validation on training set
    n_cv = min(5, X_train.shape[0])
    cv_scores = cross_val_score(rf, X_train, y_train, cv=n_cv, scoring='r2')
    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)

    metrics = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'cv_scores': cv_scores.tolist()
    }

    print(f"\n  --- Random Forest Performance ---")
    print(f"  Training R²:             {train_r2:.4f}")
    print(f"  Test R²:                 {test_r2:.4f}")
    print(f"  Cross-validation R²:     {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"  Test RMSE:               {test_rmse:.4f}")
    print(f"  Test MAE:                {test_mae:.4f}")

    # Feature importances
    importances = rf.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    print(f"\n  --- Feature Importances ---")
    for _, row in fi_df.iterrows():
        print(f"  {row['Feature']:30s} {row['Importance']:.4f}")

    # Predictions dataframe (full dataset)
    y_all_pred = rf.predict(final_df[selected_features].values)
    predictions_df = pd.DataFrame({
        'Year': final_df['Year'].values,
        'Actual': final_df[TARGET_COL].values,
        'Predicted': y_all_pred
    })

    # Also tag train/test
    train_indices = set(range(len(final_df))) - set(
        np.where(np.isin(final_df[TARGET_COL].values, y_test))[0]
    )
    predictions_df['Split'] = 'Test'
    # Use the actual train/test years for labelling
    test_years = final_df.iloc[
        np.where(np.isin(y, y_test))[0]
    ]['Year'].values
    predictions_df.loc[~predictions_df['Year'].isin(test_years), 'Split'] = 'Train'

    # Residuals
    predictions_df['Residual'] = predictions_df['Actual'] - predictions_df['Predicted']

    return {
        'model': rf,
        'metrics': metrics,
        'predictions_df': predictions_df,
        'feature_importance_df': fi_df,
        'cv_scores': cv_scores,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_train_pred': y_train_pred,
        'y_test_pred': y_test_pred,
    }


# ---------------------------------------------------------------------------
# Main entry point (standalone testing)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from data_preprocessing import merge_datasets
    from genetic_algorithm import run_ga, apply_best_solution

    master_df, _ = merge_datasets()
    print("\nRunning GA optimisation...")
    best_chrom, best_fit, history = run_ga(
        master_df, pop_size=30, n_generations=50, random_state=42
    )

    final_df, features, config = apply_best_solution(master_df, best_chrom)
    print(f"\nTraining Random Forest with {len(features)} features...")
    results = train_and_evaluate(final_df, features)
