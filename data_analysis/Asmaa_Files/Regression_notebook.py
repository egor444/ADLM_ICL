import os
import time
import csv
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge, LassoCV, ElasticNetCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Configuration settings
CONFIG = {
    'pca_variance_threshold': 0.95,
    'random_state': 42,
    'verbose': True,
    'log_path': 'model_run_log.csv',
}

# Logging utilities
def log_step(step_name):
    if CONFIG['verbose']:
        print(f"\n[STEP] {step_name}")
    return time.time()

def log_elapsed(start_time, step_name, log_dict=None):
    elapsed = time.time() - start_time
    if CONFIG['verbose']:
        print(f"[DONE] {step_name} in {elapsed:.2f} sec")
    if log_dict is not None:
        log_dict[step_name] = elapsed

# Initialize environment

def setup_environment():
    os.environ['PYTHONHASHSEED'] = str(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])
    return {'timing_log': {}}

# Load and merge datasets
def load_and_merge(paths, label):
    print(f"\nLoading and merging: {label}")
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df = df.drop(columns=[col for col in ['eid'] if col in df.columns])
    df = df.dropna()
    X = df.drop(columns=['age'])
    y = df['age']
    return {'X': X, 'y': y}

# Model selection
def get_models():
    return {
        "HistGBR": HistGradientBoostingRegressor(random_state=CONFIG['random_state']),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=CONFIG['random_state'], verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=CONFIG['random_state'], verbose=-1),
        "Ridge": Ridge(),
        "SVR": SVR(),
        "LassoCV": LassoCV(cv=5, random_state=CONFIG['random_state']),
        "ElasticNetCV": ElasticNetCV(cv=5, random_state=CONFIG['random_state']),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=CONFIG['random_state']),
    }

# Save experiment logs to CSV
def log_run_to_csv(csv_path, entry):
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(entry)

# Grouped permutation importance
def compute_grouped_feature_importance(model, X, y, feature_names, group_by='organ', scoring='r2', n_repeats=10):
    print(f"\nCalculating Permutation Importance (grouped by: {group_by})")
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=CONFIG['random_state'], scoring=scoring)
    importances = pd.Series(result.importances_mean, index=feature_names)

    def get_group_name(feature):
        parts = feature.split('_')
        if group_by == 'organ':
            return parts[0]
        elif group_by == 'type':
            return parts[2] if len(parts) > 2 else "unknown"
        elif group_by == 'full':
            return '_'.join(parts[:2]) if len(parts) >= 2 else feature
        return "unknown"

    grouped = defaultdict(list)
    for feat in feature_names:
        group = get_group_name(feat)
        grouped[group].append(importances[feat])

    aggregated = {group: np.mean(vals) for group, vals in grouped.items()}
    importance_df = pd.Series(aggregated).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importance_df.plot(kind='barh')
    plt.title(f"Grouped Feature Importance by {group_by.capitalize()}")
    plt.xlabel("Mean Permutation Importance")
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"grouped_feature_importance_by_{group_by}.png")
    plt.close()
    print(f"Saved grouped importance plot to 'grouped_feature_importance_by_{group_by}.png'")

    return importance_df

# SHAP analysis
def compute_shap_values(model, X, feature_names, output_prefix="shap_summary"):
    print("\nComputing SHAP values...")
    try:
        explainer = shap.Explainer(model, X)
    except:
        explainer = shap.Explainer(model.predict, X)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary.png")
    plt.close()
    print(f"Saved SHAP summary plot as '{output_prefix}_summary.png'")

    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_bar.png")
    plt.close()
    print(f"Saved SHAP bar plot as '{output_prefix}_bar.png'")

# Main model evaluation function
def run_model_comparison(train_paths, test_paths, label, apply_pca):
    env = setup_environment()
    results = {}

    start = log_step(f"Loading {label}")
    train_data = load_and_merge(train_paths, f"{label} - Train")
    test_data = load_and_merge(test_paths, f"{label} - Test")
    log_elapsed(start, f"Data Loading - {label}", env['timing_log'])

    if apply_pca:
        start = log_step(f"PCA - {label}")
        pca = PCA(n_components=CONFIG['pca_variance_threshold'])
        train_data['X'] = pd.DataFrame(pca.fit_transform(train_data['X']))
        test_data['X'] = pd.DataFrame(pca.transform(test_data['X']))
        log_elapsed(start, f"PCA - {label}", env['timing_log'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data['X'])
    X_test_scaled = scaler.transform(test_data['X'])

    models = get_models()
    for name, model in models.items():
        print(f"\nTraining: {name}")
        start = log_step(f"Train {name}")
        model.fit(X_train_scaled, train_data['y'])
        log_elapsed(start, f"Train {name}", env['timing_log'])

        preds = model.predict(X_test_scaled)
        mae = mean_absolute_error(test_data['y'], preds)
        print(f"{name} MAE: {mae:.2f}")
        results[name] = mae

        # Log all relevant run metadata
        run_log = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model': name,
            'dataset': label,
            'pca_applied': apply_pca,
            'mae': round(mae, 4),
            'train_paths': '|'.join(train_paths),
            'test_paths': '|'.join(test_paths),
            'time_sec': round(env['timing_log'].get(f"Train {name}", 0), 2),
        }
        log_run_to_csv(CONFIG['log_path'], run_log)

        # Extra analysis for RandomForest
        if name == "RandomForest":
            feat_names = train_data['X'].columns if isinstance(train_data['X'], pd.DataFrame) else [f'feat_{i}' for i in range(X_train_scaled.shape[1])]
            compute_grouped_feature_importance(model, X_test_scaled, test_data['y'], feat_names, group_by='full')
            compute_shap_values(model, X_test_scaled, feat_names, output_prefix=f"shap_{name.lower()}")

    return results

# Plot MAE comparison
def plot_results(raw_results, pca_results, output_path="model_mae_comparison.png"):
    models = list(raw_results.keys())
    raw_maes = [raw_results[m] for m in models]
    pca_maes = [pca_results[m] for m in models]

    x = np.arange(len(models))
    width = 0.35

    plt.figure(figsize=(12, 7))
    plt.bar(x - width/2, raw_maes, width, label='Raw', color='#1f77b4')
    plt.bar(x + width/2, pca_maes, width, label='PCA', color='#4a90e2')
    plt.xticks(x, models, rotation=45, fontsize=12)
    plt.ylabel('MAE', fontsize=14)
    plt.title('Model MAE Comparison (Raw vs PCA)', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved plot to {output_path}")

# Entry point
if __name__ == '__main__':
    print("\n### Comparing Models on Combined Water+Fat Data (Raw and PCA) ###")

    train_paths = [
        "/vol/miltank/projects/practical_sose25/in_context_learning/data/regression/train_wat.csv",
        "/vol/miltank/projects/practical_sose25/in_context_learning/data/regression/train_fat.csv"
    ]
    test_paths = [
        "/vol/miltank/projects/practical_sose25/in_context_learning/data/regression/test_wat.csv",
        "/vol/miltank/projects/practical_sose25/in_context_learning/data/regression/test_fat.csv"
    ]

    print("\nRunning without PCA")
    raw_results = run_model_comparison(train_paths, test_paths, "Raw Combined", apply_pca=False)

    print("\nRunning with PCA")
    pca_results = run_model_comparison(train_paths, test_paths, "Raw Combined + PCA", apply_pca=True)

    print("\nFinal MAE Comparison")
    print("Model\t\tRaw MAE\tPCA MAE")
    for model in raw_results:
        print(f"{model:<12}\t{raw_results[model]:.2f}\t{pca_results[model]:.2f}")

    plot_results(raw_results, pca_results)
