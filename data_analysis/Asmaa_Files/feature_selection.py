import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from collections import defaultdict
from boruta import BorutaPy
import logging
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split

class FeatureSelector:
    def __init__(self, task='regression', verbose=True):
        self.task = task
        self.verbose = verbose
        self.logger = logging.getLogger('FeatureSelector')
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[FeatureSelector] %(message)s')
        handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    def _log(self, message):
        self.logger.info(message)

    def correlation_filter(self, X, threshold=0.95):
        self._log("Applying correlation filter...")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        self._log(f"Dropping {len(to_drop)} features due to correlation > {threshold}")
        return X.drop(columns=to_drop)

    def variance_filter(self, X, threshold=0.0):
        self._log("Applying variance filter...")
        variances = X.var()
        to_keep = variances[variances > threshold].index
        self._log(f"Keeping {len(to_keep)} features with variance > {threshold}")
        return X[to_keep]

    def compute_univariate_scores(self, X, y):
        self._log("Computing univariate scores...")
        score_func = f_regression if self.task == 'regression' else f_classif
        scores = SelectKBest(score_func, k='all').fit(X, y).scores_
        return pd.DataFrame({'feature': X.columns, 'score': scores})

    def compute_mutual_info(self, X, y):
        self._log("Computing mutual information...")
        mi_func = mutual_info_regression if self.task == 'regression' else mutual_info_classif
        scores = mi_func(X, y)
        return pd.DataFrame({'feature': X.columns, 'mutual_info': scores})

    def compute_permutation_importance(self, model, X, y):
        self._log("Computing permutation importance...")
        result = permutation_importance(model, X, y, n_repeats=10, random_state=42, n_jobs=-1)
        return pd.DataFrame({
            'feature': X.columns,
            'importance_mean': result.importances_mean,
            'importance_std': result.importances_std
        })

    def compute_lasso_importance(self, X, y):
        self._log("Computing Lasso importance...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        if self.task == 'regression':
            model = LassoCV(cv=5, n_alphas=50, max_iter=10000).fit(X_scaled, y)
            importances = np.abs(model.coef_)
        else:
            model = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000).fit(X_scaled, y)
            importances = np.abs(model.coef_).flatten()
        return pd.DataFrame({'feature': X.columns, 'lasso_importance': importances})

    def compute_rfe_importance(self, X, y):
        self._log("Running RFE for importance...")
        base_model = Lasso() if self.task == 'regression' else LogisticRegression(max_iter=1000)
        step_size = max(1, X.shape[1] // 20)
        rfe = RFE(estimator=base_model, n_features_to_select=min(10, X.shape[1]), step=step_size)
        rfe.fit(X, y)
        return pd.DataFrame({'feature': X.columns, 'rfe_ranking': rfe.ranking_})

    def compute_distance_correlation(self, X, y, max_features=500):
        self._log(f"Computing distance correlation on top {max_features} features...")
        from scipy.spatial.distance import pdist, squareform

        def distance_corr(x, y):
            a = squareform(pdist(x[:, None]))
            b = squareform(pdist(y[:, None]))
            A = a - a.mean(axis=0) - a.mean(axis=1)[:, None] + a.mean()
            B = b - b.mean(axis=0) - b.mean(axis=1)[:, None] + b.mean()
            dcov = np.sqrt(np.mean(A * B))
            dvar_x = np.sqrt(np.mean(A * A))
            dvar_y = np.sqrt(np.mean(B * B))
            return dcov / np.sqrt(dvar_x * dvar_y)

        X_sub = X.iloc[:, :min(max_features, X.shape[1])]
        dcor_scores = [distance_corr(X_sub.iloc[:, i].values, y) for i in range(X_sub.shape[1])]
        return pd.DataFrame({'feature': X_sub.columns, 'distance_corr': dcor_scores})

    def compute_shap(self, model, X, output_prefix="shap_values", model_name="model"):
        self._log("Computing SHAP values...")
        try:
            if hasattr(model, "predict_proba") and self.task == 'classification':
                explainer = shap.Explainer(model, X)
            elif "tree" in model.__class__.__name__.lower():
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))

            shap_values = explainer(X)
            shap.summary_plot(shap_values, features=X, feature_names=X.columns, show=False)
            plt.title("SHAP Summary")
            plt.tight_layout()
            path = f"{output_prefix}_{model_name}_summary.png"
            plt.savefig(path)
            plt.close()
            self._log(f"SHAP plot saved to {path}")
            mean_abs = np.abs(shap_values.values).mean(axis=0)
            return pd.DataFrame({'feature': X.columns, 'mean_abs_shap': mean_abs})
        except Exception as e:
            self._log(f"SHAP computation failed: {e}")
            return pd.DataFrame({'feature': X.columns, 'mean_abs_shap': [np.nan] * X.shape[1]})

    def compute_boruta(self, X, y):
        self._log("Running Boruta...")
        rf = RandomForestRegressor(n_jobs=-1) if self.task == 'regression' else RandomForestClassifier(n_jobs=-1)
        boruta = BorutaPy(estimator=rf, n_estimators='auto', max_iter=50, random_state=42, verbose=0)
        boruta.fit(X.values, y)
        rankings = boruta.ranking_
        return pd.DataFrame({'feature': X.columns, 'boruta_ranking': rankings})

    def run_pre_model_feature_selection(self, X, y, output_dir="feature_selection_output", save_intermediate=False):
        if save_intermediate:
            os.makedirs(output_dir, exist_ok=True)
        
        X_filtered = self.variance_filter(X, threshold=1e-5)
        X_filtered = self.correlation_filter(X_filtered, threshold=0.95)
        
        if save_intermediate:
            path = os.path.join(output_dir, "pre_selected_filtered_features.csv")
            X_filtered.to_csv(path, index=False)
        
        score_func = f_regression if self.task == 'regression' else f_classif
        fast_selector = SelectKBest(score_func, k=min(500, X_filtered.shape[1]))
        X_fast = pd.DataFrame(fast_selector.fit_transform(X_filtered, y),
                              columns=X_filtered.columns[fast_selector.get_support()])
        
        if save_intermediate:
            path = os.path.join(output_dir, "pre_selected_top500.csv")
            X_fast.to_csv(path, index=False)
        
        score_dfs = [
            self.compute_univariate_scores(X_filtered, y),
            self.compute_mutual_info(X_filtered, y)
        ]
        
        if save_intermediate:
            for i, df in enumerate(score_dfs):
                path = os.path.join(output_dir, f'pre_score_{i}_{df.columns[1]}.csv')
                df.to_csv(path, index=False)
        
        return X_filtered, X_fast, score_dfs

    def run_post_model_feature_selection(self, X_fast, y, model):
        return [
            self.compute_permutation_importance(model, X_fast, y),
            self.compute_lasso_importance(X_fast, y),
            self.compute_rfe_importance(X_fast, y),
            self.compute_distance_correlation(X_fast, y),
            self.compute_shap(model, X_fast),
            self.compute_boruta(X_fast, y)
        ]

    def aggregate_feature_scores(self, score_dfs, top_k=20):
        self._log("Aggregating feature scores...")
        votes = defaultdict(int)
        direction_map = {
            'score': False,
            'mutual_info': False,
            'importance_mean': False,
            'lasso_importance': False,
            'rfe_ranking': True,
            'distance_corr': False,
            'mean_abs_shap': False,
            'boruta_ranking': True
        }

        for df in score_dfs:
            if df.empty:
                continue
            score_col = df.columns[1]
            ascending = direction_map.get(score_col, False)
            ranked = df.sort_values(score_col, ascending=ascending)
            top_feats = ranked.head(top_k)['feature']
            for feat in top_feats:
                votes[feat] += 1
        
        vote_df = pd.DataFrame(list(votes.items()), columns=['feature', 'votes'])
        vote_df = vote_df.sort_values('votes', ascending=False).reset_index(drop=True)
        self._log(f"Top features by votes:\n{vote_df.head(10)}")
        return vote_df

    def finalize_feature_selection(self, score_dfs, top_k=20, output_dir="feature_selection_output"):
        os.makedirs(output_dir, exist_ok=True)
        vote_df = self.aggregate_feature_scores(score_dfs, top_k=top_k)
        vote_path = os.path.join(output_dir, 'aggregated_feature_votes.csv')
        vote_df.to_csv(vote_path, index=False)
        self._log(f"Saved aggregated feature votes to {vote_path}")

        plt.figure(figsize=(10, 6))
        plt.bar(vote_df['feature'].head(top_k), vote_df['votes'].head(top_k))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Votes')
        plt.title('Top Features by Aggregated Votes')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'feature_vote_plot.png')
        plt.savefig(plot_path)
        plt.close()
        self._log(f"Saved feature vote plot to {plot_path}")
        return vote_df



# ----------------------
# NEW FEATURE ENGINEER
# ----------------------

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, impute_strategy='mean', scale=True):
        self.impute_strategy = impute_strategy
        self.scale = scale
        self.imputer = SimpleImputer(strategy=impute_strategy)
        self.scaler = StandardScaler() if scale else None

    def fit(self, X, y=None):
        X_numeric = X.select_dtypes(include=[np.number])
        self.imputer.fit(X_numeric)
        if self.scaler:
            self.scaler.fit(X_numeric)
        return self

    def transform(self, X):
        X_copy = X.copy()
        numeric_cols = X_copy.select_dtypes(include=[np.number]).columns
        X_copy[numeric_cols] = self.imputer.transform(X_copy[numeric_cols])
        if self.scaler:
            X_copy[numeric_cols] = self.scaler.transform(X_copy[numeric_cols])
        return X_copy
