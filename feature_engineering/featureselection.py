import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import os
import json
from pathlib import Path
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from joblib import Parallel, delayed
import signal
import time
from data_extraction.data_manager import DataManager
from feature_engineering.sampling import balance_data  # Adjust import if needed
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')

#not used not since we are using the organ selector
class FeatureSelector:
    def __init__(self, config, verbose=True):
        # Always convert config to a dict and check for modules
        if hasattr(config, "__dict__"):
            config_dict = {k: v for k, v in vars(config).items() if not k.startswith("__") and not callable(v)}
        elif isinstance(config, dict):
            config_dict = dict(config)
        else:
            raise ValueError("Config must be a dict or module with attributes.")
        # Check for module objects in config dict
        import types
        for k, v in config_dict.items():
            if isinstance(v, types.ModuleType):
                raise ValueError(f"Config key '{k}' contains a module object, which is not picklable.")
        self.config = config_dict
        self.verbose = verbose
        self.feature_groups = None
        self.organ_groups = None
        self.selected_features = None
        self.feature_importance_scores = None
        self.features_to_drop_variance = set()
        self.features_to_drop_correlation = set()
        self.selection_history = {}
        self.filtering_stats = {}
        self.logger = logging.getLogger('FeatureSelector')
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[FeatureSelector] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # Set up config values from dict
        self._log(f'FeatureSelector initialized with config: {self.config}')
        self.task = self._get_config_value('TASK')
        self.random_state = self._get_config_value('RANDOM_STATE')
        self.n_features = self._get_config_value('N_FEATURES')
        self.variance_thresholds = self._get_config_value('VARIANCE_THRESHOLDS')
        self.correlation_thresholds = self._get_config_value('CORRELATION_THRESHOLDS')

    def _get_config_value(self, key, default=None):
        # Use dict lookup, then attribute
        if isinstance(self.config, dict) and key in self.config:
            return self.config[key]
        elif hasattr(self.config, key):
            return getattr(self.config, key)
        else:
            return default

    def _log(self, msg):
        if self.verbose:
            self.logger.info(msg)

    def parse_feature_names(self, feature_names):
        self._log("Parsing feature names into groups...")
        parsed_features = {
            'radiomics_shape': [],
            'radiomics_texture': [],
            'radiomics_intensity': [],
            'embeddings': [],
            'tabular': []
        }
        organ_groups = defaultdict(list)
        organ_feature_types = defaultdict(lambda: defaultdict(list))  # organ -> feature_type -> features

        for feat in feature_names:
            feat_low = feat.lower()
            feature_type = None
            
            # Determine feature type
            if (feat.replace('.', '').replace('-', '').replace('_', '').isdigit() or
                (len(feat.split('_')) <= 2 and
                 not any(rad_term in feat_low for rad_term in ['shape', 'glcm', 'glrlm', 'glszm', 'ngtdm', 'gldm', 'firstorder']))):
                parsed_features['embeddings'].append(feat)
                feature_type = 'embeddings'
            elif '_shape_' in feat_low:
                parsed_features['radiomics_shape'].append(feat)
                feature_type = 'radiomics_shape'
            elif any(tex in feat_low for tex in ['_glcm_', '_glrlm_', '_glszm_', '_ngtdm_', '_gldm_']):
                parsed_features['radiomics_texture'].append(feat)
                feature_type = 'radiomics_texture'
            elif '_firstorder_' in feat_low:
                parsed_features['radiomics_intensity'].append(feat)
                feature_type = 'radiomics_intensity'
            else:
                parsed_features['tabular'].append(feat)
                feature_type = 'tabular'
            
            # Extract organ name more robustly
            organ = self._extract_organ_name(feat)
            organ_groups[organ].append(feat)
            
            # Group by organ and feature type
            if feature_type:
                organ_feature_types[organ][feature_type].append(feat)

        self.feature_groups = parsed_features
        self.organ_groups = dict(organ_groups)
        self.organ_feature_types = dict(organ_feature_types)  # New: organ -> feature_type -> features

        # Store parsing stats
        self.filtering_stats['original_features'] = len(feature_names)
        self.filtering_stats['feature_groups'] = {
            group: len(feats) for group, feats in parsed_features.items()
        }
        self.filtering_stats['organ_groups'] = {
            organ: len(feats) for organ, feats in organ_groups.items()
        }

        for group, feats in parsed_features.items():
            self._log(f"{group}: {len(feats)} features")
        self._log(f"Identified {len(organ_groups)} organs: {list(organ_groups.keys())}")
        
        # Log organ breakdown
        for organ, feature_types in organ_feature_types.items():
            self._log(f"Organ '{organ}': {sum(len(feats) for feats in feature_types.values())} total features")
            for feat_type, feats in feature_types.items():
                if feats:
                    self._log(f"  - {feat_type}: {len(feats)} features")

        return parsed_features

    def _extract_organ_name(self, feature_name):
        # Common organ prefixes to look for
        organ_indicators = [
            'liver', 'kidney', 'spleen', 'pancreas', 'heart', 'lung', 'brain', 
            'stomach', 'intestine', 'bladder', 'prostate', 'uterus', 'ovary',
            'breast', 'thyroid', 'adrenal', 'gallbladder', 'esophagus'
        ]
        
        feat_low = feature_name.lower()
        
        # First, try to find organ indicators in the feature name
        for organ in organ_indicators:
            if organ in feat_low:
                return organ
        
        # If no organ indicator found, try to extract from underscore pattern
        parts = feature_name.split('_')
        if len(parts) > 1:
            # Look for organ-like patterns in the first few parts
            for part in parts[:3]:  # Check first 3 parts
                if len(part) > 2 and not part.isdigit():  # Avoid numbers and very short parts
                    return part
        
        return 'unknown'

    def fit(self, X, y, n_features=None):
        logger = logging.getLogger('FeatureSelector')
        logger.info(f"[FeatureSelector] fit() called. Input shape: {X.shape}")
        self._log(f"Starting feature selection fit with {X.shape[1]} input features [Best-practice: fit only on training data]")
        self.parse_feature_names(X.columns)
        strategies = self._get_config_value('FEATURE_SELECTION_STRATEGIES', {})
        do_hierarchical = strategies.get("hierarchical_filtering", True)
        X_filtered = X
        if do_hierarchical:
            X_filtered = self._filter_variance(X_filtered)
            X_filtered = self._filter_correlation(X_filtered)
        else:
            self._log("Hierarchical filtering disabled: skipping variance and correlation filters.")
        self._log(f"After filtering: {X_filtered.shape[1]} features remaining")
        self.feature_importance_scores = self._compute_importance_scores(X_filtered, y)
        n_features = n_features or self.n_features
        self.selected_features = self._balanced_selection(X_filtered, n_features=n_features)
        self._log(f"Selected {len(self.selected_features)} features [Best-practice: fit only on training data]")
        self.selection_history[n_features] = {
            'selected_features': self.selected_features.copy(),
            'importance_scores': self.feature_importance_scores.copy() if not self.feature_importance_scores.empty else None,
            'filtering_stats': self.filtering_stats.copy()
        }
        logger.info(f"[FeatureSelector] fit() complete. Selected features: {len(self.selected_features)}")
        return self

    def transform(self, X):
        logger = logging.getLogger('FeatureSelector')
        logger.info(f"[FeatureSelector] transform() called. Input shape: {X.shape}")
        if self.selected_features is None:
            raise ValueError("Call fit() before transform()")

        # Apply same filtering as during fit
        X_filtered = self._apply_variance_filter(X)
        X_filtered = self._apply_correlation_filter(X_filtered)

        # Select only the chosen features, fill missing with NaN
        existing = [f for f in self.selected_features if f in X_filtered.columns]
        missing = [f for f in self.selected_features if f not in X_filtered.columns]

        if missing:
            self._log(f"Warning: {len(missing)} selected features missing in transform input. Filling with NaN.")
            for f in missing:
                X_filtered[f] = np.nan

        # Always return all selected_features in the same order
        result = X_filtered[self.selected_features]
        logger.info(f"[FeatureSelector] transform() complete. Output shape: {result.shape}")
        return result

    def fit_transform(self, X, y, **kwargs):
        logger = logging.getLogger('FeatureSelector')
        logger.info(f"[FeatureSelector] fit_transform() called. Input shape: {X.shape}")
        # Extract n_features from kwargs to avoid parameter conflicts
        n_features = kwargs.pop('n_features', None)
        
        # Call fit and transform separately to avoid parameter conflicts
        result = self.fit(X, y, n_features=n_features).transform(X)
        logger.info(f"[FeatureSelector] fit_transform() complete. Output shape: {result.shape}")
        return result

    def _filter_variance(self, X):
        logger = logging.getLogger('FeatureSelector')
        logger.info(f"[FeatureSelector] _filter_variance() called. Input shape: {X.shape}")
        self._log("Applying variance filter...")
        vt = self.variance_thresholds
        for group, feats in self.feature_groups.items():
            if group in vt:
                feats_exist = [f for f in feats if f in X.columns]
                if feats_exist:
                    variances = X[feats_exist].var()
                    drop_feats = variances[variances <= vt[group]].index.tolist()
                    if drop_feats:
                        self._log(f"{group}: Dropping {len(drop_feats)} features due to low variance")
                        self.features_to_drop_variance.update(drop_feats)
        # Store variance filtering stats
        self.filtering_stats['variance_filtered'] = len(self.features_to_drop_variance)
        result = self._apply_variance_filter(X)
        logger.info(f"[FeatureSelector] _filter_variance() complete. Output shape: {result.shape}")
        return result

    def _apply_variance_filter(self, X):
        if not self.features_to_drop_variance:
            return X.copy()
        
        keep_cols = [col for col in X.columns if col not in self.features_to_drop_variance]
        result = X[keep_cols]
        return result

    def _filter_correlation(self, X):
        logger = logging.getLogger('FeatureSelector')
        logger.info(f"[FeatureSelector] _filter_correlation() called. Input shape: {X.shape}")
        self._log("Applying correlation filter...")
        ct = self.correlation_thresholds
        
        for group, feats in self.feature_groups.items():
            feats_exist = [f for f in feats if f in X.columns]
            if len(feats_exist) > 1 and group in ct:
                corr_mat = X[feats_exist].corr().abs()
                upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))
                drop_feats = [col for col in upper.columns if any(upper[col] > ct[group])]
                if drop_feats:
                    self._log(f"{group}: Dropping {len(drop_feats)} features due to high correlation")
                    self.features_to_drop_correlation.update(drop_feats)
        
        # Store correlation filtering stats
        self.filtering_stats['correlation_filtered'] = len(self.features_to_drop_correlation)
        
        result = self._apply_correlation_filter(X)
        logger.info(f"[FeatureSelector] _filter_correlation() complete. Output shape: {result.shape}")
        return result

    def _apply_correlation_filter(self, X):
        if not self.features_to_drop_correlation:
            return X.copy()
        
        keep_cols = [col for col in X.columns if col not in self.features_to_drop_correlation]
        result = X[keep_cols]
        return result

    def _compute_importance_scores(self, X, y):
        logger = logging.getLogger('FeatureSelector')
        logger.info(f"[FeatureSelector] _compute_importance_scores() called. Input shape: {X.shape}")
        self._log("Computing feature importance scores...")
        strategies = self._get_config_value('FEATURE_SELECTION_STRATEGIES', {})
        self._log(f'FeatureSelector config FEATURE_SELECTION_STRATEGIES: {strategies}')
        do_multi_method = strategies.get("multi_method_scoring", True)
        do_group_based = strategies.get("group_based_selection", True)
        self._log(f"multi_method_scoring: {do_multi_method}, group_based_selection: {do_group_based}")
        if X.empty:
            self._log("Input X is empty, returning empty importance scores.")
            return pd.DataFrame(columns=['feature', 'importance', 'group'])
        if do_multi_method:
            # Run all methods (univariate, MI, LASSO)
            importances = []
            if self.task == "regression":
                scores = f_regression(X, y)
                importances.append(("univariate", scores[0] if isinstance(scores, tuple) else scores))
                from sklearn.feature_selection import mutual_info_regression
                mi = mutual_info_regression(X, y, random_state=self.random_state)
                importances.append(("mutual_info", mi))
                lasso_scores = self._compute_lasso_importance(X, y)
                importances.append(("lasso", lasso_scores))
            else:
                scores = f_classif(X, y)
                importances.append(("univariate", scores[0] if isinstance(scores, tuple) else scores))
                from sklearn.feature_selection import mutual_info_classif
                mi = mutual_info_classif(X, y, random_state=self.random_state)
                importances.append(("mutual_info", mi))
                lasso_scores = self._compute_lasso_importance(X, y)
                importances.append(("lasso", lasso_scores))
            # Average normalized importances
            df = pd.DataFrame({
                'feature': X.columns,
                'univariate': importances[0][1],
                'mutual_info': importances[1][1],
                'lasso': importances[2][1],
                'group': [self._get_group(f) for f in X.columns]
            })
            for col in ['univariate', 'mutual_info', 'lasso']:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min() + 1e-8)
            df['importance'] = df[['univariate', 'mutual_info', 'lasso']].mean(axis=1)
            result = df.sort_values('importance', ascending=False)
            logger.info(f"[FeatureSelector] _compute_importance_scores() complete.")
            return result
        else:
            # Do not run any feature selection/scoring
            self._log("multi_method_scoring is False: Skipping all feature selection/scoring. All features will be returned with equal importance.")
            importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': 1.0,
                'group': [self._get_group(f) for f in X.columns]
            })
        return importance_df

    def _compute_lasso_importance(self, X, y):
        self._log("Computing LASSO importance by group...")
        lasso_importance = []
        X_clean = X.loc[:, X.nunique() > 1]
        
        if X_clean.empty:
            self._log("No non-constant features available for LASSO")
            return pd.DataFrame()

        for group, feats in self.feature_groups.items():
            group_feats = [f for f in feats if f in X_clean.columns]
            if not group_feats:
                continue
                
            X_group = X_clean[group_feats]

            # Fit LASSO model
            if self.task == 'regression':
                model = LassoCV(cv=5, random_state=self.random_state, max_iter=5000)
            else:
                model = LogisticRegression(
                    penalty='l1', solver='liblinear', max_iter=5000,
                    random_state=self.random_state, C=1.0,
                    class_weight='balanced'
                )
            
            try:
                model.fit(X_group, y)
                coefs = np.abs(model.coef_).flatten()
                
                for f, coef in zip(group_feats, coefs):
                    lasso_importance.append({
                        'feature': f, 
                        'lasso_importance': coef, 
                        'group': group
                    })
                    
            except Exception as e:
                self._log(f"LASSO failed for group '{group}': {e}")

        if not lasso_importance:
            return pd.DataFrame()

        return pd.DataFrame(lasso_importance).groupby('feature', as_index=False).agg({
            'lasso_importance': 'max'
        })

    def _get_group(self, feature):
        for group, feats in self.feature_groups.items():
            if feature in feats:
                return group
        return 'unknown'

    def get_features_by_organ_and_type(self, organ=None, feature_type=None):
        if organ is None and feature_type is None:
            return list(self.organ_groups.values())
        
        if organ is not None and feature_type is not None:
            return self.organ_feature_types.get(organ, {}).get(feature_type, [])
        elif organ is not None:
            return self.organ_groups.get(organ, [])
        elif feature_type is not None:
            return self.feature_groups.get(feature_type, [])
        
        return []

    def get_organ_summary(self):
        summary = {}
        for organ, feature_types in self.organ_feature_types.items():
            summary[organ] = {
                'total_features': len(self.organ_groups[organ]),
                'by_type': {feat_type: len(feats) for feat_type, feats in feature_types.items() if feats}
            }
        return summary

    def _balanced_selection(self, X, n_features=None):
        n_features = n_features or self.n_features
        strategies = self._get_config_value('FEATURE_SELECTION_STRATEGIES', {})
        do_group_based = strategies.get("group_based_selection", True)
        do_organ_based = strategies.get("organ_based_selection", False)  # New option
        self._log(f"group_based_selection: {do_group_based}, organ_based_selection: {do_organ_based}")
        
        if self.feature_importance_scores is None or self.feature_importance_scores.empty:
            self._log("No importance scores available, using first N features")
            return X.columns[:n_features].tolist()
        
        df = self.feature_importance_scores.copy()
        df = df[df['feature'].isin(X.columns)]
        if df.empty:
            self._log("No scored features found in input data")
            return X.columns[:n_features].tolist()
        
        if do_organ_based and hasattr(self, 'organ_groups'):
            # Organ-based selection
            organs = list(self.organ_groups.keys())
            quota = max(1, n_features // len(organs))
            selected = []
            
            for organ in organs:
                organ_feats = [f for f in self.organ_groups[organ] if f in df['feature'].values]
                if organ_feats:
                    organ_df = df[df['feature'].isin(organ_feats)].sort_values('importance', ascending=False)
                    organ_selected = organ_df.head(quota)['feature'].tolist()
                    selected.extend(organ_selected)
                    self._log(f"Selected {len(organ_selected)} features from organ '{organ}'")
            
            if len(selected) < n_features:
                remaining_needed = n_features - len(selected)
                remaining_feats = df[~df['feature'].isin(selected)].sort_values('importance', ascending=False)
                additional = remaining_feats.head(remaining_needed)['feature'].tolist()
                selected.extend(additional)
                self._log(f"Added {len(additional)} additional top features")
            
            final_selected = selected[:n_features]
            self._log(f"Final selection: {len(final_selected)} features (organ-balanced)")
            return final_selected
            
        elif do_group_based:
            # Feature type-based selection (original)
            groups = df['group'].unique()
            quota = max(1, n_features // len(groups))
            selected = []
            for group in groups:
                group_feats = df[df['group'] == group].sort_values('importance', ascending=False)
                group_selected = group_feats.head(quota)['feature'].tolist()
                selected.extend(group_selected)
                self._log(f"Selected {len(group_selected)} features from group '{group}'")
            if len(selected) < n_features:
                remaining_needed = n_features - len(selected)
                remaining_feats = df[~df['feature'].isin(selected)].sort_values('importance', ascending=False)
                additional = remaining_feats.head(remaining_needed)['feature'].tolist()
                selected.extend(additional)
                self._log(f"Added {len(additional)} additional top features")
            final_selected = selected[:n_features]
            self._log(f"Final selection: {len(final_selected)} features")
            return final_selected
        else:
            top_feats = df.sort_values('importance', ascending=False)['feature'].head(n_features).tolist()
            self._log(f"Final selection: {len(top_feats)} features (no group balancing)")
            return top_feats

    def save_selection_summary(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save selection history
        if self.selection_history:
            with open(output_path / "selection_history.json", 'w') as f:
                # Convert DataFrames to dicts for JSON serialization
                history_copy = {}
                for n_feat, info in self.selection_history.items():
                    history_copy[str(n_feat)] = {
                        'selected_features': info['selected_features'],
                        'filtering_stats': info['filtering_stats']
                    }
                    if info['importance_scores'] is not None:
                        history_copy[str(n_feat)]['top_features'] = info['importance_scores'].head(20).to_dict('records')
                
                json.dump(history_copy, f, indent=2)

        # Save feature importance scores
        if self.feature_importance_scores is not None and not self.feature_importance_scores.empty:
            self.feature_importance_scores.to_csv(output_path / "feature_importance_scores.csv", index=False)

        # Save feature groups
        if self.feature_groups:
            with open(output_path / "feature_groups.json", 'w') as f:
                json.dump(self.feature_groups, f, indent=2)

        # Save organ groups
        if hasattr(self, 'organ_groups') and self.organ_groups:
            with open(output_path / "organ_groups.json", 'w') as f:
                json.dump(self.organ_groups, f, indent=2)

        # Save organ feature types breakdown
        if hasattr(self, 'organ_feature_types') and self.organ_feature_types:
            with open(output_path / "organ_feature_types.json", 'w') as f:
                json.dump(self.organ_feature_types, f, indent=2)

        # Save filtering statistics
        if self.filtering_stats:
            with open(output_path / "filtering_stats.json", 'w') as f:
                json.dump(self.filtering_stats, f, indent=2)

        self._log(f"Selection summary saved to {output_path}")

    def get_selection_summary(self):
        if not self.selected_features:
            return "No features selected yet"
        
        summary = {
            'total_selected': len(self.selected_features),
            'selection_by_group': {},
            'filtering_stats': self.filtering_stats.copy()
        }
        
        # Count selected features by group
        for feat in self.selected_features:
            group = self._get_group(feat)
            summary['selection_by_group'][group] = summary['selection_by_group'].get(group, 0) + 1
        
        return summary

    def apply_organ_pca(self, X, n_components_per_organ=10, random_state=None):
        
        
        self._log(f"Applying PCA to each organ group with {n_components_per_organ} components per organ")
        
        if not hasattr(self, 'organ_groups') or not self.organ_groups:
            self._log("No organ groups available. Run parse_feature_names first.")
            return X
        
        pca_results = {}
        organ_pca_models = {}
        
        for organ, features in self.organ_groups.items():
            # Get features that exist in the input data
            available_features = [f for f in features if f in X.columns]
            
            if len(available_features) < n_components_per_organ:
                self._log(f"Warning: Organ '{organ}' has only {len(available_features)} features, "
                         f"but {n_components_per_organ} components requested. "
                         f"Using {len(available_features)} components.")
                n_components = len(available_features)
            else:
                n_components = n_components_per_organ
            
            if n_components == 0:
                self._log(f"Skipping organ '{organ}' - no features available")
                continue
                
            # Extract organ features
            X_organ = X[available_features]
            
            # Apply PCA
            pca = PCA(n_components=n_components, random_state=random_state)
            X_pca = pca.fit_transform(X_organ)
            
            # Create column names for PCA components
            pca_columns = [f"{organ}_pca_{i+1}" for i in range(n_components)]
            
            # Store results
            pca_results[organ] = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
            organ_pca_models[organ] = pca
            
            # Log explained variance
            explained_variance = pca.explained_variance_ratio_
            total_variance = explained_variance.sum()
            self._log(f"Organ '{organ}': {len(available_features)} features -> {n_components} components "
                     f"(explained variance: {total_variance:.3f})")
            
            # Log individual component contributions
            for i, (col, var) in enumerate(zip(pca_columns, explained_variance)):
                self._log(f"  - {col}: {var:.3f} ({var/total_variance*100:.1f}%)")
        
        # Combine all PCA results
        if pca_results:
            combined_pca = pd.concat(pca_results.values(), axis=1)
            self._log(f"Combined PCA result: {combined_pca.shape[1]} total components from {len(pca_results)} organs")
            
            # Store PCA models for later use
            self.organ_pca_models = organ_pca_models
            
            return combined_pca
        else:
            self._log("No PCA results generated")
            return pd.DataFrame(index=X.index)

    def transform_with_organ_pca(self, X, n_components_per_organ=10, random_state=None):
        if not hasattr(self, 'organ_pca_models') or not self.organ_pca_models:
            self._log("No pre-fitted PCA models found. Running fit_transform_with_organ_pca instead.")
            return self.fit_transform_with_organ_pca(X, n_components_per_organ, random_state)
        
        self._log("Transforming data using pre-fitted organ PCA models")
        
        pca_results = {}
        
        for organ, pca_model in self.organ_pca_models.items():
            # Get features that exist in the input data
            available_features = [f for f in self.organ_groups[organ] if f in X.columns]
            
            if not available_features:
                self._log(f"Warning: No features found for organ '{organ}' in transform data")
                continue
            
            # Extract organ features
            X_organ = X[available_features]
            
            # Transform using fitted PCA
            X_pca = pca_model.transform(X_organ)
            
            # Create column names for PCA components
            n_components = X_pca.shape[1]
            pca_columns = [f"{organ}_pca_{i+1}" for i in range(n_components)]
            
            # Store results
            pca_results[organ] = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
        
        # Combine all PCA results
        if pca_results:
            combined_pca = pd.concat(pca_results.values(), axis=1)
            self._log(f"Transform result: {combined_pca.shape[1]} total components from {len(pca_results)} organs")
            return combined_pca
        else:
            self._log("No transform results generated")
            return pd.DataFrame(index=X.index)

    def fit_transform_with_organ_pca(self, X, n_components_per_organ=10, random_state=None):
        self._log(f"Fitting and transforming with organ PCA ({n_components_per_organ} components per organ)")
        
        # First, parse feature names if not already done
        if not hasattr(self, 'organ_groups'):
            self.parse_feature_names(X.columns)
        
        # Apply PCA to each organ group
        return self.apply_organ_pca(X, n_components_per_organ, random_state)

    def get_organ_pca_summary(self):
        if not hasattr(self, 'organ_pca_models'):
            return "No PCA models fitted yet"
        
        summary = {}
        
        for organ, pca_model in self.organ_pca_models.items():
            explained_variance = pca_model.explained_variance_ratio_
            total_variance = explained_variance.sum()
            
            summary[organ] = {
                'n_components': len(explained_variance),
                'total_explained_variance': total_variance,
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance': np.cumsum(explained_variance).tolist()
            }
        
        return summary

    def save_organ_pca_summary(self, output_dir):
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save PCA summary
        pca_summary = self.get_organ_pca_summary()
        if isinstance(pca_summary, dict):
            with open(output_path / "organ_pca_summary.json", 'w') as f:
                json.dump(pca_summary, f, indent=2)
        
        # Save organ groups info
        if hasattr(self, 'organ_groups'):
            with open(output_path / "organ_groups_pca.json", 'w') as f:
                json.dump(self.organ_groups, f, indent=2)
        
        self._log(f"Organ PCA summary saved to {output_path}")


def nested_cv_experiment(X, y, config, model_name, n_outer=5, n_inner=3, n_features=None, random_state=42, is_classification=False, splits=None, save_fold_indices_path=None, include_feature_engineering=False, **kwargs):
    import logging
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
    from sklearn.base import clone
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    import time
    results = []
    fold_indices = []
    logger = logging.getLogger("nested_cv_experiment")
    logger.setLevel(logging.INFO)
    n_jobs = kwargs.get('n_jobs', 1)
    # Model registry and scoring (same as before)
    if is_classification:
        scoring = {
            'accuracy': 'accuracy',
            'f1_weighted': 'f1_weighted',
            'roc_auc': 'roc_auc',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'
        }
    else:
        scoring = {'r2': 'r2'}
    param_grids = {
        'RandomForest': {'n_estimators': [100, 200], 'max_depth': [None, 10]},
        'LogisticRegression': {'C': [0.1, 1, 10]},
        'MLP': {'hidden_layer_sizes': [(50,), (100,)], 'alpha': [0.0001, 0.001]},
        'XGBoost': {'n_estimators': [100, 200], 'max_depth': [3, 6]},
        'LightGBM': {'n_estimators': [100, 200], 'max_depth': [3, 6]},
    }
    param_grid = param_grids.get(model_name, {})
    # Use DataManager for folds
    dm = DataManager(*config.DATA_FLAGS)
    k = n_outer
    def run_fold(fold):
        try:
            logger.info(f"Starting fold {fold} for model {model_name} with {n_features} features")
            t0 = time.time()
            train_folds = [i for i in range(k) if i != fold]
            test_fold = [fold]
            train_df = dm.get_fold_data_set(train_folds)
            test_df = dm.get_fold_data_set(test_fold)
            X_train = train_df.drop(columns=[config.TARGET_COL])
            y_train = train_df[config.TARGET_COL]
            X_test = test_df.drop(columns=[config.TARGET_COL])
            y_test = test_df[config.TARGET_COL]
            logger.info(f"Fold {fold}: X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            fold_indices.append({"fold": fold, "train_folds": train_folds, "test_fold": test_fold})
            # Apply sampling to training data if enabled in config
            if hasattr(config, 'ENABLE_SAMPLING') and config.ENABLE_SAMPLING:
                params = getattr(config, 'SAMPLING_PARAMS', {})
                Xb, yb = balance_data(X_train, y_train, task='classification', **params)
                X_train = pd.DataFrame(Xb, columns=X_train.columns)
                y_train = yb
            # Feature selection and tuning in inner loop (same as before)
            logger.info(f"Fold {fold}: Starting feature selection...")
            t1 = time.time()
            def make_pipeline():
                from sklearn.pipeline import Pipeline
                from feature_engineering.organ_pca_selector import OrganPCASelector
                steps = []
                if include_feature_engineering:
                    from feature_engineering.feature_engineering import FeatureEngineering
                    steps.append(('feature_engineering', FeatureEngineering(**config.FEATURE_ENGINEERING_PARAMS)))
                organ_pca_selector = OrganPCASelector(n_components_per_organ=10, random_state=random_state, verbose=False)
                steps.append(('organ_pca_selector', organ_pca_selector))
                X_train_sel_temp = organ_pca_selector.fit_transform(X_train, y_train)
                logger.info(f"Fold {fold}: Organ PCA complete. Selected features shape: {X_train_sel_temp.shape}")
                n_dims = X_train_sel_temp.shape[1]
                n_positions = n_dims
                logger.info(f"Fold {fold}: Creating model pipeline...")
                t2 = time.time()
                if is_classification:
                    from classification.models.classification_models import get_classification_models
                    models = get_classification_models(random_state=random_state, model_names=[model_name])
                else:
                    from regression.models.regression_models import get_regression_models
                    models = get_regression_models(random_state=random_state, model_names=[model_name])
                if model_name not in models:
                    logger.error(f"Model {model_name} not found in registry.")
                    raise ValueError(f"Model {model_name} not found in registry.")
                logger.info(f"Fold {fold}: Adding StandardScaler before model for all models.")
                scaler = StandardScaler()
                steps.append(('scaler', scaler))
                if model_name == 'LogisticRegression':
                    logreg = LogisticRegression(solver='saga', max_iter=200, random_state=42)
                    steps.append(('model', logreg))
                else:
                    steps.append(('model', clone(models[model_name])))
                logger.info(f"Fold {fold}: Model pipeline created.")
                t3 = time.time()
                logger.info(f"Fold {fold}: Pipeline creation time: {t3-t2:.2f}s")
                return Pipeline(steps)
            pipe = make_pipeline()
            logger.info(f"Fold {fold}: Pipeline steps and types:")
            import types
            for name, step in pipe.steps:
                logger.info(f"  - {name}: {type(step)}")
                for attr in dir(step):
                    if not attr.startswith('_'):
                        try:
                            val = getattr(step, attr)
                            if isinstance(val, types.ModuleType):
                                logger.warning(f"    * {name}.{attr} is a module: {val}")
                        except Exception:
                            pass
            logger.info(f"Fold {fold}: Fitting pipeline/model...")
            logger.info(f"Fold {fold}: About to fit model...")
            t4 = time.time()
            if param_grid:
                grid = GridSearchCV(
                    pipe,
                    param_grid={f'model__{k}': v for k, v in param_grid.items()},
                    cv=n_inner,
                    scoring=list(scoring.values())[0],
                    n_jobs=1,
                    refit=True,
                    verbose=0,
                )
            else:
                grid = pipe
            try:
                grid.fit(X_train, y_train)
                logger.info(f"Fold {fold}: Finished fitting model.")
            except Exception as e:
                logger.error(f"Fold {fold}: Model fit failed: {e}", exc_info=True)
                return {'fold': fold, 'model': model_name, 'n_features': n_features, 'error': f'model_fit_failed: {str(e)}'}
            t5 = time.time()
            logger.info(f"Fold {fold}: Model fit complete. Time: {t5-t4:.2f}s")
            if param_grid:
                best_params = grid.best_params_
                best_estimator = grid.best_estimator_
            else:
                best_params = {}
                best_estimator = grid
            logger.info(f"Fold {fold}: Transforming train/test data with best selector...")
            t6 = time.time()
            try:
                if include_feature_engineering:
                    X_train_sel = best_estimator.named_steps['organ_pca_selector'].transform(
                        best_estimator.named_steps['feature_engineering'].transform(X_train)
                    )
                    X_test_sel = best_estimator.named_steps['organ_pca_selector'].transform(
                        best_estimator.named_steps['feature_engineering'].transform(X_test)
                    )
                else:
                    X_train_sel = best_estimator.named_steps['organ_pca_selector'].transform(X_train)
                    X_test_sel = best_estimator.named_steps['organ_pca_selector'].transform(X_test)
            except Exception as e:
                logger.error(f"Fold {fold}: Data transform failed: {e}", exc_info=True)
                return {'fold': fold, 'model': model_name, 'n_features': n_features, 'error': f'transform_failed: {str(e)}'}
            t7 = time.time()
            logger.info(f"Fold {fold}: Data transform complete. Time: {t7-t6:.2f}s. X_train_sel shape: {X_train_sel.shape}, X_test_sel shape: {X_test_sel.shape}")
            logger.info(f"Fold {fold}: Fitting final model on selected features...")
            t8 = time.time()
            try:
                model_final = best_estimator.named_steps['model']
                model_final.fit(X_train_sel, y_train)
            except Exception as e:
                logger.error(f"Fold {fold}: Final model fit failed: {e}", exc_info=True)
                return {'fold': fold, 'model': model_name, 'n_features': n_features, 'error': f'final_model_fit_failed: {str(e)}'}
            t9 = time.time()
            logger.info(f"Fold {fold}: Final model fit complete. Time: {t9-t8:.2f}s")
            logger.info(f"Fold {fold}: Predicting on test set...")
            t10 = time.time()
            try:
                y_pred = model_final.predict(X_test_sel)
            except Exception as e:
                logger.error(f"Fold {fold}: Prediction failed: {e}", exc_info=True)
                return {'fold': fold, 'model': model_name, 'n_features': n_features, 'error': f'predict_failed: {str(e)}'}
            t11 = time.time()
            logger.info(f"Fold {fold}: Prediction complete. Time: {t11-t10:.2f}s")
            fold_result = {
                'fold': fold,
                'model': model_name,
                'n_features': n_features,
                'best_params': best_params
            }
            logger.info(f"Fold {fold}: Calculating metrics...")
            t12 = time.time()
            if is_classification:
                fold_result['accuracy'] = accuracy_score(y_test, y_pred)
                fold_result['f1_weighted'] = f1_score(y_test, y_pred, average='weighted')
                fold_result['precision_weighted'] = precision_score(y_test, y_pred, average='weighted')
                fold_result['recall_weighted'] = recall_score(y_test, y_pred, average='weighted')
                try:
                    if hasattr(model_final, 'predict_proba'):
                        y_prob = model_final.predict_proba(X_test_sel)[:, 1] if y_pred.ndim == 1 else model_final.predict_proba(X_test_sel)
                        fold_result['roc_auc'] = roc_auc_score(y_test, y_prob)
                except Exception as e:
                    fold_result['roc_auc'] = np.nan
                    logger.warning(f"ROC AUC computation failed for fold {fold}: {e}")
            else:
                from sklearn.metrics import r2_score
                fold_result['r2'] = r2_score(y_test, y_pred)
                fold_result['y_true'] = y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test)
                fold_result['y_pred'] = y_pred.tolist() if hasattr(y_pred, 'tolist') else list(y_pred)
            t13 = time.time()
            logger.info(f"Fold {fold}: Metrics calculation complete. Time: {t13-t12:.2f}s")
            logger.info(f"Completed fold {fold} for model {model_name} with {n_features} features. Metrics: {fold_result}")
            logger.info(f"Fold {fold}: Total time: {t13-t0:.2f}s")
            return fold_result
        except Exception as e:
            logger.error(f"Fold {fold} failed for model {model_name} with {n_features} features: {e}", exc_info=True)
            return {'fold': fold, 'model': model_name, 'n_features': n_features, 'error': str(e)}
    if n_jobs > 1:
        fold_results = Parallel(n_jobs=n_jobs)(delayed(run_fold)(fold) for fold in range(k))
    else:
        fold_results = [run_fold(fold) for fold in range(k)]
    results.extend(fold_results)
    if save_fold_indices_path is not None:
        import json
        with open(save_fold_indices_path, 'w') as f:
            json.dump(fold_indices, f, indent=2)
    return pd.DataFrame(results)