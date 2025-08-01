import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import logging
from typing import Optional

from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

import warnings
warnings.filterwarnings('ignore')

umap = None

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 impute_strategy_numeric: str = 'median', 
                 impute_strategy_categorical: str = 'most_frequent',
                 scale_method: str = 'robust', 
                 data_type: str = 'mixed',
                 epsilon: float = 1e-8,
                 problem_type: str = 'regression',
                 verbose: bool = True,
                 log_level: int = None,
                 **kwargs):
        
        mapping = {
            'polynomial_features': None,
            'polynomial_degree': None,
            'standard_scaling': None,
            'robust_scaling': None,
            'min_max_scaling': None,
            'log_transform': None,
            'sqrt_transform': None,
            'box_cox_transform': None,
            'quantile_transform': None,
            'remove_outliers': None,
            'outlier_method': None,
            'outlier_threshold': None,
        }
        
        for k, v in list(kwargs.items()):
            if k in mapping and mapping[k]:
                setattr(self, mapping[k], v)
                kwargs.pop(k)
            elif k in mapping and mapping[k] is None:
                kwargs.pop(k)
        
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        
        unused = [k for k in kwargs if not hasattr(self, k) and k not in mapping]
        if unused:
            warnings.warn(f"FeatureEngineering: Ignored config keys: {unused}")
        
        self.impute_strategy_numeric = impute_strategy_numeric
        self.impute_strategy_categorical = impute_strategy_categorical
        self.scale_method = scale_method
        self.data_type = data_type
        self.epsilon = epsilon
        self.problem_type = problem_type
        
        self.verbose = verbose
        self.log_level = log_level or (logging.DEBUG if verbose else logging.INFO)
        self.logger = logging.getLogger('FeatureEngineering')
        self.logger.setLevel(self.log_level)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[FeatureEngineering] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        #Initialize transformation_summary
        self.transformation_summary = {}

    def _log(self, *args, level=logging.INFO):
        if self.verbose:
            self.logger.log(level, *args)

    def _get_scaler(self):
        if self.scale_method == 'standard':
            return StandardScaler()
        elif self.scale_method == 'robust':
            return RobustScaler()
        elif self.scale_method == 'none':
            return None
        else:
            self._log(f"Unknown scaling method: {self.scale_method}, using robust", level=logging.WARNING)
            return RobustScaler()







    def _remove_outliers(self, X):
        if not hasattr(self, 'remove_outliers') or not self.remove_outliers:
            return X
        
        outlier_method = getattr(self, 'outlier_method', 'iqr')
        outlier_threshold = getattr(self, 'outlier_threshold', 3.0)
        
        if outlier_method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - outlier_threshold * IQR
            upper_bound = Q3 + outlier_threshold * IQR
            
            mask = ((X >= lower_bound) & (X <= upper_bound)).all(axis=1)
            X_clean = X[mask]
            
            self._log(f"Removed {len(X) - len(X_clean)} outliers using IQR method")
            return X_clean
        
        return X

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureEngineering':
        import time
        start_time = time.time()
        self.logger.info(f"Starting feature engineering with {X.shape[1]} input features")
        
        if getattr(self, 'remove_outliers', False):
            X = self._remove_outliers(X)
            if y is not None and len(y) == len(X):
                if hasattr(y, 'iloc'):
                    y = y.iloc[X.index]
                else:
                    y = y[X.index]
        
        self.engineered_features = []
        
        self.transformation_summary.update({
            'input_shape': X.shape,
            'input_features': list(X.columns),
            'parameters': {
                'impute_strategy_numeric': self.impute_strategy_numeric,
                'impute_strategy_categorical': self.impute_strategy_categorical,
                'scale_method': self.scale_method
            }
        })
        
        self.numeric_cols_fit = list(X.select_dtypes(include=[np.number]).columns)
        self.categorical_cols_fit = list(X.select_dtypes(include=['object', 'category']).columns)
        
        if self.numeric_cols_fit:
            self.imputer_num = SimpleImputer(strategy=self.impute_strategy_numeric)
            self.imputer_num.fit(X[self.numeric_cols_fit])
            self._log(f"Fitted numeric imputer on {len(self.numeric_cols_fit)} columns")
        
        if self.categorical_cols_fit:
            self.imputer_cat = SimpleImputer(strategy=self.impute_strategy_categorical)
            self.imputer_cat.fit(X[self.categorical_cols_fit])
            self._log(f"Fitted categorical imputer on {len(self.categorical_cols_fit)} columns")
        
        if self.categorical_cols_fit:
            self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            cat_imputed = self.imputer_cat.transform(X[self.categorical_cols_fit])
            self.encoder.fit(cat_imputed)
            self._log(f"Fitted one-hot encoder on {len(self.categorical_cols_fit)} columns")
        
        if self.numeric_cols_fit:
            X_num_imputed = pd.DataFrame(
                self.imputer_num.transform(X[self.numeric_cols_fit]),
                columns=self.numeric_cols_fit,
                index=X.index
            )
            self.scaler = self._get_scaler()
            if self.scaler:
                self.scaler.fit(X_num_imputed)
                self._log(f"Fitted scaler: {self.scale_method}")
        
        total_time = time.time() - start_time
        self.logger.info(f"Feature engineering fit complete in {total_time:.2f}s")
        return self

    def transform(self, X):
        import time
        start_time = time.time()
        self.logger.info(f"Transforming data with shape {X.shape}")
        X_transformed = X.copy()

        if self.numeric_cols_fit:
            self.logger.info("Applying numeric imputation...")
            impute_start = time.time()
            X_transformed[self.numeric_cols_fit] = self.imputer_num.transform(X_transformed[self.numeric_cols_fit])
            self.logger.info(f"Imputation took {time.time() - impute_start:.2f}s")
            
            if self.scaler:
                self.logger.info("Applying scaling...")
                scale_start = time.time()
                X_transformed[self.numeric_cols_fit] = self.scaler.transform(X_transformed[self.numeric_cols_fit])
                self.logger.info(f"Scaling took {time.time() - scale_start:.2f}s")

        if self.categorical_cols_fit:
            self.logger.info("Processing categorical features...")
            cat_start = time.time()
            cat_imputed = self.imputer_cat.transform(X_transformed[self.categorical_cols_fit])
            cat_encoded = pd.DataFrame(
                self.encoder.transform(cat_imputed),
                columns=self.encoder.get_feature_names_out(self.categorical_cols_fit),
                index=X.index
            )
            X_transformed = X_transformed.drop(columns=self.categorical_cols_fit)
            X_transformed = pd.concat([X_transformed, cat_encoded], axis=1)
            self.logger.info(f"Categorical processing took {time.time() - cat_start:.2f}s")

        #No feature engineering - only preprocessing
        self._log(f"Final transformed shape: {X_transformed.shape}")

        total_time = time.time() - start_time
        self.logger.info(f"Feature engineering transform complete in {total_time:.2f}s")
        return X_transformed

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def save_transformation_summary(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.transformation_summary:
            with open(output_path / "transformation_summary.json", 'w') as f:
                json.dump(self.transformation_summary, f, indent=2)

        if self.engineered_features:
            with open(output_path / "engineered_features.json", 'w') as f:
                json.dump(self.engineered_features, f, indent=2)

        self._log(f"Transformation summary saved to {output_path}")

    def plot_feature_corr(self, X, y=None, figsize=(10, 10)):
        plt.figure(figsize=figsize)
        sns.heatmap(X.corr(), annot=False, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        return plt.gcf()