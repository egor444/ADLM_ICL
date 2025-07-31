import sys
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor,
    StackingRegressor
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import torch
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from joblib import Parallel, delayed
import logging
import time
from sklearn.base import clone
from sklearn.svm import SVR

# Import context selectors
try:
    from feature_engineering.icl_context import (
        AdaptiveTabPFNContextSelector,
        AdaptiveGPT2ContextSelector
    )
    ICL_CONTEXT_AVAILABLE = True
except ImportError:
    ICL_CONTEXT_AVAILABLE = False
    AdaptiveTabPFNContextSelector = None
    AdaptiveGPT2ContextSelector = None

# Optional imports
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMRegressor = None

try:
    from tabpfn import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    TabPFNRegressor = None

try:
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    GPT2_AVAILABLE = True
except ImportError:
    GPT2_AVAILABLE = False


class TabPFNICLWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, device=None, random_state=42, k_original=30, k_inverse=30, bins=40, num_cores=8, adaptive_context=True, n_features=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.random_state = random_state
        self.k_original = k_original
        self.k_inverse = k_inverse
        self.bins = bins
        self.num_cores = num_cores
        self.adaptive_context = adaptive_context
        self.n_features = n_features
        self.X_train = None
        self.y_train = None
        self.context_selector = None
        self.feature_indices = None
        self.logger = logging.getLogger(__name__)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'device': self.device,
            'random_state': self.random_state,
            'k_original': self.k_original,
            'k_inverse': self.k_inverse,
            'bins': self.bins,
            'num_cores': self.num_cores,
            'adaptive_context': self.adaptive_context,
            'n_features': self.n_features
        }
    
    def set_params(self, **params):
        """Set parameters for this estimator."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
        
    def fit(self, X, y):
        if not hasattr(sys.modules[__name__], '_tabpfn_initialization_logged'):
            self.logger.info(f"TabPFNICLWrapper initialized with device: {self.device}")
            sys.modules[__name__]._tabpfn_initialization_logged = True
        
        self.logger.info("Starting TabPFN model fitting...")
        self.logger.info(f"Input data shape: X={X.shape}, y={y.shape}")
        
        self.X_train = X.values if hasattr(X, 'values') else np.array(X)
        self.y_train = y.values if hasattr(y, 'values') else np.array(y)
        
        if self.feature_indices is not None:
            self.logger.info(f"Using pipeline-selected features: {len(self.feature_indices)} features")
            self.X_train_context = self.X_train[:, self.feature_indices]
        elif self.n_features is not None and self.n_features < self.X_train.shape[1]:
            self.logger.info(f"Using first {self.n_features} features")
            self.X_train_context = self.X_train[:, :self.n_features]
            self.feature_indices = list(range(self.n_features))
        else:
            self.X_train_context = self.X_train
            self.feature_indices = list(range(self.X_train.shape[1]))
            self.logger.info(f"Using all {self.X_train.shape[1]} features")
        
        if not ICL_CONTEXT_AVAILABLE:
            raise ImportError("ICL context module not available")
            
        self.context_selector = AdaptiveTabPFNContextSelector(
            k_original=self.k_original,
            k_inverse=self.k_inverse,
            bins=self.bins,
            adaptive_context=self.adaptive_context
        )
        self.context_selector.fit(self.X_train_context, self.y_train)
        
        self.logger.info("TabPFN model fitted successfully")
        return self
    
    def set_feature_indices(self, feature_indices):
        if feature_indices is not None:
            self.feature_indices = feature_indices
            self.logger.info(f"Updated feature indices: using {len(feature_indices)} features")
            
            if hasattr(self, 'X_train') and self.X_train is not None:
                self.X_train_context = self.X_train[:, feature_indices]
                self.logger.info(f"Updated context data shape: {self.X_train_context.shape}")
                
                if self.context_selector is not None:
                    self.logger.info(f"Refitting context selector with {len(feature_indices)} selected features")
                    self.context_selector.fit(self.X_train_context, self.y_train)
                    self.logger.info("Context selector refitted with new feature indices")
    
    def predict(self, X):
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN not available")
        
        if self.context_selector is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test = X.values if hasattr(X, 'values') else np.array(X)
        
        self.logger.info(f"Starting TabPFN regression with adaptive context selection on {len(X_test)} samples...")
        timestart = time.time()
        
        predictions = Parallel(n_jobs=self.num_cores)(
            delayed(self._process_single_sample)(
                i, X_test, self.context_selector
            ) for i in range(len(X_test))
        )
        
        time_taken = time.time() - timestart
        self.logger.info(f"Predictions completed. Time taken: {time_taken/60:.2f} minutes")
        
        return np.array(predictions)
    
    def _process_single_sample(self, j, X_test, context_selector):
        try:
            x_query = X_test[j]
            
            if self.feature_indices is not None:
                x_query_context = x_query[self.feature_indices]
            else:
                x_query_context = x_query
            
            x_context, y_context = context_selector.select_context(x_query_context)
            
            if j < 5:
                self.logger.debug(f"Sample {j}: context_size={len(x_context)}")
            
            model = TabPFNRegressor(
                device=self.device, 
                random_state=self.random_state + j
            )
            model.fit(x_context, y_context)
            
            y_pred = model.predict(x_query.reshape(1, -1))
            return y_pred[0]
            
        except Exception as e:
            self.logger.warning(f"Error in TabPFN prediction for sample {j}: {e}")
            return np.mean(self.y_train)


class GPT2ICLRegressorWrapper(BaseEstimator, RegressorMixin):
    
    def __init__(self, device=None, n_neighbors=50, aug_factor=0.5, adaptive_context=True, n_features=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.n_neighbors = n_neighbors
        self.aug_factor = aug_factor
        self.adaptive_context = adaptive_context
        self.n_features = n_features
        self.X_train = None
        self.y_train = None
        self.context_selector = None
        self.feature_indices = None
        self.logger = logging.getLogger(__name__)
        
        self.AGE_BINS = [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]
        
        # Initialize GPT2 model and tokenizer
        if GPT2_AVAILABLE:
            try:
                self.model = GPT2LMHeadModel.from_pretrained("gpt2").to(self.device)
                self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.model.eval()
                self.GPT2_AVAILABLE = True
            except Exception as e:
                self.logger.warning(f"Could not load GPT2 model: {e}")
                self.GPT2_AVAILABLE = False
        else:
            self.GPT2_AVAILABLE = False
        
        if ICL_CONTEXT_AVAILABLE and AdaptiveGPT2ContextSelector:
            self.context_selector = AdaptiveGPT2ContextSelector(
                n_neighbors=self.n_neighbors,
                aug_factor=self.aug_factor,
                adaptive_context=self.adaptive_context
            )
        else:
            self.context_selector = None
            self.logger.warning("GPT2 context selector not available")
        
        self.is_fitted = False
    
    def get_params(self, deep=True):
        #Get parameters for this estimator
        return {
            'device': self.device,
            'n_neighbors': self.n_neighbors,
            'aug_factor': self.aug_factor,
            'adaptive_context': self.adaptive_context,
            'n_features': self.n_features
        }
    
    def set_params(self, **params):
        #Set parameters for this estimator
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
        
    def bin_label(self, age):
        for lo, hi in self.AGE_BINS:
            if lo <= age < hi:
                return f"between {lo} and {hi}"
        return "unknown"
    
    def bin_midpoint(self, label):
        import re
        match = re.match(r"between (\d+) and (\d+)", label)
        if match:
            return (int(match.group(1)) + int(match.group(2))) / 2
        return 0.0
        
    def fit(self, X, y):
        if not hasattr(sys.modules[__name__], '_gpt2_initialization_logged'):
            self.logger.info(f"GPT2ICLRegressorWrapper initialized with device: {self.device}")
            sys.modules[__name__]._gpt2_initialization_logged = True
        
        self.logger.info("Starting GPT2 model fitting...")
        self.logger.info(f"Input data shape: X={X.shape}, y={y.shape}")
        
        self.X_train = X.values if hasattr(X, 'values') else np.array(X)
        self.y_train = y.values if hasattr(y, 'values') else np.array(y)
        
        if self.feature_indices is not None:
            self.logger.info(f"Using pipeline-selected features: {len(self.feature_indices)} features")
            self.X_train_context = self.X_train[:, self.feature_indices]
        elif self.n_features is not None and self.n_features < self.X_train.shape[1]:
            self.logger.info(f"Using first {self.n_features} features")
            self.X_train_context = self.X_train[:, :self.n_features]
            self.feature_indices = list(range(self.n_features))
        else:
            self.X_train_context = self.X_train
            self.feature_indices = list(range(self.X_train.shape[1]))
            self.logger.info(f"Using all {self.X_train.shape[1]} features")
        
        if not ICL_CONTEXT_AVAILABLE:
            raise ImportError("ICL context module not available")
            
        self.context_selector = AdaptiveGPT2ContextSelector(
            n_neighbors=self.n_neighbors,
            aug_factor=self.aug_factor,
            adaptive_context=self.adaptive_context
        )
        self.context_selector.fit(self.X_train_context, self.y_train)
        
        self.logger.info("GPT2 model fitted successfully")
        return self
    
    def set_feature_indices(self, feature_indices):
        if feature_indices is not None:
            self.feature_indices = feature_indices
            self.logger.info(f"Updated feature indices: using {len(feature_indices)} features")
            
            if hasattr(self, 'X_train') and self.X_train is not None:
                self.X_train_context = self.X_train[:, feature_indices]
                self.logger.info(f"Updated context data shape: {self.X_train_context.shape}")
                
                if self.context_selector is not None:
                    self.logger.info(f"Refitting context selector with {len(feature_indices)} selected features")
                    self.context_selector.fit(self.X_train_context, self.y_train)
                    self.logger.info("Context selector refitted with new feature indices")
    
    def predict(self, X):
        if not self.GPT2_AVAILABLE:
            self.logger.warning("GPT2 not available")
            return np.array([np.mean(self.y_train)] * len(X))
            
        X_test = X.values if hasattr(X, 'values') else np.array(X)
        
        self.logger.info(f"Starting GPT2 adaptive ICL predictions on {len(X_test)} samples...")
        timestart = time.time()
        
        predictions = Parallel(n_jobs=8)(
            delayed(self._process_gpt2_sample)(
                i, X_test, self.context_selector
            ) for i in range(len(X_test))
        )
        
        time_taken = time.time() - timestart
        self.logger.info(f"GPT2 predictions completed. Time taken: {time_taken/60:.2f} minutes")
        
        return np.array(predictions)
    
    def _process_gpt2_sample(self, i, X_test, context_selector):
        try:
            query = X_test[i]
            
            if hasattr(self, 'feature_indices'):
                query_context = query[self.feature_indices]
            else:
                query_context = query
            
            X_neighbors, y_neighbors = context_selector.select_context(query_context)
            
            prompt = "The age is one of these ranges: 0-10, 10-20, ..., 90-100 years.\n"
            for x, y_val in zip(X_neighbors, y_neighbors):
                x_str = ", ".join([f"{val:.1f}" for val in x[:20]])
                age_text = self.bin_label(y_val)
                prompt += f"The patient has features: {x_str}. The age is {age_text}.\n"
            
            x_query = ", ".join([f"{val:.1f}" for val in query[:20]])
            prompt += f"The patient has features: {x_query}. The age is"
            
            trunc_prompt = prompt[-1000:]
            inputs = self.tokenizer([trunc_prompt], return_tensors="pt", padding=True, truncation=True, max_length=900).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=True,
                    top_k=50,
                    num_return_sequences=5,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            bin_preds = []
            for txt in decoded:
                import re
                match = re.search(r"between (\d+) and (\d+)", txt)
                if match:
                    mid = (int(match.group(1)) + int(match.group(2))) / 2
                    bin_preds.append(mid)
            
            pred = np.median(bin_preds) if bin_preds else np.mean(self.y_train)
            return pred
            
        except Exception as e:
            self.logger.warning(f"Error in GPT2 prediction for sample {i}: {e}")
            return np.mean(self.y_train)


def create_hybrid_ensemble(base_models, top_n=5, cv_folds=5, random_state=42):
    traditional_models = {k: v for k, v in base_models.items() 
                        if not k.endswith('ICL')}
    icl_models = {k: v for k, v in base_models.items() 
                  if k.endswith('ICL')}
    
    traditional_names = list(traditional_models.keys())[:top_n//2]
    traditional_selected = {name: traditional_models[name] for name in traditional_names}
    icl_selected = icl_models.copy()
    
    hybrid_models = {**traditional_selected, **icl_selected}
    estimators = [(name, model) for name, model in hybrid_models.items()]
    
    meta_learner = Ridge(random_state=random_state)
    hybrid_ensemble = StackingRegressor(
        estimators=estimators,
        final_estimator=meta_learner,
        cv=cv_folds,
        n_jobs=1
    )
    
    hybrid_ensemble.base_model_names = list(hybrid_models.keys())
    hybrid_ensemble.meta_learner_name = type(meta_learner).__name__
    hybrid_ensemble.traditional_models = list(traditional_selected.keys())
    hybrid_ensemble.icl_models = list(icl_selected.keys())
    
    return hybrid_ensemble


def get_regression_models(random_state=42, params=None, model_names=None):
    models = {}
    
    if model_names is not None:
        available_models = {
            'LinearRegression': lambda: LinearRegression(),
            'Ridge': lambda: Ridge(random_state=random_state),
            'Lasso': lambda: Lasso(random_state=random_state),
            'ElasticNet': lambda: ElasticNet(random_state=random_state),
            'RandomForest': lambda: RandomForestRegressor(random_state=random_state, n_jobs=-1),
            'HistGradientBoostingRegressor': lambda: HistGradientBoostingRegressor(random_state=random_state),
            'MLP': lambda: MLPRegressor(max_iter=500, random_state=random_state),
            'KNN': lambda: KNeighborsRegressor(n_jobs=-1),
            'DecisionTree': lambda: DecisionTreeRegressor(random_state=random_state),
        }
        
        if XGBOOST_AVAILABLE:
            available_models['XGBoost'] = lambda: XGBRegressor(random_state=random_state, n_jobs=-1)
        
        if LIGHTGBM_AVAILABLE:
            available_models['LightGBM'] = lambda: LGBMRegressor(random_state=random_state, n_jobs=-1)
        
        if ICL_CONTEXT_AVAILABLE:
            available_models['TabPFNICL'] = lambda: TabPFNICLWrapper(
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_state=random_state,
                k_original=30,
                k_inverse=30,
                bins=40,
                num_cores=8,
                adaptive_context=True
            )
            available_models['GPT2ICL'] = lambda: GPT2ICLRegressorWrapper(
                device="cuda" if torch.cuda.is_available() else "cpu",
                n_neighbors=50,
                aug_factor=0.5,
                adaptive_context=True
            )
        
        for model_name in model_names:
            if model_name in available_models:
                models[model_name] = available_models[model_name]()
            else:
                print(f" Model '{model_name}' not available")
    else:
        models['LinearRegression'] = LinearRegression()
        models['Ridge'] = Ridge(random_state=random_state)
        models['Lasso'] = Lasso(random_state=random_state)
        models['ElasticNet'] = ElasticNet(random_state=random_state)
        models['RandomForest'] = RandomForestRegressor(random_state=random_state, n_jobs=-1)
        models['HistGradientBoostingRegressor'] = HistGradientBoostingRegressor(random_state=random_state)
        models['MLP'] = MLPRegressor(max_iter=500, random_state=random_state)
        models['KNN'] = KNeighborsRegressor(n_jobs=-1)
        models['DecisionTree'] = DecisionTreeRegressor(random_state=random_state)
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBRegressor(random_state=random_state, n_jobs=-1)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMRegressor(random_state=random_state, n_jobs=-1)
        
        if ICL_CONTEXT_AVAILABLE:
            models['TabPFNICL'] = TabPFNICLWrapper(
                device="cuda" if torch.cuda.is_available() else "cpu",
                random_state=random_state,
                k_original=30,
                k_inverse=30,
                bins=40,
                num_cores=8,
                adaptive_context=True
            )
            
            models['GPT2ICL'] = GPT2ICLRegressorWrapper(
                device="cuda" if torch.cuda.is_available() else "cpu",
                n_neighbors=50,
                aug_factor=0.5,
                adaptive_context=True
            )
    
    return models

