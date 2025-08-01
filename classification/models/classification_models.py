import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from joblib import Parallel, delayed
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import logging
import time
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None

try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    TabPFNClassifier = None

try:
    from .gpt2_im_context import InContextClassificationDataset, GPT2ICLLabelDecoder
    GPT2ICL_AVAILABLE = True
except ImportError:
    GPT2ICL_AVAILABLE = False
    InContextClassificationDataset = None
    GPT2ICLLabelDecoder = None


class GPT2ICLClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, max_features=30, template='natural', device=None):
        self.max_features = max_features
        self.template = template
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.label_texts = ["negative", "positive"]
        
        
        try:
            self.model = GPT2ICLLabelDecoder(label_texts=self.label_texts, device=self.device)
            self.model_available = True
        except Exception as e:
            print(f"Warning: Could not initialize GPT2ICL model: {e}")
            self.model_available = False
            self.model = None
            
        self.X_train = None
        self.y_train = None
        self.feature_indices = None
        
    def set_feature_indices(self, feature_indices):
        self.feature_indices = feature_indices
        
    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, "values") else np.array(X)
        self.y_train = y.values if hasattr(y, "values") else np.array(y)
        # Set classes_ attribute required by scikit-learn
        self.classes_ = np.unique(y)
        return self
        
    def predict(self, X):
        if not self.model_available or self.model is None:
            # Fallback to random predictions if model is not available
            return np.random.choice([0, 1], size=len(X))
            
        X_test = X.values if hasattr(X, "values") else np.array(X)
        
        if self.feature_indices is not None:
            X_test = X_test[:, self.feature_indices]
            self.X_train = self.X_train[:, self.feature_indices]
        
        try:
            ds = InContextClassificationDataset(
                X_test, np.zeros(len(X_test)), self.X_train, self.y_train,
                max_features=self.max_features, template=self.template
            )
            loader = DataLoader(ds, batch_size=8, shuffle=False)
            all_preds = []
            for batch in loader:
                prompts, _, _ = batch
                preds, _ = self.model.predict(prompts)
                all_preds.extend(preds)
            return np.array(all_preds)
        except Exception as e:
            print(f"Warning: GPT2ICL prediction failed: {e}")
            # Fallback to random predictions
            return np.random.choice([0, 1], size=len(X))
        
    def predict_proba(self, X):
        if not self.model_available or self.model is None:
            # Fallback to uniform probabilities if model is not available
            return np.array([[0.5, 0.5] for _ in range(len(X))])
            
        X_test = X.values if hasattr(X, "values") else np.array(X)
        
        if self.feature_indices is not None:
            X_test = X_test[:, self.feature_indices]
            self.X_train = self.X_train[:, self.feature_indices]
        
        try:
            ds = InContextClassificationDataset(
                X_test, np.zeros(len(X_test)), self.X_train, self.y_train,
                max_features=self.max_features, template=self.template
            )
            prompts = [ds[i][0] for i in range(len(ds))]
            _, probs = self.model.predict(prompts)
            return np.array(probs)
        except Exception as e:
            print(f"Warning: GPT2ICL predict_proba failed: {e}")
            # Fallback to uniform probabilities
            return np.array([[0.5, 0.5] for _ in range(len(X))])


class TabPFNClassifierWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, context_mode="nearest", context_size=40, random_state=42, n_jobs=1):
        self.context_mode = context_mode
        self.context_size = context_size
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.X_train = None
        self.y_train = None
        self.feature_indices = None
        
    def set_feature_indices(self, feature_indices):
        self.feature_indices = feature_indices
        
    def fit(self, X, y):
        self.X_train = X.values if hasattr(X, "values") else np.array(X)
        self.y_train = y.values if hasattr(y, "values") else np.array(y)
        # Set classes_ attribute required by scikit-learn
        self.classes_ = np.unique(y)
        return self
        
    def _predict_one(self, x_query, idx):
        try:
            if self.context_mode == "nearest":
                nn = NearestNeighbors(n_neighbors=self.context_size)
                nn.fit(self.X_train)
                distances, indices = nn.kneighbors([x_query])
                X_context = self.X_train[indices[0]]
                y_context = self.y_train[indices[0]]
            else:
                X_context = self.X_train[:self.context_size]
                y_context = self.y_train[:self.context_size]
                
            model = TabPFNClassifier(device="cpu", N_ensemble_configurations=32)
            model.fit(X_context, y_context)
            pred = model.predict([x_query])
            return pred[0]
        except Exception as e:
            print(f"Warning: TabPFN prediction failed for sample {idx}: {e}")
            return np.random.choice(np.unique(self.y_train))
            
    def predict(self, X):
        X_test = X.values if hasattr(X, "values") else np.array(X)
        
        if self.feature_indices is not None:
            X_test = X_test[:, self.feature_indices]
            self.X_train = self.X_train[:, self.feature_indices]
            
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(self._predict_one)(X_test[i], i) for i in range(len(X_test))
        )
        return np.array(predictions)


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
    
    meta_learner = LogisticRegression(random_state=random_state)
    hybrid_ensemble = StackingClassifier(
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


def get_classification_models(random_state=42, params=None, model_names=None):
    models = {}
    
    if model_names is not None:
        available_models = {
            'LogisticRegression': lambda: LogisticRegression(random_state=random_state),
            'RandomForest': lambda: RandomForestClassifier(random_state=random_state, n_jobs=-1),
            'DecisionTree': lambda: DecisionTreeClassifier(random_state=random_state),
            'HistGradientBoosting': lambda: HistGradientBoostingClassifier(random_state=random_state),
            'MLP': lambda: MLPClassifier(max_iter=500, random_state=random_state),
            'KNeighbors': lambda: KNeighborsClassifier(n_jobs=-1),
            'NaiveBayes': lambda: GaussianNB(),
        }
        
        if XGBOOST_AVAILABLE:
            available_models['XGBoost'] = lambda: XGBClassifier(random_state=random_state, n_jobs=-1)
        
        if LIGHTGBM_AVAILABLE:
            available_models['LightGBM'] = lambda: LGBMClassifier(random_state=random_state, n_jobs=-1)
        
        if TABPFN_AVAILABLE:
            available_models['TabPFNICL'] = lambda: TabPFNClassifierWrapper(
                context_mode="nearest",
                context_size=40,
                random_state=random_state,
                n_jobs=8
            )
        
        if GPT2ICL_AVAILABLE:
            available_models['GPT2ICL'] = lambda: GPT2ICLClassifierWrapper(
                max_features=30,
                template='natural',
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            print("Warning: GPT2ICL not available - skipping")
        
        for model_name in model_names:
            if model_name in available_models:
                models[model_name] = available_models[model_name]()
            else:
                print(f"Warning: Model '{model_name}' not available or not supported")
    else:
        models['LogisticRegression'] = LogisticRegression(random_state=random_state)
        models['RandomForest'] = RandomForestClassifier(random_state=random_state, n_jobs=-1)
        models['DecisionTree'] = DecisionTreeClassifier(random_state=random_state)
        models['HistGradientBoosting'] = HistGradientBoostingClassifier(random_state=random_state)
        models['MLP'] = MLPClassifier(max_iter=500, random_state=random_state)
        models['KNeighbors'] = KNeighborsClassifier(n_jobs=-1)
        models['NaiveBayes'] = GaussianNB()
        
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = XGBClassifier(random_state=random_state, n_jobs=-1)
        
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = LGBMClassifier(random_state=random_state, n_jobs=-1)
        
        if TABPFN_AVAILABLE:
            models['TabPFNICL'] = TabPFNClassifierWrapper(
                context_mode="nearest",
                context_size=40,
                random_state=random_state,
                n_jobs=8
            )
        
        if GPT2ICL_AVAILABLE:
            models['GPT2ICL'] = GPT2ICLClassifierWrapper(
                max_features=30,
                template='natural',
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
    
    return models



