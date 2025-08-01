import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import re

def validate_data(X, y, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(X) == 0 or len(y) == 0:
        logger.error("Empty data detected")
        return False
    
    if X.isnull().any().any():
        logger.warning("X contains NaN values")
    
    if pd.isnull(y).any():
        logger.warning("y contains NaN values")
    
    if np.isinf(X.values).any():
        logger.error("X contains infinite values")
        return False
    
    if np.isinf(y).any():
        logger.error("y contains infinite values")
        return False
    
    return True

def detect_class_imbalance(y, threshold=1.5):
    values, counts = np.unique(y, return_counts=True)
    
    if len(counts) < 2:
        return {
            'is_imbalanced': False,
            'imbalance_ratio': 1.0,
            'minority_class': None,
            'class_distribution': dict(zip(values, counts))
        }
    
    majority_count = counts.max()
    minority_count = counts.min()
    imbalance_ratio = majority_count / minority_count
    minority_class = values[np.argmin(counts)]
    
    return {
        'is_imbalanced': imbalance_ratio > threshold,
        'imbalance_ratio': imbalance_ratio,
        'minority_class': minority_class,
        'class_distribution': dict(zip(values, counts)),
        'severity': 'severe' if imbalance_ratio > 10 else 'moderate' if imbalance_ratio > 3 else 'mild'
    }

def detect_regression_imbalance(y, threshold=3.0):
    try:
        from scipy.stats import kurtosis, skew
        kurt = float(kurtosis(y))
        skewness = float(skew(y))
        
        return {
            'is_imbalanced': kurt > threshold,
            'kurtosis': kurt,
            'skewness': skewness,
            'severity': 'very_skewed' if kurt > 5 else 'skewed' if kurt > 3 else 'normal'
        }
    except ImportError:
        mean_val = np.mean(y)
        std_val = np.std(y)
        return {
            'is_imbalanced': False,
            'mean': mean_val,
            'std': std_val,
            'severity': 'unknown'
        }

def group_features_by_organ(feature_names):
    organ_groups = {}
    
    for feature in feature_names:
        organ = extract_organ_from_feature(feature)
        
        if organ:
            if organ not in organ_groups:
                organ_groups[organ] = []
            organ_groups[organ].append(feature)
        else:
            if 'unknown' not in organ_groups:
                organ_groups['unknown'] = []
            organ_groups['unknown'].append(feature)
    
    return organ_groups

def extract_organ_from_feature(feature_name):
    organs = [
        'liver', 'spleen', 'kidney', 'lung', 'heart', 'brain', 'pancreas',
        'stomach', 'intestine', 'bladder', 'prostate', 'uterus', 'ovary',
        'breast', 'thyroid', 'adrenal', 'gallbladder', 'esophagus',
        'bone', 'muscle', 'fat', 'skin', 'blood', 'lymph'
    ]
    
    # Convert to string if it's not already
    feature_str = str(feature_name)
    feature_lower = feature_str.lower()
    
    for organ in organs:
        if organ in feature_lower:
            return organ
    
    return None

def get_organ_importance(feature_importances, feature_names):
    organ_groups = group_features_by_organ(feature_names)
    organ_importance = {}
    
    for organ, features in organ_groups.items():
        organ_features = [f for f in features if f in feature_importances.index]
        if organ_features:
            organ_importance[organ] = feature_importances.loc[organ_features].sum()
        else:
            organ_importance[organ] = 0
    
    return organ_importance

def analyze_feature_stability(feature_importance_scores, n_folds):
    if not feature_importance_scores:
        return {}
    
    feature_counts = {}
    feature_scores = {}
    
    for fold_data in feature_importance_scores:
        if 'selected_features' in fold_data:
            for feat in fold_data['selected_features']:
                feature_counts[feat] = feature_counts.get(feat, 0) + 1
                if feat not in feature_scores:
                    feature_scores[feat] = []
                feature_scores[feat].append(fold_data.get('importance_score', 0))
    
    stability_metrics = {}
    for feat, count in feature_counts.items():
        stability_metrics[feat] = {
            'selection_frequency': count / n_folds,
            'mean_importance': np.mean(feature_scores[feat]),
            'std_importance': np.std(feature_scores[feat])
        }
    
    return stability_metrics

def log_imbalance_summary(y, task='classification', logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if task == 'classification':
        imbalance_info = detect_class_imbalance(y)
        logger.info(f"Class imbalance analysis:")
        logger.info(f"  - Imbalance ratio: {imbalance_info['imbalance_ratio']:.2f}")
        logger.info(f"  - Severity: {imbalance_info['severity']}")
        logger.info(f"  - Minority class: {imbalance_info['minority_class']}")
        logger.info(f"  - Class distribution: {imbalance_info['class_distribution']}")
        
        if imbalance_info['is_imbalanced']:
            logger.warning(f"Class imbalance detected! Ratio: {imbalance_info['imbalance_ratio']:.2f}")
        else:
            logger.info("No significant class imbalance detected")
            
    else:
        imbalance_info = detect_regression_imbalance(y)
        logger.info(f"Regression imbalance analysis:")
        logger.info(f"  - Kurtosis: {imbalance_info.get('kurtosis', 'N/A'):.2f}")
        logger.info(f"  - Skewness: {imbalance_info.get('skewness', 'N/A'):.2f}")
        logger.info(f"  - Severity: {imbalance_info['severity']}")
        
        if imbalance_info['is_imbalanced']:
            logger.warning(f"Distribution imbalance detected! Kurtosis: {imbalance_info.get('kurtosis', 'N/A'):.2f}")
        else:
            logger.info("No significant distribution imbalance detected")

def recommend_sampling_strategy(y, task='classification', n_samples=None):
    if n_samples is None:
        n_samples = len(y)
    
    if task == 'classification':
        imbalance_info = detect_class_imbalance(y)
        
        if not imbalance_info['is_imbalanced']:
            return 'none'
        
        ratio = imbalance_info['imbalance_ratio']
        
        if ratio > 10:
            if n_samples < 1000:
                return 'smoteenn'
            else:
                return 'smotetomek'
        elif ratio > 3:
            if n_samples < 500:
                return 'smote'
            else:
                return 'smoteenn'
        else:
            return 'smote'
            
    else:
        imbalance_info = detect_regression_imbalance(y)
        
        if not imbalance_info['is_imbalanced']:
            return 'none'
        
        kurt = imbalance_info.get('kurtosis', 0)
        
        if kurt > 5:
            return 'smoter'
        else:
            return 'inverse_density' 