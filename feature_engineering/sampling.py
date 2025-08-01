import numpy as np
import pandas as pd
from sklearn.utils import check_random_state
import logging

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
except ImportError:
    SMOTE = None
    RandomUnderSampler = None
    SMOTEENN = None
    SMOTETomek = None

from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import KernelDensity

#not being used 
#scrapped
def detect_imbalance(y, task='classification'):
    if task == 'classification':
        values, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return {'imbalance_ratio': 1.0, 'is_imbalanced': False, 'minority_class': None}
        
        majority_count = counts.max()
        minority_count = counts.min()
        imbalance_ratio = majority_count / minority_count
        
        minority_class = values[np.argmin(counts)]
        
        return {
            'imbalance_ratio': imbalance_ratio,
            'is_imbalanced': imbalance_ratio > 1.5,
            'minority_class': minority_class,
            'class_counts': dict(zip(values, counts)),
            'total_samples': len(y)
        }
    else:
        try:
            from scipy.stats import kurtosis
            kurt = float(kurtosis(y))
            return {
                'kurtosis': kurt,
                'is_imbalanced': kurt > 3,
                'mean': np.mean(y),
                'std': np.std(y)
            }
        except ImportError:
            return {'is_imbalanced': False}


def log_distribution(y, task, msg_prefix=""):
    if task == 'classification':
        values, counts = np.unique(y, return_counts=True)
        logging.info(f"{msg_prefix}Class distribution: {dict(zip(values, counts))}")
        if len(counts) > 1:
            imbalance_ratio = counts.max() / counts.min()
            logging.info(f"{msg_prefix}Imbalance ratio: {imbalance_ratio:.2f}")
    else:
        logging.info(f"{msg_prefix}Target mean: {np.mean(y):.3f}, std: {np.std(y):.3f}, min: {np.min(y):.3f}, max: {np.max(y):.3f}")

def pdf_relevance(y, bandwidth=1):
    y = np.squeeze(np.asarray(y))
    y = y.reshape(len(y), 1)
    pdf = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    pdf.fit(y)
    pdf_vals = np.exp(pdf.score_samples(y))
    y_relevance = 1 - (pdf_vals - pdf_vals.min()) / (pdf_vals.max() - pdf_vals.min() + 1e-8)
    return y_relevance

def split_domains(X, y, relevance, relevance_threshold):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = np.squeeze(np.asarray(relevance))
    rare_indices = np.where(relevance >= relevance_threshold)[0]
    norm_indices = np.where(relevance < relevance_threshold)[0]
    X_rare, y_rare = X[rare_indices, :], y[rare_indices]
    X_norm, y_norm = X[norm_indices, :], y[norm_indices]
    return X_norm, y_norm, X_rare, y_rare

def undersample(X, y, size, random_state=None):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    if size > len(y):
        raise ValueError(f"Cannot undersample to size {size} when data has {len(y)} samples")
    rng = check_random_state(random_state)
    new_indices = rng.choice(range(len(y)), size, replace=False)
    X_new, y_new = X[new_indices, :], y[new_indices]
    return X_new, y_new

def get_neighbors(X, k):
    X = np.asarray(X)
    if len(X) <= k:
        k = len(X) - 1
    if k <= 0:
        return np.array([])
    
    distances = pdist(X, metric='euclidean')
    distance_matrix = squareform(distances)
    
    neighbors = []
    for i in range(len(X)):
        neighbor_indices = np.argsort(distance_matrix[i])[1:k+1]
        neighbors.append(neighbor_indices)
    
    return np.array(neighbors)

def smoter_interpolate(X, y, k, size, random_state=None):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    rng = check_random_state(random_state)
    
    if len(X) <= k:
        k = len(X) - 1
    
    if k <= 0:
        return X, y
    
    neighbors = get_neighbors(X, k)
    
    X_new = []
    y_new = []
    
    for i in range(len(X)):
        X_new.append(X[i])
        y_new.append(y[i])
        
        if len(X_new) >= size:
            break
        
        if len(neighbors[i]) > 0:
            for j in range(min(k, size - len(X_new))):
                if len(X_new) >= size:
                    break
                
                neighbor_idx = neighbors[i][j]
                alpha = rng.uniform(0, 1)
                
                X_interp = X[i] + alpha * (X[neighbor_idx] - X[i])
                y_interp = y[i] + alpha * (y[neighbor_idx] - y[i])
                
                X_new.append(X_interp)
                y_new.append(y_interp)
    
    return np.array(X_new), np.array(y_new)

def oversample_regression(X, y, size, method='smoter', k=5, relevance=None, random_state=None):
    if method == 'smoter':
        return smoter_interpolate(X, y, k, size, random_state)
    else:
        raise ValueError(f"Unknown oversampling method: {method}")

def random_undersample_regression(X, y, relevance_threshold=0.5, under='balance', random_state=0):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    relevance = pdf_relevance(y)
    
    X_norm, y_norm, X_rare, y_rare = split_domains(X, y, relevance, relevance_threshold)
    
    if under == 'balance':
        target_size = len(y_rare)
    elif under == 'ratio':
        target_size = int(len(y_rare) * 0.5)
    else:
        target_size = len(y_norm)
    
    if target_size < len(y_norm):
        X_norm, y_norm = undersample(X_norm, y_norm, target_size, random_state)
    
    X_balanced = np.vstack([X_norm, X_rare])
    y_balanced = np.concatenate([y_norm, y_rare])
    
    return X_balanced, y_balanced

def inverse_density_resample(X, y, bins=10):
    X, y = np.asarray(X), np.squeeze(np.asarray(y))
    
    if len(y) < bins:
        bins = len(y)
    
    hist, bin_edges = np.histogram(y, bins=bins)
    bin_indices = np.digitize(y, bin_edges[:-1])
    
    X_resampled = []
    y_resampled = []
    
    for bin_idx in range(1, len(bin_edges)):
        bin_mask = bin_indices == bin_idx
        X_bin = X[bin_mask]
        y_bin = y[bin_mask]
        
        if len(X_bin) > 0:
            if hist[bin_idx-1] == 0:
                continue
            
            target_samples = max(1, int(len(y) / bins))
            
            if len(X_bin) >= target_samples:
                indices = np.random.choice(len(X_bin), target_samples, replace=False)
                X_resampled.append(X_bin[indices])
                y_resampled.append(y_bin[indices])
            else:
                indices = np.random.choice(len(X_bin), target_samples, replace=True)
                X_resampled.append(X_bin[indices])
                y_resampled.append(y_bin[indices])
    
    if len(X_resampled) > 0:
        X_final = np.vstack(X_resampled)
        y_final = np.concatenate(y_resampled)
        return X_final, y_final
    else:
        return X, y

def balance_data(X, y, task, method='auto', random_state=42, smote_k_neighbors=5, relevance_threshold=0.8):
    if task == 'classification':
        if method == 'auto':
            imbalance_info = detect_imbalance(y, task)
            if imbalance_info['is_imbalanced']:
                if imbalance_info['imbalance_ratio'] > 3:
                    method = 'smote'
                else:
                    method = 'undersample'
            else:
                return X, y
        
        if method == 'smote' and SMOTE is not None:
            smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            return X_resampled, y_resampled
        
        elif method == 'undersample' and RandomUnderSampler is not None:
            rus = RandomUnderSampler(random_state=random_state)
            X_resampled, y_resampled = rus.fit_resample(X, y)
            return X_resampled, y_resampled
        
        elif method == 'smoteenn' and SMOTEENN is not None:
            smoteenn = SMOTEENN(random_state=random_state)
            X_resampled, y_resampled = smoteenn.fit_resample(X, y)
            return X_resampled, y_resampled
        
        elif method == 'smotetomek' and SMOTETomek is not None:
            smotetomek = SMOTETomek(random_state=random_state)
            X_resampled, y_resampled = smotetomek.fit_resample(X, y)
            return X_resampled, y_resampled
        
        else:
            logging.warning(f"Sampling method '{method}' not available, returning original data")
            return X, y
    
    elif task == 'regression':
        if method == 'auto':
            imbalance_info = detect_imbalance(y, task)
            if imbalance_info['is_imbalanced']:
                method = 'inverse_density'
            else:
                return X, y
        
        if method == 'smoter':
            target_size = int(len(y) * 1.5)
            X_resampled, y_resampled = oversample_regression(X, y, target_size, 'smoter', smote_k_neighbors, random_state=random_state)
            return X_resampled, y_resampled
        
        elif method == 'inverse_density':
            X_resampled, y_resampled = inverse_density_resample(X, y, bins=20)
            return X_resampled, y_resampled
        
        elif method == 'undersample':
            X_resampled, y_resampled = random_undersample_regression(X, y, relevance_threshold, 'balance', random_state)
            return X_resampled, y_resampled
        
        else:
            logging.warning(f"Sampling method '{method}' not available for regression, returning original data")
            return X, y
    
    else:
        logging.warning(f"Unknown task '{task}', returning original data")
        return X, y



