import numpy as np
import pandas as pd
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import random

logger = logging.getLogger(__name__)

def inverse_density_resample(X, y, bins=10):
    """
    Oversample rare label regions, undersample frequent ones
    """
    logger.debug(f"Inverse density resampling with {bins} bins, original size: {len(y)}")
    
    y_binned = pd.qcut(y, q=bins, duplicates='drop')
    counts = y_binned.value_counts()
    inverse_weights = 1.0 / counts[y_binned].values
    probs = inverse_weights / inverse_weights.sum()
    
    idx = np.arange(len(y))
    sampled_idx = np.random.choice(idx, size=len(y), replace=True, p=probs)

    X_resampled = X.iloc[sampled_idx].reset_index(drop=True)
    y_resampled = y.iloc[sampled_idx].reset_index(drop=True)
    
    logger.debug(f"Inverse density resampling completed. Resampled size: {len(y_resampled)}")
    logger.debug(f"Original target stats: mean={np.mean(y):.3f}, std={np.std(y):.3f}")
    logger.debug(f"Resampled target stats: mean={np.mean(y_resampled):.3f}, std={np.std(y_resampled):.3f}")

    return X_resampled, y_resampled

class AdaptiveTabPFNContextSelector:
    """Adaptive context selector for TabPFN with inverse density sampling."""
    
    def __init__(self, k_original=30, k_inverse=30, bins=40, adaptive_context=True):
        self.k_original = k_original
        self.k_inverse = k_inverse
        self.bins = bins
        self.adaptive_context = adaptive_context
        self.X_train = None
        self.y_train = None
        self.x_inv = None
        self.y_inv = None
        self.nn_orig = None
        self.nn_inv = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X_train, y_train):
        """Fit the adaptive TabPFN context selector."""
        self.logger.info("Fitting AdaptiveTabPFNContextSelector...")
        
        # Convert to pandas if needed for inverse density resampling
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(y_train, pd.Series):
            y_train = pd.Series(y_train)
            
        self.X_train = X_train.values
        self.y_train = y_train.values
        
        self.logger.info(f"Training data prepared: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        
        # Create inverse density resampled data
        self.logger.info(f"Creating inverse density resampled data with {self.bins} bins...")
        self.x_inv, self.y_inv = inverse_density_resample(X_train, y_train, bins=self.bins)
        self.x_inv = self.x_inv.values
        self.y_inv = self.y_inv.values
        
        self.logger.info(f"Inverse density data created: x_inv={self.x_inv.shape}, y_inv={self.y_inv.shape}")
        
        # Fit nearest neighbors for both original and inverse data
        self.logger.info(f"Fitting nearest neighbors: k_original={self.k_original}, k_inverse={self.k_inverse}")
        self.nn_orig = NearestNeighbors(n_neighbors=max(50, self.k_original), metric='cosine')
        self.nn_inv = NearestNeighbors(n_neighbors=max(50, self.k_inverse), metric='cosine')
        self.nn_orig.fit(self.X_train)
        self.nn_inv.fit(self.x_inv)
        
        self.logger.info("AdaptiveTabPFNContextSelector fitted successfully")
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context using adaptive inverse density selection."""
        if self.adaptive_context:
            # Get neighbors from both datasets for density analysis
            distances_orig, indices_orig = self.nn_orig.kneighbors(x_query.reshape(1, -1), n_neighbors=50)
            distances_inv, indices_inv = self.nn_inv.kneighbors(x_query.reshape(1, -1), n_neighbors=50)
            
            # Analyze local density to determine adaptive context size
            neighbors_y_orig = [self.y_train[i] for i in indices_orig[0]]
            neighbors_y_inv = [self.y_inv[i] for i in indices_inv[0]]
            
            # Calculate density metrics - use mean of neighbor values for comparison
            mean_y_orig = np.mean(neighbors_y_orig)
            mean_y_inv = np.mean(neighbors_y_inv)
            close_count_orig = sum(abs(np.array(neighbors_y_orig) - mean_y_orig) < 2.0)
            close_count_inv = sum(abs(np.array(neighbors_y_inv) - mean_y_inv) < 2.0)
            total_close = close_count_orig + close_count_inv
            
            # Adaptive context sizing based on local density
            if total_close <= 5:
                region = 'sparse'
                k_orig_adaptive = max(5, self.k_original // 3)
                k_inv_adaptive = max(5, self.k_inverse // 3)
            elif total_close <= 15:
                region = 'medium'
                k_orig_adaptive = max(10, self.k_original // 2)
                k_inv_adaptive = max(10, self.k_inverse // 2)
            else:
                region = 'dense'
                k_orig_adaptive = self.k_original
                k_inv_adaptive = self.k_inverse
            
            # Get adaptive neighbors
            idx_orig = self.nn_orig.kneighbors(x_query.reshape(1, -1), n_neighbors=k_orig_adaptive, return_distance=False).flatten()
            idx_inv = self.nn_inv.kneighbors(x_query.reshape(1, -1), n_neighbors=k_inv_adaptive, return_distance=False).flatten()
            
            self.logger.debug(f"Adaptive context: {region} region, total_close={total_close}, k_orig={k_orig_adaptive}, k_inv={k_inv_adaptive}")
        else:
            # Use fixed context sizes
            idx_orig = self.nn_orig.kneighbors(x_query.reshape(1, -1), n_neighbors=self.k_original, return_distance=False).flatten()
            idx_inv = self.nn_inv.kneighbors(x_query.reshape(1, -1), n_neighbors=self.k_inverse, return_distance=False).flatten()
            region = 'fixed'
            k_orig_adaptive = self.k_original
            k_inv_adaptive = self.k_inverse
        
        # Combine context
        x_context = np.vstack([self.X_train[idx_orig], self.x_inv[idx_inv]])
        y_context = np.concatenate([self.y_train[idx_orig], self.y_inv[idx_inv]])
        
        return x_context, y_context

class AdaptiveGPT2ContextSelector:
    """Adaptive context selector for GPT2 with augmented data."""
    
    def __init__(self, n_neighbors=50, aug_factor=0.5, adaptive_context=True):
        self.n_neighbors = n_neighbors
        self.aug_factor = aug_factor
        self.adaptive_context = adaptive_context
        self.X_train = None
        self.y_train = None
        self.X_aug = None
        self.y_aug = None
        self.nn_orig = None
        self.nn_aug = None
        self.logger = logging.getLogger(__name__)
        
    def fit(self, X_train, y_train):
        """Fit the adaptive GPT2 context selector."""
        self.logger.info("Fitting AdaptiveGPT2ContextSelector...")
        
        self.X_train = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        self.y_train = y_train.values if hasattr(y_train, 'values') else np.array(y_train)
        
        self.logger.info(f"Training data prepared: X_train={self.X_train.shape}, y_train={self.y_train.shape}")
        
        # Create augmented data for adaptive ICL
        self.logger.info("Creating augmented data for adaptive ICL...")
        hist, bin_edges = np.histogram(self.y_train, bins=10)
        bin_ids = np.digitize(self.y_train, bin_edges[:-1], right=True)
        inv_freq = {i: 1 / (count + 1e-6) for i, count in enumerate(hist)}
        weights = np.array([
            inv_freq.get(max(0, min(bin_id - 1, len(inv_freq) - 1)), 1.0)
            for bin_id in bin_ids
        ])
        probs = weights / weights.sum()
        
        aug_size = int(self.aug_factor * len(self.y_train))
        aug_indices = np.random.choice(len(self.y_train), size=aug_size, p=probs)
        self.X_aug = self.X_train[aug_indices]
        self.y_aug = self.y_train[aug_indices]
        
        self.logger.info(f"Augmented data created: X_aug={self.X_aug.shape}, y_aug={self.y_aug.shape}")
        
        # Create nearest neighbors for original and augmented data
        self.logger.info("Fitting nearest neighbors for context selection...")
        self.nn_orig = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_train)
        self.nn_aug = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X_aug)
        
        self.logger.info("AdaptiveGPT2ContextSelector fitted successfully")
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context using adaptive GPT2 selection."""
        if self.adaptive_context:
            # Get nearest neighbors for density analysis
            distances, indices = self.nn_orig.kneighbors([x_query], n_neighbors=self.n_neighbors)
            neighbors_y = [self.y_train[j] for j in indices[0]]
            # Calculate density based on neighbor values, not comparing with x_query
            mean_y = np.mean(neighbors_y)
            close_count = sum(abs(np.array(neighbors_y) - mean_y) < 2.0)
            
            # Adaptive context size based on region
            if close_count <= 3:
                region = 'few-shot'
                k_context = 2
                aug_k = 2
            elif close_count <= 20:
                region = 'medium-shot'
                k_context = 4
                aug_k = 4
            else:
                region = 'many-shot'
                k_context = 6
                aug_k = 6
            
            self.logger.debug(f"Adaptive context: {region} region, close_count={close_count}, k_context={k_context}, aug_k={aug_k}")
        else:
            # Use fixed context sizes
            k_context = 4
            aug_k = 4
            region = 'fixed'
        
        # Get context samples
        indices_orig = self.nn_orig.kneighbors([x_query], n_neighbors=k_context + 1)[1][0][1:]
        indices_aug = self.nn_aug.kneighbors([x_query], n_neighbors=aug_k)[1][0]
        
        X_neighbors = np.vstack([self.X_train[indices_orig], self.X_aug[indices_aug]])
        y_neighbors = np.concatenate([self.y_train[indices_orig], self.y_aug[indices_aug]])
        
        return X_neighbors, y_neighbors

class ContextSelector(nn.Module):
    """Neural network-based context selector using REINFORCE algorithm."""
    
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.item_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x_query, x_train):
        q_embed = self.query_encoder(x_query).expand(x_train.size(0), -1)
        x_embed = self.item_encoder(x_train)
        pair_embed = torch.cat([q_embed, x_embed], dim=1)
        logits = self.scorer(pair_embed).squeeze(-1)
        probs = torch.sigmoid(logits)
        return probs

class NearestNeighborContextSelector:
    """Simple nearest neighbor-based context selection."""
    
    def __init__(self, n_neighbors=30, metric='cosine'):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.nn = None
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """Fit the nearest neighbor selector."""
        self.X_train = X_train
        self.y_train = y_train
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric=self.metric)
        self.nn.fit(X_train)
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context using nearest neighbors."""
        if context_size is None:
            context_size = self.n_neighbors
            
        # Get nearest neighbors
        distances, indices = self.nn.kneighbors(x_query.reshape(1, -1))
        
        # Return top context_size neighbors
        selected_indices = indices[0][:context_size]
        return self.X_train[selected_indices], self.y_train[selected_indices]

class InverseDensityContextSelector:
    """Inverse density sampling for context selection."""
    
    def __init__(self, bins=40, k_original=30, k_inverse=30):
        self.bins = bins
        self.k_original = k_original
        self.k_inverse = k_inverse
        self.X_train = None
        self.y_train = None
        self.x_inv = None
        self.y_inv = None
        self.nn_orig = None
        self.nn_inv = None
        
    def fit(self, X_train, y_train):
        """Fit the inverse density selector."""
        self.X_train = X_train
        self.y_train = y_train
        
        # Create inverse density resampled data
        X_df = pd.DataFrame(X_train)
        y_df = pd.Series(y_train)
        
        # Create inverse density data for context selection
        y_binned = pd.qcut(y_df, q=self.bins, duplicates='drop')
        counts = y_binned.value_counts()
        inverse_weights = 1.0 / counts[y_binned].values
        probs = inverse_weights / inverse_weights.sum()
        
        idx = np.arange(len(y_df))
        sampled_idx = np.random.choice(idx, size=len(y_df), replace=True, p=probs)
        
        self.x_inv = X_df.iloc[sampled_idx].reset_index(drop=True).values
        self.y_inv = y_df.iloc[sampled_idx].reset_index(drop=True).values
        
        # Fit nearest neighbors for both original and inverse data
        self.nn_orig = NearestNeighbors(n_neighbors=self.k_original, metric='cosine')
        self.nn_inv = NearestNeighbors(n_neighbors=self.k_inverse, metric='cosine')
        self.nn_orig.fit(self.X_train)
        self.nn_inv.fit(self.x_inv)
        
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context using both original and inverse density neighbors."""
        if context_size is None:
            context_size = self.k_original + self.k_inverse
            
        # Get neighbors from both datasets
        idx_orig = self.nn_orig.kneighbors(x_query.reshape(1, -1), return_distance=False).flatten()
        idx_inv = self.nn_inv.kneighbors(x_query.reshape(1, -1), return_distance=False).flatten()
        
        # Combine context
        x_context = np.vstack([self.X_train[idx_orig], self.x_inv[idx_inv]])
        y_context = np.concatenate([self.y_train[idx_orig], self.y_inv[idx_inv]])
        
        return x_context, y_context

class ClusteringContextSelector:
    """Context selection based on clustering."""
    
    def __init__(self, n_clusters=10, n_neighbors=30):
        self.n_clusters = n_clusters
        self.n_neighbors = n_neighbors
        self.kmeans = None
        self.X_train = None
        self.y_train = None
        self.cluster_centers = None
        
    def fit(self, X_train, y_train):
        """Fit the clustering selector."""
        self.X_train = X_train
        self.y_train = y_train
        
        # Fit K-means clustering
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        cluster_labels = self.kmeans.fit_predict(X_train)
        self.cluster_centers = self.kmeans.cluster_centers_
        
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context using clustering-based selection."""
        if context_size is None:
            context_size = self.n_neighbors
            
        # Find the closest cluster center
        distances_to_centers = np.linalg.norm(self.cluster_centers - x_query, axis=1)
        closest_cluster = np.argmin(distances_to_centers)
        
        # Get samples from the closest cluster
        cluster_labels = self.kmeans.predict(self.X_train)
        cluster_samples = self.X_train[cluster_labels == closest_cluster]
        cluster_targets = self.y_train[cluster_labels == closest_cluster]
        
        # If not enough samples in cluster, add random samples
        if len(cluster_samples) < context_size:
            remaining_size = context_size - len(cluster_samples)
            other_indices = np.where(cluster_labels != closest_cluster)[0]
            if len(other_indices) > 0:
                random_indices = np.random.choice(other_indices, size=min(remaining_size, len(other_indices)), replace=False)
                cluster_samples = np.vstack([cluster_samples, self.X_train[random_indices]])
                cluster_targets = np.concatenate([cluster_targets, self.y_train[random_indices]])
        
        # Return top context_size samples
        return cluster_samples[:context_size], cluster_targets[:context_size]

class RandomContextSelector:
    """Random context selection for baseline comparison."""
    
    def __init__(self, context_size=30):
        self.context_size = context_size
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """Fit the random selector."""
        self.X_train = X_train
        self.y_train = y_train
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context randomly."""
        if context_size is None:
            context_size = self.context_size
            
        # Randomly sample context_size samples
        indices = np.random.choice(len(self.X_train), size=min(context_size, len(self.X_train)), replace=False)
        return self.X_train[indices], self.y_train[indices]

class HybridContextSelector:
    """Hybrid context selection combining multiple strategies."""
    
    def __init__(self, selectors, weights=None):
        self.selectors = selectors
        self.weights = weights if weights is not None else [1.0] * len(selectors)
        self.X_train = None
        self.y_train = None
        
    def fit(self, X_train, y_train):
        """Fit all selectors."""
        self.X_train = X_train
        self.y_train = y_train
        
        for selector in self.selectors:
            selector.fit(X_train, y_train)
        return self
        
    def select_context(self, x_query, context_size=None):
        """Select context using weighted combination of multiple strategies."""
        if context_size is None:
            context_size = 30
            
        all_x_context = []
        all_y_context = []
        
        for selector, weight in zip(self.selectors, self.weights):
            x_context, y_context = selector.select_context(x_query, context_size)
            # Repeat samples based on weight
            repeat_count = max(1, int(weight * len(x_context)))
            all_x_context.extend([x_context] * repeat_count)
            all_y_context.extend([y_context] * repeat_count)
        
        # Combine and return unique samples
        combined_x = np.vstack(all_x_context)
        combined_y = np.concatenate(all_y_context)
        
        # Remove duplicates while preserving order
        unique_indices = np.unique(combined_x, axis=0, return_index=True)[1]
        unique_indices = np.sort(unique_indices)
        
        return combined_x[unique_indices][:context_size], combined_y[unique_indices][:context_size]

# Context selection factory
def create_context_selector(method='nearest_neighbor', **kwargs):
    """
    Factory function to create context selectors.
    
    Parameters:
    -----------
    method : str
        Context selection method:
        - 'nearest_neighbor': Simple nearest neighbor selection
        - 'inverse_density': Inverse density sampling
        - 'clustering': Clustering-based selection
        - 'random': Random selection (baseline)
        - 'neural': Neural network-based selection
        - 'hybrid': Combination of multiple methods
        - 'adaptive_tabpfn': Adaptive context selector for TabPFN
        - 'adaptive_gpt2': Adaptive context selector for GPT2
    **kwargs : dict
        Additional parameters for the selector
        
    Returns:
    --------
    ContextSelector object
    """
    if method == 'nearest_neighbor':
        return NearestNeighborContextSelector(**kwargs)
    elif method == 'inverse_density':
        return InverseDensityContextSelector(**kwargs)
    elif method == 'clustering':
        return ClusteringContextSelector(**kwargs)
    elif method == 'random':
        return RandomContextSelector(**kwargs)
    elif method == 'neural':
        return ContextSelector(**kwargs)
    elif method == 'hybrid':
        # Create hybrid with multiple selectors
        selectors = [
            NearestNeighborContextSelector(n_neighbors=20),
            InverseDensityContextSelector(k_original=15, k_inverse=15),
            ClusteringContextSelector(n_clusters=5, n_neighbors=10)
        ]
        weights = [0.4, 0.4, 0.2]  # Weight for each selector
        return HybridContextSelector(selectors, weights)
    elif method == 'adaptive_tabpfn':
        return AdaptiveTabPFNContextSelector(**kwargs)
    elif method == 'adaptive_gpt2':
        return AdaptiveGPT2ContextSelector(**kwargs)
    else:
        raise ValueError(f"Unknown context selection method: {method}")

# Utility functions for context selection
def evaluate_context_quality(x_context, y_context, x_query, y_true):

    # Calculate diversity (average pairwise distance)
    if len(x_context) > 1:
        distances = np.linalg.norm(x_context[:, None, :] - x_context[None, :, :], axis=2)
        diversity = np.mean(distances[np.triu_indices_from(distances, k=1)])
    else:
        diversity = 0.0
    
    # Calculate relevance (distance to query)
    query_distances = np.linalg.norm(x_context - x_query, axis=1)
    relevance = np.mean(query_distances)
    
    # Calculate target similarity
    target_similarity = np.mean(np.abs(y_context - y_true))
    
    return {
        'diversity': diversity,
        'relevance': relevance,
        'target_similarity': target_similarity,
        'context_size': len(x_context)
    }

def get_context_selection_config():
    """Get default configuration for context selection."""
    return {
        'nearest_neighbor': {
            'n_neighbors': 30,
            'metric': 'cosine'
        },
        'inverse_density': {
            'bins': 40,
            'k_original': 30,
            'k_inverse': 30
        },
        'clustering': {
            'n_clusters': 10,
            'n_neighbors': 30
        },
        'random': {
            'context_size': 30
        },
        'neural': {
            'hidden_dim': 64
        },
        'hybrid': {
            'weights': [0.4, 0.4, 0.2]
        },
        'adaptive_tabpfn': {
            'k_original': 30,
            'k_inverse': 30,
            'bins': 40,
            'adaptive_context': True
        },
        'adaptive_gpt2': {
            'n_neighbors': 50,
            'aug_factor': 0.5,
            'adaptive_context': True
        }
    }
