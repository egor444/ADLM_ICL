
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import os
import json
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.base import BaseEstimator, TransformerMixin

warnings.filterwarnings('ignore')


class OrganPCASelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components_per_organ=10, random_state=42, verbose=True, organ_indicators=None):
        self.n_components_per_organ = n_components_per_organ
        self.random_state = random_state
        self.verbose = verbose

        #  organ mapping based on csv
        if organ_indicators is None:
            self.organ_indicators = {
                'spleen': ['spleen'],
                'kidney': ['kidney_right', 'kidney_left', 'kidney'],
                'digestive': ['gallbladder', 'stomach', 'esophagus', 'intestine', 'duodenum'],
                'liver': ['liver'],
                'pancreas': ['pancreas'],
                'endocrine': ['adrenal_gland_right', 'adrenal_gland_left', 'thyroid_gland'],
                'respiratory': ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right', 
                               'lung_middle_lobe_right', 'lung_lower_lobe_right', 'trachea'],
                'urinary_bladder': ['urinary_bladder'],
                'prostate': ['prostate'],
                'bone': ['sacrum', 'humerus_left', 'humerus_right', 'scapula_left', 'scapula_right', 
                        'clavicula_left', 'clavicula_right', 'femur_left', 'femur_right', 'hip_left', 
                        'hip_right', 'sternum', 'costal_cartilages', 'bone_other'],
                'heart': ['heart'],
                'vascular': ['aorta', 'pulmonary_vein', 'brachiocephalic_trunk', 'subclavian_artery_right',
                            'subclavian_artery_left', 'common_carotid_artery_right', 'common_carotid_artery_left',
                            'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'atrial_appendage_left',
                            'superior_vena_cava', 'inferior_vena_cava', 'portal_vein_and_splenic_vein',
                            'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right'],
                'spine': ['spinal_cord', 'IVD', 'vertebra_body', 'vertebra_posterior_elements', 'spinal_channel'],
                'muscle': ['gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right',
                          'gluteus_minimus_left', 'gluteus_minimus_right', 'autochthon_left', 'autochthon_right',
                          'iliopsoas_left', 'iliopsoas_right', 'muscle'],
                'fat': ['subcutaneous_fat', 'inner_fat'],
                'unused': ['unused']
            }
        elif isinstance(organ_indicators, dict):
            self.organ_indicators = organ_indicators
        else:
            self.organ_indicators = organ_indicators
        
        # Initialize attributes
        self.organ_groups = {}
        self.pca_models = {}
        self.scalers = {}
        self.feature_names_ = None
        
        # Setup logging
        self.logger = logging.getLogger('OrganPCASelector')
        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[OrganPCASelector] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def _log(self, msg):
        #Log message if verbose is enabled
        if self.verbose:
            self.logger.info(msg)
    
    def _extract_organ_name(self, feature_name):
        #Extract organ name from feature name using the above mapping
        feat_low = feature_name.lower()
        
        # If organ_indicators is a dict, use pattern matching
        if isinstance(self.organ_indicators, dict):
            for organ, patterns in self.organ_indicators.items():
                if isinstance(patterns, str):
                    patterns = [patterns]
                for pattern in patterns:
                    if pattern.lower() in feat_low:
                        return organ
        
        # If organ_indicators is a list, use simple substring matching
        elif isinstance(self.organ_indicators, list):
            for organ in self.organ_indicators:
                if organ.lower() in feat_low:
                    return organ
        
        # If no organ indicator found, try to extract from underscore pattern
        parts = feature_name.split('_')
        if len(parts) > 1:
            # Look for organ-like patterns in the first few parts
            for part in parts[:3]:  # Check first 3 parts
                if len(part) > 2 and not part.isdigit():  # Avoid numbers and very short parts
                    return part
        
        return 'unknown'
    
    def _group_features_by_organ(self, feature_names):
        #Group features by organ name.
        self._log("Grouping features by organ...")
        
        organ_groups = defaultdict(list)
        
        for feature in feature_names:
            organ = self._extract_organ_name(feature)
            organ_groups[organ].append(feature)
        
        # Log organ grouping results
        for organ, features in organ_groups.items():
            self._log(f"Organ '{organ}': {len(features)} features")
        
        return dict(organ_groups)
    
    def fit(self, X, y=None):

        #Fit the organ PCA selector.
        self._log(f"Fitting Organ PCA Selector with {self.n_components_per_organ} components per organ")
        self._log(f"Input shape: {X.shape}")
        
        # Group features by organ
        self.organ_groups = self._group_features_by_organ(X.columns)
        
        # Apply PCA to each organ group
        pca_results = {}
        self.pca_models = {}
        self.scalers = {}
        
        for organ, features in self.organ_groups.items():
            # Get features that exist in the input data
            available_features = [f for f in features if f in X.columns]
            
            if len(available_features) == 0:
                self._log(f"Skipping organ '{organ}' - no features available")
                continue
            
            # Determine number of components for this organ
            if len(available_features) < self.n_components_per_organ:
                self._log(f"Warning: Organ '{organ}' has only {len(available_features)} features, "
                         f"but {self.n_components_per_organ} components requested. "
                         f"Using {len(available_features)} components.")
                n_components = len(available_features)
            else:
                n_components = self.n_components_per_organ
            
            # Extract organ features
            X_organ = X[available_features]
            
            # Scale the features
            scaler = StandardScaler()
            X_organ_scaled = scaler.fit_transform(X_organ)
            self.scalers[organ] = scaler
            
            # Apply PCA
            pca = PCA(n_components=n_components, random_state=self.random_state)
            X_pca = pca.fit_transform(X_organ_scaled)
            
            # Store PCA model
            self.pca_models[organ] = pca
            
            # Create column names for PCA components
            pca_columns = [f"{organ}_pca_{i+1}" for i in range(n_components)]
            
            # Store results
            pca_results[organ] = pd.DataFrame(X_pca, columns=pca_columns, index=X.index)
            
            # Log explained variance
            explained_variance = pca.explained_variance_ratio_
            total_variance = explained_variance.sum()
            self._log(f"Organ '{organ}': {len(available_features)} features -> {n_components} components "
                     f"(explained variance: {total_variance:.3f})")
            
            # Log individual component contributions
            for i, (col, var) in enumerate(zip(pca_columns, explained_variance)):
                self._log(f"  - {col}: {var:.3f} ({var/total_variance*100:.1f}%)")
        
        # Combine all PCA results to get feature names
        if pca_results:
            combined_pca = pd.concat(pca_results.values(), axis=1)
            self.feature_names_ = combined_pca.columns.tolist()
            self._log(f"Combined PCA result: {len(self.feature_names_)} total components from {len(pca_results)} organs")
        else:
            self._log("No PCA results generated")
            self.feature_names_ = []
        
        return self
    
    def transform(self, X):
        #Transform data using fitted organ PCA models.
        
        
        if not self.pca_models:
            raise ValueError("Call fit() before transform()")
        
        self._log("Transforming data using fitted organ PCA models")
        
        pca_results = {}
        
        for organ, pca_model in self.pca_models.items():
            # Get features that exist in the input data
            available_features = [f for f in self.organ_groups[organ] if f in X.columns]
            
            if not available_features:
                self._log(f"Warning: No features found for organ '{organ}' in transform data")
                continue
            
            # Extract organ features
            X_organ = X[available_features]
            
            # Scale using fitted scaler
            scaler = self.scalers[organ]
            X_organ_scaled = scaler.transform(X_organ)
            
            # Transform using fitted PCA
            X_pca = pca_model.transform(X_organ_scaled)
            
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
    
    def fit_transform(self, X, y=None):
        
        return self.fit(X, y).transform(X)
    
    def get_organ_summary(self):
        if not self.pca_models:
            return "No PCA models fitted yet"
        
        summary = {}
        
        for organ, pca_model in self.pca_models.items():
            explained_variance = pca_model.explained_variance_ratio_
            total_variance = explained_variance.sum()
            
            summary[organ] = {
                'n_components': len(explained_variance),
                'total_explained_variance': total_variance,
                'explained_variance_ratio': explained_variance.tolist(),
                'cumulative_variance': np.cumsum(explained_variance).tolist(),
                'n_original_features': len(self.organ_groups[organ])
            }
        
        return summary
    
    def save_summary(self, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save PCA summary
        pca_summary = self.get_organ_summary()
        if isinstance(pca_summary, dict):
            with open(output_path / "organ_pca_summary.json", 'w') as f:
                json.dump(pca_summary, f, indent=2)
        
        # Save organ groups info
        if self.organ_groups:
            with open(output_path / "organ_groups.json", 'w') as f:
                json.dump(self.organ_groups, f, indent=2)
        
        # Save organ mapping information
        if self.organ_indicators:
            with open(output_path / "organ_mapping.json", 'w') as f:
                json.dump(self.organ_indicators, f, indent=2)
        
        # Save feature names
        if self.feature_names_:
            with open(output_path / "pca_feature_names.json", 'w') as f:
                json.dump(self.feature_names_, f, indent=2)
        
        self._log(f"Organ PCA summary saved to {output_path}")
    
    def get_feature_names(self):
        return self.feature_names_ if self.feature_names_ else [] 