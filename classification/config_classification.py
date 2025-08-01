
import torch

# ============================================================================
# BASIC CONFIGURATION
# ============================================================================

OUTPUT_DIR = "/vol/miltank/projects/practical_sose25/in_context_learning/classification/results"

RANDOM_STATE = 42
TARGET_COL = "target"
DATA_FLAGS = ["classification", "cancer", "rboth"]

# ============================================================================
# FEATURE ENGINEERING - IMPROVED FOR LINEAR MODELS
# ============================================================================

FEATURE_ENGINEERING_PARAMS = {
    "impute_strategy_numeric": "median",
    "impute_strategy_categorical": "most_frequent",
    "scale_method": "robust",  # Changed back to robust scaling for better outlier handling
    "remove_outliers": True,    # Enable outlier removal for better regularization
    "outlier_method": "iqr",    
    "outlier_threshold": 2.5,   # More aggressive threshold for regularization
    "verbose": False
}

# ============================================================================
# ADDITIONAL REGULARIZATION SETTINGS
# ============================================================================

# Cross-validation settings for better regularization assessment
REGULARIZATION_CV_SETTINGS = {
    "use_shuffle_split": True,  # Use shuffle split for better regularization assessment
    "test_size": 0.2,           # 20% test size
    "n_splits": 5,              # 5-fold CV
    "random_state": RANDOM_STATE
}

# Early stopping settings for gradient boosting models
EARLY_STOPPING_SETTINGS = {
    "patience": 10,             # Number of iterations without improvement
    "min_delta": 0.001,         # Minimum improvement threshold
    "restore_best_weights": True # Restore best weights after early stopping
}

# Ensemble regularization settings
ENSEMBLE_REGULARIZATION = {
    "use_bagging": True,        # Enable bagging for ensemble regularization
    "bagging_fraction": 0.8,    # Fraction of samples for bagging
    "bagging_freq": 5,          # Bagging frequency
    "feature_fraction": 0.8,    # Feature fraction for regularization
    "drop_rate": 0.1            # Drop rate for regularization
}

# Data validation settings to prevent numerical issues
DATA_VALIDATION_SETTINGS = {
    "check_nan": True,          # Check for NaN values
    "check_inf": True,          # Check for infinite values
    "check_extreme": True,      # Check for extreme values (> 1e10)
    "check_constant": True,     # Check for constant features
    "max_abs_value": 1e6,       # Maximum absolute value threshold
    "min_std": 1e-8,           # Minimum standard deviation threshold
    "remove_outliers": True,    # Remove outliers during validation
    "outlier_threshold": 3.0    # Outlier threshold (standard deviations)
}

# ============================================================================
# ORGAN PCA FEATURE SELECTION
# ============================================================================

# Organ PCA settings - replaces traditional feature selection
ORGAN_PCA_SETTINGS = {
    "n_components_per_organ": 10,  # 10 PCA components per organ
    "random_state": RANDOM_STATE,
    "verbose": True
}

# Example: 12 organs * 10 components = 120 total features

# Legacy settings (kept for compatibility but not used)
N_FEATURES = 500  # Not used with organ PCA
TASK = "classification"

# ============================================================================
# MODELS - ENHANCED REGULARIZATION PARAMETERS
# ============================================================================

MODELS_TO_TEST = [
    "LogisticRegression", "RandomForest", "DecisionTree", "HistGradientBoosting",
    "MLP", "KNeighbors", "NaiveBayes", "XGBoost", "LightGBM", "TabPFNICL", "GPT2ICL"
]

# Models that need hyperparameter tuning - ENHANCED REGULARIZATION
MODELS_WITH_HYPERPARAMETERS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"]
    },
    "RandomForest": {
        "n_estimators": [50, 100],  # Further reduced for speed
        "max_depth": [10, None],  # Reduced options
        "min_samples_split": [2, 5],  # Reduced options
        "min_samples_leaf": [1, 2],  # Reduced options
        "max_features": ["sqrt", "log2"],  # Reduced options
        "bootstrap": [True],  # Default
        "oob_score": [False]  # Disable OOB scoring for speed
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)],  # Reduced options
        "alpha": [0.001, 0.01],  # Reduced range
        "learning_rate_init": [0.001, 0.01],  # Reduced range
        "early_stopping": [True],  # Enable early stopping
        "validation_fraction": [0.1]  # Standard validation fraction
    },
    "HistGradientBoosting": {
        "max_iter": [100, 200],  # Reduced range
        "learning_rate": [0.05, 0.1],  # Reduced range
        "max_depth": [5, 10],  # Reduced range
        "min_samples_leaf": [10, 20],  # Reduced range
        "l2_regularization": [0.1, 0.5],  # Reduced range
        "max_bins": [128],  # Single option
        "early_stopping": ["auto"]  # Enable early stopping
    },
    "XGBoost": {
        "n_estimators": [100, 200],  # Reduced range
        "max_depth": [5, 7],  # Reduced range
        "learning_rate": [0.05, 0.1],  # Reduced range
        "subsample": [0.8, 0.9],  # Reduced range
        "colsample_bytree": [0.8, 0.9],  # Reduced range
        "reg_alpha": [0.1, 0.5],  # Reduced range
        "reg_lambda": [0.1, 0.5],  # Reduced range
        "min_child_weight": [1, 3]  # Reduced range
    },
    "LightGBM": {
        "n_estimators": [100, 200],  # Reduced range
        "max_depth": [5, 7],  # Reduced range
        "learning_rate": [0.05, 0.1],  # Reduced range
        "num_leaves": [25, 31],  # Reduced range
        "subsample": [0.8, 0.9],  # Reduced range
        "colsample_bytree": [0.8, 0.9],  # Reduced range
        "reg_alpha": [0.1, 0.5],  # Reduced range
        "reg_lambda": [0.1, 0.5],  # Reduced range
        "min_child_samples": [15, 20]  # Reduced range
    },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7, 9, 11, 15],  # Standard range
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # Standard distance metrics
    },
    # ICL MODELS - ADDED TO HYPERPARAMETER TUNING
    "TabPFNICL": {
        "k_original": [20, 30, 40],
        "k_inverse": [20, 30, 40],
        "bins": [30, 40, 50],
        "adaptive_context": [True, False]
    },
    "GPT2ICL": {
        "n_neighbors": [30, 50, 70],
        "aug_factor": [0.3, 0.5, 0.7],
        "adaptive_context": [True, False]
    }
}

# Hyperparameter spaces for Optuna optimization - ENHANCED REGULARIZATION
HYPERPARAMETER_SPACES = {
    "LogisticRegression": {
        "C": ["loguniform", 0.01, 100.0],
        "penalty": ["categorical", ["l1", "l2"]],
        "solver": ["categorical", ["liblinear", "saga"]]
    },
    "RandomForest": {
        "n_estimators": ["int", 50, 150],  # Increased range
        "max_depth": ["int", 5, 15],  # Increased range
        "min_samples_split": ["int", 2, 10],  # Allow default (2)
        "min_samples_leaf": ["int", 1, 4],  # Allow default (1)
        "max_features": ["categorical", ["sqrt", "log2", 0.3, 0.5]]  # More options
    },
    "MLP": {
        "hidden_layer_sizes": ["categorical", [(50,), (100,), (200,), (50, 25), (100, 50), (200, 100)]],  # More options
        "alpha": ["loguniform", 0.0001, 0.1],  # Standard L2 regularization range
        "learning_rate_init": ["loguniform", 0.001, 0.1],  # Standard range
        "max_iter": ["int", 500, 1000],  # More iterations
        "early_stopping": ["categorical", [True]],  # Enable early stopping
        "validation_fraction": ["uniform", 0.1, 0.2]  # Validation fraction
    },
    "HistGradientBoosting": {
        "max_iter": ["int", 100, 300],  # Standard range
        "learning_rate": ["loguniform", 0.01, 0.2],  # Standard range
        "max_depth": ["int", 3, 15],  # Standard range
        "min_samples_leaf": ["int", 5, 20],  # Standard range
        "l2_regularization": ["loguniform", 0.01, 1.0],  # Standard range
        "max_bins": ["int", 128, 256]  # Standard range
    },
    "XGBoost": {
        "n_estimators": ["int", 100, 300],  # Standard range
        "max_depth": ["int", 3, 10],  # Standard range
        "learning_rate": ["loguniform", 0.01, 0.2],  # Standard range
        "subsample": ["uniform", 0.7, 1.0],  # Standard range
        "colsample_bytree": ["uniform", 0.7, 1.0],  # Standard range
        "reg_alpha": ["loguniform", 0.01, 1.0],  # Standard L1 regularization
        "reg_lambda": ["loguniform", 0.01, 1.0],  # Standard L2 regularization
        "min_child_weight": ["int", 1, 5]  # Standard range
    },
    "LightGBM": {
        "n_estimators": ["int", 100, 300],  # Standard range
        "max_depth": ["int", 3, 10],  # Standard range
        "learning_rate": ["loguniform", 0.01, 0.2],  # Standard range
        "num_leaves": ["int", 15, 50],  # Standard range
        "subsample": ["uniform", 0.7, 1.0],  # Standard range
        "colsample_bytree": ["uniform", 0.7, 1.0],  # Standard range
        "reg_alpha": ["loguniform", 0.01, 1.0],  # Standard L1 regularization
        "reg_lambda": ["loguniform", 0.01, 1.0],  # Standard L2 regularization
        "min_child_samples": ["int", 10, 30]  # Standard range
    },
    "KNeighbors": {
        "n_neighbors": ["int", 3, 15],  # Standard range
        "weights": ["categorical", ["uniform", "distance"]],
        "p": ["int", 1, 2]  # Standard distance metrics
    },
    # ICL MODELS - ADDED TO OPTUNA SPACES
    "TabPFNICL": {
        "k_original": ["int", 20, 40],
        "k_inverse": ["int", 20, 40],
        "bins": ["int", 30, 50],
        "adaptive_context": ["categorical", [True, False]]
    },
    "GPT2ICL": {
        "n_neighbors": ["int", 30, 70],
        "aug_factor": ["uniform", 0.3, 0.7],
        "adaptive_context": ["categorical", [True, False]]
    }
}

# Models that don't need hyperparameter tuning (simple models)
MODELS_WITHOUT_HYPERPARAMETERS = [
    "DecisionTree",     # Can use default parameters
    "NaiveBayes",      # Can use default parameters
    "TabPFNICL",       # ICL models use default parameters
    "GPT2ICL"          # ICL models use default parameters
]

# ============================================================================
# ICL SETTINGS - IMPROVED
# ============================================================================

GPT2ICL_CONTEXT_MODE = "adaptive_nearest_neighbor"
GPT2ICL_CONTEXT_SIZE = 50  # Increased context size
TABPFN_CONTEXT_MODE = "adaptive_inverse_density"
TABPFN_CONTEXT_SIZE = 50  # Increased context size

# ICL-specific parameters
ICL_PARAMS = {
    "TabPFNICL": {
        "k_original": [20, 30, 40],
        "k_inverse": [20, 30, 40],
        "bins": [30, 40, 50],
        "adaptive_context": [True, False]
    },
    "GPT2ICL": {
        "n_neighbors": [30, 50, 70],
        "aug_factor": [0.3, 0.5, 0.7],
        "adaptive_context": [True, False]
    }
}

# ============================================================================
# CROSS-VALIDATION SETTINGS
# ============================================================================

CV_FOLDS = 5
CV_STRATEGY = "stratified"  # Use stratified k-fold for classification
HYPERPARAMETER_TUNING = True
TUNING_METRIC = "f1_weighted"  # Optimize for F1 score
USE_NESTED_CV = True  # Enable proper nested cross-validation
INNER_CV_FOLDS = 3  # Reduced for faster completion

# ============================================================================
# OPTUNA SETTINGS
# ============================================================================

OPTUNA_N_TRIALS = 5  #reduced for faster completion
OPTUNA_N_TRIALS_ICL = 3  # Reduced for ICL models because of computational cost
OPTUNA_N_TRIALS_RF = 3   # Reduced for Random Forest due to computational cost
OPTUNA_TIMEOUT = 120  # Reduced timeout for faster completion
OPTUNA_N_JOBS = 1  # Use single job for nested CV compatibility

# ============================================================================
# ENSEMBLE SETTINGS
# ============================================================================

ENSEMBLE_METHODS = ["hybrid"]
TOP_MODELS_FOR_ENSEMBLE = 3

# ============================================================================
# CONFIG CLASS
# ============================================================================

class Config:
    def __init__(self):
        # Basic settings
        self.RANDOM_STATE = RANDOM_STATE
        self.TARGET_COL = TARGET_COL
        self.DATA_FLAGS = DATA_FLAGS
        self.OUTPUT_DIR = OUTPUT_DIR
        
        # Feature settings
        self.N_FEATURES = N_FEATURES
        self.FEATURE_ENGINEERING_PARAMS = FEATURE_ENGINEERING_PARAMS
        self.ORGAN_PCA_SETTINGS = ORGAN_PCA_SETTINGS  # New organ PCA settings
        
        # Feature selection specific settings
        self.TASK = TASK
        
        # Model settings
        self.MODELS_TO_TEST = MODELS_TO_TEST
        self.MODELS_WITH_HYPERPARAMETERS = MODELS_WITH_HYPERPARAMETERS
        self.MODELS_WITHOUT_HYPERPARAMETERS = MODELS_WITHOUT_HYPERPARAMETERS
        self.HYPERPARAMETER_SPACES = HYPERPARAMETER_SPACES
        self.ICL_PARAMS = ICL_PARAMS
        
        # CV settings
        self.CV_FOLDS = CV_FOLDS
        self.CV_STRATEGY = CV_STRATEGY
        self.HYPERPARAMETER_TUNING = HYPERPARAMETER_TUNING
        self.TUNING_METRIC = TUNING_METRIC
        self.USE_NESTED_CV = USE_NESTED_CV
        self.INNER_CV_FOLDS = INNER_CV_FOLDS
        
        # Optuna settings
        self.OPTUNA_N_TRIALS = OPTUNA_N_TRIALS
        self.OPTUNA_N_TRIALS_ICL = OPTUNA_N_TRIALS_ICL
        self.OPTUNA_N_TRIALS_RF = OPTUNA_N_TRIALS_RF
        self.OPTUNA_TIMEOUT = OPTUNA_TIMEOUT
        self.OPTUNA_N_JOBS = OPTUNA_N_JOBS
        
        # Ensemble settings
        self.ENSEMBLE_METHODS = ENSEMBLE_METHODS
        self.TOP_MODELS_FOR_ENSEMBLE = TOP_MODELS_FOR_ENSEMBLE
        
        # ICL settings
        self.GPT2ICL_CONTEXT_MODE = GPT2ICL_CONTEXT_MODE
        self.GPT2ICL_CONTEXT_SIZE = GPT2ICL_CONTEXT_SIZE
        self.TABPFN_CONTEXT_MODE = TABPFN_CONTEXT_MODE
        self.TABPFN_CONTEXT_SIZE = TABPFN_CONTEXT_SIZE
        
        # Regularization settings
        self.REGULARIZATION_CV_SETTINGS = REGULARIZATION_CV_SETTINGS
        self.EARLY_STOPPING_SETTINGS = EARLY_STOPPING_SETTINGS
        self.ENSEMBLE_REGULARIZATION = ENSEMBLE_REGULARIZATION
        self.DATA_VALIDATION_SETTINGS = DATA_VALIDATION_SETTINGS

config = Config()