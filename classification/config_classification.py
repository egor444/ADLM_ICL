
import torch

# ============================================================================
# BASIC CONFIGURATION
# ============================================================================

OUTPUT_DIR = "/vol/miltank/projects/practical_sose25/in_context_learning/classification/results"

RANDOM_STATE = 42
TARGET_COL = "target"
DATA_FLAGS = ["classification", "cancer", "rboth"]

# ============================================================================
# FEATURE ENGINEERING 
# ============================================================================

FEATURE_ENGINEERING_PARAMS = {
    "impute_strategy_numeric": "median",
    "impute_strategy_categorical": "most_frequent",
    "scale_method": "robust",  
    "remove_outliers": True,   
    "outlier_method": "iqr",    
    "outlier_threshold": 2.5,   
    "verbose": False
}

# ============================================================================
# ADDITIONAL REGULARIZATION SETTINGS
# ============================================================================

# Cross-validation settings 
REGULARIZATION_CV_SETTINGS = {
    "use_shuffle_split": True,  # Use shuffle split for better regularization assessment
    "test_size": 0.2,           # 20% test size
    "n_splits": 5,              # 5-fold CV
    "random_state": RANDOM_STATE
}

# Early stopping settings for gradient boosting models
EARLY_STOPPING_SETTINGS = {
    "patience": 10,            
    "min_delta": 0.001,        
    "restore_best_weights": True 
}

# Ensemble regularization settings
ENSEMBLE_REGULARIZATION = {
    "use_bagging": True,       
    "bagging_fraction": 0.8,   
    "bagging_freq": 5,         
    "feature_fraction": 0.8,    
    "drop_rate": 0.1           
}

# Data validation settings to prevent numerical issues
DATA_VALIDATION_SETTINGS = {
    "check_nan": True,         
    "check_inf": True,         
    "check_extreme": True,     
    "check_constant": True,    
    "max_abs_value": 1e6,       
    "min_std": 1e-8,          
    "remove_outliers": True,   
    "outlier_threshold": 3.0   
}

# ============================================================================
# ORGAN PCA FEATURE SELECTION
# ============================================================================

# Organ PCA settings
ORGAN_PCA_SETTINGS = {
    "n_components_per_organ": 10,  # 10 PCA components per organ
    "random_state": RANDOM_STATE,
    "verbose": True
}

# Example: 12 organs * 10 components = 120 total features

# Legacy settings (kept for compatibility but not used)
N_FEATURES = 500  # Not used with organ PCA, old for the unused feature selector
TASK = "classification"

# ============================================================================
# MODELS 
# ============================================================================

MODELS_TO_TEST = [
    "LogisticRegression", "RandomForest", "DecisionTree", "HistGradientBoosting",
    "MLP", "KNeighbors", "NaiveBayes", "XGBoost", "LightGBM", "TabPFNICL", "GPT2ICL"
]

#Models that need hyperparameter tuning
MODELS_WITH_HYPERPARAMETERS = {
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear", "saga"]
    },
    "RandomForest": {
        "n_estimators": [50, 100],
        "max_depth": [10, None],  
        "min_samples_split": [2, 5], 
        "min_samples_leaf": [1, 2], 
        "max_features": ["sqrt", "log2"], 
        "bootstrap": [True], 
        "oob_score": [False] 
    },
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,)], 
        "alpha": [0.001, 0.01],  
        "learning_rate_init": [0.001, 0.01],  
        "early_stopping": [True],  
        "validation_fraction": [0.1]  
    },
    "HistGradientBoosting": {
        "max_iter": [100, 200], 
        "learning_rate": [0.05, 0.1], 
        "max_depth": [5, 10], 
        "min_samples_leaf": [10, 20], 
        "l2_regularization": [0.1, 0.5], 
        "max_bins": [128],  
        "early_stopping": ["auto"]  
    },
    "XGBoost": {
        "n_estimators": [100, 200],  
        "max_depth": [5, 7],
        "learning_rate": [0.05, 0.1], 
        "subsample": [0.8, 0.9],  
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0.1, 0.5], 
        "reg_lambda": [0.1, 0.5], 
        "min_child_weight": [1, 3] 
    },
    "LightGBM": {
        "n_estimators": [100, 200], 
        "max_depth": [5, 7], 
        "learning_rate": [0.05, 0.1], 
        "num_leaves": [25, 31],  
        "subsample": [0.8, 0.9],  
        "colsample_bytree": [0.8, 0.9],
        "reg_alpha": [0.1, 0.5], 
        "reg_lambda": [0.1, 0.5], 
        "min_child_samples": [15, 20] 
    },
    "KNeighbors": {
        "n_neighbors": [3, 5, 7, 9, 11, 15], 
        "weights": ["uniform", "distance"],
        "p": [1, 2]  
    },
    # ICL MODELS 
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

# Hyperparameter spaces for Optuna optimization 
HYPERPARAMETER_SPACES = {
    "LogisticRegression": {
        "C": ["loguniform", 0.01, 100.0],
        "penalty": ["categorical", ["l1", "l2"]],
        "solver": ["categorical", ["liblinear", "saga"]]
    },
    "RandomForest": {
        "n_estimators": ["int", 50, 150], 
        "max_depth": ["int", 5, 15], 
        "min_samples_split": ["int", 2, 10],  
        "min_samples_leaf": ["int", 1, 4], 
        "max_features": ["categorical", ["sqrt", "log2", 0.3, 0.5]] 
    },
    "MLP": {
        "hidden_layer_sizes": ["categorical", [(50,), (100,), (200,), (50, 25), (100, 50), (200, 100)]],
        "alpha": ["loguniform", 0.0001, 0.1], 
        "learning_rate_init": ["loguniform", 0.001, 0.1], 
        "max_iter": ["int", 500, 1000], 
        "early_stopping": ["categorical", [True]],  
        "validation_fraction": ["uniform", 0.1, 0.2]  
    },
    "HistGradientBoosting": {
        "max_iter": ["int", 100, 300], 
        "learning_rate": ["loguniform", 0.01, 0.2],
        "max_depth": ["int", 3, 15], 
        "min_samples_leaf": ["int", 5, 20], 
        "l2_regularization": ["loguniform", 0.01, 1.0], 
        "max_bins": ["int", 128, 256] 
    },
    "XGBoost": {
        "n_estimators": ["int", 100, 300], 
        "max_depth": ["int", 3, 10],  
        "learning_rate": ["loguniform", 0.01, 0.2],  
        "subsample": ["uniform", 0.7, 1.0], 
        "colsample_bytree": ["uniform", 0.7, 1.0], 
        "reg_alpha": ["loguniform", 0.01, 1.0],  
        "reg_lambda": ["loguniform", 0.01, 1.0], 
        "min_child_weight": ["int", 1, 5] 
    },
    "LightGBM": {
        "n_estimators": ["int", 100, 300],  
        "max_depth": ["int", 3, 10], 
        "learning_rate": ["loguniform", 0.01, 0.2], 
        "num_leaves": ["int", 15, 50], 
        "subsample": ["uniform", 0.7, 1.0],  
        "colsample_bytree": ["uniform", 0.7, 1.0],  
        "reg_alpha": ["loguniform", 0.01, 1.0], 
        "reg_lambda": ["loguniform", 0.01, 1.0],  
        "min_child_samples": ["int", 10, 30] 
    },
    "KNeighbors": {
        "n_neighbors": ["int", 3, 15],  
        "weights": ["categorical", ["uniform", "distance"]],
        "p": ["int", 1, 2] 
    },
    # ICL MODELS
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
    "DecisionTree",    
    "NaiveBayes",      
    "TabPFNICL",       
    "GPT2ICL"         
]

# ============================================================================
# ICL SETTINGS 
# ============================================================================

GPT2ICL_CONTEXT_MODE = "adaptive_nearest_neighbor"
GPT2ICL_CONTEXT_SIZE = 50  
TABPFN_CONTEXT_MODE = "adaptive_inverse_density"
TABPFN_CONTEXT_SIZE = 50 

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
CV_STRATEGY = "stratified"  
HYPERPARAMETER_TUNING = True
TUNING_METRIC = "f1_weighted" 
USE_NESTED_CV = True  
INNER_CV_FOLDS = 3  

# ============================================================================
# OPTUNA SETTINGS
# ============================================================================

OPTUNA_N_TRIALS = 5  
OPTUNA_N_TRIALS_ICL = 3  
OPTUNA_N_TRIALS_RF = 3   
OPTUNA_TIMEOUT = 120  
OPTUNA_N_JOBS = 1  

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