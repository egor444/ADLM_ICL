import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
)
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from data_extraction.data_manager import DataManager
import logging
import time
import signal


def get_model(trial, model_name, model_type, random_state, config=None, **kwargs):
    if config is None:
        if model_type == "regression":
            import sys
            import os
            # Add parent directory to path to find config module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                import config_regression as config
            except ImportError:
                # Try alternative path if the first one doesn't work
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                import config_regression as config
        else:
            import sys
            import os
            # Add parent directory to path to find classification module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from classification import config_classification as config
            except ImportError:
                # Try alternative path if the first one doesn't work
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from classification import config_classification as config
    
    try:
        if model_type == "regression":
            import sys
            import os
            # Add parent directory to path to find models module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from models.regression_models import get_regression_models
            except ImportError:
                # Try alternative path if the first one doesn't work
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from models.regression_models import get_regression_models
            
            # Get models for both trial=None and trial!=None cases
            models = get_regression_models(random_state=random_state)
            
            if trial is None:
                # Don't use MODEL_PARAMS - let scikit-learn defaults work
                if model_name not in models:
                    raise ValueError(f"Model {model_name} not found in get_regression_models.")
                if model_name == "TabPFNICL":
                    return models[model_name]
                # Return the model as-is without applying config parameters
                return models[model_name]
            
            search_space = config.HYPERPARAMETER_SPACES.get(model_name, {})
            if not search_space:
                logging.warning(f"[OptunaTuner] No hyperparameter space found for {model_name}")
                return float('-inf')
            
            params = {}
            for param, spec in search_space.items():
                try:
                    if spec[0] == "int":
                        params[param] = trial.suggest_int(param, spec[1], spec[2])
                    elif spec[0] == "loguniform":
                        params[param] = trial.suggest_float(param, spec[1], spec[2], log=True)
                    elif spec[0] == "uniform":
                        params[param] = trial.suggest_float(param, spec[1], spec[2])
                    elif spec[0] == "categorical":
                        params[param] = trial.suggest_categorical(param, spec[1])
                    else:
                        logging.warning(f"[OptunaTuner] Unknown parameter type {spec[0]} for {param}")
                        return float('-inf')
                except Exception as e:
                    logging.error(f"[OptunaTuner] Error suggesting parameter {param}: {e}")
                    return float('-inf')
            
            if "random_state" in getattr(config, 'MODEL_PARAMS', {}).get(model_name, {}):
                params["random_state"] = random_state
            
            # Create model directly with parameters instead of using get_regression_models
            try:
                if model_name == "Ridge":
                    from sklearn.linear_model import Ridge
                    return Ridge(**params)
                elif model_name == "Lasso":
                    from sklearn.linear_model import Lasso
                    return Lasso(**params)
                elif model_name == "ElasticNet":
                    from sklearn.linear_model import ElasticNet
                    return ElasticNet(**params)
                elif model_name == "RandomForest":
                    from sklearn.ensemble import RandomForestRegressor
                    return RandomForestRegressor(**params)

                elif model_name == "MLP":
                    from sklearn.neural_network import MLPRegressor
                    return MLPRegressor(**params)
                elif model_name == "HistGradientBoostingRegressor":
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    return HistGradientBoostingRegressor(**params)
                elif model_name == "XGBoost":
                    from xgboost import XGBRegressor
                    # Add specific validation for XGBoost parameters
                    if 'n_estimators' in params and params['n_estimators'] <= 0:
                        logging.warning(f"[OptunaTuner] Invalid n_estimators for XGBoost: {params['n_estimators']}")
                        return float('-inf')
                    if 'max_depth' in params and params['max_depth'] <= 0:
                        logging.warning(f"[OptunaTuner] Invalid max_depth for XGBoost: {params['max_depth']}")
                        return float('-inf')
                    if 'learning_rate' in params and (params['learning_rate'] <= 0 or params['learning_rate'] > 1):
                        logging.warning(f"[OptunaTuner] Invalid learning_rate for XGBoost: {params['learning_rate']}")
                        return float('-inf')
                    
                    # Add random_state for reproducibility
                    params['random_state'] = random_state
                    return XGBRegressor(**params)
                elif model_name == "LightGBM":
                    from lightgbm import LGBMRegressor
                    return LGBMRegressor(**params)
                elif model_name == "KNN":
                    from sklearn.neighbors import KNeighborsRegressor
                    return KNeighborsRegressor(**params)
                elif model_name == "TabPFNICL":
                    import sys
                    import os
                    # Add parent directory to path to find models module
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    try:
                        from models.regression_models import TabPFNICLWrapper
                    except ImportError:
                        # Try alternative path if the first one doesn't work
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from models.regression_models import TabPFNICLWrapper
                    return TabPFNICLWrapper(**params)
                elif model_name == "GPT2ICL":
                    import sys
                    import os
                    # Add parent directory to path to find models module
                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    try:
                        from models.regression_models import GPT2ICLRegressorWrapper
                    except ImportError:
                        # Try alternative path if the first one doesn't work
                        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                        from models.regression_models import GPT2ICLRegressorWrapper
                    return GPT2ICLRegressorWrapper(**params)
                else:
                    # Fallback to get_regression_models for other models
                    # models is already available from the import at the top
                    if model_name not in models:
                        raise ValueError(f"Model {model_name} not found in get_regression_models.")
                    return models[model_name]
            except Exception as e:
                logging.error(f"Error creating model {model_name} with params {params}: {e}")
                import traceback
                logging.error(f"Full traceback: {traceback.format_exc()}")
                raise
        else:
            import sys
            import os
            # Add parent directory to path to find classification module
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            try:
                from classification.models.classification_models import get_classification_models
            except ImportError:
                # Try alternative path if the first one doesn't work
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from classification.models.classification_models import get_classification_models
            if trial is None:
                # Don't use MODEL_PARAMS - let scikit-learn defaults work
                models = get_classification_models(random_state=random_state, model_names=[model_name], **kwargs)
                if model_name not in models:
                    raise ValueError(f"Model {model_name} not found in get_classification_models.")
                return models[model_name]
            
            search_space = config.HYPERPARAMETER_SPACES.get(model_name, {})
            params = {}
            for param, spec in search_space.items():
                if spec[0] == "int":
                    params[param] = trial.suggest_int(param, spec[1], spec[2])
                elif spec[0] == "loguniform":
                    params[param] = trial.suggest_float(param, spec[1], spec[2], log=True)
                elif spec[0] == "uniform":
                    params[param] = trial.suggest_float(param, spec[1], spec[2])
                elif spec[0] == "categorical":
                    params[param] = trial.suggest_categorical(param, spec[1])
            
            if "random_state" in getattr(config, 'MODEL_PARAMS', {}).get(model_name, {}):
                params["random_state"] = random_state
            
            models = get_classification_models(random_state=random_state, model_names=[model_name], params=params, **kwargs)
            if model_name == 'XGBoost':
                return XGBClassifier(**params)
            elif model_name == 'LightGBM':
                return LGBMClassifier(**params)
            elif model_name == 'RandomForest':
                return RandomForestClassifier(**params)

            elif model_name == 'MLP':
                return MLPClassifier(**params)
            elif model_name == 'HistGradientBoosting':
                return HistGradientBoostingClassifier(**params)
            elif model_name == 'KNeighbors':
                return KNeighborsClassifier(**params)
            elif model_name == 'NaiveBayes':
                return GaussianNB(**params)
            elif model_name == 'DecisionTree':
                return DecisionTreeClassifier(**params)
            elif model_name == 'LogisticRegression':
                return LogisticRegression(**params)
            elif model_name == 'TabPFNICL':
                return models[model_name]
            elif model_name == 'GPT2ICL':
                return models[model_name]
            else:
                return models[model_name]
    except Exception as e:
        logging.error(f"Error creating model {model_name}: {e}")
        import traceback
        logging.error(f"Full traceback: {traceback.format_exc()}")
        raise ValueError(f"Model '{model_name}' could not be created.")


def tune_model(model_name, model_type, X, y, config, n_trials=30, random_state=42, scoring=None, n_jobs=1):
    logger: logging.Logger = logging.getLogger('OptunaTuner')
    logger.info(f"[OptunaTuner] tune_model() called for model_name={model_name}, model_type={model_type}, n_trials={n_trials}")
    
    # Validate inputs
    if X is None or y is None:
        raise ValueError("X and y cannot be None")
    
    if len(X) == 0 or len(y) == 0:
        raise ValueError("X and y cannot be empty")
    
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")
    
    # Validate data types
    if not isinstance(X, (np.ndarray, pd.DataFrame)):
        raise ValueError("X must be numpy array or pandas DataFrame")
    
    if not isinstance(y, (np.ndarray, pd.Series)):
        raise ValueError("y must be numpy array or pandas Series")
    
    # Validate model name
    if model_name is None or model_name == "":
        raise ValueError("model_name cannot be None or empty")
    
    # Validate model name against known models
    known_models = ["LinearRegression", "Ridge", "Lasso", "ElasticNet", "RandomForest", "MLP", "HistGradientBoostingRegressor", "XGBoost", "LightGBM", "KNN", "DecisionTree", "TabPFNICL", "GPT2ICL", "LogisticRegression", "HistGradientBoosting", "XGBoost", "LightGBM", "KNeighbors", "NaiveBayes", "DecisionTree"]
    if model_name not in known_models:
        logger.warning(f"[OptunaTuner] Unknown model name: {model_name}")
        # Don't raise error, just log warning
    
    # Validate config
    if config is None:
        raise ValueError("config cannot be None")
    
    # Validate config has required attributes
    required_attrs = ["HYPERPARAMETER_SPACES", "RANDOM_STATE"]
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"config missing required attribute: {attr}")
    
    # ICL models should use the same nested CV approach as other models
    # The context selection is handled internally by the ICL model wrappers
    # during the cross_val_score evaluation in the objective function
    
    maximize = True
    if scoring is not None:
        if scoring.startswith("neg_"):
            maximize = False
        elif scoring in ["accuracy", "f1", "f1_weighted", "roc_auc", "r2"]:
            maximize = True
        else:
            maximize = False
    
    # Validate scoring metric
    valid_scoring_metrics = ["accuracy", "f1", "f1_weighted", "roc_auc", "r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
    if scoring is not None and scoring not in valid_scoring_metrics:
        logger.warning(f"[OptunaTuner] Unknown scoring metric: {scoring}")
        # Don't raise error, just log warning

    direction = "maximize" if maximize else "minimize"

    logger.info(f"Starting Optuna tuning for {model_name} [{model_type}] [Best-practice: use only training data]")

    try:
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=random_state))
    except Exception as e:
        logger.error(f"[OptunaTuner] Error creating study for {model_name}: {e}")
        raise
    
    def objective(trial):
        # Access maximize from outer scope
        nonlocal maximize
        try:
            try:
                model = get_model(trial, model_name, model_type, random_state, config)
                if model is None:
                    raise ValueError(f"Model '{model_name}' could not be created.")
                
                # Validate model parameters
                if hasattr(model, 'alpha') and model.alpha <= 0:
                    logger.warning(f"[OptunaTuner] Invalid alpha value for {model_name}: {model.alpha}")
                    return float('-inf') if maximize else float('inf')
                
                if hasattr(model, 'C') and model.C <= 0:
                    logger.warning(f"[OptunaTuner] Invalid C value for {model_name}: {model.C}")
                    return float('-inf') if maximize else float('inf')
                
                # Additional validation for other parameters
                if hasattr(model, 'learning_rate'):
                    lr = model.learning_rate
                    if isinstance(lr, (int, float)) and (lr <= 0 or lr > 1):
                        logger.warning(f"[OptunaTuner] Invalid learning_rate value for {model_name}: {lr}")
                        return float('-inf') if maximize else float('inf')
                    elif isinstance(lr, str) and lr not in ['constant', 'adaptive', 'invscaling']:
                        logger.warning(f"[OptunaTuner] Invalid learning_rate string for {model_name}: {lr}")
                        return float('-inf') if maximize else float('inf')
                
                if hasattr(model, 'max_depth') and isinstance(model.max_depth, (int, float)) and model.max_depth <= 0:
                    logger.warning(f"[OptunaTuner] Invalid max_depth value for {model_name}: {model.max_depth}")
                    return float('-inf') if maximize else float('inf')
                
                # Additional validation for n_estimators
                if hasattr(model, 'n_estimators') and isinstance(model.n_estimators, (int, float)) and model.n_estimators <= 0:
                    logger.warning(f"[OptunaTuner] Invalid n_estimators value for {model_name}: {model.n_estimators}")
                    return float('-inf') if maximize else float('inf')
                
                # Additional validation for min_samples_split
                if hasattr(model, 'min_samples_split') and isinstance(model.min_samples_split, (int, float)) and model.min_samples_split < 2:
                    logger.warning(f"[OptunaTuner] Invalid min_samples_split value for {model_name}: {model.min_samples_split}")
                    return float('-inf') if maximize else float('inf')
                
                logger.info(f"[OptunaTuner] About to evaluate model {model_name} in Optuna objective on data shape {X.shape}")
                
            except Exception as e:
                logger.error(f"[OptunaTuner] Error creating model {model_name}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return float('-inf') if maximize else float('inf')
            
            from sklearn.model_selection import cross_val_score
            from sklearn.metrics import mean_absolute_error
            
            # Validate data before cross-validation
            if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning(f"[OptunaTuner] Invalid values in X for {model_name}")
                return float('-inf') if maximize else float('inf')
            
            if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                logger.warning(f"[OptunaTuner] Invalid values in y for {model_name}")
                return float('-inf') if maximize else float('inf')
            
            # Check for extreme values that might cause numerical issues
            if np.any(np.abs(X) > 1e10):
                logger.warning(f"[OptunaTuner] Extreme values in X for {model_name}")
                return float('-inf') if maximize else float('inf')
            
            if np.any(np.abs(y) > 1e10):
                logger.warning(f"[OptunaTuner] Extreme values in y for {model_name}")
                return float('-inf') if maximize else float('inf')
            
            # Check for constant features that might cause issues
            if np.any(np.std(X, axis=0) == 0):
                logger.warning(f"[OptunaTuner] Constant features detected in X for {model_name}")
                return float('-inf') if maximize else float('inf')
            
            # Validate model parameters before cross-validation
            if hasattr(model, 'alpha') and model.alpha <= 0:
                logger.warning(f"[OptunaTuner] Invalid alpha value for {model_name}: {model.alpha}")
                return float('-inf') if maximize else float('inf')
            
            if hasattr(model, 'C') and model.C <= 0:
                logger.warning(f"[OptunaTuner] Invalid C value for {model_name}: {model.C}")
                return float('-inf') if maximize else float('inf')
            
            # Additional validation for other parameters
            if hasattr(model, 'learning_rate'):
                lr = model.learning_rate
                if isinstance(lr, (int, float)) and (lr <= 0 or lr > 1):
                    logger.warning(f"[OptunaTuner] Invalid learning_rate value for {model_name}: {lr}")
                    return float('-inf') if maximize else float('inf')
                elif isinstance(lr, str) and lr not in ['constant', 'adaptive', 'invscaling']:
                    logger.warning(f"[OptunaTuner] Invalid learning_rate string for {model_name}: {lr}")
                    return float('-inf') if maximize else float('inf')
            
            if hasattr(model, 'max_depth') and isinstance(model.max_depth, (int, float)) and model.max_depth <= 0:
                logger.warning(f"[OptunaTuner] Invalid max_depth value for {model_name}: {model.max_depth}")
                return float('-inf') if maximize else float('inf')
            
            # Additional validation for n_estimators
            if hasattr(model, 'n_estimators') and isinstance(model.n_estimators, (int, float)) and model.n_estimators <= 0:
                logger.warning(f"[OptunaTuner] Invalid n_estimators value for {model_name}: {model.n_estimators}")
                return float('-inf') if maximize else float('inf')
            
            # Additional validation for min_samples_split
            if hasattr(model, 'min_samples_split') and isinstance(model.min_samples_split, (int, float)) and model.min_samples_split < 2:
                logger.warning(f"[OptunaTuner] Invalid min_samples_split value for {model_name}: {model.min_samples_split}")
                return float('-inf') if maximize else float('inf')
            
            logger.info(f"[OptunaTuner] About to evaluate model {model_name} in Optuna objective on data shape {X.shape}")
            
            # Add timeout for Random Forest to prevent excessive training time
            if model_name == "RandomForest":
                import signal
                def timeout_handler(signum, frame):
                    raise TimeoutError("Random Forest training timed out")
                
                # Set timeout to 2 minutes for Random Forest
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(120)  # 2 minutes timeout
                
                try:
                    if scoring:
                        # Use fewer CV folds for Random Forest to speed up training
                        cv_folds = 3 if model_name == "RandomForest" else 5
                        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
                    else:
                        # Use fewer CV folds for Random Forest to speed up training
                        cv_folds = 3 if model_name == "RandomForest" else 5
                        scores = cross_val_score(model, X, y, cv=cv_folds, n_jobs=1)
                    signal.alarm(0)  # Cancel timeout
                except TimeoutError:
                    logger.warning(f"[OptunaTuner] Random Forest training timed out for {model_name}")
                    return float('-inf') if maximize else float('inf')
                except Exception as e:
                    signal.alarm(0)  # Cancel timeout
                    raise e
            else:
                try:
                    if scoring:
                        # Use fewer CV folds for Random Forest to speed up training
                        cv_folds = 3 if model_name == "RandomForest" else 5
                        scores = cross_val_score(model, X, y, cv=cv_folds, scoring=scoring, n_jobs=1)
                    else:
                        # Use fewer CV folds for Random Forest to speed up training
                        cv_folds = 3 if model_name == "RandomForest" else 5
                        scores = cross_val_score(model, X, y, cv=cv_folds, n_jobs=1)
                except Exception as e:
                    logger.error(f"[OptunaTuner] Error in cross_val_score for {model_name}: {e}")
                    logger.error(f"[OptunaTuner] Model parameters: {model.get_params()}")
                    logger.error(f"[OptunaTuner] Data shape: X={X.shape}, y={y.shape}")
                    logger.error(f"[OptunaTuner] Data types: X={X.dtype}, y={y.dtype}")
                    logger.error(f"[OptunaTuner] Data range: X=[{X.min():.3f}, {X.max():.3f}], y=[{y.min():.3f}, {y.max():.3f}]")
                    
                    # Try fallback with simpler CV
                    try:
                        logger.info(f"[OptunaTuner] Trying fallback CV for {model_name}")
                        if scoring:
                            scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=1)
                        else:
                            scores = cross_val_score(model, X, y, cv=3, n_jobs=1)
                        logger.info(f"[OptunaTuner] Fallback CV successful for {model_name}")
                    except Exception as e2:
                        logger.error(f"[OptunaTuner] Fallback CV also failed for {model_name}: {e2}")
                        import traceback
                        logger.error(f"Full traceback: {traceback.format_exc()}")
                        return float('-inf') if maximize else float('inf')
            
            # Additional validation of scores
            if not isinstance(scores, np.ndarray):
                logger.warning(f"[OptunaTuner] Scores not numpy array for {model_name}: {type(scores)}")
                return float('-inf') if maximize else float('inf')
            
            # Check for invalid scores
            if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                logger.warning(f"[OptunaTuner] Invalid scores detected for {model_name}: {scores}")
                return float('-inf') if maximize else float('inf')
            
            # Check for empty scores
            if len(scores) == 0:
                logger.warning(f"[OptunaTuner] Empty scores for {model_name}")
                return float('-inf') if maximize else float('inf')
            
            # Check for extreme scores that might indicate issues
            if np.any(np.abs(scores) > 1e6):
                logger.warning(f"[OptunaTuner] Extreme scores detected for {model_name}: {scores}")
                return float('-inf') if maximize else float('inf')
            
            # Log scores for debugging
            logger.info(f"[OptunaTuner] {model_name} CV scores: {scores}")
            
            # Simplified trial reporting - report all scores at once
            try:
                for i, score in enumerate(scores):
                    # Check if score is valid before reporting
                    if np.isnan(score) or np.isinf(score):
                        logger.warning(f"[OptunaTuner] Invalid score at fold {i} for {model_name}: {score}")
                        return float('-inf') if maximize else float('inf')
                    
                    # Report score to trial
                    trial.report(score, i)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        logger.info(f"[OptunaTuner] Trial pruned for {model_name} at fold {i}")
                        raise optuna.TrialPruned()
                        
            except optuna.TrialPruned:
                # Re-raise TrialPruned exception
                raise
            except Exception as e:
                logger.error(f"[OptunaTuner] Error in trial reporting for {model_name}: {e}")
                logger.error(f"[OptunaTuner] Scores: {scores}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                return float('-inf') if maximize else float('inf')
            
            try:
                mean_score = np.mean(scores)
            except Exception as e:
                logger.error(f"[OptunaTuner] Error calculating mean score for {model_name}: {e}")
                return float('-inf') if maximize else float('inf')
            
            # Calculate MAE for additional insight
            try:
                mae_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=1)
                mean_mae = -np.mean(mae_scores)  # Convert back to positive MAE
                logger.info(f"[OptunaTuner] Model {model_name} CV score: {mean_score:.5f}, MAE: {mean_mae:.5f}")
            except Exception as e:
                logger.warning(f"[OptunaTuner] Error calculating MAE for {model_name}: {e}")
                import traceback
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                logger.info(f"[OptunaTuner] Model {model_name} CV score: {mean_score:.5f}")
            
            try:
                return mean_score
            except Exception as e:
                logger.error(f"[OptunaTuner] Error returning mean score for {model_name}: {e}")
                return float('-inf') if maximize else float('inf')
            
        except Exception as e:
            logger.error(f"Error in objective function: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return float('-inf') if maximize else float('inf')
        except KeyboardInterrupt:
            logger.warning("Keyboard interrupt detected in objective function")
            return float('-inf') if maximize else float('inf')
    
    try:
        study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)
    except Exception as e:
        logger.error(f"[OptunaTuner] Error in study optimization for {model_name}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected in study optimization")
        raise

    logger.info(f"Best params for {model_name} ({model_type}): {study.best_params}")
    logger.info(f"Best CV score: {study.best_value:.5f}")

    try:
        best_model = get_model(optuna.trial.FixedTrial(study.best_params), model_name, model_type, random_state, config)
    except Exception as e:
        logger.error(f"[OptunaTuner] Error creating best_model {model_name}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise
    
    logger.info(f"[OptunaTuner] About to fit best_model {model_name} on data shape {X.shape}")
    try:
        best_model.fit(X, y)
        logger.info(f"[OptunaTuner] best_model {model_name} fit complete.")
    except Exception as e:
        logger.error(f"[OptunaTuner] Error fitting best_model {model_name}: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

    logger.info(f"[OptunaTuner] tune_model() complete for model_name={model_name}")
    try:
        return best_model, study.best_params, study.best_value
    except Exception as e:
        logger.error(f"[OptunaTuner] Error returning results for {model_name}: {e}")
        raise


def plot_optuna_results(study, output_dir=None, model_name="model"):
    """Plot Optuna study results and save to files"""
    import os
    import matplotlib.pyplot as plt
    
    if output_dir is None:
        output_dir = "."
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Optimization history
        fig = plot_optimization_history(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_optimization_history.html'))
        fig.write_image(os.path.join(output_dir, f'{model_name}_optimization_history.png'))
        plt.close()
        
        # Parameter importances
        fig = plot_param_importances(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_param_importances.html'))
        fig.write_image(os.path.join(output_dir, f'{model_name}_param_importances.png'))
        plt.close()
        
        # Parallel coordinate
        fig = plot_parallel_coordinate(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_parallel_coordinate.html'))
        fig.write_image(os.path.join(output_dir, f'{model_name}_parallel_coordinate.png'))
        plt.close()
        
        # Contour plots
        fig = plot_contour(study)
        fig.write_html(os.path.join(output_dir, f'{model_name}_contour.html'))
        fig.write_image(os.path.join(output_dir, f'{model_name}_contour.png'))
        plt.close()
        
        print(f"Optuna plots saved to {output_dir}")
        
    except Exception as e:
        print(f"Error saving Optuna plots: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
