import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import sys
import time
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config_classification import config, OUTPUT_DIR, DATA_FLAGS, TARGET_COL
from feature_engineering.organ_pca_selector import OrganPCASelector
from feature_engineering.feature_engineering import FeatureEngineering
from feature_engineering.optuna_tuner import tune_model
from models.classification_models import (
    get_classification_models, TabPFNClassifierWrapper,
    GPT2ICLClassifierWrapper, create_hybrid_ensemble
)
from classification_utils import log_timing_info, display_results_table, save_results
from sklearn.base import clone

from feature_engineering.plotting_utils import (
    save_model, plot_pred_vs_true, plot_residuals, plot_model_comparison,
    plot_final_summary, plot_comprehensive_model_comparison, plot_model_performance_radar,
    plot_learning_curves_comparison, plot_statistical_significance_matrix,
    plot_target_distribution, plot_feature_quality_analysis, plot_model_complexity_analysis,
    plot_cv_stability_analysis, plot_organ_pca_analysis, plot_prediction_error_analysis,
    plot_feature_importance_analysis, plot_learning_curves_detailed,
    plot_icl_benchmark_analysis, plot_icl_detailed_comparison, plot_model_category_analysis,
    plot_icl_advantage_analysis, plot_ensemble_analysis, plot_ensemble_details,
    plot_three_way_ensemble_comparison, plot_performance_vs_time_tradeoff,
    plot_ensemble_advantage_analysis, save_metrics_and_predictions, plot_fold_performance_comparison
)

try:
    from data_extraction.data_manager import DataManager
except ImportError:
    class DataManager:
        def __init__(self, *args):
            pass
        def get_train(self):
            n_samples, n_features = 1000, 50
            X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                           columns=[f'feature_{i}' for i in range(n_features)])
            y = pd.Series(np.random.randint(0, 2, n_samples), name='target')
            return pd.concat([X, y], axis=1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_classification.log')
    ]
)
logger = logging.getLogger(__name__)

def run_classification_pipeline(X, y, config, output_dir):
    logger.info("="*50)
    logger.info("CLASSIFICATION PIPELINE STARTING")
    logger.info("="*50)
    logger.info(f"Output directory: {output_dir}")
    
    results = {}
    for model_name in config.MODELS_TO_TEST:
        results[model_name] = {
            'accuracy_scores': [],
            'f1_scores': [],
            'precision_scores': [],
            'recall_scores': [],
            'roc_auc_scores': [],
            'best_params': [],
            'training_times': [],
            'inner_cv_scores': []
        }
    
    base_models = get_classification_models(config.RANDOM_STATE)
    logger.info(f"Loaded {len(base_models)} models: {list(base_models.keys())}")
    
    #Check which models are actually available
    logger.info("Model availability check:")
    for model_name in config.MODELS_TO_TEST:
        if model_name in base_models:
            logger.info(f"{model_name} - Available")
        else:
            logger.info(f"{model_name} - Not available")
    
    logger.info(f"Using nested CV: Outer {config.CV_FOLDS}-fold, Inner {config.INNER_CV_FOLDS}-fold")
    logger.info(f"Hyperparameter optimization: Optuna with {config.OPTUNA_N_TRIALS} trials (ICL: {config.OPTUNA_N_TRIALS_ICL})")
    logger.info(f"Models with hyperparameter tuning: {list(config.MODELS_WITH_HYPERPARAMETERS.keys())}")
    logger.info(f"Models without hyperparameter tuning: {config.MODELS_WITHOUT_HYPERPARAMETERS}")
    logger.info(f"Total operations: {config.CV_FOLDS} outer folds × {len(base_models)} models = {config.CV_FOLDS * len(base_models)} evaluations")
    logger.info("")
    
    outer_cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
        logger.info("="*42)
        logger.info(f"Outer Fold {fold}/{config.CV_FOLDS}")
        logger.info("="*42)
        
        X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
        y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
        
        logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        
        logger.info(f"Data Distribution Analysis:")
        logger.info(f"Train target - Classes: {y_train_outer.value_counts().to_dict()}")
        logger.info(f"Test target - Classes: {y_test_outer.value_counts().to_dict()}")
        
        train_feature_means = X_train_outer.mean()
        test_feature_means = X_test_outer.mean()
        feature_diff = np.abs(train_feature_means - test_feature_means)
        logger.info(f"Feature distribution difference - Mean: {feature_diff.mean():.4f}, Max: {feature_diff.max():.4f}")
        
        if feature_diff.max() > 1.0:
            logger.warning(f"Large feature distribution differences detected")
        else:
            logger.info(f"Feature distributions are similar between train and test sets")
        
        logger.info("Starting feature engineering...")
        logger.info(f"Input features: {X_train_outer.shape[1]}")
        logger.info(f"Training samples: {X_train_outer.shape[0]}")
        logger.info(f"Test samples: {X_test_outer.shape[0]}")
        
        fe = FeatureEngineering(**config.FEATURE_ENGINEERING_PARAMS)
        X_train_eng = fe.fit_transform(X_train_outer, y_train_outer)
        X_test_eng = fe.transform(X_test_outer)
        logger.info(f"Feature engineering complete: {X_train_eng.shape[1]} features")
        
        logger.info(f"Starting organ PCA feature selection: {X_train_eng.shape[1]} features")
        logger.info(f"Components per organ: {config.ORGAN_PCA_SETTINGS['n_components_per_organ']}")
        
        organ_pca_selector = OrganPCASelector(**config.ORGAN_PCA_SETTINGS)
        X_train_sel = organ_pca_selector.fit_transform(X_train_eng, y_train_outer)
        X_test_sel = organ_pca_selector.transform(X_test_eng)
        logger.info(f"Organ PCA complete: {X_train_sel.shape[1]} PCA components selected")
        logger.info(f"Final training shape: {X_train_sel.shape}")
        logger.info(f"Final test shape: {X_test_sel.shape}")
        
        logger.info(f"Organ PCA analysis:")
        logger.info(f"Original features: {X_train_eng.shape[1]}")
        logger.info(f"PCA components: {X_train_sel.shape[1]}")
        logger.info(f"Organs found: {len(organ_pca_selector.organ_groups)}")
        
        organ_summary = organ_pca_selector.get_organ_summary()
        if isinstance(organ_summary, dict):
            for organ, info in organ_summary.items():
                logger.info(f"{organ}: {info['n_original_features']} features -> {info['n_components']} PCA components")
        
        if X_train_sel.shape[1] < 50:
            logger.warning(f"Very few PCA components - may cause underfitting")
        elif X_train_sel.shape[1] > 1000:
            logger.warning(f"Many PCA components - may cause overfitting")
        else:
            logger.info(f"Reasonable number of PCA components")
        
        inner_cv = StratifiedKFold(n_splits=config.INNER_CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        organ_pca_selector.save_summary(f"{output_dir}/organ_pca_summary")
        
        plot_organ_pca_analysis(organ_pca_selector, f"{output_dir}/analysis")
        
        feature_indices = list(range(X_train_sel.shape[1]))
        logger.info(f"Setting feature indices for ICL models: {len(feature_indices)} PCA components")
        
        for model_name in ['TabPFNICL', 'GPT2ICL']:
            if model_name in base_models:
                if hasattr(base_models[model_name], 'set_feature_indices'):
                    base_models[model_name].set_feature_indices(feature_indices)
        
        for model_name, base_model in base_models.items():
            logger.info(f"Training {model_name} with nested CV")
            logger.info(f"Outer fold: {fold}/{config.CV_FOLDS}")
            logger.info(f"Training data shape: {X_train_sel.shape}")
            logger.info(f"Test data shape: {X_test_sel.shape}")
            start_time = time.time()
            
            try:
                if model_name in ["TabPFNICL", "GPT2ICL"]:
                    logger.info(f"for ICL model: {model_name}")
                    logger.info(f"Using default parameters (no hyperparameter tuning)")
                    
                    best_model = base_models[model_name]
                    
                    if hasattr(best_model, 'set_feature_indices'):
                        feature_indices = list(range(X_train_sel.shape[1]))
                        best_model.set_feature_indices(feature_indices)
                    
                    logger.info(f"Fitting {model_name} model on outer training data...")
                    best_model.fit(X_train_sel, y_train_outer)
                    model = best_model
                    y_pred = model.predict(X_test_sel)
                    
                    best_params = {}
                    best_score = 0.0
                    
                    logger.info(f"{model_name} using default parameters")
                    
                elif model_name in config.MODELS_WITH_HYPERPARAMETERS and config.HYPERPARAMETER_TUNING:
                    logger.info(f"Optimizing {model_name} with hyperparameter tuning (inner CV)...")
                    logger.info(f"Inner CV folds: {config.INNER_CV_FOLDS}")
                    logger.info(f"Tuning metric: {config.TUNING_METRIC}")
                    
                    n_trials = config.OPTUNA_N_TRIALS
                    logger.info(f"Using Optuna for {model_name} (trials: {n_trials})")
                    
                    best_model, best_params, best_score = tune_model(
                        model_name=model_name,
                        model_type="classification",
                        X=X_train_sel,
                        y=y_train_outer,
                        config=config,
                        n_trials=n_trials,
                        random_state=config.RANDOM_STATE,
                        scoring=config.TUNING_METRIC,
                        n_jobs=config.OPTUNA_N_JOBS
                    )
                    
                    logger.info(f"Fitting best {model_name} model on outer training data...")
                    best_model.fit(X_train_sel, y_train_outer)
                    model = best_model
                    y_pred = model.predict(X_test_sel)
                    
                    logger.info(f"{model_name} best score (inner CV): {best_score:.3f}")
                    logger.info(f"{model_name} best parameters: {best_params}")
                    
                elif model_name in config.MODELS_WITHOUT_HYPERPARAMETERS:
                    logger.info(f"Training {model_name} (no hyperparameter tuning needed)")
                    logger.info(f"Inner CV folds: {config.INNER_CV_FOLDS}")
                    logger.info(f"Tuning metric: {config.TUNING_METRIC}")
                    
                    logger.info(f"Running inner cross-validation...")
                    scores = cross_val_score(base_model, X_train_sel, y_train_outer, cv=inner_cv, scoring=config.TUNING_METRIC)
                    mean_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    logger.info(f"Inner CV scores: {scores}")
                    logger.info(f"Inner CV mean ± std: {mean_score:.3f} ± {std_score:.3f}")
                    
                    logger.info(f"Fitting {model_name} on outer training data...")
                    model = base_model
                    model.fit(X_train_sel, y_train_outer)
                    y_pred = model.predict(X_test_sel)
                    best_params = {}
                    best_score = mean_score
                    
                    logger.info(f"    {model_name} inner CV score: {mean_score:.3f}")
                else:
                    logger.info(f"    {model_name} - No hyperparameter tuning")
                    model = base_model
                    model.fit(X_train_sel, y_train_outer)
                    y_pred = model.predict(X_test_sel)
                    best_params = {}
                    best_score = 0.0
                
                logger.info(f"Evaluating {model_name} on outer test data...")
                accuracy = accuracy_score(y_test_outer, y_pred)
                f1 = f1_score(y_test_outer, y_pred, average='weighted')
                precision = precision_score(y_test_outer, y_pred, average='weighted')
                recall = recall_score(y_test_outer, y_pred, average='weighted')
                
                training_time = time.time() - start_time
                
                logger.info(f"Outer test metrics:")
                logger.info(f"Accuracy = {accuracy:.3f}")
                logger.info(f"F1 = {f1:.3f}")
                logger.info(f"Precision = {precision:.3f}")
                logger.info(f"Recall = {recall:.3f}")
                logger.info(f"Training time: {training_time:.2f}s")
                
                inner_score = best_score
                
                overfitting_ratio = inner_score - accuracy
                logger.info(f"Overfitting analysis:")
                logger.info(f"Inner CV score: {inner_score:.3f}")
                logger.info(f"Outer test score: {accuracy:.3f}")
                logger.info(f"Performance drop: {overfitting_ratio:.3f}")
                
                if overfitting_ratio > 0.1:
                    logger.warning(f"Significant overfitting detected!")
                elif overfitting_ratio > 0.05:
                    logger.warning(f"Moderate overfitting detected")
                else:
                    logger.info(f"  Good generalization")
                
                results[model_name]['accuracy_scores'].append(accuracy)
                results[model_name]['f1_scores'].append(f1)
                results[model_name]['precision_scores'].append(precision)
                results[model_name]['recall_scores'].append(recall)
                results[model_name]['best_params'].append(best_params)
                results[model_name]['training_times'].append(training_time)
                results[model_name]['inner_cv_scores'].append(inner_score)
                
                model_analysis_dir = os.path.join(output_dir, "analysis", "individual_models")
                os.makedirs(model_analysis_dir, exist_ok=True)
                
                plot_pred_vs_true(y_test_outer, y_pred, model_analysis_dir, model_name, task_type="classification")
                plot_residuals(y_test_outer, y_pred, model_analysis_dir, model_name, task_type="classification")
                plot_prediction_error_analysis(y_test_outer, y_pred, model_name, model_analysis_dir, task_type="classification")
                
                if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_'):
                    feature_names = [f"PCA_Component_{i}" for i in range(X_train_sel.shape[1])]
                    plot_feature_importance_analysis(model, feature_names, model_analysis_dir, model_name, task_type="classification")
                
                try:
                    plot_learning_curves_detailed(X_train_sel, y_train_outer, model, model_analysis_dir, model_name, task_type="classification")
                except Exception as e:
                    logger.warning(f"Could not create learning curves for {model_name}: {e}")
                
                fold_analysis_dir = os.path.join(output_dir, "analysis", "fold_results")
                os.makedirs(fold_analysis_dir, exist_ok=True)
                save_metrics_and_predictions(
                    y_test_outer, y_pred, fold_analysis_dir, 
                    f"{model_name}_fold_{fold}", task_type="classification"
                )
                
                logger.info(f"{model_name} completed successfully")
                
                if hasattr(model, 'coef_'):
                    non_zero_coefs = np.sum(model.coef_ != 0)
                    total_coefs = len(model.coef_)
                    sparsity = 1 - (non_zero_coefs / total_coefs)
                    logger.info(f"Model complexity - Non-zero coefficients: {non_zero_coefs}/{total_coefs} ({sparsity:.1%} sparse)")
                
                if hasattr(model, 'feature_importances_'):
                    important_features = np.sum(model.feature_importances_ > 0.01)
                    top_features = np.argsort(model.feature_importances_)[-5:]
                    logger.info(f"Model complexity, Important features: {important_features}/{len(model.feature_importances_)}")
                    logger.info(f"Top 5 feature importances: {model.feature_importances_[top_features]}")
                
            except Exception as e:
                logger.warning(f"Error in {model_name}: {e}")
                import traceback
                logger.warning(f"Full traceback: {traceback.format_exc()}")
                results[model_name]['accuracy_scores'].append(0.0)
                results[model_name]['f1_scores'].append(0.0)
                results[model_name]['precision_scores'].append(0.0)
                results[model_name]['recall_scores'].append(0.0)
                results[model_name]['best_params'].append({})
                results[model_name]['training_times'].append(0.0)
                results[model_name]['inner_cv_scores'].append(0.0)
        
        logger.info(f"")
        logger.info(f"Outer Fold {fold}/{config.CV_FOLDS} Summary:")
        logger.info(f"==========================================")
        
        logger.info(f"Model Performance:")
        for model_name in base_models.keys():
            if len(results[model_name]['accuracy_scores']) > 0:
                latest_accuracy = results[model_name]['accuracy_scores'][-1]
                latest_f1 = results[model_name]['f1_scores'][-1]
                latest_precision = results[model_name]['precision_scores'][-1]
                latest_recall = results[model_name]['recall_scores'][-1]
                latest_time = results[model_name]['training_times'][-1]
                
                if latest_accuracy > 0.8:
                    status = "GOOD"
                elif latest_accuracy > 0.6:
                    status = "Meh"
                else:
                    status = "POOOO"
                
                logger.info(f"{model_name}: {status} Accuracy = {latest_accuracy:.3f}, F1 = {latest_f1:.3f}, Precision = {latest_precision:.3f}, Recall = {latest_recall:.3f}, Time = {latest_time:.2f}s")
        
        logger.info(f"Overfitting Analysis:")
        best_model = None
        best_score = -np.inf
        for model_name in base_models.keys():
            if len(results[model_name]['accuracy_scores']) > 0:
                score = results[model_name]['accuracy_scores'][-1]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        if best_model:
            logger.info(f"Best model: {best_model} (Accuracy = {best_score:.3f})")
            if best_score < 0.3:
                logger.warning(f"All models performing poorly - consider feature engineering or data quality")
            elif best_score < 0.5:
                logger.warning(f"Poor performance - may need more features or different models")
            else:
                logger.info(f"Reasonable performance achieved")
        
        logger.info(f"Outer Fold {fold}/{config.CV_FOLDS} completed")
        logger.info(f"")
    
    save_results(results, output_dir, config)
    
    create_comprehensive_analysis_plots(results, output_dir)
    
    if config.ENSEMBLE_METHODS and len(results) > 0:
        logger.info("")
        logger.info("="*50)
        logger.info("ENSEMBLE MODEL EVALUATION")
        logger.info("="*50)
        
        temp_results_data = []
        for model_name, result in results.items():
            if len(result['accuracy_scores']) > 0:
                mean_accuracy = np.mean(result['accuracy_scores'])
                std_accuracy = np.std(result['accuracy_scores'])
                mean_f1 = np.mean(result['f1_scores'])
                std_f1 = np.std(result['f1_scores'])
                mean_precision = np.mean(result['precision_scores'])
                std_precision = np.std(result['precision_scores'])
                mean_recall = np.mean(result['recall_scores'])
                std_recall = np.std(result['recall_scores'])
                mean_time = np.mean(result['training_times'])
                
                row = {
                    'Model': model_name,
                    'Mean_Accuracy': mean_accuracy,
                    'Std_Accuracy': std_accuracy,
                    'Mean_F1': mean_f1,
                    'Std_F1': std_f1,
                    'Mean_Precision': mean_precision,
                    'Std_Precision': std_precision,
                    'Mean_Recall': mean_recall,
                    'Std_Recall': std_recall,
                    'Mean_Time': mean_time
                }
                temp_results_data.append(row)
        
        temp_results_df = pd.DataFrame(temp_results_data)
        top_models = temp_results_df.head(config.TOP_MODELS_FOR_ENSEMBLE)['Model'].tolist()
        logger.info(f"Creating ensembles from top {len(top_models)} models: {top_models}")
        
        ensemble_models = {}
        
        if "hybrid" in config.ENSEMBLE_METHODS:
            #select top traditional models and best ICL model
            if len(base_models) >= 2:
                # Get top traditional models
                traditional_models = {name: base_models[name] for name in top_models 
                                   if not name.endswith('ICL')}
                
                # Get best ICL model based on performance
                icl_results = temp_results_df[temp_results_df['Model'].str.endswith('ICL')]
                best_icl_model = None
                if len(icl_results) > 0:
                    best_icl_model = icl_results.loc[icl_results['Mean_Accuracy'].idxmax(), 'Model']
                    logger.info(f"Best ICL model: {best_icl_model} (Accuracy = {icl_results['Mean_Accuracy'].max():.3f})")
                
                # Create hybrid models dict with top traditional + best ICL
                hybrid_models = traditional_models.copy()
                if best_icl_model and best_icl_model in base_models:
                    hybrid_models[best_icl_model] = base_models[best_icl_model]
                    logger.info(f"Hybrid ensemble: {list(traditional_models.keys())} + {best_icl_model}")
                else:
                    logger.info(f"Hybrid ensemble: {list(traditional_models.keys())} (no ICL models available)")
                
                # Only create hybrid ensemble if we have at least 2 models
                if len(hybrid_models) >= 2:
                    try:
                        hybrid_ensemble = create_hybrid_ensemble(
                            hybrid_models,  # Pass selected models
                            top_n=len(hybrid_models),
                            cv_folds=config.INNER_CV_FOLDS,
                            random_state=config.RANDOM_STATE
                        )
                        ensemble_models['HybridEnsemble'] = hybrid_ensemble
                        logger.info("Created hybrid ensemble with top traditional + best ICL model")
                    except Exception as e:
                        logger.warning(f"Failed to create hybrid ensemble: {e}")
                        logger.info("Falling back to traditional ensemble only")
                else:
                    logger.warning("Not enough models for hybrid ensemble (need at least 2)")
            else:
                logger.warning("Need at least 2 models for hybrid ensemble")
        
        if ensemble_models:
            logger.info("Evaluating ensemble models with nested CV...")
            
            for ensemble_name, ensemble_model in ensemble_models.items():
                logger.info(f"Evaluating {ensemble_name}")
                
                ensemble_results = {
                    'accuracy_scores': [],
                    'f1_scores': [],
                    'precision_scores': [],
                    'recall_scores': [],
                    'training_times': []
                }
                
                for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
                    logger.info(f"Fold {fold}/{config.CV_FOLDS}")
                    
                    X_train_outer, X_test_outer = X.iloc[train_idx], X.iloc[test_idx]
                    y_train_outer, y_test_outer = y.iloc[train_idx], y.iloc[test_idx]
                    
                    fe = FeatureEngineering(**config.FEATURE_ENGINEERING_PARAMS)
                    X_train_eng = fe.fit_transform(X_train_outer, y_train_outer)
                    X_test_eng = fe.transform(X_test_outer)
                    
                    organ_pca_selector = OrganPCASelector(**config.ORGAN_PCA_SETTINGS)
                    X_train_sel = organ_pca_selector.fit_transform(X_train_eng, y_train_outer)
                    X_test_sel = organ_pca_selector.transform(X_test_eng)
                    
                    start_time = time.time()
                    ensemble_model.fit(X_train_sel, y_train_outer)
                    y_pred = ensemble_model.predict(X_test_sel)
                    training_time = time.time() - start_time
                    
                    accuracy = accuracy_score(y_test_outer, y_pred)
                    f1 = f1_score(y_test_outer, y_pred, average='weighted')
                    precision = precision_score(y_test_outer, y_pred, average='weighted')
                    recall = recall_score(y_test_outer, y_pred, average='weighted')
                    
                    ensemble_results['accuracy_scores'].append(accuracy)
                    ensemble_results['f1_scores'].append(f1)
                    ensemble_results['precision_scores'].append(precision)
                    ensemble_results['recall_scores'].append(recall)
                    ensemble_results['training_times'].append(training_time)
                    
                    logger.info(f"Accuracy = {accuracy:.3f}, F1 = {f1:.3f}, Precision = {precision:.3f}, Recall = {recall:.3f}, Time = {training_time:.2f}s")
                
                results[ensemble_name] = ensemble_results
                
                mean_accuracy = np.mean(ensemble_results['accuracy_scores'])
                std_accuracy = np.std(ensemble_results['accuracy_scores'])
                mean_f1 = np.mean(ensemble_results['f1_scores'])
                std_f1 = np.std(ensemble_results['f1_scores'])
                mean_precision = np.mean(ensemble_results['precision_scores'])
                std_precision = np.std(ensemble_results['precision_scores'])
                mean_recall = np.mean(ensemble_results['recall_scores'])
                std_recall = np.std(ensemble_results['recall_scores'])
                mean_time = np.mean(ensemble_results['training_times'])
                
                logger.info(f"{ensemble_name} Performance:")
                logger.info(f"Accuracy = {mean_accuracy:.3f} ± {std_accuracy:.3f}")
                logger.info(f"F1 = {mean_f1:.3f} ± {std_f1:.3f}")
                logger.info(f"Precision = {mean_precision:.3f} ± {std_precision:.3f}")
                logger.info(f"Recall = {mean_recall:.3f} ± {std_recall:.3f}")
                logger.info(f"Time = {mean_time:.2f}s")
                logger.info("")
        
        logger.info("Ensemble evaluation complete!")
        logger.info("")
        
        final_results_data = []
        for model_name, result in results.items():
            if len(result['accuracy_scores']) > 0:
                mean_accuracy = np.mean(result['accuracy_scores'])
                std_accuracy = np.std(result['accuracy_scores'])
                mean_f1 = np.mean(result['f1_scores'])
                std_f1 = np.std(result['f1_scores'])
                mean_precision = np.mean(result['precision_scores'])
                std_precision = np.std(result['precision_scores'])
                mean_recall = np.mean(result['recall_scores'])
                std_recall = np.std(result['recall_scores'])
                mean_time = np.mean(result['training_times'])
                
                row = {
                    'Model': model_name,
                    'Mean_Accuracy': mean_accuracy,
                    'Std_Accuracy': std_accuracy,
                    'Mean_F1': mean_f1,
                    'Std_F1': std_f1,
                    'Mean_Precision': mean_precision,
                    'Std_Precision': std_precision,
                    'Mean_Recall': mean_recall,
                    'Std_Recall': std_recall,
                    'Mean_Time': mean_time
                }
                final_results_data.append(row)
        
        if final_results_data:
            final_results_df = pd.DataFrame(final_results_data)
            final_results_df = final_results_df.sort_values('Mean_Accuracy', ascending=False)
            
            individual_models = final_results_df[~final_results_df['Model'].str.contains('Ensemble')]
            ensemble_models = final_results_df[final_results_df['Model'].str.contains('Ensemble')]
            
            if len(individual_models) > 0 and len(ensemble_models) > 0:
                try:
                    from feature_engineering.plotting_utils import (
                        plot_three_way_ensemble_comparison, 
                        plot_performance_vs_time_tradeoff, 
                        plot_ensemble_advantage_analysis
                    )
                    
                    comparison_df = pd.concat([individual_models.head(1), ensemble_models], ignore_index=True)
                    comparison_df = comparison_df.rename(columns={
                        'Mean_Accuracy': 'Mean_Accuracy',
                        'Mean_F1': 'Mean_F1',
                        'Mean_Precision': 'Mean_Precision',
                        'Mean_Recall': 'Mean_Recall',
                        'Mean_Time': 'Mean_Time'
                    })
                    
                    plot_three_way_ensemble_comparison(comparison_df, os.path.join(output_dir, "analysis"), task_type="classification")
                    plot_performance_vs_time_tradeoff(final_results_df, os.path.join(output_dir, "analysis"), task_type="classification")
                    plot_ensemble_advantage_analysis(individual_models, ensemble_models, os.path.join(output_dir, "analysis"), task_type="classification")
                    
                    logger.info("ensemble comparison plots created!")
                except Exception as e:
                    logger.warning(f"Could not do ensemble comparison plots: {e}")

    
    results_df = pd.DataFrame()
    for model_name, result in results.items():
        if len(result['accuracy_scores']) > 0:
            mean_accuracy = np.mean(result['accuracy_scores'])
            std_accuracy = np.std(result['accuracy_scores'])
            mean_f1 = np.mean(result['f1_scores'])
            std_f1 = np.std(result['f1_scores'])
            mean_precision = np.mean(result['precision_scores'])
            std_precision = np.std(result['precision_scores'])
            mean_recall = np.mean(result['recall_scores'])
            std_recall = np.std(result['recall_scores'])
            mean_time = np.mean(result['training_times'])
            
            row = {
                'Model': model_name,
                'Mean_Accuracy': mean_accuracy,
                'Std_Accuracy': std_accuracy,
                'Mean_F1': mean_f1,
                'Std_F1': std_f1,
                'Mean_Precision': mean_precision,
                'Std_Precision': std_precision,
                'Mean_Recall': mean_recall,
                'Std_Recall': std_recall,
                'Mean_Time': mean_time
            }
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    
    if not results_df.empty:
        results_df = results_df.sort_values('Mean_Accuracy', ascending=False)
        results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        
        display_results_table(results_df, logger)
        
        best_model_name = results_df.iloc[0]['Model']
        logger.info(f"\n" + "="*50)
        logger.info(f"FINAL MODEL TRAINING")
        logger.info(f"="*50)
        logger.info(f"Best model: {best_model_name}")
        logger.info(f"Best Accuracy score: {results_df.iloc[0]['Mean_Accuracy']:.3f} ± {results_df.iloc[0]['Std_Accuracy']:.3f}")
        
        if best_model_name in config.MODELS_WITH_HYPERPARAMETERS:
            best_params_list = results[best_model_name]['best_params']
            valid_params = [params for params in best_params_list if params]
            
            if valid_params:
                from collections import Counter
                param_strings = [str(sorted(params.items())) for params in valid_params]
                most_common = Counter(param_strings).most_common(1)[0][0]
                final_params = dict(eval(most_common))
                
                logger.info(f"Best parameters: {final_params}")
                
                final_model = base_models[best_model_name]
                final_model.set_params(**final_params)
            else:
                final_model = base_models[best_model_name]
        elif best_model_name in config.MODELS_WITHOUT_HYPERPARAMETERS:
            if best_model_name == "TabPFNICL":
                final_model = TabPFNClassifierWrapper(random_state=config.RANDOM_STATE)
            elif best_model_name == "GPT2ICL":
                final_model = GPT2ICLClassifierWrapper()
            else:
                final_model = base_models[best_model_name]
        elif best_model_name.endswith('Ensemble'):
            logger.info(f"Recreating {best_model_name} for final training...")
            if best_model_name == "HybridEnsemble":
                temp_results_data = []
                for model_name, result in results.items():
                    if len(result['accuracy_scores']) > 0 and not model_name.endswith('Ensemble'):
                        mean_accuracy = np.mean(result['accuracy_scores'])
                        temp_results_data.append({'Model': model_name, 'Mean_Accuracy': mean_accuracy})
                
                if temp_results_data:
                    temp_results_df = pd.DataFrame(temp_results_data)
                    temp_results_df = temp_results_df.sort_values('Mean_Accuracy', ascending=False)
                    top_models = temp_results_df.head(config.TOP_MODELS_FOR_ENSEMBLE)['Model'].tolist()
                    
                    # Get top traditional models
                    traditional_models = {name: base_models[name] for name in top_models 
                                       if not name.endswith('ICL')}
                    
                    # Get best ICL model based on performance
                    icl_results = temp_results_df[temp_results_df['Model'].str.endswith('ICL')]
                    best_icl_model = None
                    if len(icl_results) > 0:
                        best_icl_model = icl_results.loc[icl_results['Mean_Accuracy'].idxmax(), 'Model']
                        logger.info(f"Best ICL model for final training: {best_icl_model} (Accuracy = {icl_results['Mean_Accuracy'].max():.3f})")
                    
                    # Create hybrid models dict with top traditional + best ICL
                    hybrid_models = traditional_models.copy()
                    if best_icl_model and best_icl_model in base_models:
                        hybrid_models[best_icl_model] = base_models[best_icl_model]
                        logger.info(f"Final hybrid ensemble: {list(traditional_models.keys())} + {best_icl_model}")
                    else:
                        logger.info(f"Final hybrid ensemble: {list(traditional_models.keys())} (no ICL models available)")
                    
                    if len(hybrid_models) >= 2:
                        final_model = create_hybrid_ensemble(
                            hybrid_models,  # Pass selected models
                            top_n=len(hybrid_models),
                            cv_folds=config.INNER_CV_FOLDS,
                            random_state=config.RANDOM_STATE
                        )
                    else:
                        logger.warning(f"Not enough models for {best_model_name}, using best individual model")
                        best_individual = temp_results_df.iloc[0]['Model']
                        final_model = base_models[best_individual]
                else:
                    logger.warning(f"No valid models found for {best_model_name}, using first available model")
                    final_model = list(base_models.values())[0]
            else:
                logger.warning(f"Unknown ensemble type {best_model_name}, using best individual model")
                best_individual = None
                best_score = -np.inf
                for model_name, result in results.items():
                    if len(result['accuracy_scores']) > 0 and not model_name.endswith('Ensemble'):
                        score = np.mean(result['accuracy_scores'])
                        if score > best_score:
                            best_score = score
                            best_individual = model_name
                
                if best_individual and best_individual in base_models:
                    final_model = base_models[best_individual]
                else:
                    final_model = list(base_models.values())[0]
        else:
            final_model = base_models[best_model_name]
            
            fe_final = FeatureEngineering(**config.FEATURE_ENGINEERING_PARAMS)
            X_eng_final = fe_final.fit_transform(X, y)
            
            organ_pca_selector_final = OrganPCASelector(**config.ORGAN_PCA_SETTINGS)
            X_sel_final = organ_pca_selector_final.fit_transform(X_eng_final, y)
            
            final_model.fit(X_sel_final, y)
            
            save_model(final_model, output_dir, best_model_name)
            logger.info(f"Final model saved: {best_model_name}")
        
        try:
            final_results_data = []
            for model_name, result in results.items():
                if len(result['accuracy_scores']) > 0:
                    mean_accuracy = np.mean(result['accuracy_scores'])
                    std_accuracy = np.std(result['accuracy_scores'])
                    mean_f1 = np.mean(result['f1_scores'])
                    std_f1 = np.std(result['f1_scores'])
                    mean_precision = np.mean(result['precision_scores'])
                    std_precision = np.std(result['precision_scores'])
                    mean_recall = np.mean(result['recall_scores'])
                    std_recall = np.std(result['recall_scores'])
                    mean_time = np.mean(result['training_times'])
                    
                    row = {
                        'Model': model_name,
                        'Mean_Accuracy': mean_accuracy,
                        'Std_Accuracy': std_accuracy,
                        'Mean_F1': mean_f1,
                        'Std_F1': std_f1,
                        'Mean_Precision': mean_precision,
                        'Std_Precision': std_precision,
                        'Mean_Recall': mean_recall,
                        'Std_Recall': std_recall,
                        'Mean_Time': mean_time
                    }
                    final_results_data.append(row)
            
            if final_results_data:
                final_results_df = pd.DataFrame(final_results_data)
                final_results_df = final_results_df.sort_values('Mean_Accuracy', ascending=False)
                plot_final_summary(final_results_df, os.path.join(output_dir, "analysis"), metric="accuracy", task_type="classification")
                logger.info("Final summary plot created!")
        except Exception as e:
            logger.warning(f"Could not create final summary plot: {e}")
    
    return results_df

def create_comprehensive_analysis_plots(results, output_dir):
    analysis_dir = os.path.join(output_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    results_data = []
    for model_name, result in results.items():
        if len(result['accuracy_scores']) > 0:
            mean_accuracy = np.mean(result['accuracy_scores'])
            std_accuracy = np.std(result['accuracy_scores'])
            mean_f1 = np.mean(result['f1_scores'])
            std_f1 = np.std(result['f1_scores'])
            mean_precision = np.mean(result['precision_scores'])
            std_precision = np.std(result['precision_scores'])
            mean_recall = np.mean(result['recall_scores'])
            std_recall = np.std(result['recall_scores'])
            mean_time = np.mean(result['training_times'])
            
            row = {
                'Model': model_name,
                'Mean_Accuracy': mean_accuracy,
                'Std_Accuracy': std_accuracy,
                'Mean_F1': mean_f1,
                'Std_F1': std_f1,
                'Mean_Precision': mean_precision,
                'Std_Precision': std_precision,
                'Mean_Recall': mean_recall,
                'Std_Recall': std_recall,
                'Mean_Time': mean_time
            }
            results_data.append(row)
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Mean_Accuracy', ascending=False)
        
        logger.info("Creating model comparison plots...")
        plot_model_comparison(results_df, 'Accuracy', analysis_dir, task_type="classification")
        plot_comprehensive_model_comparison(results_df, analysis_dir, task_type="classification")
        plot_model_performance_radar(results_df, analysis_dir, task_type="classification")
        plot_learning_curves_comparison(results_df, analysis_dir, task_type="classification")
        plot_statistical_significance_matrix(results_df, analysis_dir, task_type="classification")
        plot_model_complexity_analysis(results_df, analysis_dir, task_type="classification")
        plot_cv_stability_analysis(results, analysis_dir, task_type="classification")
        plot_fold_performance_comparison(results, analysis_dir, task_type="classification")
        
        logger.info("Creating model category analysis...")
        plot_model_category_analysis(results_df, analysis_dir, task_type="classification")
        
        icl_models = results_df[results_df['Model'].str.contains('GPT2|TabPFN')]
        if len(icl_models) > 0:
            logger.info("Creating ICL benchmark analysis...")
            plot_icl_benchmark_analysis(results_df, analysis_dir, task_type="classification")
            plot_icl_detailed_comparison(results_df, analysis_dir, task_type="classification")
            plot_icl_advantage_analysis(results_df, analysis_dir, task_type="classification")
        
        ensemble_models = results_df[results_df['Model'].str.contains('Ensemble')]
        if len(ensemble_models) > 0:
            logger.info("Creating ensemble analysis...")
            ensemble_dict = {}
            for _, row in ensemble_models.iterrows():
                ensemble_dict[row['Model']] = {
                    'score': row['Mean_Accuracy'],
                    'time': row['Mean_Time']
                }
            plot_ensemble_analysis(results_df, ensemble_dict, analysis_dir, task_type="classification")
            plot_ensemble_details(ensemble_dict, analysis_dir, task_type="classification")
        
        logger.info("Comprehensive analysis plots Done")

def main():
    start_time = time.time()
    
    logger.info("=" * 50)
    logger.info("CLASSIFICATION PIPELINE STARTING")
    logger.info("=" * 50)
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{OUTPUT_DIR}/classification_comparison_{timestamp}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        
        logger.info("Loading data...")
        dm = DataManager(*DATA_FLAGS)
        train_data = dm.get_train()
        X = train_data.drop(columns=[TARGET_COL])
        y = train_data[TARGET_COL]
        
        logger.info("Validating and cleaning data...")
        
        if np.any(np.isinf(X.values)):
            logger.warning("Infinite values detected, replacing with NaN")
            X = X.replace([np.inf, -np.inf], np.nan)
        
        if np.any(np.abs(X.values) > 1e10):
            logger.warning("Extreme values detected, clipping")
            X = X.clip(-1e10, 1e10)
        
        if np.any(np.isinf(y.values)) or np.any(np.isnan(y.values)):
            logger.warning("Invalid values in target, removing affected samples")
            valid_mask = ~(np.isinf(y.values) | np.isnan(y.values))
            X = X[valid_mask]
            y = y[valid_mask]
        
        logger.info("Performing additional outlier detection...")
        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            X[col] = X[col].clip(lower_bound, upper_bound)
        
        logger.info("Data validation and cleaning complete")
        
        logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        analysis_dir = os.path.join(output_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        logger.info("Creating target distribution analysis...")
        plot_target_distribution(y, analysis_dir, task_type="classification")
        
        logger.info("Creating feature quality analysis...")
        plot_feature_quality_analysis(X, y, analysis_dir, task_type="classification")
        
        results_df = run_classification_pipeline(X, y, config, output_dir)
        
        logger.info("NESTED CROSS-VALIDATION PIPELINE DONE")
        
        logger.info("")
        logger.info("="*60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("="*60)
        logger.info(f"Target distribution analysis: {os.path.join(output_dir, 'analysis', 'target_distribution_analysis.png')}")
        logger.info(f"Feature quality analysis: {os.path.join(output_dir, 'analysis', 'feature_quality_analysis.png')}")
        logger.info(f"Organ PCA analysis: {os.path.join(output_dir, 'analysis', 'organ_pca_analysis.png')}")
        logger.info(f"Model comparison plots: {os.path.join(output_dir, 'analysis', 'model_comparison_accuracy.png')}")
        logger.info(f"Model comparison: {os.path.join(output_dir, 'analysis', 'comprehensive_model_comparison.png')}")
        logger.info(f"Model performance radar: {os.path.join(output_dir, 'analysis', 'model_performance_radar.png')}")
        logger.info(f"Statistical significance matrix: {os.path.join(output_dir, 'analysis', 'statistical_significance_matrix.png')}")
        logger.info(f"CV stability analysis: {os.path.join(output_dir, 'analysis', 'cv_stability_analysis.png')}")
        logger.info(f"Model complexity analysis: {os.path.join(output_dir, 'analysis', 'model_complexity_analysis.png')}")
        logger.info(f"Model category analysis: {os.path.join(output_dir, 'analysis', 'model_category_analysis.png')}")
        logger.info(f"Final summary plots: {os.path.join(output_dir, 'analysis', 'final_summary_plots.png')}")
        logger.info("")
        logger.info("Individual model analysis plots saved in: analysis/individual_models/")
        logger.info("Fold results saved in: analysis/fold_results/")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()