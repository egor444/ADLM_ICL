import numpy as np
import pandas as pd
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')
from classification.models.classification_models import get_classification_models
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score,
    classification_report
)
from feature_engineering.feature_engineering import FeatureEngineering
from feature_engineering.organ_pca_selector import OrganPCASelector
def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def setup_logging(output_dir=None, log_level="INFO"):
    handlers = [logging.StreamHandler()]
    
    if output_dir:
        ensure_dir(output_dir)
        handlers.append(logging.FileHandler(os.path.join(output_dir, 'pipeline.log')))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

def calculate_classification_metrics(y_true, y_pred, y_prob=None, metrics_list=None):
    if metrics_list is None:
        metrics_list = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc']
    
    metrics = {}
    
    if 'accuracy' in metrics_list:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    if 'f1_weighted' in metrics_list:
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
    if 'precision_weighted' in metrics_list:
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
    if 'recall_weighted' in metrics_list:
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
    if 'roc_auc' in metrics_list and y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            metrics['roc_auc'] = 0.0
    
    return metrics

def log_timing_info(start_time, operation_name, logger=None):
    elapsed_time = time.time() - start_time
    message = f"{operation_name} completed in {elapsed_time:.2f} seconds"
    
    if logger:
        logger.info(message)
    else:
        print(message)

def get_base_models(random_state=42):
    
    return get_classification_models(random_state=random_state)

def train_final_model_nested(best_model_name, X, y, config, results):
    logger = logging.getLogger(__name__)
    logger.info(f"Training final {best_model_name} model with best hyperparameters...")
    
    fe = FeatureEngineering(**config.FEATURE_ENGINEERING_PARAMS)
    X_eng = fe.fit_transform(X, y)
    
    logger.info(f"Final model organ PCA: {X_eng.shape[1]} features")
    
    organ_pca_selector = OrganPCASelector(**config.ORGAN_PCA_SETTINGS)
    X_sel = organ_pca_selector.fit_transform(X_eng, y)
    logger.info(f"Final model organ PCA complete: {X_sel.shape[1]} PCA components")
    
    logger.info("Final model: feature engineering already applied scaling")
    
    best_params = results[best_model_name].get('most_common_best_params', {})
    
    base_models = get_base_models(random_state=config.RANDOM_STATE)
    final_model = base_models[best_model_name]
    
    if best_params:
        logger.info(f"Using best hyperparameters: {best_params}")
        for param, value in best_params.items():
            if hasattr(final_model, param):
                setattr(final_model, param, value)
            elif hasattr(final_model, 'set_params'):
                try:
                    final_model.set_params(**{param: value})
                except:
                    pass
    
    final_model.fit(X_sel, y)
    return final_model

def display_results_table(results_df, logger):
    logger.info("\n" + "="*80)
    logger.info("CLASSIFICATION RESULTS SUMMARY")
    logger.info("="*80)
    
    for idx, row in results_df.iterrows():
        model_name = row['Model']
        accuracy = row['Mean_Accuracy']
        std_accuracy = row['Std_Accuracy']
        f1 = row['Mean_F1']
        std_f1 = row['Std_F1']
        precision = row['Mean_Precision']
        std_precision = row['Std_Precision']
        recall = row['Mean_Recall']
        std_recall = row['Std_Recall']
        time_taken = row['Mean_Time']
        
        logger.info(f"{model_name:25} | "
                   f"Accuracy: {accuracy:.3f}±{std_accuracy:.3f} | "
                   f"F1: {f1:.3f}±{std_f1:.3f} | "
                   f"Precision: {precision:.3f}±{std_precision:.3f} | "
                   f"Recall: {recall:.3f}±{std_recall:.3f} | "
                   f"Time: {time_taken:.2f}s")
    
    logger.info("="*80)

def save_results(results, output_dir, config):
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
        results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        
        config_data = {
            'random_state': config.RANDOM_STATE,
            'cv_folds': config.CV_FOLDS,
            'inner_cv_folds': config.INNER_CV_FOLDS,
            'models_tested': config.MODELS_TO_TEST,
            'ensemble_methods': config.ENSEMBLE_METHODS,
            'feature_engineering_params': config.FEATURE_ENGINEERING_PARAMS,
            'organ_pca_settings': config.ORGAN_PCA_SETTINGS
        }
        
        with open(os.path.join(output_dir, 'config_summary.json'), 'w') as f:
            json.dump(config_data, f, indent=2)
        
        return results_df
    return None

def train_final_model(best_model_name, X, y, config, models):
    logger = logging.getLogger(__name__)
    logger.info(f"Training final {best_model_name} model...")
    
    
    fe = FeatureEngineering(**config.FEATURE_ENGINEERING_PARAMS)
    X_eng = fe.fit_transform(X, y)
    
    
    organ_pca_selector = OrganPCASelector(**config.ORGAN_PCA_SETTINGS)
    X_sel = organ_pca_selector.fit_transform(X_eng, y)
    
    final_model = models[best_model_name]
    final_model.fit(X_sel, y)
    
    return final_model 