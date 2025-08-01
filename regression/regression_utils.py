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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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

def calculate_regression_metrics(y_true, y_pred, metrics_list=None):
    if metrics_list is None:
        metrics_list = ['r2', 'mae', 'mse', 'rmse']
        
    metrics = {}
    
    if 'r2' in metrics_list:
        metrics['r2'] = r2_score(y_true, y_pred)
    if 'mae' in metrics_list:
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
    if 'mse' in metrics_list:
        metrics['mse'] = mean_squared_error(y_true, y_pred)
    if 'rmse' in metrics_list:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
    return metrics

def log_timing_info(start_time, operation_name, logger=None):
    elapsed_time = time.time() - start_time
    message = f"{operation_name} completed in {elapsed_time:.2f} seconds"
    
    if logger:
        logger.info(message)
    else:
        print(message)



def display_results_table(results_df, logger):
    """Display results table with all variance information"""
    logger.info("="*80)
    logger.info("COMPLETE RESULTS TABLE WITH VARIANCE")
    logger.info("="*80)
    logger.info(f"{'Model':<20} {'R²':<12} {'MAE':<12} {'MSE':<12} {'Time':<8}")
    logger.info(f"{'':<20} {'Mean±Std':<12} {'Mean±Std':<12} {'Mean±Std':<12} {'(s)':<8}")
    logger.info("-" * 80)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        model_name = row['Model']
        mean_r2 = row['Mean_R2']
        std_r2 = row['Std_R2']
        mean_mae = row['Mean_MAE']
        std_mae = row['Std_MAE']
        mean_mse = row['Mean_MSE']
        std_mse = row['Std_MSE']
        mean_time = row['Mean_Time']
        
        # Format the metrics with variance
        r2_str = f"{mean_r2:.3f}±{std_r2:.3f}"
        mae_str = f"{mean_mae:.3f}±{std_mae:.3f}"
        mse_str = f"{mean_mse:.3f}±{std_mse:.3f}"
        time_str = f"{mean_time:.2f}"
        
        logger.info(f"{model_name:<20} {r2_str:<12} {mae_str:<12} {mse_str:<12} {time_str:<8}")
    
    logger.info("-" * 80)
    logger.info("Note: All metrics include mean ± standard deviation from nested CV")
    logger.info("="*80)
    logger.info("")

def save_results(results, output_dir, config):
    """Save results to CSV with proper nested CV statistics"""
    logger = logging.getLogger(__name__)
    results_data = []
    
    for model_name, result in results.items():
        if len(result['r2_scores']) > 0:
            # Calculate summary statistics
            mean_r2 = np.mean(result['r2_scores'])
            std_r2 = np.std(result['r2_scores'])
            mean_mae = np.mean(result['mae_scores'])
            std_mae = np.std(result['mae_scores'])
            mean_mse = np.mean(result['mse_scores'])
            std_mse = np.std(result['mse_scores'])
            mean_time = np.mean(result['training_times'])
            mean_inner_cv = np.mean(result['inner_cv_scores'])
            
            # Find most common best parameters
            valid_params = [params for params in result['best_params'] if params]
            if valid_params:
                from collections import Counter
                param_strings = [str(sorted(params.items())) for params in valid_params]
                most_common = Counter(param_strings).most_common(1)[0][0]
                most_common_params = dict(eval(most_common))
            else:
                most_common_params = {}
            
            row = {
                'Model': model_name,
                'Mean_R2': mean_r2,
                'Std_R2': std_r2,
                'Mean_MAE': mean_mae,
                'Std_MAE': std_mae,
                'Mean_MSE': mean_mse,
                'Std_MSE': std_mse,
                'Mean_Time': mean_time,
                'Mean_Inner_CV': mean_inner_cv,
                'Best_Hyperparameters': str(most_common_params)
            }
            results_data.append(row)
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Mean_R2', ascending=False)
        results_path = os.path.join(output_dir, 'nested_cv_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
        
        # Log summary
        logger.info("\n" + "="*50)
        logger.info("NESTED CROSS-VALIDATION RESULTS")
        logger.info("="*50)
        logger.info(f"Total models evaluated: {len(results_data)}")
        logger.info(f"Cross-validation folds: {config.CV_FOLDS} outer, {config.INNER_CV_FOLDS} inner")
        # Check if using Organ PCA or traditional feature selection
        if hasattr(config, 'ORGAN_PCA_SETTINGS') and config.ORGAN_PCA_SETTINGS:
            n_components = config.ORGAN_PCA_SETTINGS.get('n_components_per_organ', 10)
            logger.info(f"Feature selection: Organ PCA ({n_components} components per organ)")
        else:
            logger.info(f"Feature selection: {config.N_FEATURES} features")
        logger.info("")
        
        for i, row in results_df.head(5).iterrows():
            logger.info(f"{i+1}. {row['Model']}:")
            logger.info(f"   R² = {row['Mean_R2']:.3f} ± {row['Std_R2']:.3f}")
            logger.info(f"   MAE = {row['Mean_MAE']:.3f} ± {row['Std_MAE']:.3f}")
            logger.info(f"   MSE = {row['Mean_MSE']:.3f} ± {row['Std_MSE']:.3f}")
            logger.info(f"   Inner CV = {row['Mean_Inner_CV']:.3f}")
            logger.info(f"   Avg Time = {row['Mean_Time']:.2f}s")
            logger.info("")
        
        # Save detailed hyperparameter results
        hyperparam_results = {}
        for model_name, result in results.items():
            if len(result['best_params']) > 0:
                valid_params = [params for params in result['best_params'] if params]
                if valid_params:
                    from collections import Counter
                    param_strings = [str(sorted(params.items())) for params in valid_params]
                    most_common = Counter(param_strings).most_common(1)[0][0]
                    most_common_params = dict(eval(most_common))
                    hyperparam_results[model_name] = {
                        'best_params': most_common_params,
                        'all_params': result['best_params']
                    }
        
        if hyperparam_results:
            hyperparam_path = os.path.join(output_dir, 'hyperparameter_results.json')
            with open(hyperparam_path, 'w') as f:
                json.dump(hyperparam_results, f, indent=2)
            logger.info(f"Hyperparameter results saved to {hyperparam_path}")
        
        return results_df
    else:
        logger.warning("No valid results to save")
        return None

 