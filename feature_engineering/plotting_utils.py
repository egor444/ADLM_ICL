import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import label_binarize
from itertools import cycle
from sklearn.metrics import roc_curve, auc

# Set modern plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

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

def log_timing_info(start_time, operation_name, logger=None):
    elapsed_time = time.time() - start_time
    message = f"{operation_name} completed in {elapsed_time:.2f} seconds"
    if logger:
        logger.info(message)
    else:
        print(message)

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

def calculate_classification_metrics(y_true, y_pred, y_prob=None, metrics_list=None):
    if metrics_list is None:
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        
    metrics = {}
    
    if 'accuracy' in metrics_list:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    if 'precision' in metrics_list:
        metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    if 'recall' in metrics_list:
        metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    if 'f1' in metrics_list:
        metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    if 'roc_auc' in metrics_list and y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        except:
            metrics['roc_auc'] = 0.0
        
    return metrics

def save_model(model, output_dir, model_name):
    model_path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, model_path)

def load_model(model_path):
    return joblib.load(model_path)

def save_metrics_and_predictions(y_true, y_pred, output_dir, model_name, task_type="regression",
                                y_prob=None, metrics_list=None, save_predictions=True):
    if task_type == "regression":
        metrics = calculate_regression_metrics(y_true, y_pred, metrics_list)
    else:
        metrics = calculate_classification_metrics(y_true, y_pred, y_prob, metrics_list)
    
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if save_predictions:
        predictions_df = pd.DataFrame({
            'true': y_true,
            'predicted': y_pred
        })
        if y_prob is not None:
            if len(y_prob.shape) == 2:
                for i in range(y_prob.shape[1]):
                    predictions_df[f'prob_class_{i}'] = y_prob[:, i]
            else:
                predictions_df['probability'] = y_prob
            
        predictions_path = os.path.join(output_dir, f"{model_name}_predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

def plot_pred_vs_true(y_true, y_pred, output_dir, model_name="model", task_type="regression"):
    plt.figure(figsize=(12, 10))
    
    if task_type == "regression":
        # Create a more sophisticated scatter plot
        plt.scatter(y_true, y_pred, alpha=0.7, s=50, c='#2E8B57', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
        
        plt.xlabel('True Values', fontsize=12, fontweight='bold')
        plt.ylabel('Predicted Values', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Predicted vs True Values', fontsize=14, fontweight='bold', pad=20)
        
        # Add R² score with better styling
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#E8F5E8", 
                         edgecolor="#2E8B57", linewidth=2), fontsize=12, fontweight='bold')
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=10)
        
        # Style the plot
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
    else:
        cm = confusion_matrix(y_true, y_pred)
        
        # Use a better colormap
        plt.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
        
        # Add colorbar with better styling
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center", fontsize=12, fontweight='bold',
                        color="white" if cm[i, j] > thresh else "black",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_pred_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_residuals(y_true, y_pred, output_dir, model_name="model", task_type="regression"):
    if task_type == "regression":
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.7, s=40, c='#2E8B57', edgecolors='black', linewidth=0.5)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=11, fontweight='bold')
        axes[0, 0].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Residuals vs Predicted', fontsize=12, fontweight='bold', pad=15)
        axes[0, 0].grid(True, alpha=0.2, linestyle='--')
        axes[0, 0].set_facecolor('#f8f9fa')
        axes[0, 0].spines['top'].set_visible(False)
        axes[0, 0].spines['right'].set_visible(False)
        
        # Residuals Distribution
        axes[0, 1].hist(residuals, bins=30, alpha=0.8, edgecolor='black', linewidth=0.5, 
                        color='#4ECDC4', density=True)
        axes[0, 1].set_xlabel('Residuals', fontsize=11, fontweight='bold')
        axes[0, 1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold', pad=15)
        axes[0, 1].grid(True, alpha=0.2, linestyle='--')
        axes[0, 1].set_facecolor('#f8f9fa')
        axes[0, 1].spines['top'].set_visible(False)
        axes[0, 1].spines['right'].set_visible(False)
        
        # Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normality Test)', fontsize=12, fontweight='bold', pad=15)
        axes[1, 0].grid(True, alpha=0.2, linestyle='--')
        axes[1, 0].set_facecolor('#f8f9fa')
        axes[1, 0].spines['top'].set_visible(False)
        axes[1, 0].spines['right'].set_visible(False)
        
        # Residuals vs Index
        axes[1, 1].plot(residuals, alpha=0.8, color='#FF6B6B', linewidth=1)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.8)
        axes[1, 1].set_xlabel('Sample Index', fontsize=11, fontweight='bold')
        axes[1, 1].set_ylabel('Residuals', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Residuals vs Index', fontsize=12, fontweight='bold', pad=15)
        axes[1, 1].grid(True, alpha=0.2, linestyle='--')
        axes[1, 1].set_facecolor('#f8f9fa')
        axes[1, 1].spines['top'].set_visible(False)
        axes[1, 1].spines['right'].set_visible(False)
    
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_residuals.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    if task_type == "classification":
        report = classification_report(y_true, y_pred, output_dict=True)

        plt.figure(figsize=(12, 8))
        classes = list(report.keys())[:-3]
        precision = [report[cls]['precision'] for cls in classes]
        recall = [report[cls]['recall'] for cls in classes]
        f1 = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        # Use better colors
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Green, Red, Teal
        
        bars1 = plt.bar(x - width, precision, width, label='Precision', alpha=0.8, 
                        color=colors[0], edgecolor='black', linewidth=0.5)
        bars2 = plt.bar(x, recall, width, label='Recall', alpha=0.8, 
                        color=colors[1], edgecolor='black', linewidth=0.5)
        bars3 = plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8, 
                        color=colors[2], edgecolor='black', linewidth=0.5)
        
        plt.xlabel('Classes', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Classification Report', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(x, classes, fontsize=10)
        plt.legend(fontsize=10, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.2, linestyle='--')
        
        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.2f}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                        facecolor="white", alpha=0.8, edgecolor='gray'))
        
        # Style the plot
        plt.gca().set_facecolor('#f8f9fa')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{model_name}_classification_report.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_model_comparison(results_df, metric, output_dir, task_type="regression"):
    plt.figure(figsize=(14, 8))
    
    results_df = results_df.sort_values(f'Mean_{metric}', ascending=False)
    
    # Create gradient colors based on performance
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    
    bars = plt.bar(range(len(results_df)), results_df[f'Mean_{metric}'], 
                   yerr=results_df[f'Std_{metric}'], capsize=8, alpha=0.8, 
                   color=colors, edgecolor='black', linewidth=0.5)
    
    plt.xlabel('Models', fontsize=12, fontweight='bold')
    plt.ylabel(f'{metric.upper()} Score', fontsize=12, fontweight='bold')
    plt.title(f'Model Performance Comparison - {metric.upper()}', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(range(len(results_df)), results_df['Model'], rotation=45, ha='right', fontsize=10)
    plt.grid(True, alpha=0.2, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, results_df[f'Mean_{metric}'], results_df[f'Std_{metric}'])):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.002,
                f'{mean_val:.3f}±{std_val:.3f}', ha='center', va='bottom', 
                fontweight='bold', fontsize=9, bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8, edgecolor='gray'))
    
    # Add subtle background
    plt.gca().set_facecolor('#f8f9fa')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'model_comparison_{metric.lower()}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_comprehensive_model_comparison(results_df, output_dir, task_type="regression"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Performance Analysis', fontsize=18, fontweight='bold', y=0.95)
    
    if task_type == "regression":
        metrics = ['R2', 'MAE', 'MSE']
        metric_cols = ['Mean_R2', 'Mean_MAE', 'Mean_MSE']
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Green, Red, Teal
    else:  # classification
        metrics = ['Accuracy', 'F1', 'Precision']
        metric_cols = ['Mean_Accuracy', 'Mean_F1', 'Mean_Precision']
        colors = ['#2E8B57', '#FF6B6B', '#4ECDC4']  # Green, Red, Teal
    
    for i, (metric, col, color) in enumerate(zip(metrics, metric_cols, colors)):
        row, col_idx = i // 2, i % 2
        ax = axes[row, col_idx]
        
        if col in results_df.columns:
            sorted_df = results_df.sort_values(col, ascending=(metric == 'R2' if task_type == "regression" else metric == 'Accuracy'))
            
            # Create gradient colors
            bar_colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_df)))
            bars = ax.bar(range(len(sorted_df)), sorted_df[col], alpha=0.8, 
                         color=bar_colors, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'{metric} Performance Comparison', fontsize=12, fontweight='bold', pad=15)
            ax.set_ylabel(f'{metric} Score', fontsize=10, fontweight='bold')
            ax.set_xticks(range(len(sorted_df)))
            ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right', fontsize=9)
            ax.grid(True, alpha=0.2, linestyle='--')
            
            # Add value labels
            for bar, val in zip(bars, sorted_df[col]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{val:.3f}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                       facecolor="white", alpha=0.8, edgecolor='gray'))
            
            # Style the subplot
            ax.set_facecolor('#f8f9fa')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        else:
            ax.text(0.5, 0.5, f'{metric} data not available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10, style='italic')
            ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    
    # Time comparison
    ax = axes[1, 1]
    if 'Mean_Time' in results_df.columns:
        sorted_df = results_df.sort_values('Mean_Time', ascending=True)
        bar_colors = plt.cm.autumn(np.linspace(0, 1, len(sorted_df)))
        bars = ax.bar(range(len(sorted_df)), sorted_df['Mean_Time'], alpha=0.8, 
                     color=bar_colors, edgecolor='black', linewidth=0.5)
        
        ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel('Time (seconds)', fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(sorted_df)))
        ax.set_xticklabels(sorted_df['Model'], rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        
        for bar, val in zip(bars, sorted_df['Mean_Time']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{val:.1f}s', ha='center', va='bottom', 
                   fontweight='bold', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", 
                   facecolor="white", alpha=0.8, edgecolor='gray'))
        
        # Style the subplot
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    else:
        ax.text(0.5, 0.5, 'Time data not available', ha='center', va='center', 
               transform=ax.transAxes, fontsize=10, style='italic')
        ax.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_model_performance_radar(results_df, output_dir, task_type="regression"):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='polar')
    
    if task_type == "regression":
        metrics = ['R2', 'MAE', 'MSE', 'Time']
        metric_cols = ['Mean_R2', 'Mean_MAE', 'Mean_MSE', 'Mean_Time']
    else:  # classification
        metrics = ['Accuracy', 'F1', 'Precision', 'Time']
        metric_cols = ['Mean_Accuracy', 'Mean_F1', 'Mean_Precision', 'Mean_Time']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    # Use a better color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_df)))
    
    # Add grid lines for better readability
    ax.set_rticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_rlabel_position(0)
    ax.grid(True, alpha=0.3)
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        values = []
        for col in metric_cols:
            if col not in results_df.columns:
                values.append(0)
                continue
                
            if col == 'Mean_Time':
                # Normalize time (lower is better)
                max_time = results_df[col].max()
                values.append(1 - (row[col] / max_time))
            elif col == 'Mean_MAE' or col == 'Mean_MSE':
                # Lower is better for MAE/MSE
                max_val = results_df[col].max()
                values.append(1 - (row[col] / max_val))
            else:
                # Higher is better for R2/Accuracy/F1/Precision
                max_val = results_df[col].max()
                values.append(row[col] / max_val)
        
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=3, label=row['Model'], color=colors[i], markersize=6)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_title('Model Performance Radar Chart', size=18, fontweight='bold', y=1.15, pad=20)
    
    # Improve legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.0), 
                      fontsize=10, frameon=True, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Add subtle background
    ax.set_facecolor('#f8f9fa')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_target_distribution(y, output_dir, task_type="regression"):
    plt.figure(figsize=(12, 8))
    
    if task_type == "regression":
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Target Variable Distribution Analysis', fontsize=14)
        
        axes[0, 0].hist(y, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Target Values')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Histogram')
        axes[0, 0].grid(True, alpha=0.3)
        
        stats.probplot(y, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].boxplot(y)
        axes[1, 0].set_title('Box Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(y, alpha=0.6)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Target Values')
        axes[1, 1].set_title('Target vs Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'target_distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    if task_type == "classification":
        plt.figure(figsize=(10, 6))
        value_counts = y.value_counts()
        plt.bar(value_counts.index, value_counts.values, alpha=0.7)
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.title('Class Distribution')
        plt.grid(True, alpha=0.3)
        
        for i, v in enumerate(value_counts.values):
            plt.text(i, v + 0.01 * max(value_counts.values), str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_feature_quality_analysis(X, y, output_dir, task_type="regression"):
    plt.figure(figsize=(15, 10))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Feature Quality Analysis', fontsize=14)
    
    # Feature correlations with target
    correlations = []
    feature_names = []
    
    for col in X.columns:
        if X[col].dtype in ['int64', 'float64']:
            corr = np.corrcoef(X[col], y)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
                feature_names.append(col)
    
    # Top correlations
    top_corr_idx = np.argsort(correlations)[-10:]
    top_correlations = [correlations[i] for i in top_corr_idx]
    top_features = [feature_names[i] for i in top_corr_idx]
    
    axes[0, 0].barh(range(len(top_correlations)), top_correlations)
    axes[0, 0].set_yticks(range(len(top_correlations)))
    axes[0, 0].set_yticklabels(top_features)
    axes[0, 0].set_xlabel('|Correlation|')
    axes[0, 0].set_title('Top 10 Feature Correlations')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Feature variance
    variances = X.var().sort_values(ascending=False)
    top_var_features = variances.head(10)
    
    axes[0, 1].barh(range(len(top_var_features)), top_var_features.values)
    axes[0, 1].set_yticks(range(len(top_var_features)))
    axes[0, 1].set_yticklabels(top_var_features.index)
    axes[0, 1].set_xlabel('Variance')
    axes[0, 1].set_title('Top 10 Feature Variances')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Missing values
    missing_counts = X.isnull().sum()
    missing_pct = (missing_counts / len(X)) * 100
    missing_data = pd.DataFrame({'Feature': missing_counts.index, 'Missing_Pct': missing_pct.values})
    missing_data = missing_data.sort_values('Missing_Pct', ascending=False).head(10)
    
    axes[0, 2].barh(range(len(missing_data)), missing_data['Missing_Pct'])
    axes[0, 2].set_yticks(range(len(missing_data)))
    axes[0, 2].set_yticklabels(missing_data['Feature'])
    axes[0, 2].set_xlabel('Missing %')
    axes[0, 2].set_title('Top 10 Missing Values')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Feature distributions (sample)
    sample_features = X.columns[:6] if len(X.columns) >= 6 else X.columns
    
    for i, feature in enumerate(sample_features):
        row, col = (i + 3) // 3, (i + 3) % 3
        if row < 2:
            axes[row, col].hist(X[feature].dropna(), bins=20, alpha=0.7)
            axes[row, col].set_title(f'{feature} Distribution')
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_quality_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_three_way_ensemble_comparison(comparison_df, output_dir, task_type="classification"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ensemble Comparison: Individual vs Voting', fontsize=16, fontweight='bold')
    
    colors = {'Best_Individual': '#1f77b4', 'Voting_Ensemble': '#ff7f0e'}
    
    metrics = ['Accuracy', 'F1', 'Precision', 'Recall', 'Time']
    metric_labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Training Time (s)']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        row, col = i // 3, i % 3
        
        models = comparison_df['Model'].values
        values = comparison_df[f'Mean_{metric}'].values
        errors = comparison_df[f'Std_{metric}'].values if f'Std_{metric}' in comparison_df.columns else None
        
        if errors is not None:
            bars = axes[row, col].bar(models, values, yerr=errors, capsize=5, 
                                    color=[colors.get(m, '#666666') for m in models], alpha=0.7)
        else:
            bars = axes[row, col].bar(models, values, color=[colors.get(m, '#666666') for m in models], alpha=0.7)
        
        axes[row, col].set_title(f'{label} Comparison')
        axes[row, col].set_ylabel(label)
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 2].axis('off')
    summary_text = "Performance Summary:\n\n"
    for _, row in comparison_df.iterrows():
        model = row['Model']
        acc = row['Mean_Accuracy']
        f1 = row['Mean_F1']
        time_val = row['Mean_Time']
        summary_text += f"{model}:\n"
        summary_text += f"  Accuracy: {acc:.3f}\n"
        summary_text += f"  F1: {f1:.3f}\n"
        summary_text += f"  Time: {time_val:.2f}s\n\n"
    
    axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_vs_time_tradeoff(results_df, output_dir, task_type="classification"):
    individual_models = results_df[~results_df['Model'].str.contains('Ensemble')]
    voting_ensemble = results_df[results_df['Model'].str.contains('Voting')]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    if len(individual_models) > 0:
        ax1.scatter(individual_models['Mean_Time'], individual_models['Mean_Accuracy'], 
                   s=100, alpha=0.7, label='Individual Models', color='blue')
        for _, row in individual_models.iterrows():
            ax1.annotate(row['Model'], (row['Mean_Time'], row['Mean_Accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    if len(voting_ensemble) > 0:
        ax1.scatter(voting_ensemble['Mean_Time'], voting_ensemble['Mean_Accuracy'], 
                   s=150, alpha=0.8, label='Voting Ensemble', color='orange', marker='s')
        for _, row in voting_ensemble.iterrows():
            ax1.annotate(row['Model'], (row['Mean_Time'], row['Mean_Accuracy']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy vs Training Time Trade-off')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if len(individual_models) > 0:
        ax2.scatter(individual_models['Mean_Time'], individual_models['Mean_F1'], 
                   s=100, alpha=0.7, label='Individual Models', color='blue')
        for _, row in individual_models.iterrows():
            ax2.annotate(row['Model'], (row['Mean_Time'], row['Mean_F1']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    if len(voting_ensemble) > 0:
        ax2.scatter(voting_ensemble['Mean_Time'], voting_ensemble['Mean_F1'], 
                   s=150, alpha=0.8, label='Voting Ensemble', color='orange', marker='s')
        for _, row in voting_ensemble.iterrows():
            ax2.annotate(row['Model'], (row['Mean_Time'], row['Mean_F1']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
    
    ax2.set_xlabel('Training Time (seconds)')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score vs Training Time Trade-off')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_vs_time_tradeoff.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_ensemble_advantage_analysis(individual_models, ensemble_models, output_dir, task_type="classification"):
    if len(individual_models) == 0 or len(ensemble_models) == 0:
        return
    
    best_individual = individual_models.iloc[0]
    
    ensemble_advantages = []
    for _, ensemble in ensemble_models.iterrows():
        acc_advantage = ensemble['Mean_Accuracy'] - best_individual['Mean_Accuracy']
        f1_advantage = ensemble['Mean_F1'] - best_individual['Mean_F1']
        time_penalty = ensemble['Mean_Time'] - best_individual['Mean_Time']
        
        ensemble_advantages.append({
            'Model': ensemble['Model'],
            'Accuracy_Advantage': acc_advantage,
            'F1_Advantage': f1_advantage,
            'Time_Penalty': time_penalty
        })
    
    advantages_df = pd.DataFrame(ensemble_advantages)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Ensemble Advantage Analysis', fontsize=16, fontweight='bold')
    
    metrics = ['Accuracy_Advantage', 'F1_Advantage', 'Time_Penalty']
    titles = ['Accuracy Advantage', 'F1 Advantage', 'Time Penalty']
    colors = ['green', 'green', 'red']
    
    for i, (metric, title, color) in enumerate(zip(metrics, titles, colors)):
        row, col = i // 2, i % 2
        
        values = advantages_df[metric].values
        bars = axes[row, col].bar(advantages_df['Model'], values, 
                                 color=[color if x > 0 else 'red' for x in values], alpha=0.7)
        axes[row, col].set_title(title)
        axes[row, col].set_ylabel('Improvement' if 'Advantage' in metric else 'Additional Time (s)')
        axes[row, col].tick_params(axis='x', rotation=45)
        axes[row, col].axhline(y=0, color='black', linestyle='--')
        axes[row, col].grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                               f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    
    axes[1, 1].axis('off')
    summary_text = "Ensemble Advantage Summary:\n\n"
    summary_text += f"Best Individual Model: {best_individual['Model']}\n"
    summary_text += f"Best Individual Accuracy: {best_individual['Mean_Accuracy']:.3f}\n"
    summary_text += f"Best Individual F1: {best_individual['Mean_F1']:.3f}\n"
    summary_text += f"Best Individual Time: {best_individual['Mean_Time']:.2f}s\n\n"
    
    for _, ensemble in ensemble_models.iterrows():
        summary_text += f"{ensemble['Model']}:\n"
        summary_text += f"  Accuracy: {ensemble['Mean_Accuracy']:.3f} "
        summary_text += f"({'↑' if ensemble['Mean_Accuracy'] > best_individual['Mean_Accuracy'] else '↓'})\n"
        summary_text += f"  F1: {ensemble['Mean_F1']:.3f} "
        summary_text += f"({'↑' if ensemble['Mean_F1'] > best_individual['Mean_F1'] else '↓'})\n"
        summary_text += f"  Time: {ensemble['Mean_Time']:.2f}s\n\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_advantage_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Legacy functions for backward compatibility
def plot_final_summary(results_df, output_dir, metric="r2", task_type="regression"):
    return plot_comprehensive_model_comparison(results_df, output_dir, task_type)

def plot_learning_curves_comparison(results_df, output_dir, task_type="regression"):
    """Create detailed fold-wise performance comparison with multiple metrics"""
    
    # This function will be enhanced to show learning curves
    # For now, we'll create a comprehensive fold analysis
    pass

def plot_fold_performance_comparison(results, output_dir, task_type="regression"):
    """Create comprehensive fold-wise performance comparison plots"""
    
    # Extract all fold-wise results
    fold_data = []
    for model_name, result in results.items():
        if task_type == "regression":
            metrics = ['r2_scores', 'mae_scores', 'mse_scores']
            metric_names = ['R²', 'MAE', 'MSE']
        else:
            metrics = ['accuracy_scores', 'f1_scores', 'precision_scores']
            metric_names = ['Accuracy', 'F1', 'Precision']
        
        for metric, metric_name in zip(metrics, metric_names):
            if metric in result and len(result[metric]) > 0:
                for fold, score in enumerate(result[metric], 1):
                    fold_data.append({
                        'Model': model_name,
                        'Fold': fold,
                        'Score': score,
                        'Metric': metric_name
                    })
    
    if not fold_data:
        return
    
    df = pd.DataFrame(fold_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle('Comprehensive Fold-wise Performance Analysis', fontsize=18, fontweight='bold', y=0.95)
    
    # Get unique models and metrics
    models = df['Model'].unique()
    metrics = df['Metric'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    # 1. Box plot for each metric
    for i, metric in enumerate(metrics):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        metric_data = df[df['Metric'] == metric]
        box_data = [metric_data[metric_data['Model'] == model]['Score'].values for model in models]
        
        bp = ax.boxplot(box_data, labels=models, patch_artist=True,
                       medianprops=dict(color='black', linewidth=2),
                       flierprops=dict(marker='o', markerfacecolor='red', markersize=4))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{metric} Performance Across Folds', fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel(f'{metric} Score', fontsize=10, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(True, alpha=0.2, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 2. Heatmap of fold performance
    ax = axes[1, 1]
    pivot_data = df.pivot_table(values='Score', index='Model', columns='Fold', aggfunc='mean')
    
    im = ax.imshow(pivot_data.values, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(pivot_data.columns)))
    ax.set_xticklabels([f'Fold {i}' for i in pivot_data.columns], fontsize=9)
    ax.set_yticks(range(len(pivot_data.index)))
    ax.set_yticklabels(pivot_data.index, fontsize=9)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax.text(j, i, f'{pivot_data.iloc[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')
    
    ax.set_title('Fold-wise Performance Heatmap', fontsize=12, fontweight='bold', pad=15)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fold_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed summary
    summary_stats = df.groupby(['Model', 'Metric'])['Score'].agg(['mean', 'std', 'min', 'max']).round(4)
    summary_stats.columns = ['Mean', 'Std', 'Min', 'Max']
    summary_stats = summary_stats.sort_values(['Metric', 'Mean'], ascending=[True, False])
    
    # Save detailed summary
    summary_stats.to_csv(os.path.join(output_dir, 'detailed_fold_summary.csv'))
    
    return summary_stats

def plot_statistical_significance_matrix(results_df, output_dir, task_type="regression"):
    pass

def plot_model_complexity_analysis(results_df, output_dir, task_type="regression"):
    pass

def plot_cv_stability_analysis(results, output_dir, task_type="regression"):
    # Extract fold-wise results
    fold_data = []
    for model_name, result in results.items():
        if len(result.get('r2_scores' if task_type == "regression" else 'accuracy_scores', [])) > 0:
            if task_type == "regression":
                scores = result['r2_scores']
                metric_name = 'R²'
            else:
                scores = result['accuracy_scores']
                metric_name = 'Accuracy'
            
            for fold, score in enumerate(scores, 1):
                fold_data.append({
                    'Model': model_name,
                    'Fold': fold,
                    'Score': score,
                    'Metric': metric_name
                })
    
    if not fold_data:
        return
    
    df = pd.DataFrame(fold_data)
    models = df['Model'].unique()
    
    # Create color mapping with special treatment for ICL models
    colors = {
        'cornflowerblue': ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet'],
        'royalblue': ['RandomForest', 'HistGradientBoosting', 'XGBoost', 'LightGBM'],
        'lightgray': ['MLP', 'SVR', 'KNeighbors', 'DecisionTree'],
        'gray': ['TabPFNICL'],
        'black': ['GPT2ICL', 'HybridEnsemble']
    }
    
    # Assign colors to models
    model_colors = {}
    for model in models:
        assigned = False
        for color, model_list in colors.items():
            for pattern in model_list:
                if pattern in model:
                    model_colors[model] = color
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            model_colors[model] = 'cornflowerblue'  # default
    
    plt.style.use('seaborn-v0_8-white')
    plt.figure(figsize=(8, 6))
    
    # Plot each model
    for model in models:
        model_data = df[df['Model'] == model]
        x = model_data['Fold'].values
        y = model_data['Score'].values
        color = model_colors[model]
        
        # Special styling for ICL models
        if 'GPT2ICL' in model or 'HybridEnsemble' in model:
            plt.plot(x, y, label=model, color=color, linewidth=3, marker='*', markersize=12)
        else:
            plt.plot(x, y, label=model, color=color, linewidth=2, marker='o', markersize=6)
    
    # Axis limits and labels
    metric = df['Metric'].iloc[0]
    plt.ylim(0, 1.0)
    plt.xlabel('Fold Number')
    plt.ylabel(f'{metric} Performance')
    plt.title(f'Model Performance Across Folds', fontsize=14, fontweight='bold')
    
    # Legend
    plt.legend(title='Models', loc='center right', bbox_to_anchor=(1.35, 0.5))
    
    # Remove top/right spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cv_stability_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    summary_stats = df.groupby('Model')['Score'].agg(['mean', 'std', 'min', 'max']).round(4)
    summary_stats.columns = ['Mean', 'Std', 'Min', 'Max']
    summary_stats = summary_stats.sort_values('Mean', ascending=False)
    
    # Save summary to CSV
    summary_stats.to_csv(os.path.join(output_dir, 'cv_fold_summary.csv'))
    
    return summary_stats

def plot_organ_pca_analysis(organ_pca_selector, output_dir):
    pass

def plot_prediction_error_analysis(y_true, y_pred, model_name, output_dir, task_type="regression"):
    return plot_residuals(y_true, y_pred, output_dir, model_name, task_type)

def plot_feature_importance_analysis(model, feature_names, output_dir, model_name, task_type="regression"):
    pass

def plot_learning_curves_detailed(X, y, model, output_dir, model_name, task_type="regression"):
    pass

def plot_icl_benchmark_analysis(results_df, output_dir, task_type="regression"):
    pass

def plot_icl_detailed_comparison(results_df, output_dir, task_type="regression"):
    pass

def plot_model_category_analysis(results_df, output_dir, task_type="regression"):
    pass

def plot_icl_advantage_analysis(results_df, output_dir, task_type="regression"):
    pass

def plot_ensemble_analysis(results_df, ensemble_models, output_dir, task_type="regression"):
    ensemble_df = results_df[results_df['Model'].str.contains('Ensemble')]
    individual_df = results_df[~results_df['Model'].str.contains('Ensemble')]
    
    if len(ensemble_df) == 0:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ensemble Analysis', fontsize=16, fontweight='bold')
    
    metric_col = 'Mean_R2' if task_type == "regression" else 'Mean_Accuracy'
    metric_name = 'R² Score' if task_type == "regression" else 'Accuracy Score'
    
    # Performance comparison
    ax1 = axes[0, 0]
    all_models = pd.concat([individual_df, ensemble_df])
    colors = ['lightblue' if 'Ensemble' not in model else 'orange' for model in all_models['Model']]
    
    if metric_col in all_models.columns:
        bars = ax1.bar(range(len(all_models)), all_models[metric_col], color=colors, alpha=0.7)
        ax1.set_title(f'{metric_name} Comparison')
        ax1.set_ylabel(metric_name)
        ax1.set_xticks(range(len(all_models)))
        ax1.set_xticklabels(all_models['Model'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, all_models[metric_col]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    else:
        ax1.text(0.5, 0.5, f'{metric_name} data not available', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title(f'{metric_name} Comparison')
    
    # Time vs performance trade-off
    ax2 = axes[0, 1]
    if 'Mean_Time' in individual_df.columns and metric_col in individual_df.columns:
        ax2.scatter(individual_df['Mean_Time'], individual_df[metric_col], 
                    c='lightblue', s=100, alpha=0.7, label='Individual Models')
        ax2.scatter(ensemble_df['Mean_Time'], ensemble_df[metric_col], 
                    c='orange', s=150, alpha=0.8, marker='s', label='Ensemble Models')
        
        for _, row in individual_df.iterrows():
            ax2.annotate(row['Model'], (row['Mean_Time'], row[metric_col]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        for _, row in ensemble_df.iterrows():
            ax2.annotate(row['Model'], (row['Mean_Time'], row[metric_col]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, fontweight='bold')
        
        ax2.set_xlabel('Training Time (s)')
        ax2.set_ylabel(metric_name)
        ax2.set_title('Performance vs Time Trade-off')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Time/Performance data not available', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Performance vs Time Trade-off')
    
    # Ensemble advantage analysis
    ax3 = axes[1, 0]
    if metric_col in individual_df.columns and metric_col in ensemble_df.columns:
        best_individual = individual_df.loc[individual_df[metric_col].idxmax()]
        ensemble_df['Improvement'] = ensemble_df[metric_col] - best_individual[metric_col]
        colors = ['green' if x > 0 else 'red' for x in ensemble_df['Improvement']]
        bars = ax3.bar(ensemble_df['Model'], ensemble_df['Improvement'], color=colors, alpha=0.7)
        ax3.set_title(f'Improvement over Best Individual Model\n({best_individual["Model"]}: {best_individual[metric_col]:.3f})')
        ax3.set_ylabel(f'{metric_name} Improvement')
        ax3.axhline(y=0, color='black', linestyle='--')
        ax3.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, ensemble_df['Improvement']):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.001 if val > 0 else -0.001),
                    f'{val:.3f}', ha='center', va='bottom' if val > 0 else 'top', fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'Improvement data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Improvement over Best Individual Model')
    
    # Error analysis
    ax4 = axes[1, 1]
    if task_type == "regression":
        metrics = ['Mean_R2', 'Mean_MAE', 'Mean_MSE']
        metric_names = ['R² Score', 'MAE', 'MSE']
    else:
        metrics = ['Mean_Accuracy', 'Mean_F1', 'Mean_Precision']
        metric_names = ['Accuracy', 'F1', 'Precision']
    
    available_metrics = [m for m in metrics if m in individual_df.columns and m in ensemble_df.columns]
    if available_metrics:
        x = np.arange(len(available_metrics))
        width = 0.35
        
        individual_avg = individual_df[available_metrics].mean()
        ensemble_avg = ensemble_df[available_metrics].mean()
        
        ax4.bar(x - width/2, individual_avg, width, label='Individual Models (Avg)', 
                color='lightblue', alpha=0.7)
        ax4.bar(x + width/2, ensemble_avg, width, label='Ensemble Models (Avg)', 
                color='orange', alpha=0.7)
        
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Score')
        ax4.set_title('Average Performance by Metric')
        ax4.set_xticks(x)
        ax4.set_xticklabels([metric_names[metrics.index(m)] for m in available_metrics])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Metric data not available', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Average Performance by Metric')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary text file
    summary_path = os.path.join(output_dir, 'ensemble_analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("ENSEMBLE ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        if metric_col in individual_df.columns:
            best_individual = individual_df.loc[individual_df[metric_col].idxmax()]
            f.write(f"Best Individual Model: {best_individual['Model']}\n")
            f.write(f"Best Individual {metric_name}: {best_individual[metric_col]:.3f}\n")
            if 'Mean_Time' in best_individual:
                f.write(f"Best Individual Time: {best_individual['Mean_Time']:.2f}s\n\n")
        
        f.write("ENSEMBLE MODELS:\n")
        f.write("-" * 20 + "\n")
        for _, ensemble in ensemble_df.iterrows():
            f.write(f"{ensemble['Model']}:\n")
            if metric_col in ensemble:
                f.write(f"  {metric_name}: {ensemble[metric_col]:.3f}\n")
            if 'Mean_Time' in ensemble:
                f.write(f"  Time: {ensemble['Mean_Time']:.2f}s\n")
            if 'Improvement' in ensemble:
                f.write(f"  Improvement: {ensemble['Improvement']:.3f} ({ensemble['Improvement_Percent']:.1f}%)\n")
            f.write("\n")


def plot_ensemble_details(ensemble_models, output_dir, task_type="regression"):
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ensemble Details Analysis', fontsize=16, fontweight='bold')
    
    # Performance breakdown
    ax1 = axes[0, 0]
    ensemble_names = list(ensemble_models.keys())
    scores = [ensemble_models[name]['score'] for name in ensemble_names]
    times = [ensemble_models[name]['time'] for name in ensemble_names]
    
    bars = ax1.bar(ensemble_names, scores, color='orange', alpha=0.7)
    ax1.set_title('Ensemble Performance')
    ax1.set_ylabel('R² Score')
    ax1.set_xticklabels(ensemble_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time analysis
    ax2 = axes[0, 1]
    bars = ax2.bar(ensemble_names, times, color='lightgreen', alpha=0.7)
    ax2.set_title('Ensemble Training Time')
    ax2.set_ylabel('Time (s)')
    ax2.set_xticklabels(ensemble_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Performance efficiency
    ax3 = axes[1, 0]
    efficiency = [score/time if time > 0 else 0 for score, time in zip(scores, times)]
    
    bars = ax3.bar(ensemble_names, efficiency, color='purple', alpha=0.7)
    ax3.set_title('Performance per Second (Efficiency)')
    ax3.set_ylabel('R² Score / Second')
    ax3.set_xticklabels(ensemble_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Ensemble composition
    ax4 = axes[1, 1]
    
    composition_info = {}
    for ensemble_name in ensemble_names:
        if 'Hybrid' in ensemble_name:
            composition_info[ensemble_name] = {
                'Traditional Models': 'Top 2-3 traditional ML models',
                'ICL Models': 'TabPFNICL only',
                'Meta-learner': 'Ridge Regression',
                'Method': 'Stacking with CV'
            }
        else:
            composition_info[ensemble_name] = {
                'Traditional Models': 'Unknown',
                'ICL Models': 'Unknown', 
                'Meta-learner': 'Unknown',
                'Method': 'Unknown'
            }
    
    ax4.text(0.1, 0.8, 'Ensemble Composition:', fontsize=12, fontweight='bold', transform=ax4.transAxes)
    
    y_offset = 0.7
    for i, (ensemble_name, comp) in enumerate(composition_info.items()):
        ax4.text(0.1, y_offset - i*0.15, f'{ensemble_name}:', fontsize=10, fontweight='bold', transform=ax4.transAxes)
        ax4.text(0.15, y_offset - i*0.15 - 0.05, f'• {comp["Traditional Models"]}', fontsize=9, transform=ax4.transAxes)
        ax4.text(0.15, y_offset - i*0.15 - 0.08, f'• {comp["ICL Models"]}', fontsize=9, transform=ax4.transAxes)
        ax4.text(0.15, y_offset - i*0.15 - 0.11, f'• Meta-learner: {comp["Meta-learner"]}', fontsize=9, transform=ax4.transAxes)
    
    ax4.set_title('Ensemble Model Composition')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ensemble_details.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Detailed summary
    details_path = os.path.join(output_dir, 'ensemble_details_summary.txt')
    with open(details_path, 'w') as f:
        f.write("ENSEMBLE DETAILS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for name, metrics in ensemble_models.items():
            f.write(f"{name}:\n")
            f.write(f"  R² Score: {metrics['score']:.3f}\n")
            f.write(f"  Training Time: {metrics['time']:.2f}s\n")
            f.write(f"  Efficiency: {metrics['score']/metrics['time']:.4f} R²/s\n")
            
            if name in composition_info:
                comp = composition_info[name]
                f.write(f"  Composition:\n")
                f.write(f"    Traditional Models: {comp['Traditional Models']}\n")
                f.write(f"    ICL Models: {comp['ICL Models']}\n")
                f.write(f"    Meta-learner: {comp['Meta-learner']}\n")
                f.write(f"    Method: {comp['Method']}\n")
            f.write("\n")
        
        best_ensemble = max(ensemble_models.items(), key=lambda x: x[1]['score'])
        f.write(f"Best Ensemble: {best_ensemble[0]}\n")
        f.write(f"Best Score: {best_ensemble[1]['score']:.3f}\n")


def plot_confusion_matrix(y_true, y_pred, output_dir, model_name="model", task_type="classification"):
    return plot_pred_vs_true(y_true, y_pred, output_dir, model_name, task_type)

def plot_classification_report(y_true, y_pred, output_dir, model_name="model", task_type="classification"):
    return plot_residuals(y_true, y_pred, output_dir, model_name, task_type)

def plot_roc_curves(y_true, y_prob, output_dir, model_name="model", task_type="classification"):
    pass

def plot_class_distribution(y_true, output_dir, model_name="model", task_type="classification"):
    return plot_target_distribution(y_true, output_dir, task_type)

def plot_multiclass_analysis(y_true, y_pred, output_dir, model_name="model", task_type="classification", y_prob=None):
    pass



