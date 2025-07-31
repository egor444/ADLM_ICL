
import logging
import random
import re
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed

from data_extraction.data_manager import DataManager
from feature_engineering.featureselection import FeatureSelector
from models.gpt2_regressor import InContextRegressionDataset, GPT2Regressor
from tabpfn import TabPFNRegressor
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# CONFIGURATION for sampling
k_original = 30   # Neighbors from original training set
k_inverse = 30    # Neighbors from inverse density set
total_k = k_original + k_inverse

def inverse_density_resample(X, y, bins=10):
    """
    Oversample rare label regions, undersample frequent ones
    """
    y_binned = pd.qcut(y, q=bins, duplicates='drop')
    counts = y_binned.value_counts()
    inverse_weights = 1.0 / counts[y_binned].values
    probs = inverse_weights / inverse_weights.sum()
    
    idx = np.arange(len(y))
    sampled_idx = np.random.choice(idx, size=len(y), replace=True, p=probs)

    return X.iloc[sampled_idx].reset_index(drop=True), y.iloc[sampled_idx].reset_index(drop=True)

def retrieve_neighbors(x_query, X_train, k):
    """
    Retrieve k nearest neighbors based on cosine similarity
    """

    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X_train)
    distances, indices = nn.kneighbors(x_query)
    return indices.flatten()
    
def plot_predictions(y_true, y_pred, title, filename):
    """
    Plot predictions vs actual values
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], color='red', linestyle='--')
    plt.title(title)
    plt.xlabel('Actual Age')
    plt.ylabel('Predicted Age')
    plt.xlim(y_true.min(), y_true.max())
    plt.ylim(y_true.min(), y_true.max())
    plt.grid()
    plt.savefig(filename)
    plt.close()

### Main Functions ###

def run_cross_validation(model_type, dm, fold_count=5):
    """
    Run cross-validation for the given model and data manager
    """
    if model_type == 'tabpfn':
        model_t = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42)

    fold_preds_t = []  # List to store predictions for each fold
    fold_preds_g = []
    for i in range(fold_count):
        logging.info(f"Testing fold {i+1}/{fold_count}...")
        train_set = dm.get_fold_data_set([j for j in range(fold_count) if j != i])
        test_set = dm.get_fold_data_set([i])
        X_train = train_set.drop(columns=['age']).values
        y_train = train_set['age'].values
        X_test = test_set.drop(columns=['age']).values
        y_test = test_set['age'].values
        logging.info(f"Fold {i} - Training set size: {len(X_train)}, Test set size: {len(X_test)}, Fitting {model_type}...")
        model_t.fit(X_train, y_train)
        preds_t = model_t.predict(X_test)

        fold_preds_t.append((y_test, preds_t))

    # save predictions for each fold
    fold_preds_t_df = pd.DataFrame({
        'Fold': [],
        'Actual': [],
        'Predicted': []
    })

    for i, (y_test, preds_t) in enumerate(fold_preds_t):
        fold_preds_t_df = fold_preds_t_df.append({
            'Fold': i,
            'Actual': y_test,
            'Predicted': preds_t
        }, ignore_index=True)

    fold_preds_g_df = pd.DataFrame({
        'Fold': [],
        'Actual': [],
        'Predicted': []
    })

    for i, (y_test, preds_g) in enumerate(fold_preds_g):
        fold_preds_g_df = fold_preds_g_df.append({
            'Fold': i,
            'Actual': y_test,
            'Predicted': preds_g
        }, ignore_index=True)
    fold_preds_t_df.to_csv('results/icl_results/tabpfn_fold_predictions.csv', index=False)
    fold_preds_g_df.to_csv('results/icl_results/gpt2_fold_predictions.csv', index=False)
    return fold_preds_t, fold_preds_g

def test_icl_with_localized_selection(dm):
    train_set = dm.get_train()
    test_set = dm.get_test()
    num_cores = 8
    
    X_train = train_set.drop(columns=['age'])
    y_train = train_set['age']
    X_test = test_set.drop(columns=['age'])
    y_test = test_set['age']

    test_sample_size = 200 # len(X_test)
    divide_size = 10
    # 10% to 100% of 60% of training set size + total training set size
    context_sizes = [20,50,80] + [int((len(X_train)*0.6) // divide_size * i) for i in range(1, divide_size + 1)] + [len(X_train)]


    logging.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    x_inv, y_inv = inverse_density_resample(X_train, y_train, bins=40)

    # pd to numpy
    X_train = X_train.values
    y_train = y_train.values
    X_test = X_test.values[:test_sample_size]
    y_test = y_test.values[:test_sample_size]

    x_inv = x_inv.values
    y_inv = y_inv.values

    preds_tabpfn = []
    trues_tabpfn = []
    maes_tabpfn = []
    mses_tabpfn = []
    r2s_tabpfn = []
    times = []

    #context_sizes = [len(X_train)]
    logging.info("Starting TabPFN regression with localized selection...")

    for context_size in context_sizes:
        preds_tabpfn.append(np.array([]))
        trues_tabpfn.append(np.array([]))
        logging.info(f"Context size: {context_size}/{len(X_train)} = {context_size*100/len(X_train):.2f}% of training set")
        if not context_size == len(X_train):
            nn_orig = NearestNeighbors(n_neighbors=context_size, metric='cosine')
            nn_inv = NearestNeighbors(n_neighbors=context_size, metric='cosine')
            nn_orig.fit(X_train)
            nn_inv.fit(x_inv)
        timestart = time.time()
        logging.info(f"Nearest neighbors fitted. Starting predictions on {len(X_test)} test samples...")
        # Fit TabPFN for this query
        if context_size == len(X_train):
            tabpfn = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42)
            full_context = np.vstack([X_train, x_inv])
            full_target = np.concatenate([y_train, y_inv])
            tabpfn.fit(full_context, full_target)
            preds = tabpfn.predict(X_test)
            y_true = y_test
        else:
            outs = Parallel(n_jobs=num_cores)(delayed(process_single_sample)(
                j, X_test, X_train, y_train, x_inv, y_inv, y_test, nn_orig, nn_inv
            ) for j in range(len(X_test)))
            preds = [out[1] for out in outs]
            y_true = [out[0] for out in outs]
        times.append(time.time() - timestart)
        logging.info(f"Predictions completed. Time taken: {((time.time() - timestart) / 60):.2f} minutes")

        mae = mean_absolute_error(y_true, preds)
        mse = mean_squared_error(y_true, preds)
        r2 = r2_score(y_true, preds)
        logging.info(f"Context size {context_size} - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

        preds_tabpfn[-1] = np.append(preds_tabpfn[-1], preds)
        trues_tabpfn[-1] = np.append(trues_tabpfn[-1], y_true)
        maes_tabpfn.append(mae) 
        mses_tabpfn.append(mse)
        r2s_tabpfn.append(r2)


    for i, context_size in enumerate(context_sizes):
        title = f'TabPFN Regression, Context Size: {context_size/len(X_train):.2f}% ({context_size} samples), R2: {r2s_tabpfn[i]:.2f}, MAE: {maes_tabpfn[i]:.2f}'
        plot_predictions(trues_tabpfn[i], preds_tabpfn[i], title, f'results/icl_results/knn_analysis_results/tabpfn_regression_predictions_{(context_size*100)//len(X_train)}.png')

    logging.info("TabPFN regression with localized selection completed.")

    # dataframe to save results with columns for every context size predictions and true values like preds_10, preds_20, ..., true
    result_df = pd.DataFrame({
        'true': [int(age) for age in trues_tabpfn[0]],
        **{f'preds_{context_size}': preds_tabpfn[i] for i, context_size in enumerate(context_sizes)}
    })
    result_df.to_csv('results/icl_results/knn_analysis_results/tabpfn_context_results.csv', index=False)

    # plot maes and time taken for each context size
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(context_sizes, maes_tabpfn, marker='o')
    plt.title('MAE vs Context Size')
    plt.xlabel('Context Size')
    plt.ylabel('Mean Absolute Error (MAE)')

    plt.subplot(1, 2, 2)
    plt.plot(context_sizes, times, marker='o')
    plt.title('Time Taken vs Context Size')
    plt.xlabel('Context Size')
    plt.ylabel('Time Taken (seconds)')
    plt.tight_layout()
    plt.savefig('results/icl_results/knn_analysis_results/tabpfn_regression_mae_time.png')
    plt.close()

def process_single_sample(j, X_test, X_train, y_train, x_inv, y_inv, y_test, nn_orig, nn_inv):
    x_query = X_test[j].reshape(1, -1)

    idx_orig = nn_orig.kneighbors(x_query, return_distance=False).flatten()
    idx_inv = nn_inv.kneighbors(x_query, return_distance=False).flatten()

    x_context = np.vstack([X_train[idx_orig], x_inv[idx_inv]])
    y_context = np.concatenate([y_train[idx_orig], y_inv[idx_inv]])

    model = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42 + j)
    model.fit(x_context, y_context)
    y_pred = model.predict(x_query)
    y_true = y_test[j]
    return (y_true, y_pred[0])

def test_icl_folds(dm):
    fold_count = 5
    fold_preds = [] # [[(true, pred), ...], ...]
    num_cores = 16 # multiprocessing.cpu_count()
    fold_preds_whole = []
    logging.info(f"Number of CPU cores available: {num_cores}")
    for i in range(fold_count):
        logging.info(f"Testing fold {i+1}/{fold_count}...")
        # use fold i as test set, others as train set
        train_set = dm.get_fold_data_set([j for j in range(fold_count) if j != i])
        test_set = dm.get_fold_data_set([i])
        X_train = train_set.drop(columns=['age'])
        y_train = train_set['age']
        X_test = test_set.drop(columns=['age'])
        y_test = test_set['age']

        x_inv, y_inv = inverse_density_resample(X_train, y_train, bins=10)
        x_inv, y_inv = x_inv.values, y_inv.values
        X_train, y_train, X_test, y_test = X_train.values, y_train.values, X_test.values, y_test.values
        logging.info(f"[FOLD {i}] FS and scaling complete, training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # nearest neighbors
        nn_orig = NearestNeighbors(n_neighbors=k_original, metric='cosine')
        nn_inv = NearestNeighbors(n_neighbors=k_inverse, metric='cosine')
        nn_orig.fit(X_train)
        nn_inv.fit(x_inv)

        preds = []
        logging.info(f"[FOLD {i}] Starting TabPFN regression with localized selection...")
        
        preds = Parallel(n_jobs=num_cores)(delayed(process_single_sample)(
            j, X_test, X_train, y_train, x_inv, y_inv, y_test, nn_orig, nn_inv
        ) for j in range(len(X_test)))
            
        fold_preds.append(preds)
        logging.info(f"[FOLD {i}] Completed TabPFN regression with localized selection.")
        whole_fold_tabpfn = TabPFNRegressor(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42 + i)
        newtrain = np.vstack([X_train, x_inv])
        newy = np.concatenate([y_train, y_inv])
        whole_fold_tabpfn.fit(newtrain, newy)
        y_pred_whole = whole_fold_tabpfn.predict(X_test)
        y_true_whole = y_test
        fold_preds_whole.append([(y_true_whole[j], y_pred_whole[j]) for j in range(len(y_true_whole))])

    # Save fold predictions
    fold_preds_df = pd.DataFrame({
        'Fold': [],
        'Actual': [],
        'Predicted NN': [],
        'Predicted whole': [],
        'MAE NN': [],
        'MAE whole': []
    })

    logging.info(f"nnpreds: {len(fold_preds)}, fold_preds_whole: {len(fold_preds_whole)}")
    logging.info(f"Fold predictions length: {len(fold_preds[0])}, Fold whole predictions length: {len(fold_preds_whole[0])}")
    
    for i, (preds, fold_pred_whole) in enumerate(zip(fold_preds, fold_preds_whole)):
        fold_data = {
            'Fold': [i] * len(preds),
            'Actual': [p[0] for p in preds],
            'Predicted NN': [p[1] for p in preds],
            'Predicted whole': [p[1] for p in fold_pred_whole],
            'MAE NN': [mean_absolute_error([p[0]], [p[1]]) for p in preds],
            'MAE whole': [mean_absolute_error([p[0]], [p[1]]) for p in fold_pred_whole]
        }
        fold_data = pd.DataFrame(fold_data)
        fold_preds_df = pd.concat([fold_preds_df, fold_data], ignore_index=True)
    
    fold_preds_df.to_csv('/vol/miltank/projects/practical_sose25/in_context_learning/regression/results/icl_results/knn_analysis_results/tabpfn_fold_predictions.csv', index=False)
    logging.info("All folds completed. Calculating overall metrics...")
    # Calculate overall MAE and R2 for each fold
    fold_maes_nn = []
    fold_r2s_nn = []
    fold_variances_nn = []
    fold_maes_whole = []
    fold_r2s_whole = []
    fold_variances_whole = []
    for i, (preds, fold_pred_whole) in enumerate(zip(fold_preds, fold_preds_whole)):
        y_true = np.array([p[0] for p in preds])
        y_pred = np.array([p[1] for p in preds])
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        variance = np.var(y_pred)
        fold_maes_nn.append(mae)
        fold_r2s_nn.append(r2)
        fold_variances_nn.append(variance)
        logging.info(f"[FOLD {i}] [Nearest Neighbor] MAE: {mae}, R2: {r2}, Variance: {variance}")
        y_pred_whole = np.array([p[1] for p in fold_pred_whole])
        mae_whole = mean_absolute_error(y_true, y_pred_whole)
        r2_whole = r2_score(y_true, y_pred_whole)
        variance_whole = np.var(y_pred_whole)
        fold_maes_whole.append(mae_whole)
        fold_r2s_whole.append(r2_whole)
        fold_variances_whole.append(variance_whole)
        logging.info(f"[FOLD {i}] [Whole] MAE: {mae_whole}, R2: {r2_whole}, Variance: {variance_whole}")

    # Save overall metrics
    overall_metrics_df = pd.DataFrame({
        'Fold': list(range(fold_count)),
        'MAE NN': fold_maes_nn,
        'R2 NN': fold_r2s_nn,
        'Variance NN': fold_variances_nn,
        'MAE whole': fold_maes_whole,
        'R2 whole': fold_r2s_whole,
        'Variance whole': fold_variances_whole
    })
    overall_metrics_df.to_csv('results/icl_results/tabpfn_fold_overall_metrics.csv', index=False)
    logging.info("Overall metrics saved.")
    

def plot_fold_metrics():
    """
    Plot the fold metrics from the CSV file
    """
    df = pd.read_csv('results/icl_results/tabpfn_fold_predictions.csv')
    fold_preds = df.groupby('Fold').apply(lambda x: list(zip(x['Actual'], x['Predicted NN']))).tolist()
    y_preds_whole = df["Predicted whole"].tolist()
    y_trues = np.array([p[0] for preds in fold_preds for p in preds])
    y_preds = np.array([p[1] for preds in fold_preds for p in preds])
    title = f'TabPFN Regression Predictions vs Actual across all folds NN, r2: {r2_score(y_trues, y_preds)}, MAE: {mean_absolute_error(y_trues, y_preds)}'
    plot_predictions(y_trues, y_preds, title, 'results/icl_results/tabpfn_regression_fold_predictions_nn.png')
    fold_preds_whole = df.groupby('Fold').apply(lambda x: list(zip(x['Actual'], x['Predicted whole']))).tolist()
    title_whole = f'TabPFN Regression Predictions vs Actual across all folds Whole, r2: {r2_score(y_trues, y_preds_whole)}, MAE: {mean_absolute_error(y_trues, y_preds_whole)}'
    plot_predictions(y_trues, y_preds_whole, title_whole, 'results/icl_results/tabpfn_regression_fold_predictions_whole.png')
    # bar chart for fold maes and r2s nn and whole
    fold_count = len(fold_preds)
    fold_maes_nn = [mean_absolute_error([p[0] for p in preds], [p[1] for p in preds]) for preds in fold_preds]
    fold_r2s_nn = [r2_score([p[0] for p in preds], [p[1] for p in preds]) for preds in fold_preds]
    fold_maes_whole = [mean_absolute_error([p[0] for p in preds], [p[1] for p in preds]) for preds in fold_preds_whole]
    fold_r2s_whole = [r2_score([p[0] for p in preds], [p[1] for p in preds]) for preds in fold_preds_whole]
    logging.info(f"Fold MAEs NN: {fold_maes_nn}, R2s NN: {fold_r2s_nn}, MAEs Whole: {fold_maes_whole}, R2s Whole: {fold_r2s_whole}")
    bar_data = {
        'Fold': list(range(fold_count)),
        'MAE NN': fold_maes_nn,
        'R2 NN': fold_r2s_nn,
        'MAE whole': fold_maes_whole,
        'R2 whole': fold_r2s_whole
    }
    bar_df = pd.DataFrame(bar_data)
    bar_df.plot(x='Fold', kind='bar', figsize=(12, 6),
                title='TabPFN Regression Fold Metrics',
                ylabel='Metric Value',
                xlabel='Fold',
                rot=0)
    plt.savefig('results/icl_results/tabpfn_regression_fold_metrics.png')
    plt.close()

def plot_fold_whole_vs_nn():
    """
    Plot the fold metrics for whole vs nn predictions at the distribution tails
    """
    df = pd.read_csv('results/icl_results/tabpfn_fold_predictions.csv')
    preds_nn_total = df["Predicted NN"].tolist()
    preds_whole_total = df["Predicted whole"].tolist()
    actuals_total = df["Actual"].tolist()

    age_range_tail = (np.percentile(actuals_total, 15), np.percentile(actuals_total, 85))
    tail_indices = [i for i, age in enumerate(actuals_total) if age < age_range_tail[0] or age > age_range_tail[1]]
    logging.info(f"Calculating tail indices for ages outside {age_range_tail[0]} and {age_range_tail[1]}, found {len(tail_indices)} samples.")
    logging.info(f"Number of samples not in tail: {len(actuals_total) - len(tail_indices)}")
    preds_nn_tail = [preds_nn_total[i] for i in tail_indices]
    preds_whole_tail = [preds_whole_total[i] for i in tail_indices]
    actuals_tail = [actuals_total[i] for i in tail_indices]

    mae_nn_tail = mean_absolute_error(actuals_tail, preds_nn_tail)
    mae_whole_tail = mean_absolute_error(actuals_tail, preds_whole_tail)

    logging.info(f"MAE NN Tail: {mae_nn_tail}, MAE Whole Tail: {mae_whole_tail}")
    # histogram of actuals, nn preds and whole preds in the tail
    bins = 40
    plt.figure(figsize=(12, 6))
    plt.hist(actuals_tail, bins=bins, alpha=0.5, label='Actuals Tail', color='blue')
    plt.hist(preds_nn_tail, bins=bins, alpha=0.5, label='Predicted NN Tail', color='orange')
    plt.hist(preds_whole_tail, bins=bins, alpha=0.5, label='Predicted Whole Tail', color='green')
    plt.title(f'Tail Distribution of Actuals vs Predictions\nMAE NN Tail: {mae_nn_tail}, MAE Whole Tail: {mae_whole_tail}')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig('results/icl_results/tabpfn_regression_tail_distribution.png')
    plt.close()

def save_tail_data():
    """
    Save the tail data for further analysis
    """
    df = pd.read_csv('results/icl_results/emb_set/tabpfn_fold_emb_predictions.csv')
    preds_nn_total = df["Predicted NN"].tolist()
    preds_whole_total = df["Predicted whole"].tolist()
    actuals_total = df["Actual"].tolist()

    age_range_tail = (np.percentile(actuals_total, 15), np.percentile(actuals_total, 85))
    tail_indices = [i for i, age in enumerate(actuals_total) if age < age_range_tail[0] or age > age_range_tail[1]]
    
    tail_data = {
        'Actual': [int(actuals_total[i]) for i in tail_indices],
        'Predicted NN': [int(preds_nn_total[i]) for i in tail_indices],
        'Predicted Whole': [int(preds_whole_total[i]) for i in tail_indices]
    }
    
    tail_df = pd.DataFrame(tail_data)
    tail_df.to_csv('results/icl_results/emb_set/tabpfn_regression_tail_data.csv', index=False)
    logging.info("Tail data saved.")

def save_ages_total():
    """
    Save the total ages for further analysis
    """
    df = pd.read_csv('results/icl_results/emb_set/tabpfn_fold_emb_predictions.csv')
    actuals_total = df["Actual"].tolist()
    
    ages_df = pd.DataFrame({'Age': [int(age) for age in actuals_total]})
    ages_df.to_csv('results/icl_results/tabpfn_regression_ages_total.csv', index=False)
    logging.info("Total ages saved.")

def plot_nearest_neighbors(dm):
    """
    Plot the nearest neighbors for a sample from the test set
    """
    n_neighbours = 200
    bins = 50

    train = dm.get_fold_data_set([1,2,3,4])
    test = dm.get_fold_data_set([0])
    X_train = train.drop(columns=['age'])
    y_train = train['age']
    X_test = test.drop(columns=['age'])
    y_test = test['age']
    min_test_age = y_test.min()
    max_test_age = y_test.max()
    
    min_age_x = X_test[y_test == min_test_age].values[0]
    max_age_x = X_test[y_test == max_test_age].values[0]
    logging.info(f"Minimum test age: {min_test_age}, Maximum test age: {max_test_age}")

    x_inv, y_inv = inverse_density_resample(X_train, y_train, bins=10)
    
    nn_orig = NearestNeighbors(n_neighbors=n_neighbours, metric='cosine')
    nn_inv = NearestNeighbors(n_neighbors=n_neighbours, metric='cosine')
    nn_orig.fit(X_train)
    nn_inv.fit(x_inv)
    logging.info("Nearest neighbors fitted.")
    # Retrieve neighbors for min and max age samples

    idx_min_orig = nn_orig.kneighbors(min_age_x.reshape(1, -1), return_distance=False).flatten()
    idx_min_inv = nn_inv.kneighbors(min_age_x.reshape(1, -1), return_distance=False).flatten()
    idx_max_orig = nn_orig.kneighbors(max_age_x.reshape(1, -1), return_distance=False).flatten()
    idx_max_inv = nn_inv.kneighbors(max_age_x.reshape(1, -1), return_distance=False).flatten()

    
    # plot ages of neighbors in stacked histogram
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist([y_train[idx_min_orig], y_inv[idx_min_inv]], bins=bins, label=[f'Original {n_neighbours} Neighbors', f'Inverse {n_neighbours} Neighbors'], stacked=True, color=['blue', 'orange'])
    plt.axvline(min_test_age, color='red', linestyle='--', label='Min Test Age')
    plt.title(f'Neighbors of Min Age Sample ({min_test_age})')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.hist([y_train[idx_max_orig], y_inv[idx_max_inv]], bins=bins, label=[f'Original {n_neighbours} Neighbors', f'Inverse {n_neighbours} Neighbors'], stacked=True, color=['blue', 'orange'])
    plt.axvline(max_test_age, color='red', linestyle='--', label='Max Test Age')
    plt.title(f'Neighbors of Max Age Sample ({max_test_age})')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/icl_results/tabpfn_regression_neighbors_distribution.png')
    plt.close()

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='outfiles2/reg2.log', filemode='w')
    logging.info("Testing TabPFN with localized selection...")
    dm_logger = logging.getLogger('DATAMANAGER')
    dm = DataManager("regression","emb","rboth","verbose",logger = dm_logger)
    dm.apply_pca(n_components=500)
    
    run_cross_validation(dm, fold_count=5)
    #test_icl_folds(dm)
    #test_icl_with_localized_selection(dm)
    #plot_fold_metrics()
    #plot_fold_whole_vs_nn()
    #save_tail_data()
    #save_ages_total()
    #plot_nearest_neighbors(dm)
    logging.info("Test completed.")

if __name__ == "__main__":
    main()
    