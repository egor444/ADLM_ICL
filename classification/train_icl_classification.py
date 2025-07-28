import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             auc, classification_report, f1_score,
                             roc_curve)
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

from data_extraction.data_manager import DataManager
from feature_engineering.featureselection import FeatureSelector
from tabpfn import TabPFNClassifier


config = {
    "TASK": "classification",
    "N_FEATURES": 200,
    "N_NEIGHBORS": 20,
    "VARIANCE_THRESHOLDS": {
        'default': 1e-6
    },
    "CORRELATION_THRESHOLDS": {
        'default': 0.90
    },
    "RANDOM_STATE": 42,
    "FEATURE_SELECTION_STRATEGIES": {
        "multi_method_scoring": False,
        "hierarchical_filtering": True,
    },
}

k_original = 20
k_inverse = 20
fold_count = 5
num_cores = 8

def process_single_sample(j, X_test, X_train, y_train, x_inv, y_inv, y_test, nn_orig, nn_inv):
    x_query = X_test[j].reshape(1, -1)

    idx_orig = nn_orig.kneighbors(x_query, return_distance=False).flatten()
    idx_inv = nn_inv.kneighbors(x_query, return_distance=False).flatten()

    x_context = np.vstack([X_train[idx_orig], x_inv[idx_inv]])
    y_context = np.concatenate([y_train[idx_orig], y_inv[idx_inv]])

    model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42 + j)
    model.fit(x_context, y_context)
    y_pred = model.predict(x_query)
    y_true = y_test[j]

    return (y_true, y_pred[0])


def inverse_density_resample(X, y, bins=30):
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

def balanced_resample(X, y):
    df = pd.concat([X, y], axis=1)
    class_counts = y.value_counts()
    max_count = class_counts.max()
    balanced = pd.concat([
        resample(df[df[y.name] == cls], replace=True, n_samples=max_count, random_state=42)
        for cls in class_counts.index
    ])
    return balanced.drop(columns=[y.name]).reset_index(drop=True), balanced[y.name].reset_index(drop=True)

def retrieve_neighbors(x_query, X_train, k):
    nn = NearestNeighbors(n_neighbors=k, metric='cosine')
    nn.fit(X_train)
    return nn.kneighbors(x_query, return_distance=False).flatten()

def plot_roc_curve(y_true, y_pred, title, filename):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(filename)
    plt.close()

def process_classification_sample(j, X_test, X_train, y_train, x_inv, y_inv, y_test, nn_orig, nn_inv):
    x_query = X_test[j].reshape(1, -1)
    idx_orig = nn_orig.kneighbors(x_query, return_distance=False).flatten()
    idx_inv = nn_inv.kneighbors(x_query, return_distance=False).flatten()

    x_context = np.vstack([X_train[idx_orig], x_inv[idx_inv]])
    y_context = np.concatenate([y_train[idx_orig], y_inv[idx_inv]])

    model = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=config["RANDOM_STATE"] + j)
    model.fit(x_context, y_context)
    pred = model.predict(x_query)[0]
    return (y_test[j], pred)



def test_classification_folds(dm):
    fold_preds = []
    os.makedirs('results/tabpfn_classification/test_set/', exist_ok=True)

    for i in range(fold_count):
        logging.info(f"Processing fold {i + 1}/{fold_count}")
        train_set = dm.get_fold_data_set([j for j in range(fold_count) if j != i])
        test_set = dm.get_fold_data_set([i])

        X_train = train_set.drop(columns=['target'])
        y_train = train_set['target'].astype(str)
        X_test = test_set.drop(columns=['target'])
        y_test = test_set['target'].astype(str)

        fs = FeatureSelector(config, verbose=False)
        X_train = fs.fit_transform(X_train, y_train)
        X_test = fs.transform(X_test)

        x_inv, y_inv = balanced_resample(X_train, y_train)
        X_train, y_train = X_train.values, y_train.values
        X_test, y_test = X_test.values, y_test.values
        x_inv, y_inv = x_inv.values, y_inv.values

        nn_orig = NearestNeighbors(n_neighbors=k_original, metric='cosine')
        nn_inv = NearestNeighbors(n_neighbors=k_inverse, metric='cosine')
        nn_orig.fit(X_train)
        nn_inv.fit(x_inv)

        preds = Parallel(n_jobs=num_cores)(
            delayed(process_classification_sample)(j, X_test, X_train, y_train, x_inv, y_inv, y_test, nn_orig, nn_inv)
            for j in range(len(X_test))
        )

        y_true = [p[0] for p in preds]
        y_pred = [p[1] for p in preds]

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)

        logging.info(f"FOLD {i} - Accuracy: {acc}, F1: {f1}\n{report}")

        pd.DataFrame({
            'Fold': [i] * len(y_true),
            'Actual': y_true,
            'Predicted': y_pred
        }).to_csv(f'results/tabpfn_classification/test_set/fold_{i}_results.csv', index=False)

        fold_preds.extend(list(zip(y_true, y_pred)))

    y_true_all = [p[0] for p in fold_preds]
    y_pred_all = [p[1] for p in fold_preds]
    ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_all)
    plt.title("Overall Confusion Matrix")
    plt.savefig("results/tabpfn_classification/confusion_matrix.png")
    plt.close()

    pd.DataFrame({
        'Actual': y_true_all,
        'Predicted': y_pred_all
    }).to_csv("results/tabpfn_classification/classification_all_folds.csv", index=False)

def test_icl_with_localized_selection(dm):
    train_set = dm.get_train()
    test_set = dm.get_test()
    num_cores = 8
    
    X_train = train_set.drop(columns=['target'])
    y_train = train_set['target']
    X_test = test_set.drop(columns=['target'])
    y_test = test_set['target']

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
    accs_tabpfn = []
    f1s_tabpfn = []
    aucs_tabpfn = []
    weighted_accs_tabpfn = []
    times = []

    #context_sizes = [len(X_train)]
    logging.info("Starting TabPFN classification with localized selection...")

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
            tabpfn = TabPFNClassifier(device='cuda' if torch.cuda.is_available() else 'cpu', random_state=42)
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

        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average='weighted')
        weighted_acc = accuracy_score(y_true, preds, normalize=False) / len(y_true)
        auc_out = auc(*roc_curve(y_true, preds)[:2])
        accs_tabpfn.append(acc)
        f1s_tabpfn.append(f1)
        weighted_accs_tabpfn.append(weighted_acc)
        aucs_tabpfn.append(auc_out)
        logging.info(f"Context size {context_size} - Accuracy: {acc:.2f}, F1: {f1:.2f}, Weighted Accuracy: {weighted_acc:.2f} AUC: {auc_out:.2f}")
        preds_tabpfn[-1] = np.append(preds_tabpfn[-1], preds)
        trues_tabpfn[-1] = np.append(trues_tabpfn[-1], y_true)
        title = f'TabPFN Classififcation, Context Size: {context_size/len(X_train):.2f}% ({context_size} samples)\nAcc: {accs_tabpfn[-1]:.2f}, F1: {f1s_tabpfn[-1]:.2f}, WAcc: {weighted_accs_tabpfn[-1]:.2f}, AUC: {aucs_tabpfn[-1]:.2f}'
        plot_roc_curve(trues_tabpfn[-1], preds_tabpfn[-1], title, f'results/tabpfn_classification/rocs/tabpfn_classification_predictions_{(context_size*100)//len(X_train)}.png')

        # flush gpu memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    #for i, context_size in enumerate(context_sizes):
    #    title = f'TabPFN Classififcation, Context Size: {context_size/len(X_train):.2f}% ({context_size} samples), Accuracy: {accs_tabpfn[i]:.2f}, F1: {f1s_tabpfn[i]:.2f}, Weighted Acc: {weighted_accs_tabpfn[i]:.2f}, AUC: {aucs_tabpfn[i]:.2f}'
    #    plot_roc_curve(trues_tabpfn[i], preds_tabpfn[i], title, f'results/tabpfn_classification/rocs/tabpfn_classification_predictions_{(context_size*100)//len(X_train)}.png')

    logging.info("TabPFN classification with localized selection completed.")

    # dataframe to save results with columns for every context size predictions and true values like preds_10, preds_20, ..., true
    result_df = pd.DataFrame({
        'true': [int(age) for age in trues_tabpfn[0]],
        **{f'preds_{context_size}': preds_tabpfn[i] for i, context_size in enumerate(context_sizes)}
    })
    result_df.to_csv('results/tabpfn_classification/rocs/tabpfn_context_results.csv', index=False)

    # plot aucs and times
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(context_sizes, aucs_tabpfn, marker='o')
    plt.title('AUC vs Context Size')
    plt.xlabel('Context Size')
    plt.ylabel('Area Under Curve (AUC)')

    plt.subplot(1, 2, 2)
    plt.plot(context_sizes, times, marker='o')
    plt.title('Time Taken vs Context Size')
    plt.xlabel('Context Size')
    plt.ylabel('Time Taken (seconds)')
    plt.tight_layout()
    plt.savefig('results/tabpfn_classification/rocs/tabpfn_classification_time.png')
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='outfiles2/classification.log', filemode='w')
    logging.info("Starting TabPFN Classification with Localized Selection")
    dm_logger = logging.getLogger('DATAMANAGER')
    dm = DataManager("classification", "liver", "emb", "rboth", logger=dm_logger)
    dm.apply_pca(n_components=100)

    test_icl_with_localized_selection(dm)

    #test_classification_folds(dm)
    logging.info("Completed TabPFN Classification.")

if __name__ == "__main__":
    main()


