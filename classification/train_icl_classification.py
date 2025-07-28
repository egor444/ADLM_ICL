from data_extraction.data_manager import DataManager
from tabpfn import TabPFNClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from feature_engineering.featureselection import FeatureSelector
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import torch
import logging
import os


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
        "group_based_selection": False
    },
}

k_original = 20
k_inverse = 20
fold_count = 5
num_cores = 8



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



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='outfiles2/classification3.log', filemode='w')
    logging.info("Starting TabPFN Classification with Localized Selection")
    dm_logger = logging.getLogger('DATAMANAGER')
    dm = DataManager("classification", "cancer", "rboth", logger=dm_logger)
    test_classification_folds(dm)
    logging.info("Completed TabPFN Classification.")

if __name__ == "__main__":
    main()


