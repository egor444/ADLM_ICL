import os
import logging
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

from data_extraction.data_manager import DataManager
from feature_engineering.featureselection import FeatureSelector
from classification.models.gpt2_im_context import GPT2ICLLabelDecoder, InContextClassificationDataset


config = {
    "TASK": "classification",
    "N_FEATURES": 200,
    "N_NEIGHBORS": 20,
    "VARIANCE_THRESHOLDS": {'default': 1e-6},
    "CORRELATION_THRESHOLDS": {'default': 0.90},
    "RANDOM_STATE": 42,
    "FEATURE_SELECTION_STRATEGIES": {
        "multi_method_scoring": False,
        "hierarchical_filtering": True,
    },
}

fold_count = 5


def test_classification_folds_gpt2(dm, template="natural"):
    os.makedirs('results/gpt2_icl/test_set/', exist_ok=True)

    model = GPT2ICLLabelDecoder(label_texts=["negative", "positive"])
    fold_preds = []

    for i in range(fold_count):
        logging.info(f"Processing fold {i + 1}/{fold_count}")
        train_set = dm.get_fold_data_set([j for j in range(fold_count) if j != i])
        test_set = dm.get_fold_data_set([i])

        X_train = train_set.drop(columns=['target'])
        y_train = train_set['target'].astype(str)
        X_test = test_set.drop(columns=['target'])
        y_test = test_set['target'].astype(str)

        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)
        y_test_enc = le.transform(y_test)

        fs = FeatureSelector(config, verbose=False)
        X_train = fs.fit_transform(X_train, y_train)
        X_test = fs.transform(X_test)

        scaler = StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        ds = InContextClassificationDataset(X_test, y_test_enc, X_train, y_train_enc, template=template)
        loader = DataLoader(ds, batch_size=4, shuffle=False)

        all_preds, all_trues, all_regions = [], [], []

        for batch in tqdm(loader, desc=f"Fold {i + 1}"):
            prompts, targets, regions = batch
            preds, _ = model.predict(prompts)
            all_preds.extend(preds)
            all_trues.extend(targets.numpy())
            all_regions.extend(regions)

        y_true = le.inverse_transform(all_trues)
        y_pred = le.inverse_transform(all_preds)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        report = classification_report(y_true, y_pred)

        logging.info(f"FOLD {i} - Accuracy: {acc:.4f}, F1: {f1:.4f}\n{report}")

        pd.DataFrame({
            'Fold': [i] * len(y_true),
            'Actual': y_true,
            'Predicted': y_pred,
            'Region': all_regions
        }).to_csv(f'results/gpt2_icl/test_set/fold_{i}_results.csv', index=False)

        fold_preds.extend(list(zip(y_true, y_pred)))

    y_true_all = [p[0] for p in fold_preds]
    y_pred_all = [p[1] for p in fold_preds]

    ConfusionMatrixDisplay.from_predictions(y_true_all, y_pred_all)
    plt.title("Overall Confusion Matrix - GPT-2 ICL")
    plt.savefig("results/gpt2_icl/confusion_matrix.png")
    plt.close()

    pd.DataFrame({
        'Actual': y_true_all,
        'Predicted': y_pred_all
    }).to_csv("results/gpt2_icl/classification_all_folds.csv", index=False)



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        filename='outfiles3/gpt2_icl.log', filemode='w')
    logging.info("Starting GPT-2 ICL Classification with Localized Contexts")
    dm_logger = logging.getLogger('DATAMANAGER')
    dm = DataManager("classification", "cancer", "rboth", logger=dm_logger)
    test_classification_folds_gpt2(dm, template="natural")
    logging.info("Completed GPT-2 ICL Classification.")


if __name__ == "__main__":
    main()
