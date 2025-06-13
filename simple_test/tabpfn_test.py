from tabpfn import TabPFNRegressor, TabPFNClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, roc_auc_score, accuracy_score, roc_curve

import time


def get_pca_data(data_df, n_components=0.95):
    """Apply PCA to the data and return the transformed data."""
    print("\t\tApplying PCA to the data...")
    age = data_df['age'].values
    eids = data_df['eid'].values
    data = data_df.drop(columns=['eid','age']) 

    pca = PCA(n_components=n_components)  
    print(f"\t\tPCA fitting..")
    pca.fit(data)
    pca_data = pca.transform(data)
    pca_data = pd.DataFrame(pca_data, columns=[f'PC{i+1}' for i in range(pca.n_components_)])
    pca_data['age'] = age
    pca_data['eid'] = eids
    print(f"\t\tPCA applied, total features: {pca_data.shape[1]-2}")
    return pca_data, pca

def fit_model(model, data_path, target_col='age',use_pca=False):
    """Fitting the TabPFN model to the data."""
    print("Begginging model fit\n\tLoading data..")
    data = pd.read_csv(data_path)
    if use_pca:
        data, pca = get_pca_data(data, n_components=0.95)
    print("\tData loaded, shape:", data.shape)
    cols = data.columns.tolist()
    cols.remove(target_col)

    X = data[cols].values
    y = data[target_col].values
    print(f"\tPreprocessing done. X shape: {X.shape}, y shape: {y.shape}")
    print("\tFitting model..")
    model.fit(X, y)
    print("\tModel fitting complete.")
    return model, pca if use_pca else None

def test_model(model, data_path, target_col='age', use_pca=None):
    """Testing the TabPFN model on the data."""
    filename = data_path.split("/")[-1]
    print(f"Beginning model test. Loading test data from {filename}..")
    test_data = pd.read_csv(data_path)
    y_test = test_data[target_col].values
    if use_pca:
        test_data = use_pca.transform(test_data.drop(columns=[target_col]))
    else:
        test_data = test_data.drop(columns=[target_col]).values
    X_test = test_data
    print(f"\tTest data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

    print("\tMaking predictions..")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Test complete; MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")
    return y_pred, y_test, mse, mae, r2

def plot_predictions(y_test, y_pred, mse, mae, r2, test_name):
    """Plotting predictions vs true values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'TabPFN Predictions\nMAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}')
    plt.savefig(f'outputs/tabpfn_{test_name}.png')
    plt.show()
    

def run_regression():
    print("Running TabPFN")
    data_path = "../data/regression/ALL_pca_train.csv"
    test_path = "../data/regression/ALL_pca_test.csv"
    USE_PCA = False
    model = TabPFNRegressor(device='cuda', ignore_pretraining_limits=True)
    
    print("\tFitting model..")
    train_time = time.time()
    model, pca = fit_model(model, data_path, use_pca=USE_PCA)
    train_time = time.time() - train_time
    print(f"\tModel fitted in {train_time:.2f} seconds.")

    test_time_1 = time.time()
    ypred, ytest, mse, mae, r2 = test_model(model, test_path, use_pca=pca)
    test_time_1 = time.time() - test_time_1
    print(f"\tModel tested on ALL data in {test_time_1:.2f} seconds.")
    plot_predictions(ytest, ypred, mse, mae, r2, "ALL_regression")
    print("TabPFN regression test finished.")


def run_test_classification():
    print("Running TabPFN classification test; loading data..")
    data_path_train = "../data/classification/ALL_pca_train.csv"
    data_train = pd.read_csv(data_path_train)

    # balance classes 
    #print("\tBalancing classes..")
    #class_counts = data_train['target'].value_counts()
    #min_class_count = class_counts.min()
    #balanced_data = pd.concat([
    #    data_train[data_train['target'] == cls].sample(min_class_count, random_state=42)
    #    for cls in class_counts.index
    #])
    #data_train = balanced_data.reset_index(drop=True)

    X = data_train.drop(columns=['target']).values
    y = data_train['target'].values

    model = TabPFNClassifier(device='cuda', ignore_pretraining_limits=True)
    print("\tFitting model..")
    train_time = time.time()
    model.fit(X, y)
    train_time = time.time() - train_time
    print(f"\tModel fitted in {train_time:.2f} seconds.")
    print("\tModel fitting complete. Testing model..")
    data_path_test = "../data/classification/ALL_pca_test.csv"
    data_test = pd.read_csv(data_path_test)

    
    print(f"\tTest data shape: {data_test.shape}")

    X_test = data_test.drop(columns=['target']).values
    y_test = data_test['target'].values
    test_time = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - test_time
    print(f"\tModel testing complete in {test_time:.2f} seconds. Calculating metrics..")
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'TabPFN Classification Confusion Matrix\nAccuracy: {acc:.2f}')
    plt.savefig('outputs/tabpfn_classification_ALL_cm.png')
    plt.show()

    y_prob = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)

    # plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for TabPFN')
    plt.legend(loc='lower right')
    plt.savefig('outputs/tabpfn_classification_ALL_roc.png')
    plt.show()
    print(f"TabPFN classification test finished. Accuracy: {acc:.2f}, ROC AUC: {roc_auc:.2f}")



if __name__ == "__main__":
    run_regression()
    run_test_classification()
    print("TabPFN test finished.")