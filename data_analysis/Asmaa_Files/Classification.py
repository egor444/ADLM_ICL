import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import GridSearchCV

CONFIG = {
    'pca_variance_threshold': 0.95,
    'random_state': 42,
    'verbose': True,
}


def log_step(step_name):
    if CONFIG['verbose']:
        print(f"\n[STEP] {step_name}")
    return time.time()


def log_elapsed(start_time, step_name, log_dict=None):
    elapsed = time.time() - start_time
    msg = f"[DONE] {step_name} in {elapsed:.2f} sec"
    if CONFIG['verbose']:
        print(msg)
    if log_dict is not None:
        log_dict[step_name] = elapsed


def setup_environment():
    os.environ['PYTHONHASHSEED'] = str(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])
    return {'timing_log': {}}


def load_and_merge(paths, label):
    print(f"\nLoading and merging: {label}")
    dfs = [pd.read_csv(p) for p in paths]
    df_combined = pd.concat(dfs, axis=0).reset_index(drop=True)
    df_combined = df_combined.drop(columns=[col for col in ['eid'] if col in df_combined.columns])
    df_combined = df_combined.dropna()
    X = df_combined.drop(columns=['target'])
    y = df_combined['target']
    return {'X': X.values, 'y': y.values}


def get_classification_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=CONFIG['random_state']),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=CONFIG['random_state']),
        "SVC": SVC(probability=True),
        "KNeighbors": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=CONFIG['random_state']),
    }


def run_classification(train_paths, test_paths, label, apply_pca):
    env = setup_environment()
    results = {}
    start = log_step(f"Loading {label}")
    train_data = load_and_merge(train_paths, f"{label} - Train")
    test_data = load_and_merge(test_paths, f"{label} - Test")
    log_elapsed(start, f"Data Loading - {label}", env['timing_log'])

    if apply_pca:
        start = log_step(f"PCA - {label}")
        pca = PCA(n_components=CONFIG['pca_variance_threshold'])
        train_data['X'] = pca.fit_transform(train_data['X'])
        test_data['X'] = pca.transform(test_data['X'])
        log_elapsed(start, f"PCA - {label}", env['timing_log'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data['X'])
    X_test_scaled = scaler.transform(test_data['X'])

    models = get_classification_models()
    for name, model in models.items():
        print(f"\nTraining: {name}")
        start = log_step(f"Train {name}")
        model.fit(X_train_scaled, train_data['y'])
        log_elapsed(start, f"Train {name}", env['timing_log'])

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(test_data['y'], y_pred)
        roc = roc_auc_score(test_data['y'], y_prob)

        print(f"{name} Accuracy: {acc:.2f}, ROC AUC: {roc:.2f}")

        cm = confusion_matrix(test_data['y'], y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f'{name} Confusion Matrix\nAccuracy: {acc:.2f}')
        os.makedirs("outputs/confusion_matrices", exist_ok=True)
        plt.savefig(f'outputs/confusion_matrices/{name}_cm.png')
        plt.close()

        results[name] = {'Accuracy': acc, 'ROC_AUC': roc}

    return results


def save_results_to_csv(raw_results, pca_results, output_path="outputs/classification_results_summary.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []
    for model in raw_results:
        rows.append({
            'Model': model,
            'Raw_Accuracy': raw_results[model]['Accuracy'],
            'Raw_ROC_AUC': raw_results[model]['ROC_AUC'],
            'PCA_Accuracy': pca_results[model]['Accuracy'],
            'PCA_ROC_AUC': pca_results[model]['ROC_AUC']
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n\U0001F4C4 Classification summary saved to {output_path}")


if __name__ == '__main__':
    print("\n### Comparing Classification Models on Binary Risk Data (Raw and PCA) ###")

    train_paths = [
        "/vol/miltank/projects/practical_sose25/in_context_learning/data/classification/train_fat.csv"
    ]
    test_paths = [
        "/vol/miltank/projects/practical_sose25/in_context_learning/data/classification/test_fat.csv"
    ]

    print("\nRunning classification without PCA")
    raw_results = run_classification(train_paths, test_paths, "Raw Combined", apply_pca=False)

    print("\nRunning classification with PCA")
    pca_results = run_classification(train_paths, test_paths, "Raw Combined + PCA", apply_pca=True)

    print("\nFinal Accuracy and ROC AUC Comparison")
    print("Model\t\tRaw Acc\tRaw AUC\tPCA Acc\tPCA AUC")
    for model in raw_results:
        print(f"{model:<12}\t{raw_results[model]['Accuracy']:.2f}\t{raw_results[model]['ROC_AUC']:.2f}\t{pca_results[model]['Accuracy']:.2f}\t{pca_results[model]['ROC_AUC']:.2f}")

    save_results_to_csv(raw_results, pca_results)
