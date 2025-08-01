import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.model_selection import KFold
from sklearn.base import clone
import shap

# === Logger Setup ===
logging.basicConfig(
    filename="cv_debug.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode='w'
)



def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_metrics(metrics, output_dir, model_name):
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)


def save_predictions(y_true, y_pred, output_dir, model_name):
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    path = os.path.join(output_dir, f"{model_name}_predictions.csv")
    df.to_csv(path, index=False)


def save_model(model, output_dir, model_name):
    model_path = os.path.join(output_dir, f"{model_name}_model.joblib")
    joblib.dump(model, model_path)


def save_feature_importance(model, features, output_dir, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        df = pd.DataFrame({'feature': features, 'importance': importances})
        df = df.sort_values(by='importance', ascending=False)
        path = os.path.join(output_dir, f"{model_name}_feature_importance.csv")
        df.to_csv(path, index=False)

        top_n = min(20, len(df))
        plt.figure(figsize=(10, 6))
        plt.barh(df['feature'][:top_n][::-1], df['importance'][:top_n][::-1])
        plt.xlabel('Importance')
        plt.title(f'{model_name} - Top {top_n} Feature Importances')
        fig_dir = os.path.join(output_dir, "figures")
        ensure_dir(fig_dir)
        try:
            plt.tight_layout()
        except Exception as e:
            logging.warning(f"Tight layout not applied: {e}")
        plt.savefig(os.path.join(fig_dir, f"{model_name}_feature_importance.png"))
        plt.close()


def run_cross_validation(
    model, X, y, engineer, selector,
    task='regression', k=5, top_k=20,
    output_dir="cv_output", final_test=None
):
    os.makedirs(output_dir, exist_ok=True)
    scores = []
    models = []
    fold_data = []

    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\n[CV] Fold {fold}")
        logging.info(f"[CV] Fold {fold} starting")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Feature engineering
        X_train_eng = engineer.fit_transform(X_train)
        X_val_eng = engineer.transform(X_val)

        logging.info(f"[Fold {fold}] X_train_eng stats: min={X_train_eng.min().min():.4f}, max={X_train_eng.max().max():.4f}, mean={X_train_eng.mean().mean():.4f}")
        logging.info(f"[Fold {fold}] X_val_eng stats: min={X_val_eng.min().min():.4f}, max={X_val_eng.max().max():.4f}, mean={X_val_eng.mean().mean():.4f}")

        if np.any(np.isnan(X_train_eng)) or np.any(np.isinf(X_train_eng)):
            logging.warning(f"[Fold {fold}] NaNs or infs found in X_train_eng")
        if np.any(np.isnan(X_val_eng)) or np.any(np.isinf(X_val_eng)):
            logging.warning(f"[Fold {fold}] NaNs or infs found in X_val_eng")

        # Feature selection (no saving)
        _, X_fast, _ = selector.run_pre_model_feature_selection(
            X_train_eng, y_train,
            output_dir=None  # Prevent saving
        )

        selected_features = X_fast.columns.tolist()
        logging.info(f"[Fold {fold}] Selected features: {selected_features}")

        X_train_sel = X_train_eng[selected_features]
        X_val_sel = X_val_eng[selected_features]

        try:
            model_copy = clone(model)
            model_copy.fit(X_train_sel, y_train)
            preds = model_copy.predict(X_val_sel)

            if task == 'regression':
                score = r2_score(y_val, preds)
                print(f"[CV] Fold {fold} R²: {score:.4f}")
            else:
                score = accuracy_score(y_val, preds)
                print(f"[CV] Fold {fold} Accuracy: {score:.4f}")

            if task == 'regression' and score < -10:
                logging.error(f"[Fold {fold}] ABNORMAL R² score: {score:.4f}")
                logging.warning(f"[Fold {fold}] y_val sample: {y_val[:5]}")
                logging.warning(f"[Fold {fold}] preds sample: {preds[:5]}")

        except Exception as e:
            score = float('-inf')
            logging.exception(f"[Fold {fold}] Failed during model training or prediction: {e}")
            preds = np.zeros_like(y_val)

        scores.append(score)
        models.append(model_copy)
        fold_data.append({
            "fold": fold,
            "score": score,
            "model": model_copy,
            "features": selected_features
        })

        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        ensure_dir(fold_dir)

        save_model(model_copy, fold_dir, model_name=f"fold{fold}")
        if task == 'regression':
            run_regression_tests(model_copy, X_val_sel, y_val, fold_dir, model_name=f"fold{fold}")
        else:
            run_classification_tests(model_copy, X_val_sel, y_val, fold_dir, model_name=f"fold{fold}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"\n[CV] Average Score: {mean_score:.4f} ± {std_score:.4f}")
    logging.info(f"[CV] Mean ± Std: {mean_score:.4f} ± {std_score:.4f}")

    best_fold = max(fold_data, key=lambda x: x["score"])
    print(f"\n[CV] Best Fold: {best_fold['fold']} (Score: {best_fold['score']:.4f})")
    logging.info(f"[CV] Best Fold: {best_fold['fold']} (Score: {best_fold['score']:.4f})")

    if final_test:
        X_test, y_test = final_test
        engineer.fit(X)
        X_test_eng = engineer.transform(X_test)
        X_test_sel = X_test_eng[best_fold["features"]]

        test_dir = os.path.join(output_dir, "final_test")
        ensure_dir(test_dir)

        if task == 'regression':
            run_regression_tests(best_fold["model"], X_test_sel, y_test, test_dir, "final_model")
        else:
            run_classification_tests(best_fold["model"], X_test_sel, y_test, test_dir, "final_model")

    return {
        "scores": scores,
        "models": models,
        "best_model": best_fold["model"],
        "best_features": best_fold["features"],
        "best_fold": best_fold["fold"]
    }

def plot_and_save(y_true, y_pred, output_dir, model_name, task, y_proba=None):
    fig_dir = os.path.join(output_dir, "figures")
    ensure_dir(fig_dir)

    if task == 'regression':
        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"{model_name} Predictions vs True Values")
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{model_name}_pred_vs_true.png"))
        plt.close()

    elif task == 'classification':
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{model_name} Confusion Matrix")
        plt.colorbar()
        tick_marks = range(len(np.unique(y_true)))
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{model_name}_confusion_matrix.png"))
        plt.close()

        if y_proba is not None and len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = roc_auc_score(y_true, y_proba)
            plt.figure(figsize=(6, 5))
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.title(f"{model_name} ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{model_name}_roc_curve.png"))
            plt.close()


def run_regression_tests(model, X_test, y_test, output_dir, model_name):
    ensure_dir(output_dir)
    y_pred = model.predict(X_test)

    metrics = {
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MedianAE": median_absolute_error(y_test, y_pred),
        "ExplainedVariance": explained_variance_score(y_test, y_pred),
        "R2": r2_score(y_test, y_pred)
    }

    save_metrics(metrics, output_dir, model_name)
    save_predictions(y_test, y_pred, output_dir, model_name)
    save_model(model, output_dir, model_name)
    plot_and_save(y_test, y_pred, output_dir, model_name, task='regression')
    save_feature_importance(model, X_test.columns, output_dir, model_name)
    
def run_classification_tests(model, X_test, y_test, output_dir, model_name):
    ensure_dir(output_dir)
    y_pred = model.predict(X_test)
    y_proba = None

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception as e:
            logging.warning(f"predict_proba failed: {e}")
            y_proba = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0, average='weighted'),
        "Recall": recall_score(y_test, y_pred, zero_division=0, average='weighted'),
        "F1": f1_score(y_test, y_pred, zero_division=0, average='weighted')
    }

    if y_proba is not None:
        metrics["ROC_AUC"] = roc_auc_score(y_test, y_proba)

    save_metrics(metrics, output_dir, model_name)
    save_predictions(y_test, y_pred, output_dir, model_name)
    save_model(model, output_dir, model_name)
    plot_and_save(y_test, y_pred, output_dir, model_name, task='classification', y_proba=y_proba)
    save_feature_importance(model, X_test.columns, output_dir, model_name)


def save_shap_analysis(model, X_train, output_dir, model_name):
    shap_dir = os.path.join(output_dir, "shap", model_name)
    ensure_dir(shap_dir)

    try:
        explainer = shap.Explainer(model, X_train)
    except Exception:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer(X_train)

    shap_df = pd.DataFrame(shap_values.values, columns=X_train.columns)
    shap_df.to_csv(os.path.join(shap_dir, "shap_values.csv"), index=False)

    plt.figure()
    shap.summary_plot(shap_values, X_train, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_dir, "shap_summary_plot.png"))
    plt.close()
