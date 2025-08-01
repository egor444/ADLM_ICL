import os
import logging
import pandas as pd
from datetime import datetime
from sklearn.base import clone

from models import get_regression_models
from optuna_tuner import tune_model_with_optuna  # Your Optuna tuning function
from feature_selection import FeatureSelector, FeatureEngineer
from utils import (
    run_regression_tests,
    save_shap_analysis,
    run_cross_validation
)

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_regression.log")
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Logging initialized")

def load_and_process(path, label):
    print(f"\nLoading and processing: {label}")
    df = pd.read_csv(path).dropna()
    if 'eid' in df.columns:
        df = df.drop(columns=['eid'])
    X = df.drop(columns=['age'])
    y = df['age']
    return X, y

def load_data():
    train_path = '/vol/miltank/projects/practical_sose25/in_context_learning/data/regression/train_ALL.csv'
    test_path = '/vol/miltank/projects/practical_sose25/in_context_learning/data/regression/test_ALL.csv'
    X_train, y_train = load_and_process(train_path, "Training Data")
    X_test, y_test = load_and_process(test_path, "Testing Data")
    return X_train, y_train.values, X_test, y_test.values

def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = "results/regression"
    output_dir = os.path.join(base_output_dir, f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logging(output_dir)
    logging.info("Starting regression pipeline")

    X_train, y_train, X_test, y_test = load_data()

    engineer = FeatureEngineer(impute_strategy='mean', scale=True)
    selector = FeatureSelector(task='regression', verbose=True)

    base_models = get_regression_models()

    tuned_models = {}
    for name, model in base_models.items():
        logging.info(f"Starting hyperparameter tuning for model: {name}")
        
        # Use internal CV inside tune_model_with_optuna (no cv_func passed)
        tuned_model = tune_model_with_optuna(
            model_name=name,
            task='regression',
            X=X_train,
            y=y_train,
            n_trials=30  # adjust as needed
        )
        tuned_models[name] = tuned_model
        logging.info(f"Completed tuning for {name}")

    for name, model in tuned_models.items():
        logging.info(f"Running cross-validation for tuned model: {name}")

        cv_output_dir = os.path.join(output_dir, f"cross_validation_{name}")
        cv_results = run_cross_validation(
            model=clone(model),
            X=X_train,
            y=y_train,
            engineer=engineer,
            selector=selector,
            task='regression',
            k=5,
            output_dir=cv_output_dir
        )

        logging.info(f"[{name}] CV scores: {cv_results['scores']}")
        logging.info(f"[{name}] Best fold: {cv_results['best_fold']} with score {max(cv_results['scores'])}")

        best_features = cv_results['best_features']
        logging.info(f"[{name}] Number of features selected by CV: {len(best_features)}")

        logging.info(f"[{name}] Training final model on full training data")

        X_train_eng = engineer.fit_transform(X_train)
        X_test_eng = engineer.transform(X_test)

        X_train_sel = X_train_eng[best_features]
        X_test_sel = X_test_eng[best_features]

        final_model = clone(model)
        final_model.fit(X_train_sel, y_train)

        save_shap_analysis(final_model, X_train_sel, output_dir, model_name=name)

        logging.info(f"[{name}] Running post-model feature importance")
        post_scores = selector.run_post_model_feature_selection(X_train_sel, y_train, final_model)

        post_scores_dir = os.path.join(output_dir, f'post_training_importance_{name}')
        os.makedirs(post_scores_dir, exist_ok=True)

        figures_dir_post = os.path.join(post_scores_dir, "figures")
        os.makedirs(figures_dir_post, exist_ok=True)
        
        selector.finalize_feature_selection(
            post_scores,
            top_k=20,
            output_dir=post_scores_dir,
            save_plot_path=os.path.join(figures_dir_post, f"{name}_aggregated_features.png")
        )

        run_regression_tests(final_model, X_test_sel, y_test, output_dir=output_dir, model_name=name)

    logging.info("Regression pipeline complete.")

if __name__ == "__main__":
    main()
