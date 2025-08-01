import os
import logging
from datetime import datetime
from sklearn.base import clone

from models import get_classification_models
from optuna_tuner import tune_model_with_optuna  # Your Optuna tuning function
from feature_selection import FeatureSelector, FeatureEngineer
from utils import run_classification_tests, run_cross_validation
from data_manager import DataManager  # Adjust import as needed


def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train_classification.log")
    logging.root.handlers.clear()
    logging.basicConfig(
        filename=log_path,
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logging.info("Logging initialized")


def load_data_with_manager():
    logging.info("Initializing DataManager to load data")
    # Customize DataManager args as needed
    dm = DataManager('classification', 'cancer', 'emb', 'rboth', pca=False, logger=logging)

    train_df = dm.get_train()
    test_df = dm.get_test()

    for df in [train_df, test_df]:
        if 'eid' in df.columns:
            df.drop(columns=['eid'], inplace=True)

    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target'].values

    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target'].values

    return X_train, y_train, X_test, y_test


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", "classification", f"run_{run_id}")
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    logging.info("Starting classification pipeline")

    X_train, y_train, X_test, y_test = load_data_with_manager()

    engineer = FeatureEngineer(impute_strategy='mean', scale=True)
    selector = FeatureSelector(task='classification', verbose=True)

    base_models = get_classification_models()

    tuned_models = {}
    for name, model in base_models.items():
        logging.info(f"Starting hyperparameter tuning for model: {name}")
        tuned_model = tune_model_with_optuna(
            model_name=name,
            task='classification',
            X=X_train,
            y=y_train,
            n_trials=30  # Adjust as needed
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
            task='classification',
            k=5,
            output_dir=cv_output_dir
        )

        logging.info(f"[{name}] CV scores: {cv_results['scores']}")
        logging.info(f"[{name}] Best fold: {cv_results['best_fold']} with score {max(cv_results['scores'])}")

        best_features = cv_results['best_features']
        logging.info(f"[{name}] Number of features selected by CV: {len(best_features)}")

        logging.info(f"[{name}] Training final model on full training data")

        X_train_eng = engineer.fit_transform(X_train, y_train)
        X_test_eng = engineer.transform(X_test)

        X_train_sel = X_train_eng[best_features]
        X_test_sel = X_test_eng[best_features]

        final_model = clone(model)
        final_model.fit(X_train_sel, y_train)

        logging.info(f"[{name}] Running post-model feature importance")
        post_scores = selector.run_post_model_feature_selection(X_train_sel, y_train, final_model)

        post_scores_dir = os.path.join(output_dir, f'post_training_importance_{name}')
        os.makedirs(post_scores_dir, exist_ok=True)

        for df in post_scores:
            if not df.empty:
                score_name = df.columns[1]
                df.to_csv(os.path.join(post_scores_dir, f'{score_name}.csv'), index=False)
                logging.info(f"Saved post-training feature importance: {score_name}")

        selector.finalize_feature_selection(post_scores, top_k=20, output_dir=post_scores_dir)

        run_classification_tests(final_model, X_test_sel, y_test, output_dir=output_dir, model_name=name)

    logging.info("Classification pipeline complete.")


if __name__ == "__main__":
    main()
