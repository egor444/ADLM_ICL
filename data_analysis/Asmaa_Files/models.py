# models.py

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


def get_regression_models(random_state=42, tuned=False, scoring='r2', n_iter=10, cv=3):
    if not tuned:
        return {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
            "SVR": SVR(),
            "MLP": MLPRegressor(random_state=random_state, max_iter=500),
            "HistGBR": HistGradientBoostingRegressor(random_state=random_state),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0),
            "LightGBM": LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1),
            "Ridge": Ridge(),
            "Lasso": Lasso(max_iter=10000),
            "ElasticNet": ElasticNet(max_iter=10000),
        }

    return {
        "RandomForest": RandomizedSearchCV(
            RandomForestRegressor(random_state=random_state, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "SVR": RandomizedSearchCV(
            SVR(),
            {
                'C': [0.1, 1, 10],
                'epsilon': [0.01, 0.1, 1],
                'kernel': ['linear', 'rbf']
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "Ridge": RandomizedSearchCV(
            Ridge(),
            {'alpha': np.logspace(-3, 3, 10)},
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "Lasso": RandomizedSearchCV(
            Lasso(max_iter=10000),
            {'alpha': np.logspace(-3, 3, 10)},
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "ElasticNet": RandomizedSearchCV(
            ElasticNet(max_iter=10000),
            {
                'alpha': np.logspace(-3, 3, 10),
                'l1_ratio': np.linspace(0.1, 1.0, 5)
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "MLP": RandomizedSearchCV(
            MLPRegressor(random_state=random_state, max_iter=500),
            {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "HistGBR": HistGradientBoostingRegressor(random_state=random_state),
        "XGBoost": XGBRegressor(n_estimators=100, random_state=random_state, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=random_state, verbose=-1),
    }


def get_classification_models(random_state=42, tuned=False, scoring='f1_weighted', n_iter=10, cv=3):
    if not tuned:
        return {
            "LogisticRegression": LogisticRegression(max_iter=500, random_state=random_state),
            "RandomForest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
            "SVC": SVC(probability=True, random_state=random_state),
            "MLP": MLPClassifier(random_state=random_state, max_iter=500),
            "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
            "KNeighbors": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB(),
            "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        }

    return {
        "LogisticRegression": RandomizedSearchCV(
            LogisticRegression(max_iter=1000, random_state=random_state),
            {
                'C': np.logspace(-3, 3, 10),
                'penalty': ['l2'],
                'solver': ['lbfgs', 'liblinear']
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "RandomForest": RandomizedSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "SVC": RandomizedSearchCV(
            SVC(probability=True, random_state=random_state),
            {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 0.01, 0.001]
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "MLP": RandomizedSearchCV(
            MLPClassifier(random_state=random_state, max_iter=500),
            {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            },
            n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=-1, random_state=random_state
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=random_state),
        "KNeighbors": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state)
    }
