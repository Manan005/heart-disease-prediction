"""Train and compare Logistic Regression, Random Forest and XGBoost.

Each model is a Pipeline (preprocessor + classifier), tuned with a small grid
search using 5-fold stratified cross-validation on the TRAINING set only.
The test set is not touched here; it is used once, in the evaluation phase.

Run from the project root:
    python -m src.train
"""

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocess import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    PROJECT_ROOT,
    RANDOM_STATE,
    build_preprocessor,
    load_data,
)

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "heart_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

SCORING = ["roc_auc", "recall", "precision", "f1", "accuracy"]
SELECTION_METRIC = "roc_auc"  # threshold-independent; the threshold is tuned in evaluation


def get_model_configs():
    """Each model with a small hyperparameter grid (param names prefixed with 'clf__')."""
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),  # GridSearchCV parallelizes
            {
                "clf__n_estimators": [200, 500],
                "clf__max_depth": [None, 5, 10],
                "clf__min_samples_leaf": [1, 3, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", n_jobs=1),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [2, 3, 4],
                "clf__learning_rate": [0.05, 0.1],
                "clf__subsample": [0.8, 1.0],
            },
        ),
    }


def make_pipeline(classifier):
    return Pipeline([("prep", build_preprocessor()), ("clf", classifier)])


def tune_models(X_train, y_train):
    """Grid-search every model; return a results table and the fitted best pipelines."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, best_models = [], {}

    for name, (classifier, grid) in get_model_configs().items():
        search = GridSearchCV(
            make_pipeline(classifier),
            param_grid=grid,
            scoring=SCORING,
            refit=SELECTION_METRIC,  # refit the best params on the full training set
            cv=cv,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)

        i = search.best_index_
        row = {"model": name}
        for metric in SCORING:
            row[f"cv_{metric}"] = search.cv_results_[f"mean_test_{metric}"][i]
            row[f"cv_{metric}_std"] = search.cv_results_[f"std_test_{metric}"][i]
        row["best_params"] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        rows.append(row)
        best_models[name] = search.best_estimator_
        print(f"{name:20s} CV ROC-AUC = {row['cv_roc_auc']:.3f} +/- {row['cv_roc_auc_std']:.3f}")

    results = pd.DataFrame(rows).sort_values(f"cv_{SELECTION_METRIC}", ascending=False)
    return results.reset_index(drop=True), best_models


def save_model(model, name, cv_row, path=MODEL_PATH, metadata_path=METADATA_PATH):
    """Save the full pipeline plus a metadata file describing it."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    metadata = {
        "model_name": name,
        "best_params": cv_row["best_params"],
        "cv_scores": {m: round(float(cv_row[f"cv_{m}"]), 4) for m in SCORING},
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "threshold": 0.5,  # updated in the evaluation phase
        "trained_on": date.today().isoformat(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
    }
    Path(metadata_path).write_text(json.dumps(metadata, indent=2, default=str))
    return metadata


def main():
    X_train, X_test, y_train, y_test = load_data()
    print(f"Training on {len(X_train)} patients (test set of {len(X_test)} held out)\n")

    results, best_models = tune_models(X_train, y_train)
    best = results.iloc[0]
    print(f"\nBest model: {best['model']} (CV ROC-AUC {best['cv_roc_auc']:.3f})")

    save_model(best_models[best["model"]], best["model"], best)
    print(f"Saved to {MODEL_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
