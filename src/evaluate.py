"""Evaluate the saved model on the held-out test set and choose a decision threshold.

In a screening tool, missing a sick patient (false negative) is worse than a
false alarm (false positive), so the threshold is lowered from 0.5 until recall
reaches TARGET_RECALL. The threshold is chosen with cross-validated predictions
on the TRAINING set; the test set is only used once, for the final numbers.

Run from the project root (after `python -m src.train`):
    python -m src.evaluate
"""

import json

import joblib
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.preprocess import RANDOM_STATE, load_data
from src.train import METADATA_PATH, MODEL_PATH

TARGET_RECALL = 0.90


def load_model(model_path=MODEL_PATH, metadata_path=METADATA_PATH):
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    return model, metadata


def out_of_fold_proba(model, X_train, y_train):
    """Disease probabilities for every training patient, each predicted by a
    model that did not see that patient (5-fold cross-validation)."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(clone(model), X_train, y_train, cv=cv, method="predict_proba")[:, 1]


def choose_threshold(y_true, proba, target_recall=TARGET_RECALL):
    """Highest threshold whose recall is still >= target_recall
    (the highest such threshold keeps as much precision as possible)."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = recall[:-1] >= target_recall  # last precision/recall point has no threshold
    return float(thresholds[ok].max())


def compute_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(threshold, 4),
        "roc_auc": roc_auc_score(y_true, proba),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "accuracy": accuracy_score(y_true, pred),
        "specificity": tn / (tn + fp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main():
    X_train, X_test, y_train, y_test = load_data()
    model, metadata = load_model()
    print(f"Model: {metadata['model_name']}\n")

    # 1. Choose the threshold on the training set (cross-validated)
    oof = out_of_fold_proba(model, X_train, y_train)
    threshold = choose_threshold(y_train, oof)
    print(f"Threshold for >= {TARGET_RECALL:.0%} recall (chosen on train CV): {threshold:.3f}\n")

    # 2. One-time evaluation on the held-out test set
    test_proba = model.predict_proba(X_test)[:, 1]
    default = compute_metrics(y_test, test_proba, 0.5)
    tuned = compute_metrics(y_test, test_proba, threshold)

    print(f"{'Test set':14s}{'thr 0.5':>10s}{f'thr {threshold:.2f}':>12s}")
    for m in ["roc_auc", "recall", "precision", "f1", "accuracy", "specificity"]:
        print(f"{m:14s}{default[m]:>10.3f}{tuned[m]:>12.3f}")
    for m in ["false_negatives", "false_positives"]:
        print(f"{m:14s}{default[m]:>10d}{tuned[m]:>12d}")

    # 3. Store the chosen threshold and test results with the model
    metadata["threshold"] = round(threshold, 4)
    metadata["target_recall"] = TARGET_RECALL
    metadata["test_metrics"] = {
        k: round(float(v), 4) if isinstance(v, float) else v for k, v in tuned.items()
    }
    metadata["test_metrics_default_threshold"] = {
        k: round(float(v), 4) if isinstance(v, float) else v for k, v in default.items()
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nUpdated {METADATA_PATH.name} with threshold and test metrics")


if __name__ == "__main__":
    main()
