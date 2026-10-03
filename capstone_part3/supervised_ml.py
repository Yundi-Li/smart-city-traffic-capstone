"""
Supervised Machine Learning for Traffic Volume Prediction and Accident-Risk Classification.

Uses Part 2 featured dataset to:
1. Classify high-risk traffic conditions (LogisticRegression, RandomForestClassifier)
2. Predict traffic volume (LinearRegression, GradientBoostingRegressor)
"""

import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
    precision_score, r2_score, recall_score, roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import load_featured_data, get_feature_columns, add_proxy_label, FIGURES_DIR

logger = logging.getLogger(__name__)


def train_classifiers(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate classification models for high_risk prediction."""
    logger.info("Training classification models")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"), True),
        "RandomForest": (RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1), False),
    }
    results = {}
    for name, (model, use_scaled) in models.items():
        logger.info("Training %s", name)
        xtr = X_train_s if use_scaled else X_train
        xte = X_test_s if use_scaled else X_test
        model.fit(xtr, y_train)
        y_pred = model.predict(xte)
        y_prob = model.predict_proba(xte)[:, 1]
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "model": model,
        }
        logger.info("%s ROC AUC: %.4f", name, results[name]["roc_auc"])

    _save_confusion_matrices(results)
    _save_feature_importance(results["RandomForest"]["model"], feature_names, "classification")
    return results


def train_regressors(X_train, X_test, y_train, y_test, feature_names):
    """Train and evaluate regression models for traffic_volume prediction."""
    logger.info("Training regression models")
    models = {
        "LinearRegression": LinearRegression(),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, random_state=42, learning_rate=0.1),
    }
    results = {}
    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[name] = {
            "mae": mean_absolute_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
            "model": model,
        }
        logger.info("%s R²: %.4f", name, results[name]["r2"])

    _save_feature_importance(results["GradientBoosting"]["model"], feature_names, "regression")
    return results


def _save_confusion_matrices(results):
    import matplotlib
    matplotlib.use("Agg")
    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        cm = res["confusion_matrix"]
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} Confusion Matrix")
        path = os.path.join(FIGURES_DIR, f"confusion_matrix_{name.lower()}.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved confusion matrix to %s", path)


def _save_feature_importance(model, feature_names, task):
    importances = model.feature_importances_
    indices = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title(f"Top 15 Feature Importance ({task})")
    path = os.path.join(FIGURES_DIR, f"feature_importance_{task}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance to %s", path)


def print_classification_results(results):
    """Print classification comparison table (user-facing output)."""
    header = f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC AUC':>10}"
    print("\n" + "=" * 80)
    print("CLASSIFICATION RESULTS — High-Risk Prediction")
    print("=" * 80)
    print(header)
    print("-" * 80)
    for name, res in results.items():
        print(f"{name:<25} {res['accuracy']:>10.4f} {res['precision']:>10.4f} "
              f"{res['recall']:>10.4f} {res['f1']:>10.4f} {res['roc_auc']:>10.4f}")
    print("=" * 80)


def print_regression_results(results):
    """Print regression comparison table (user-facing output)."""
    header = f"{'Model':<25} {'MAE':>12} {'R²':>10}"
    print("\n" + "=" * 60)
    print("REGRESSION RESULTS — Traffic Volume Prediction")
    print("=" * 60)
    print(header)
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<25} {res['mae']:>12.2f} {res['r2']:>10.4f}")
    print("=" * 60)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")

    df = load_featured_data()
    df = add_proxy_label(df)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y_cls = df["high_risk"].values
    y_reg = df["traffic_volume"].values

    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:5])
    logger.info("High-risk distribution: %d positive (%.1f%%)", y_cls.sum(), 100 * y_cls.mean())

    # Chronological split: train on 2012-2016, test on 2017
    years = df["date_time"].dt.year
    train_mask = years <= 2016
    test_mask = years == 2017

    X_train, X_test = X[train_mask], X[test_mask]
    y_cls_train, y_cls_test = y_cls[train_mask], y_cls[test_mask]
    y_reg_train, y_reg_test = y_reg[train_mask], y_reg[test_mask]

    logger.info("Chronological split: train 2012-2016 (%d), test 2017 (%d)", len(X_train), len(X_test))

    cls_results = train_classifiers(X_train, X_test, y_cls_train, y_cls_test, feature_cols)
    print_classification_results(cls_results)

    reg_results = train_regressors(X_train, X_test, y_reg_train, y_reg_test, feature_cols)
    print_regression_results(reg_results)


if __name__ == "__main__":
    main()
