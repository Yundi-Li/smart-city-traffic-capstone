# MLflow Experiment Summary

**Experiment:** traffic_volume_prediction
**Tracking URI:** sqlite:///capstone_part3/mlflow_logs/mlflow.db
**Artifact location:** capstone_part3/mlflow_logs/artifacts/

## Regression Runs

| Model | MAE | R² | Registry |
|-------|-----|-----|----------|
| LinearRegression | 826.36 | 0.7105 | v1 |
| GradientBoostingRegressor | 246.80 | 0.9593 | v2 (production) |
| RandomForestRegressor | 251.25 | 0.9550 | — |
| PyTorchNeuralNet | 345.85 | 0.9385 | — |

## Classification Runs

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|-----|---------|
| LogisticRegression_cls | 0.9706 | 0.8478 | 0.9724 | 0.9058 | 0.9960 |
| RandomForestClassifier | 0.9885 | 0.9649 | 0.9558 | 0.9603 | 0.9976 |

## Model Registry

- `traffic_volume_regressor` v1 = LinearRegression (baseline)
- `traffic_volume_regressor` v2 = GradientBoosting (alias: **production**)

## Portable Fallback

Production model also saved to `capstone_part3/models/production_model/` via joblib for deployment portability.
