# Model Version Registry

## MLflow Model Registry: `traffic_volume_regressor`

| Registry Version | Algorithm | Key Parameters | MAE | R² | Status |
|-----------------|-----------|---------------|-----|-----|--------|
| v1 | LinearRegression | default | 826.36 | 0.7105 | Baseline |
| v2 | GradientBoostingRegressor | n_estimators=200, max_depth=5, lr=0.1 | 246.80 | 0.9593 | **Production** (alias "production") |

## All Regression Models (tracked in MLflow)

| Algorithm | MAE | R² | Registry |
|-----------|-----|-----|----------|
| LinearRegression | 826.36 | 0.7105 | v1 |
| GradientBoostingRegressor | 246.80 | 0.9593 | v2 (production) |
| RandomForestRegressor | 251.25 | 0.9550 | — |
| PyTorch Neural Net (64→32→1, 30 epochs) | 345.85 | 0.9385 | — |

## Classification Models (tracked in MLflow)

| Algorithm | Accuracy | Precision | Recall | F1 | ROC AUC |
|-----------|----------|-----------|--------|-----|---------|
| LogisticRegression | 0.9706 | 0.8478 | 0.9724 | 0.9058 | 0.9960 |
| RandomForestClassifier | 0.9885 | 0.9649 | 0.9558 | 0.9603 | 0.9976 |

## Deployment Portability

MLflow stores absolute paths in its SQLite DB, so the registry only works on the machine where it was created. For portability, the production model is also saved to `capstone_part3/models/production_model/` via joblib. The FastAPI deployment app loads from the MLflow registry first and falls back to the joblib folder if the registry is unavailable.

## Notes

- Chronological split: train 2012-2016, test 2017.
- Classification scores are inflated by proxy label circularity (see responsible_ai.md).
- Speed settings: n_estimators ≤ 200, NN 30 epochs max, SHAP on 1,000-row sample.
