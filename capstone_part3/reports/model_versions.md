# Model Version Registry

## Regression Models — Traffic Volume Prediction

| Version | Algorithm | Key Parameters | MAE | R² | Status |
|---------|-----------|---------------|-----|-----|--------|
| v1 | LinearRegression | default | 831.87 | 0.7036 | Baseline |
| v2 | GradientBoostingRegressor | n_estimators=200, max_depth=5, lr=0.1 | 273.68 | 0.9425 | **Production** |
| v3 | RandomForestRegressor | n_estimators=200, max_depth=15 | 282.88 | 0.9376 | Candidate |
| v4 | PyTorch Neural Net | 128-64-32, dropout 0.3/0.2, 50 epochs | 347.58 | 0.9227 | Experimental |

## Classification Models — Proxy Accident-Risk Prediction

| Version | Algorithm | Key Parameters | Accuracy | Precision | Recall | F1 | ROC AUC | Status |
|---------|-----------|---------------|----------|-----------|--------|-----|---------|--------|
| v1 | LogisticRegression | max_iter=1000, balanced weights | 0.9646 | 0.8003 | 0.9874 | 0.8840 | 0.9942 | Baseline |
| v2 | RandomForestClassifier | n_estimators=100, balanced weights | 0.9866 | 0.9317 | 0.9729 | 0.9519 | 0.9973 | **Production** |

## Notes

- All models trained on 80/20 train/test split with random_state=42.
- Feature set: cyclical hour/day encodings, is_weekend, is_holiday, is_low_visibility, temp, rain_1h, snow_1h, clouds_all, weather one-hot encoding.
- GradientBoostingRegressor (v2) is the production regression model: best R² (0.9425) and lowest MAE (273.68).
- RandomForestClassifier (v2) is the production classifier: best F1 (0.9519) and ROC AUC (0.9973).
- The high classification scores are partly inflated because the proxy label is derived from features the model can see (congestion category from traffic_volume quartiles combined with weather). This circularity means real-world performance would be lower.
- All experiments tracked in MLflow (sqlite:///capstone_part3/mlflow_logs/mlflow.db).
- Data: Part 2 cleaned dataset (40,575 rows, deduplicated and outlier-imputed).
