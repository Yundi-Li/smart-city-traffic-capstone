# Model Version Registry

## Regression Models — Traffic Volume Prediction

| Version | Algorithm | Key Parameters | MAE | R² | Status |
|---------|-----------|---------------|-----|-----|--------|
| v1 | LinearRegression | default | 826.36 | 0.7105 | Baseline |
| v2 | GradientBoostingRegressor | n_estimators=200, max_depth=5, lr=0.1 | 246.67 | 0.9593 | **Production** (MLflow Registry v2, alias "production") |
| v3 | RandomForestRegressor | n_estimators=200, max_depth=15 | 248.75 | 0.9562 | Candidate (MLflow Registry v1) |
| v4 | PyTorch Neural Net | 128-64-32, dropout 0.3/0.2, 30 epochs | 465.24 | 0.8972 | Experimental |

## Classification Models — Proxy Accident-Risk Prediction

| Version | Algorithm | Key Parameters | Accuracy | Precision | Recall | F1 | ROC AUC | Status |
|---------|-----------|---------------|----------|-----------|--------|-----|---------|--------|
| v1 | LogisticRegression | max_iter=1000, balanced weights | 0.9705 | 0.8472 | 0.9724 | 0.9055 | 0.9960 | Baseline |
| v2 | RandomForestClassifier | n_estimators=100, balanced weights | 0.9885 | 0.9627 | 0.9581 | 0.9604 | 0.9981 | **Production** |

## Notes

- All models trained on a chronological split: train 2012-2016, test 2017 (random splits leak neighbouring hours and inflate scores).
- Feature set: cyclical hour/day encodings, is_weekend, is_holiday, is_low_visibility, temp, rain_1h, snow_1h, clouds_all, weather one-hot encoding.
- GradientBoostingRegressor (v2) is the production regression model: best R² (0.9593) and lowest MAE (246.67). Registered in MLflow Model Registry as `traffic_volume_regressor` v2 with alias "production".
- RandomForestClassifier (v2) is the production classifier: best F1 (0.9604) and ROC AUC (0.9981).
- The high classification scores are partly inflated because the proxy label is derived from features the model can see (congestion category from traffic_volume quartiles combined with weather). This circularity means real-world performance would be lower.
- All experiments tracked in MLflow (sqlite:///capstone_part3/mlflow_logs/mlflow.db). Model Registry contains 2 versions: v1=RF, v2=GBR (alias "production").
- Data: Part 2 cleaned dataset (40,575 rows, deduplicated and outlier-imputed).
