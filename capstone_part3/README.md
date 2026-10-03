# Part 3 — Machine Learning and AI: Building an Intelligent Mobility Solution

## Proxy Accident-Risk Label

**No real accident dataset was sourced.** A documented proxy label was used for the classification task:

```
is_low_visibility = (clouds_all > 80) AND weather_main in [Fog, Mist, Haze, Smoke]
high_risk = (congestion_category in [High, Severe]) AND (weather_main in SEVERE_WEATHER OR is_low_visibility)
```

Where `congestion_category` is derived from `traffic_volume` quartiles (Q1=1,248.5, Q2=3,427, Q3=4,952). This label should not be interpreted as a prediction of actual accidents.

## Project Structure

```
capstone_part3/
├── supervised_ml.py       # Classification (high_risk) + regression (traffic volume)
├── unsupervised_ml.py     # K-means clustering + association rule mining
├── deep_learning.py       # PyTorch neural net + SHAP explainability
├── advanced_ai.py         # MLflow experiment tracking
├── recommendation.py      # Travel timing recommendation system
├── monitoring.py          # Time-based drift detection (2012-2017 vs 2018)
├── data_loader.py         # Shared data-loading utility
├── deployment/
│   ├── app.py             # FastAPI prediction API
│   └── test_api.py        # API test script
├── mlflow_logs/
│   └── mlflow.db          # MLflow tracking database
├── reports/
│   ├── model_versions.md  # Model registry with metrics
│   └── monitoring_report.txt
├── responsible_ai.md      # Bias, fairness, governance report
├── capstone_report.md     # Final Part 3 report
└── requirements.txt
```

## How to Run

Run from the **repository root** in this order (Part 2 pipeline must be run first):

```bash
# Supervised ML
python capstone_part3/supervised_ml.py

# Unsupervised ML
python capstone_part3/unsupervised_ml.py

# Deep learning + SHAP
python capstone_part3/deep_learning.py

# MLflow experiment tracking
python capstone_part3/advanced_ai.py

# Recommendation system
python capstone_part3/recommendation.py

# Drift monitoring
python capstone_part3/monitoring.py
```

### FastAPI Deployment

```bash
# Start the API server
uvicorn capstone_part3.deployment.app:app --reload

# In another terminal, test it
python capstone_part3/deployment/test_api.py
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///capstone_part3/mlflow_logs/mlflow.db
```

Then open http://localhost:5000 to browse experiments.

## Data Source

All Part 3 scripts load from `capstone_part2/cleaned_traffic.csv` (40,575 rows, deduplicated and outlier-imputed), ensuring the data quality fixes from Part 2 carry forward.
