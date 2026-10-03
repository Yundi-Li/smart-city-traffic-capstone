# Smart City Traffic Intelligence: From Data Analytics to AI-Powered Mobility

An end-to-end traffic intelligence solution using the Metro Interstate Traffic Volume dataset (~48,000 hourly records of westbound I-94 near Minneapolis-St Paul, 2012–2018).

## Project Structure

```
smart-city-traffic-capstone/
├── data/
│   └── Metro_Interstate_Traffic_Volume.csv
├── capstone_part1/                  # Part 1 — Data Analytics
│   ├── sql_analysis.py              # SQLite analysis (Tasks 1.1-1.3)
│   ├── queries.sql                  # All SQL queries
│   ├── traffic.db                   # SQLite database
│   ├── statistics_probability.py    # Descriptive stats & probability (Tasks 2-3)
│   ├── dashboard.py                 # Supplementary interactive Plotly dashboard (Task 4)
│   ├── traffic_dashboard.html       # Generated dashboard output
│   ├── insights_report.md           # Part 1 findings report
│   └── powerbi_instructions.md
├── capstone_part2/                  # Part 2 — Python Pipeline
│   ├── pipeline.py                  # Data cleaning pipeline
│   ├── feature_engineering.py       # Feature creation
│   ├── visualizations.py            # Matplotlib chart generation
│   ├── app.py                       # CLI mini-application
│   ├── logging_config.py            # Centralised logging setup
│   ├── cleaned_traffic.csv          # Pipeline output (40,575 rows × 10 cols) — generated, reproducible via pipeline
│   ├── featured_traffic.csv         # Featured output (40,575 × 37) — generated, reproducible via pipeline
│   ├── pipeline.log                 # Sample log (normal run)
│   ├── pipeline_debug_sample.log    # Sample log (debug run)
│   ├── logs/pipeline.log            # Runtime log
│   ├── figures/                     # Part 2 figures
│   ├── report.md                    # Methodology report
│   ├── README.md                    # Part 2 documentation
│   └── requirements.txt
├── capstone_part3/                  # Part 3 — ML & AI
│   ├── supervised_ml.py             # Classification & regression
│   ├── unsupervised_ml.py           # K-means & association rules
│   ├── deep_learning.py             # PyTorch neural net + SHAP
│   ├── advanced_ai.py               # MLflow experiment tracking
│   ├── recommendation.py            # Travel timing recommender
│   ├── monitoring.py                # Time-based drift detection
│   ├── data_loader.py               # Shared data-loading utility
│   ├── deployment/
│   │   ├── app.py                   # FastAPI prediction API
│   │   └── test_api.py              # API test script
│   ├── mlflow_logs/mlflow.db        # MLflow tracking database
│   ├── reports/
│   │   ├── model_versions.md        # Model registry
│   │   └── monitoring_report.txt    # Drift monitoring results
│   ├── responsible_ai.md            # Bias, fairness & governance
│   ├── capstone_report.md           # Final Part 3 report
│   ├── README.md                    # Part 3 documentation
│   └── requirements.txt
└── figures/                         # All generated visualisations
```

## How to Run

### Prerequisites

```bash
pip install -r capstone_part2/requirements.txt
pip install -r capstone_part3/requirements.txt
```

### Part 1: Data Analytics

```bash
python capstone_part1/sql_analysis.py
python capstone_part1/statistics_probability.py
python capstone_part1/dashboard.py
```

### Part 2: Python Pipeline (run in order)

```bash
python capstone_part2/pipeline.py
python capstone_part2/feature_engineering.py
python capstone_part2/visualizations.py
```

Launch the CLI application:

```bash
python capstone_part2/app.py
```

All scripts accept `--debug` for DEBUG-level logging.

### Part 3: Machine Learning & AI

```bash
python capstone_part3/supervised_ml.py
python capstone_part3/unsupervised_ml.py
python capstone_part3/deep_learning.py
python capstone_part3/advanced_ai.py
python capstone_part3/recommendation.py
python capstone_part3/monitoring.py
```

Deploy the prediction API:

```bash
uvicorn capstone_part3.deployment.app:app --reload
python capstone_part3/deployment/test_api.py
```

### MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///capstone_part3/mlflow_logs/mlflow.db
```

Then open http://localhost:5000 to browse experiments and the Model Registry.

## Logging

All Python scripts use the `logging` module for internal status reporting. `print()` is used only for user-facing result tables.

- **Log file:** `capstone_part2/logs/pipeline.log`
- **Log format:** `timestamp | level | module | message`
- **Levels:** DEBUG (intermediate values, `--debug` only), INFO (milestones), WARNING (rows dropped/imputed), ERROR (failures)
- **Configuration:** Centralised in `capstone_part2/logging_config.py`

## Dataset

Metro Interstate Traffic Volume dataset from Kaggle. ~48,204 raw hourly records of westbound I-94 traffic (2012–2018) with weather and holiday information. After deduplication and cleaning: 40,575 unique hourly records.

## Mapping to Suggested Structure

Folder names follow the deliverable wording in the capstone instructions (`capstone_part1/`, `capstone_part2/`, `capstone_part3/`) rather than the suggested structure names.

| Suggested Item | Location in This Repo |
|---|---|
| part1_data_analytics/sql | capstone_part1/sql_analysis.py, queries.sql, traffic.db |
| part1_data_analytics/statistics | capstone_part1/statistics_probability.py |
| part1_data_analytics/dashboard | capstone_part1/dashboard.py, traffic_dashboard.html (interactive traffic dashboard) |
| part1_data_analytics/report | capstone_part1/insights_report.md |
| part2_python/pipeline | capstone_part2/pipeline.py |
| part2_python/feature_engineering | capstone_part2/feature_engineering.py |
| part2_python/visualizations | capstone_part2/visualizations.py, figures/ |
| part2_python/cli_app | capstone_part2/app.py |
| part2_python/logs | capstone_part2/pipeline.log, pipeline_debug_sample.log, logs/ |
| part2_python/report | capstone_part2/report.md |
| part2_python/README | capstone_part2/README.md |
| part3_machine_learning/supervised | capstone_part3/supervised_ml.py |
| part3_machine_learning/unsupervised | capstone_part3/unsupervised_ml.py |
| part3_machine_learning/deep_learning | capstone_part3/deep_learning.py |
| part3_machine_learning/mlflow | capstone_part3/advanced_ai.py, mlflow_logs/ |
| part3_machine_learning/recommendation | capstone_part3/recommendation.py |
| part3_machine_learning/deployment | capstone_part3/deployment/app.py, test_api.py |
| part3_machine_learning/monitoring | capstone_part3/monitoring.py |
| part3_machine_learning/responsible_ai | capstone_part3/responsible_ai.md |
| part3_machine_learning/report | capstone_part3/capstone_report.md |
| part3_machine_learning/README | capstone_part3/README.md |
| final_capstone_report | final_capstone_report.md (PDF if generated) |

## Proxy Accident-Risk Label

No real accident dataset was provided. A proxy label was created: `high_risk = (congestion_category in [High, Severe]) AND (severe_weather OR is_low_visibility)`. This demonstrates the ML classification workflow and should not be interpreted as actual accident prediction. See `capstone_part3/README.md` for the exact definition.
