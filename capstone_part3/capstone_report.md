# Part 3 Final Report: Machine Learning and AI for Traffic Intelligence

## 1. Supervised Machine Learning

Two supervised learning tasks were developed using a common feature set: cyclical hour/day encodings, is_weekend, is_holiday, is_low_visibility, temp, rain_1h, snow_1h, clouds_all, and one-hot encoded weather_main. Data source: Part 2 cleaned dataset (40,575 rows). A **chronological split** was used (train: 2012-2016, test: 2017) rather than random splitting, because random splits leak neighbouring hours and inflate scores by allowing the model to memorise temporal autocorrelation.

### Classification — Proxy Accident-Risk Prediction

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|-----|---------|
| LogisticRegression | 0.9705 | 0.8472 | 0.9724 | 0.9055 | 0.9960 |
| RandomForestClassifier | 0.9885 | 0.9627 | 0.9581 | 0.9604 | 0.9981 |

**Important caveat — proxy label circularity:** These scores are inflated because the proxy label is constructed from features the model can see. The `high_risk` label depends on `congestion_category` (derived from `traffic_volume` quartiles) and `weather_main` — both of which influence the feature space. This circularity means real-world accident prediction performance would be substantially lower. The classification task demonstrates the ML workflow, not a production-ready accident predictor.

### Regression — Traffic Volume Prediction

| Model | MAE | R² |
|-------|-----|-----|
| LinearRegression | 826.36 | 0.7105 |
| GradientBoostingRegressor | 246.67 | 0.9593 |

GradientBoosting achieves R² = 0.9593 and MAE = 246.67 vehicles/hour, substantially outperforming the linear baseline. Feature importance analysis shows hour-related features dominate, consistent with the Part 1 finding that traffic is primarily time-driven.

## 2. Unsupervised Machine Learning

### K-Means Clustering (k=4)

Clustering used cyclical hour encoding, an ordinal weather severity score (Clear=0, Clouds=1, Mist/Haze/Drizzle=2, Rain/Fog=3, Snow/Thunderstorm/Squall/Smoke=4), and traffic volume, all min-max scaled. The elbow method and silhouette analysis both supported k=4.

| Cluster | Name | Size | Avg Hour | Avg Severity | Avg Volume | Operational Meaning |
|---------|------|------|----------|-------------|------------|---------------------|
| 0 | Night lull | 10,958 | 2.8 | 1.30 | 842 | Schedule roadwork and lane closures |
| 1 | Shoulder hours | 8,700 | 20.9 | 1.19 | 2,515 | Transitional; ramp metering smooths flow |
| 2 | Peak commute (AM) | 10,956 | 9.2 | 1.36 | 4,723 | Signal priority and congestion pricing |
| 3 | Peak commute (PM) | 9,961 | 15.5 | 1.27 | 5,086 | Signal priority and congestion pricing |

### Association Rule Mining

Top 5 rules (filtered to congestion-only consequents) by lift:

| Rule | Support | Confidence | Lift |
|------|---------|------------|------|
| weekend + afternoon + clear → High | 0.014 | 0.84 | 3.36 |
| weekend + afternoon + cloudy → High | 0.026 | 0.84 | 3.34 |
| weekend + afternoon → High | 0.050 | 0.83 | 3.33 |
| weekday + afternoon + clear → Severe | 0.030 | 0.74 | 2.97 |
| weekday + afternoon + cloudy → Severe | 0.056 | 0.73 | 2.94 |

In plain English: weekend afternoons almost always produce High congestion regardless of weather (84% confidence, lift ~3.3x). Weekday afternoons push into Severe congestion (73-74% confidence, lift ~3x), reflecting the commute peak. Afternoon timing is the dominant factor; weather type has minimal additional effect.

## 3. Deep Learning and Explainability

A PyTorch feedforward neural network (128→64→32→1, ReLU, dropout 0.3/0.2) was trained for 30 epochs max with early stopping.

| Model | MAE | R² |
|-------|-----|-----|
| Neural Network (PyTorch) | 465.24 | 0.8972 |
| GBR Surrogate | 246.60 | 0.9593 |

SHAP was applied to a GradientBoostingRegressor surrogate (on a 1,000-row sample via TreeExplainer) rather than the neural network directly because: (1) neural nets are opaque — DeepExplainer/GradientExplainer can be unstable; (2) TreeExplainer provides exact Shapley values in polynomial time; (3) the GBR achieves substantially better accuracy (R²=0.9593 vs 0.8972), so feature importance insights transfer.

**Top SHAP features:** hour_cos, hour_sin, and day_of_week cyclical encodings dominate — confirming that time-of-day is the primary traffic driver. Temperature and cloud cover have secondary effects. Weather one-hot features contribute marginally.

## 4. Advanced AI Technique: MLflow Experiment Tracking

MLflow was selected because it connects naturally to the MLOps pipeline. Six models were tracked — four regression and two classification:

**Regression runs:**

| Model | MAE | R² | Registry |
|-------|-----|-----|----------|
| LinearRegression | 826.36 | 0.7105 | v1 (baseline) |
| GradientBoostingRegressor | 246.80 | 0.9593 | v2 (production) |
| RandomForestRegressor | 251.25 | 0.9550 | — |
| PyTorchNeuralNet | 345.85 | 0.9385 | — |

**Classification runs:**

| Model | Accuracy | F1 | ROC AUC |
|-------|----------|-----|---------|
| LogisticRegression | 0.9706 | 0.9058 | 0.9960 |
| RandomForestClassifier | 0.9885 | 0.9603 | 0.9976 |

**Why MLflow:** Reproducibility (exact parameters/metrics for every run), structured model comparison, artifact management (models stored with provenance), standardised API across frameworks.

**Model Registry:** Two versions registered under `traffic_volume_regressor`: v1 = LinearRegression (baseline), v2 = GradientBoosting (alias "production"). The registry enables stage-based promotion and rollback.

**Portability:** MLflow stores absolute paths in its SQLite DB, so the registry only works on the original machine. For portability, the production model is also saved to `capstone_part3/models/production_model/` via joblib. The FastAPI app loads from the registry first and falls back to the joblib folder.

**Limitations:** Adds I/O overhead, requires learning MLflow concepts, local SQLite-backed store doesn't scale to teams.

## 5. Recommendation System

The travel-timing recommender analyses historical traffic by hour, day type, and weather condition, restricted to realistic hours (06:00–22:00). Example output:

> "For a weekday journey in clear weather, consider travelling between 10:00 PM – 11:00 PM, when traffic is typically ~2,223 vehicles/hour, about 65% below the 4:00 PM peak."

Recommendations are grounded in historical averages from the cleaned dataset, not model predictions, because the averages are transparent and verifiable.

## 6. MLOps and Deployment Simulation

**Full flow:** MLflow experiment tracking → Model Registry (2 versions, production alias on GBR) → FastAPI prediction API → drift monitoring.

**Model Versioning:** Documented in `reports/model_versions.md`. GradientBoostingRegressor (MAE=246.67, R²=0.9593) designated as production model via MLflow Model Registry alias.

**Deployment:** FastAPI app with `/predict` endpoint (accepts hour, day_of_week, temp, weather, etc., returns predicted volume and congestion level) and `/health` endpoint. Test script validates both endpoints.

**Monitoring:** Time-based drift detection trains on 2012-2016, treats 2018 as production data. Results:
- Prediction error drift: holdout MAE vs production MAE → **ALERT** (requires investigation)
- Feature drift: temp (KS=0.1251) → **ALERT**; clouds_all (KS=0.1006) → **ALERT**; rain_1h → PASS; hour → PASS
- Overall status: **ALERT / Requires investigation**

The temp and clouds_all alerts are expected: the 2018 data only covers Jan–Sep, so its seasonal distribution differs from training data that includes complete years. This would trigger a retraining review in a production system.

**Speed settings:** 30 epochs max for NN, SHAP computed on 1,000-row sample, silhouette analysis uses sample_size=5,000, n_estimators capped at 200 for tree models.

## 7. Responsible and Sustainable AI

See `responsible_ai.md` for the full report. Key points:

- **Data limitations:** Single corridor (I-94 westbound), single direction, 10-month sensor gap (Aug 2014 – Jun 2015), no demographic or incident data.
- **Proxy label risk:** The accident-risk label is synthetic and should not be used for real-world safety decisions. Classifier scores are inflated by circularity.
- **Error distribution:** Regression errors vary by hour (higher MAE during rush hours due to greater variance) and by weather (rarer conditions have fewer training examples).
- **Governance:** Human oversight required before deployment; regular retraining cadence; transparent model documentation.
- **Sustainability:** GBR trains in ~6 seconds vs ~8 seconds for the neural net with comparable accuracy — tree models offer a better performance-to-compute ratio for this tabular dataset.

## 8. Integration

All components form a coherent pipeline: Part 2's cleaned data feeds Part 3's models. Supervised models provide the predictive foundation. Unsupervised learning reveals traffic regimes (4 clusters) and congestion patterns (association rules). SHAP explains what drives predictions (time features dominate). The recommendation system translates insights into actionable travel advice. MLflow tracks all experiments. FastAPI serves predictions. Monitoring validates deployment readiness and flags distribution drift. The responsible AI assessment provides the governance framework for trustworthy deployment.
