# Part 3 Final Report: Machine Learning and AI for Traffic Intelligence

## 1. Supervised Machine Learning

Two supervised learning tasks were developed using a common feature set: cyclical hour/day encodings, is_weekend, is_holiday, is_low_visibility, temp, rain_1h, snow_1h, clouds_all, and one-hot encoded weather_main. Data source: Part 2 cleaned dataset (40,575 rows).

### Classification — Proxy Accident-Risk Prediction

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|-------|----------|-----------|--------|-----|---------|
| LogisticRegression | 0.9646 | 0.8003 | 0.9874 | 0.8840 | 0.9942 |
| RandomForestClassifier | 0.9866 | 0.9317 | 0.9729 | 0.9519 | 0.9973 |

**Important caveat:** These scores are inflated because the proxy label is constructed from features the model can see. The `high_risk` label depends on `congestion_category` (derived from `traffic_volume` quartiles) and `weather_main` — both of which influence the feature space. This circularity means real-world accident prediction performance would be substantially lower. The classification task demonstrates the ML workflow, not a production-ready accident predictor.

### Regression — Traffic Volume Prediction

| Model | MAE | R² |
|-------|-----|-----|
| LinearRegression | 831.87 | 0.7036 |
| GradientBoostingRegressor | 273.68 | 0.9425 |

GradientBoosting achieves R² = 0.9425 and MAE = 273.68 vehicles/hour, substantially outperforming the linear baseline. Feature importance analysis shows hour-related features dominate, consistent with the Part 1 finding that traffic is primarily time-driven.

## 2. Unsupervised Machine Learning

### K-Means Clustering (k=4)

Clustering used cyclical hour encoding, an ordinal weather severity score (Clear=0, Clouds=1, Mist/Haze/Drizzle=2, Rain/Fog=3, Snow/Thunderstorm/Squall/Smoke=4), and traffic volume, all min-max scaled. The elbow method and silhouette analysis both supported k=4.

| Cluster | Name | Size | Avg Hour | Avg Severity | Avg Volume | Operational Meaning |
|---------|------|------|----------|-------------|------------|---------------------|
| 0 | Night lull | 10,966 | 2.8 | 1.22 | 843 | Schedule roadwork and lane closures |
| 1 | Shoulder hours | 8,695 | 20.9 | 1.15 | 2,516 | Transitional; ramp metering smooths flow |
| 2 | Peak commute (AM) | 10,955 | 9.2 | 1.29 | 4,723 | Signal priority and congestion pricing |
| 3 | Peak commute (PM) | 9,959 | 15.5 | 1.23 | 5,086 | Signal priority and congestion pricing |

### Association Rule Mining

Top 5 rules by lift:

| Rule | Support | Confidence | Lift |
|------|---------|------------|------|
| High congestion + cloudy → weekend + afternoon | 0.027 | 0.21 | 3.56 |
| weekend + afternoon → High congestion + cloudy | 0.027 | 0.45 | 3.56 |
| weekend + afternoon + clear → High congestion | 0.014 | 0.84 | 3.37 |
| weekend + afternoon + cloudy → High congestion | 0.027 | 0.83 | 3.33 |
| weekend + afternoon → High congestion | 0.050 | 0.83 | 3.33 |

Weekend afternoons are strongly associated with "High" congestion (lift > 3.3), and this holds across weather types. The highest-lift rule (3.56) links High congestion in cloudy weather specifically to weekend afternoons.

## 3. Deep Learning and Explainability

A PyTorch feedforward neural network (128→64→32→1, ReLU, dropout 0.3/0.2) was trained for 50 epochs with early stopping.

| Model | MAE | R² |
|-------|-----|-----|
| Neural Network (PyTorch) | 347.58 | 0.9227 |
| GBR Surrogate | 273.33 | 0.9428 |

SHAP was applied to a GradientBoostingRegressor surrogate rather than the neural network directly because: (1) neural nets are opaque — DeepExplainer/GradientExplainer can be unstable; (2) TreeExplainer provides exact Shapley values in polynomial time; (3) the GBR achieves comparable accuracy (R²=0.9428 vs 0.9227), so feature importance insights transfer.

**Top SHAP features:** hour_cos, hour_sin, and day_of_week cyclical encodings dominate — confirming that time-of-day is the primary traffic driver. Temperature and cloud cover have secondary effects. Weather one-hot features contribute marginally.

## 4. Advanced AI Technique: MLflow Experiment Tracking

MLflow was selected because it connects naturally to the MLOps pipeline. Three models were tracked:

| Model | MAE | R² |
|-------|-----|-----|
| RandomForestRegressor | 282.88 | 0.9376 |
| GradientBoostingRegressor | 273.20 | 0.9428 |
| PyTorchNeuralNet | 300.56 | 0.9348 |

**Why MLflow:** Reproducibility (exact parameters/metrics for every run), structured model comparison, artifact management (models stored with provenance), standardised API across frameworks.

**Limitations:** Adds I/O overhead, requires learning MLflow concepts, local file-backed store doesn't scale to teams, no Model Registry used here.

## 5. Recommendation System

The travel-timing recommender analyses historical traffic by hour, day type, and weather condition, restricted to realistic hours (06:00–22:00). Example output:

> "For a weekday journey in clear weather, consider travelling between 10:00 PM – 11:00 PM, when traffic is typically ~2,223 vehicles/hour, about 65% below the 4:00 PM peak."

Recommendations are grounded in historical averages from the cleaned dataset, not model predictions, because the averages are transparent and verifiable.

## 6. MLOps and Deployment Simulation

**Model Versioning:** Documented in `reports/model_versions.md`. GradientBoostingRegressor (MAE=273.68, R²=0.9425) designated as production model.

**Deployment:** FastAPI app with `/predict` endpoint (accepts hour, day_of_week, temp, weather, etc., returns predicted volume and congestion level) and `/health` endpoint. Test script validates both endpoints.

**Monitoring:** Time-based drift detection trains on 2012–2017, treats 2018 as production data. Results:
- Prediction error drift: holdout MAE=303.60, production MAE=283.04, change=−6.77% → **PASS**
- Feature drift: temp (KS=0.1303, p≈0) → **ALERT**; clouds_all (KS=0.0858, p≈0) → **ALERT**; rain_1h → PASS; hour → PASS

The temp and clouds_all alerts are expected: the 2018 data only covers Jan–Sep, so its seasonal distribution differs from training data that includes complete years. This would trigger a retraining review in a production system.

## 7. Responsible and Sustainable AI

See `responsible_ai.md` for the full report. Key points:

- **Data limitations:** Single corridor (I-94 westbound), single direction, 10-month sensor gap (Aug 2014 – Jun 2015), no demographic or incident data.
- **Proxy label risk:** The accident-risk label is synthetic and should not be used for real-world safety decisions. Classifier scores are inflated by circularity.
- **Error distribution:** Regression errors vary by hour (higher MAE during rush hours due to greater variance) and by weather (rarer conditions have fewer training examples).
- **Governance:** Human oversight required before deployment; regular retraining cadence; transparent model documentation.
- **Sustainability:** GBR trains in ~6 seconds vs ~8 seconds for the neural net with comparable accuracy — tree models offer a better performance-to-compute ratio for this tabular dataset.

## 8. Integration

All components form a coherent pipeline: Part 2's cleaned data feeds Part 3's models. Supervised models provide the predictive foundation. Unsupervised learning reveals traffic regimes (4 clusters) and congestion patterns (association rules). SHAP explains what drives predictions (time features dominate). The recommendation system translates insights into actionable travel advice. MLflow tracks all experiments. FastAPI serves predictions. Monitoring validates deployment readiness and flags distribution drift. The responsible AI assessment provides the governance framework for trustworthy deployment.
