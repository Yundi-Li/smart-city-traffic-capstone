# Smart City Traffic Intelligence: Final Capstone Report

## Executive Summary

This capstone project builds an end-to-end traffic intelligence solution using approximately 48,000 hourly records of westbound I-94 traffic near Minneapolis--St Paul (2012--2018). Part 1 establishes that traffic is overwhelmingly time-driven (r = 0.1303 for temperature--volume correlation, congestion odds ratio of 0.74 for clear vs cloudy weather). Part 2 constructs a reproducible Python pipeline that cleans 48,204 raw rows down to 40,575 unique hourly records and engineers 37 features. Part 3 applies supervised, unsupervised, and deep learning models -- with GradientBoosting achieving MAE = 246.80 and R squared = 0.9593 -- and deploys a FastAPI prediction API with MLflow tracking and drift monitoring.

---

## Part 1 -- Data Analytics

### 1.1 SQL-Based Traffic Analysis

The dataset contains 48,204 raw rows spanning October 2012 to September 2018. After removing 17 exact duplicates and aggregating 5,430 timestamps that appear more than once (accounting for 7,612 extra rows from multiple weather readings per hour), the dataset contains 40,575 unique hourly records. Coverage is uneven: 2012 has 2,103 hours (partial), 2013 has 7,294, 2014 has 4,501, 2015 has 3,593, 2016 has 7,838, and 2017 has 8,713. A major sensor gap runs from 2014-08-08 to 2015-06-11.

**Annual trends:** Raw yearly totals swing wildly due to uneven coverage, but the coverage-adjusted average is stable at 3,194--3,377 vehicles/hour, with the largest year-on-year change only +5.73% (2016 to 2017).

**Holiday temperature patterns:** Labor Day temperatures are consistently warm (18--22 degrees C) with average traffic of 2,152--2,430 vehicles/hour. New Year's Day temperatures are sub-zero (-6 to -1 degrees C) with lower traffic of 1,832--2,091 vehicles/hour.

### 1.2 Descriptive Statistics and Correlation

| Statistic | Value |
|-----------|-------|
| Mean | 3,259.82 vehicles/hour |
| Median | 3,380.00 vehicles/hour |
| Std Dev | 1,986.86 |
| Variance | 3,947,615.32 |
| Range | 0 to 7,280 |

The mean is lower than the median, indicating a left-skewed distribution caused by bimodality: daytime hours cluster around 4,000--6,000 vehicles/hour while overnight hours regularly drop below 1,000. The standard deviation of 1,987 (61% of the mean) reflects this wide day--night spread.

The Pearson correlation between temperature and traffic volume is r = 0.1303, a weak positive relationship likely confounded by seasonality and time of day.

### 1.3 Probability and Conditional Probability

Defining congestion as traffic volume > 5,500 vehicles/hour:

| Probability | Value |
|-------------|-------|
| P(Congestion) | 14.73% |
| P(Clear Weather) | 27.78% |
| P(Congestion and Clear) | 3.66% |
| P(Clear given Congestion) | 24.83% |

**Independence test:** P(Congestion) x P(Clear) = 0.0409 vs P(Congestion and Clear) = 0.0366. The small difference indicates approximate independence -- weather type alone does not strongly predict congestion.

**Odds ratio (clear vs cloudy):** 0.74. Equivalently, congestion odds are 1.36x higher in cloudy weather than clear -- a modest effect compared to time-of-day drivers.

### 1.4 Dashboard

An interactive HTML dashboard was built using Plotly as a macOS-compatible alternative to Power BI Desktop. It includes daily traffic trends, hourly patterns, weather impact analysis, temperature--traffic scatter plots, KPI cards, and a weather condition filter. Clouds has the highest average traffic (3,617 vehicles/hour); Squall has the lowest (420).

---

## Part 2 -- Python Pipeline

### 2.1 Data Pipeline Construction

The pipeline processes 48,204 raw rows through the following logged steps:

1. Schema validation (9 expected columns confirmed)
2. Weather standardisation (1,730 description entries normalised)
3. Exact duplicate removal (17 rows removed)
4. Timestamp aggregation (5,430 timestamps with 7,612 extra rows, reduced to 40,575 unique hourly records by keeping the most severe weather per hour)
5. Missing-value check (none found; 40,522 rows carry "No Holiday" as expected)
6. Outlier imputation (10 rows with 0 K temperature, 1 row with rain_1h > 9,000 mm, all imputed using per-month medians)

Final cleaned dataset: **40,575 rows x 10 columns**.

### 2.2 Feature Engineering

Starting from the cleaned dataset, 27 new columns were created:

- **Time features (6):** hour, day_of_week, is_weekend, plus cyclical sine/cosine encodings
- **Holiday feature (1):** is_holiday flagging all 24 hours of each holiday date (1,203 rows across 53 dates)
- **Weather features (15):** 11 one-hot encoded weather_main categories, binary rain/snow indicators, is_low_visibility, weather_severity ordinal (0--4)
- **Normalised features (3):** min-max scaled temp_norm, clouds_all_norm, rain_1h_norm
- **Congestion category (1):** quartile-based classification (Low/Medium/High/Severe, approximately 10,144 each)

Final featured dataset: **40,575 rows x 37 columns**.

### 2.3 Visualisations

Four Matplotlib visualisations were generated:

1. **Hourly traffic by day type:** weekdays show bimodal peaks at 07:00--08:00 (~5,800) and 16:00--17:00 (~5,900); weekends show a single plateau at ~3,500--4,000
2. **Traffic distribution:** bimodal histogram reflecting the day/night split, with the 5,500 congestion threshold between modes
3. **Temperature vs traffic scatter:** stable traffic across 260--300 K with slight drops at extreme cold
4. **Day-hour heatmap:** confirms weekday commute pattern dominance

### 2.4 Mini-Application

An interactive CLI tool with five commands: `traffic <date>`, `peak <weekday|weekend>`, `compare <month>`, `recommend <day_type> <weather>`, and `help`. All invocations are logged; invalid input triggers ERROR-level logging with user-friendly messages.

### 2.5 Logging and Reproducibility

All scripts use Python's `logging` module with centralised configuration. Format: `timestamp | level | module | message`. Running `pipeline.py -> feature_engineering.py -> visualizations.py` regenerates all outputs from the raw CSV. Every step is logged with timestamps, levels, and row counts.

---

## Part 3 -- Machine Learning and AI

### 3.1 Supervised Machine Learning

A chronological split (train: 2012--2016, test: 2017) was used instead of random splitting to avoid leaking temporal autocorrelation.

**Regression -- Traffic Volume Prediction:**

| Model | MAE | R squared |
|-------|-----|-----------|
| LinearRegression | 826.36 | 0.7105 |
| GradientBoostingRegressor | 246.80 | 0.9593 |

**Classification -- Proxy Accident-Risk Prediction:**

| Model | Accuracy | F1 | ROC AUC |
|-------|----------|-----|---------|
| LogisticRegression | 0.9705 | 0.9055 | 0.9960 |
| RandomForestClassifier | 0.9885 | 0.9603 | 0.9976 |

**Proxy label caveat:** The high_risk label is derived from traffic_volume quartiles and weather_main, both of which are features in the model. This circularity inflates classification scores far beyond what a real accident predictor would achieve. The classification task demonstrates the ML workflow, not a production-ready accident predictor.

### 3.2 Unsupervised Machine Learning

**K-Means Clustering (k=4):**

| Cluster | Name | Size | Avg Volume | Operational Meaning |
|---------|------|------|------------|---------------------|
| 0 | Night lull | 10,958 | 842 | Schedule roadwork |
| 1 | Shoulder hours | 8,700 | 2,515 | Ramp metering |
| 2 | Peak AM | 10,956 | 4,723 | Signal priority |
| 3 | Peak PM | 9,961 | 5,086 | Signal priority |

**Association Rule Mining (top rule):** weekend + afternoon + clear -> High congestion, with confidence = 0.84 and lift = 3.36. Weekend afternoons almost always produce High congestion regardless of weather. Weekday afternoons push into Severe congestion (73--74% confidence, lift ~3x). Afternoon timing is the dominant factor; weather type has minimal additional effect.

### 3.3 Deep Learning and Explainability

A PyTorch feedforward neural network (128->64->32->1, ReLU, dropout 0.3/0.2) was trained for 30 epochs with early stopping.

| Model | MAE | R squared |
|-------|-----|-----------|
| Neural Network | 345.85 | 0.9385 |
| GBR Surrogate | 246.60 | 0.9593 |

SHAP was applied to a GradientBoostingRegressor surrogate (1,000-row sample via TreeExplainer) rather than the neural network directly, because TreeExplainer provides exact Shapley values and the GBR achieves substantially better accuracy. Top SHAP features: hour_cos, hour_sin, and day_of_week cyclical encodings dominate, confirming time-of-day as the primary traffic driver.

### 3.4 Advanced AI Technique (MLflow)

Six runs were tracked -- four regression and two classification. Model Registry: v1 = LinearRegression (baseline), v2 = GradientBoostingRegressor (production alias). MLflow enables reproducibility (exact parameters/metrics for every run), structured model comparison, and stage-based promotion with rollback. The production model is also saved via joblib for portability.

### 3.5 Recommendation System

The travel-timing recommender analyses historical traffic by hour, day type, and weather condition (restricted to 06:00--22:00). Recommendations are grounded in historical averages from the cleaned dataset rather than model predictions, ensuring transparency and verifiability.

### 3.6 MLOps and Deployment

**Full flow:** MLflow experiment tracking -> Model Registry (2 versions) -> FastAPI prediction API -> drift monitoring.

**Deployment:** FastAPI app with `/predict` endpoint (accepts hour, day_of_week, temp, weather, etc.) and `/health` endpoint. Test script validates both.

**Monitoring (train: 2012--2016, production: 2018):**

| Check | Result | Status |
|-------|--------|--------|
| Prediction error drift | -35.71% change (error decreased) | PASS |
| Feature drift: temp | KS = 0.1251 | ALERT |
| Feature drift: clouds_all | KS = 0.1006 | ALERT |
| Feature drift: rain_1h | -- | PASS |
| Feature drift: hour | -- | PASS |
| Overall | Requires investigation | ALERT |

The temp and clouds_all alerts are expected: 2018 data covers only Jan--Sep, so seasonal distributions differ from the full-year training data. This would trigger a retraining review in production.

### 3.7 Responsible and Sustainable AI

**Data limitations:** Single corridor (I-94 westbound), single direction, 10-month sensor gap, no demographic or incident data.

**Proxy label risk:** The synthetic accident-risk label should not be used for real-world safety decisions. Classifier scores are inflated by circularity.

**Error distribution:** Regression errors are higher during rush hours (greater variance) and for rare weather conditions (fewer training examples).

**Governance:** Human-in-the-loop oversight, A/B testing before system-wide deployment, quarterly retraining cadence, transparent model documentation.

**Sustainability:** GBR trains in ~6 seconds vs ~8 seconds for the neural net with comparable or better accuracy. Tree-based models offer the best performance-to-compute ratio for this tabular dataset.

---

## Integration

The three parts form a coherent end-to-end traffic intelligence pipeline:

1. **Part 1 (Analytics)** establishes the foundational insight that traffic is time-driven, not weather-driven, through SQL analysis, descriptive statistics (mean = 3,260, median = 3,380, std = 1,987), correlation analysis (r = 0.13), and probability analysis (congestion odds ratio = 0.74). These findings set expectations for what features will matter in modelling.

2. **Part 2 (Pipeline)** builds the reproducible data infrastructure: cleaning 48,204 raw rows to 40,575 unique records, engineering 37 features including cyclical time encodings and weather severity scores, and producing the visualisations and CLI tools that make the data accessible to stakeholders.

3. **Part 3 (ML/AI)** validates the Part 1 insights quantitatively: GradientBoosting achieves R squared = 0.9593 using primarily time features (confirmed by SHAP), K-means discovers four operationally meaningful traffic regimes, and association rules show that afternoon timing dominates congestion patterns regardless of weather. MLflow tracks all experiments, FastAPI serves predictions, drift monitoring validates deployment readiness, and the responsible AI assessment provides the governance framework.

The pipeline flows naturally: raw data -> cleaned data (Part 2) -> featured data (Part 2) -> models (Part 3) -> predictions (Part 3 API) -> monitoring (Part 3). Each part's outputs are the next part's inputs, and findings reinforce each other consistently.
