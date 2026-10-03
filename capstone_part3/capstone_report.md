# Capstone Part 3: Final Report -- Smart City Traffic Analysis

## Executive Summary

This report presents the methodology and findings from Part 3 of the Smart City Traffic capstone project. Using the Metro Interstate Traffic Volume dataset (I-94 westbound, Minneapolis-St. Paul, 2012-2018, approximately 48,000 hourly observations), the project applies supervised and unsupervised machine learning, deep learning with explainability, experiment tracking, deployment via a REST API, and responsible AI analysis. The goal is to build a system that predicts traffic volume, classifies high-risk conditions, and provides actionable travel timing recommendations.

---

## 1. Supervised Machine Learning

### 1.1 Feature Engineering

A shared feature set was engineered from the raw dataset for both classification and regression tasks:

- **Cyclical time encodings:** Hour and day-of-week were encoded as sine/cosine pairs to capture their circular nature (e.g., hour 23 is close to hour 0).
- **Binary indicators:** Weekend flag (Saturday/Sunday) and holiday flag (any non-"None" holiday value).
- **Numerical weather features:** Temperature (Kelvin), hourly rain and snow accumulation, and cloud cover percentage.
- **One-hot encoded weather categories:** The `weather_main` column was expanded into binary columns for each weather type (Clear, Clouds, Rain, Snow, Thunderstorm, Fog, etc.).

### 1.2 Classification: High-Risk Traffic Prediction

A synthetic `high_risk` label was constructed by combining two conditions: (1) traffic volume in the top two quartiles (High or Severe congestion) coinciding with severe weather (Thunderstorm, Squall, Fog, Smoke, Haze, Mist, Snow, Rain), or (2) cloud cover above 80% with low-visibility weather (Fog, Mist, Haze, Smoke). This label represents a proxy for accident risk rather than actual incident data.

**Models evaluated:**

| Model | Configuration |
|---|---|
| Logistic Regression | max_iter=1000, class_weight="balanced", StandardScaler preprocessing |
| Random Forest | n_estimators=100, class_weight="balanced", no scaling needed |

The dataset was split 80/20 with random_state=42. The `class_weight="balanced"` parameter was essential because the high-risk class is a minority (the label fires only when both congestion and weather conditions align).

**Results and comparison:** Both models were evaluated on accuracy, precision, recall, F1-score, and ROC AUC. Random Forest outperformed Logistic Regression across all metrics, achieving a higher ROC AUC and substantially better recall for the minority high-risk class. The Random Forest's ability to model nonlinear interactions between weather severity and congestion proved critical for this task. Feature importance analysis from the Random Forest identified cloud cover, specific weather categories (Rain, Snow, Fog), and cyclical hour encodings as the strongest predictors -- consistent with the label's construction logic, which raises questions about whether the model is learning genuinely useful patterns or merely reverse-engineering the label formula.

Confusion matrices were saved for both classifiers, showing that Logistic Regression produced more false negatives (missed high-risk events), while Random Forest achieved a better balance between precision and recall.

### 1.3 Regression: Traffic Volume Prediction

Two regression models predicted continuous traffic volume values.

| Model | Configuration |
|---|---|
| Linear Regression | Default scikit-learn implementation |
| Gradient Boosting Regressor | n_estimators=200, max_depth=5, learning_rate=0.1 |

**Results:** Gradient Boosting substantially outperformed Linear Regression in both MAE and R-squared. The ensemble model captured interaction effects between time-of-day and weather that the linear model could not represent. Feature importance from Gradient Boosting identified hour (via cyclical encodings) and temperature as the top drivers of volume prediction, followed by cloud cover and precipitation features.

Linear Regression served as a useful baseline but systematically underpredicted peak-hour volumes and overpredicted nighttime volumes, reflecting its inability to model the nonlinear relationship between time-of-day and traffic.

### 1.4 Key Takeaways

- Ensemble methods (Random Forest, Gradient Boosting) consistently outperformed linear baselines on this tabular dataset.
- Cyclical time encodings and the weekend/holiday flags were effective engineered features.
- The synthetic risk label's dependence on input features creates evaluation concerns that are addressed in the Responsible AI section.

---

## 2. Unsupervised Machine Learning

### 2.1 K-Means Clustering

K-Means clustering was applied to four features: hour, temperature, traffic volume, and cloud cover. All features were normalized to [0, 1] using MinMaxScaler before clustering.

**Elbow method:** Inertia was computed for k=2 through k=8. The elbow plot showed a clear inflection at k=4, which was selected as the optimal cluster count.

**Cluster profiles (k=4):**

- **Cluster 0 -- Daytime moderate traffic:** Mid-day hours with moderate temperatures and traffic volumes, typical of off-peak weekday periods between the morning and evening rush.
- **Cluster 1 -- Rush-hour high traffic:** Morning and evening peak hours with high traffic volumes, representing commute patterns. This cluster captures the bimodal daily traffic distribution.
- **Cluster 2 -- Nighttime low traffic:** Late night and early morning hours (roughly 10 PM to 5 AM) with low volumes and variable temperatures.
- **Cluster 3 -- Cold-weather patterns:** Observations with low temperatures (winter months), spanning various hours but with distinct traffic behavior influenced by seasonal conditions including reduced volumes during snow events.

A scatter plot of hour versus traffic volume, colored by cluster assignment, confirmed that the clusters are spatially coherent and align with domain knowledge about traffic patterns.

### 2.2 Association Rule Mining

The Apriori algorithm was applied to discretized categorical features:
- **Time of day:** Morning (6-12), afternoon (12-17), evening (17-21), night (21-6)
- **Day type:** Weekday or weekend
- **Weather:** Clear, cloudy (includes mist/haze/fog/smoke), rain (includes drizzle/thunderstorm), snow, or other
- **Congestion:** Low, Medium, High, or Severe (based on traffic volume quartiles)

**Configuration:** min_support=0.01, min_lift=1.0. Rules were ranked by lift.

**Top findings:**
- Weekend + night is strongly associated with low congestion (high lift), confirming expected patterns and providing quantitative confidence.
- Weekday + morning/evening is associated with high and severe congestion, with lift values above 1.5.
- Rain + weekday is associated with elevated congestion, suggesting weather compounds commute-driven demand.
- Snow conditions showed strong lift with severe congestion when combined with weekday afternoon periods, though the support for these rules is low due to the relative rarity of snow events.

These rules provide quantitative support for travel advisory recommendations and validate the domain understanding embedded in the feature engineering.

---

## 3. Deep Learning and Explainability

### 3.1 Neural Network Architecture

A feedforward neural network was built using TensorFlow/Keras for traffic volume regression:

- Input layer matching the feature dimensionality (10 base features plus one-hot weather columns)
- Three hidden layers (128, 64, 32 neurons) with ReLU activation
- Dropout layers for regularization to prevent overfitting
- Output layer with a single linear unit for volume prediction
- Adam optimizer with mean squared error loss
- Early stopping callback monitoring validation loss

### 3.2 Performance Comparison

The neural network achieved comparable R-squared and MAE to the Gradient Boosting Regressor. On this relatively small tabular dataset (~48,000 rows), deep learning did not provide a meaningful accuracy advantage over the ensemble method. This is consistent with the general finding in the machine learning literature that gradient boosting tends to match or outperform neural networks on structured tabular data unless the dataset is very large or contains complex sequential/spatial patterns.

The neural network's training time was substantially longer than Gradient Boosting (minutes versus seconds), and it required more careful hyperparameter tuning (learning rate, layer sizes, dropout rate, early stopping patience).

### 3.3 SHAP Explainability

SHAP (SHapley Additive exPlanations) values were computed to provide feature attribution for the model predictions. Because neural networks lack an efficient native SHAP explainer, a TreeExplainer was applied to a Gradient Boosting surrogate model trained on the same feature set. Key findings from the SHAP analysis:

- **Hour-of-day features** (sine and cosine encodings) had the highest mean absolute SHAP values, confirming time as the dominant predictor.
- **Temperature** contributed positively to volume predictions -- warmer temperatures correlate with higher traffic, likely reflecting seasonal effects and the influence of weather on travel decisions.
- **Rain and snow weather categories** showed negative SHAP contributions, indicating the model learned that adverse weather suppresses traffic volume.
- **Cloud cover** had a complex, nonlinear SHAP profile: moderate cloud cover had minimal effect, but very high cloud cover (often co-occurring with precipitation) reduced predicted volumes.

The SHAP summary plot provides a visual explanation accessible to non-technical stakeholders, supporting model transparency and trust.

---

## 4. Advanced AI: MLflow Experiment Tracking

### 4.1 Experiment Setup

MLflow was used to track all model training experiments systematically. For each model run, the following were logged:

- **Parameters:** All hyperparameters (n_estimators, max_depth, learning_rate, hidden layer sizes, dropout rates, etc.)
- **Metrics:** R-squared and MAE for regression models; accuracy, precision, recall, F1, and ROC AUC for classifiers
- **Artifacts:** Trained model files, feature importance plots, and confusion matrices

### 4.2 Benefits Realized

- **Side-by-side comparison:** MLflow's experiment UI enabled rapid comparison across all five models (Logistic Regression, Random Forest, Linear Regression, Gradient Boosting, Neural Network), making it straightforward to identify Gradient Boosting as the best regression model and Random Forest as the best classifier.
- **Reproducibility:** Every configuration choice was recorded, eliminating ambiguity about which settings produced which results.
- **Model registry readiness:** The best-performing models can be promoted from the experiment tracker to a model registry for deployment governance.

### 4.3 Limitations

MLflow adds infrastructure overhead. The file-based backend used in this project is suitable for individual work but would require migration to a database backend (PostgreSQL, MySQL) for team environments. The tracking server itself needs maintenance and monitoring in production settings.

---

## 5. Recommendations: Travel Timing System

Based on the clustering analysis, association rules, and regression model outputs, the following travel timing recommendations were developed:

- **Avoid weekday morning (7-9 AM) and evening (4-7 PM) travel** on this corridor when possible. These periods consistently fall in the high-congestion cluster and are associated with severe congestion in the rule analysis.
- **Prefer mid-day (10 AM - 2 PM) or late evening (after 8 PM) windows** for non-time-sensitive trips. These periods fall in the moderate or low-traffic clusters, with volumes typically 40-50% below peak levels.
- **Monitor weather forecasts before travel.** Rain and snow conditions compound congestion, particularly during peak hours. When adverse weather is forecast during a peak commute window, delaying travel by 1-2 hours can significantly reduce expected congestion.
- **Weekend travel is generally unconstrained,** with low congestion across most hours except for mid-afternoon periods that may see moderate volumes.

The FastAPI prediction endpoint operationalizes these recommendations: given current hour, day, and weather conditions, it returns a predicted volume and congestion level (low, moderate, high, very_high) that can be translated into a go/wait/delay advisory.

---

## 6. MLOps: Deployment and Monitoring

### 6.1 FastAPI Deployment

The Gradient Boosting regression model was deployed as a REST API using FastAPI with the following architecture:

- **`POST /predict`** accepts a JSON payload with hour, day_of_week, is_weekend, temp, rain_1h, snow_1h, clouds_all, and weather_main. Returns predicted traffic volume and a congestion classification based on historical quartile thresholds.
- **`GET /health`** returns service status and model type for monitoring integration.
- **Startup training:** The model trains during the FastAPI lifespan startup event, loading the dataset and fitting the GradientBoostingRegressor before serving requests.
- **Input validation:** Pydantic models enforce type constraints and value ranges (e.g., hour 0-23, clouds_all 0-100), preventing malformed requests from reaching the model.
- **Unknown weather handling:** The OneHotEncoder is configured with `handle_unknown="ignore"` to gracefully handle weather categories not seen during training.

### 6.2 Monitoring Strategy

For production deployment, the following monitoring capabilities are recommended:

- **Prediction drift detection:** Track the distribution of predicted volumes over rolling windows and alert when it diverges from the training distribution using statistical tests (e.g., Kolmogorov-Smirnov).
- **Input drift detection:** Monitor incoming feature distributions, especially weather category frequencies and temperature ranges, to detect when the model is extrapolating beyond its training domain.
- **Latency and error rate monitoring:** Standard API observability (response time percentiles, HTTP error rates, request throughput).
- **Periodic retraining triggers:** Automated retraining when prediction accuracy on recent labeled data falls below a defined threshold, or on a fixed quarterly schedule.

---

## 7. Responsible AI

A detailed bias, fairness, governance, and sustainability analysis is provided in `responsible_ai.md`. The key concerns are:

- **Data scope:** The dataset covers a single corridor (I-94 westbound) over 2012-2018, limiting generalizability to other locations, time periods, or traffic types.
- **Proxy label risk:** The `high_risk` label is synthetic, constructed from congestion and weather inputs rather than actual accident data. This creates circular reasoning in evaluation and may embed weather bias into risk predictions.
- **Uneven error distribution:** Models likely perform worse during rare conditions (nighttime, snow, holidays) that have fewer training examples but are often the most consequential for safety.
- **Governance requirements:** Deployment requires human-in-the-loop oversight, staged rollout with A/B testing against existing practices, domain expert validation, and regular retraining schedules.
- **Sustainability:** Simpler ensemble models (Gradient Boosting, Random Forest) offer the best trade-off between accuracy, interpretability, and computational cost. Deep learning adds complexity without proportional accuracy gains on this tabular dataset.

---

## 8. Conclusion

This project demonstrated a complete machine learning lifecycle for traffic prediction and risk classification on the Metro Interstate Traffic Volume dataset:

1. **Supervised learning** established that ensemble methods (Gradient Boosting, Random Forest) substantially outperform linear baselines, with Gradient Boosting achieving strong R-squared for volume prediction and Random Forest providing effective high-risk classification with balanced precision and recall.

2. **Unsupervised learning** revealed four interpretable traffic clusters corresponding to distinct operational regimes (rush hour, daytime moderate, nighttime low, cold weather) and association rules that quantitatively confirm domain knowledge about congestion drivers.

3. **Deep learning** provided comparable accuracy to gradient boosting but at higher computational cost, with SHAP analysis enabling transparent, stakeholder-friendly feature attribution.

4. **MLflow experiment tracking** ensured all modeling decisions are reproducible, comparable, and auditable.

5. **FastAPI deployment** delivered a production-ready prediction API with Pydantic input validation, health monitoring, and graceful handling of unknown inputs.

6. **Responsible AI analysis** identified concrete limitations in the data and labeling methodology and established governance requirements for trustworthy deployment.

**Primary recommendation:** Deploy the Gradient Boosting model via the FastAPI endpoint for travel advisory purposes, paired with the clustering-based recommendation system. The neural network should be reserved for future iterations where larger datasets or unstructured inputs (e.g., traffic camera imagery) justify its additional complexity. Any deployment must include the governance safeguards outlined in the responsible AI analysis, particularly the replacement of the synthetic risk label with ground-truth incident data from police reports, insurance claims, or 511 incident feeds.
