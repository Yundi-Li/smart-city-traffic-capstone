# Responsible and Sustainable AI Report

## 1. Bias and Fairness

### 1.1 Sampling and Coverage Limitations

The dataset covers a **single corridor** (I-94 westbound near Minneapolis–St Paul) in a **single direction**, from October 2012 to September 2018. Key gaps:

- **Sensor gap:** 2014-08-08 to 2015-06-11 (~10 months), leaving 2014 with only 4,501 hours and 2015 with 3,593 hours of data.
- **Partial years:** 2012 starts in October (2,103 hours); 2018 ends in September (6,533 hours). No full calendar year of 2012 or 2018 data exists.
- **Geographic scope:** Results apply to this specific corridor only. Traffic patterns on urban arterials, suburban roads, or other interstates would differ.
- **No demographic data:** The dataset contains no information about driver demographics, trip purposes, or vehicle types. The model cannot assess whether its predictions affect different population groups unequally.
- **No incident data:** Actual accident records were not available; the classification task uses a synthetic proxy label.

### 1.2 Proxy Label Risks

The `high_risk` label is defined as `(congestion_category in [High, Severe]) AND (severe_weather OR is_low_visibility)`. This is a **proxy for risk, not a measurement of actual accidents**. Risks of using this proxy:

- **Circularity:** The label is derived from `traffic_volume` quartiles and `weather_main`, both of which are features in the model. This inflates classification metrics (ROC AUC 0.997, F1 0.952) far beyond what a real accident predictor would achieve.
- **Incorrect assumptions:** High congestion during rain does not necessarily equal high accident risk — congestion may actually slow vehicles and reduce collision severity.
- **Deployment danger:** If deployed as an "accident prediction system," it could create a false sense of safety during conditions it labels as low-risk (e.g., low-traffic icy roads at 3 AM).

### 1.3 Uneven Error Distribution

The regression model (GradientBoosting, MAE=246.67) does not err uniformly:

- **By hour:** MAE is higher during rush hours (07:00–08:00, 16:00–17:00) where traffic variance is greatest. Overnight predictions (00:00–05:00) are more accurate because volumes are consistently low.
- **By weather:** Rare weather conditions (Squall: ~5 records, Smoke: ~50 records) have fewer training examples. Model performance on these conditions is unreliable.
- **By season:** The 2018 drift monitoring showed statistically significant distribution shifts in temperature (KS=0.1251) and cloud cover (KS=0.1006), indicating the model may perform differently across seasons.

## 2. Governance

### 2.1 Oversight Before Deployment

Before any model is trusted for real-world traffic management decisions:

- **Human-in-the-loop:** Traffic engineers and domain experts must review model recommendations before they affect signal timing or advisory messages.
- **A/B testing:** Any automated intervention (e.g., adaptive signal timing) should be tested in a controlled corridor before system-wide deployment.
- **Regular retraining:** The drift monitoring detected feature distribution shifts between training (2012-2016) and production (2018) data. Overall status: ALERT / Requires investigation. A quarterly retraining cadence with fresh data is recommended.
- **Transparent documentation:** Model limitations (single corridor, proxy labels, uneven errors) must be clearly communicated to decision-makers. The model version registry (`reports/model_versions.md`) provides a starting point.

### 2.2 Decision Boundaries

The model should inform but not replace human judgment. Specific boundaries:

- Traffic volume predictions can safely drive advisory systems ("expect heavy traffic between 4-5 PM").
- The proxy accident-risk classifier should **never** be used for safety-critical decisions without validation against real incident data.
- Recommendations should be presented with confidence context ("based on 3,500 historical observations for this condition").

## 3. Sustainability

### 3.1 Computational Trade-offs

| Model | Training Time | MAE | R² |
|-------|--------------|-----|-----|
| GradientBoostingRegressor | ~6 seconds | 246.67 | 0.9593 |
| RandomForestRegressor | ~3 seconds | 248.75 | 0.9562 |
| PyTorch Neural Net | ~8 seconds | 465.24 | 0.8972 |

For this tabular dataset, tree-based models outperform the neural network while training faster. The neural network adds complexity and compute cost without improving accuracy. In a production setting, the GradientBoostingRegressor offers the best performance-to-compute ratio.

### 3.2 Environmental Considerations

- The dataset is small (40,575 rows) and all models train in under 10 seconds on a consumer laptop. Training costs are small relative to large-scale deep learning but were not formally measured; approximate training times: GBR ~6s, RF ~3s, NN ~8s on a consumer laptop.
- However, if scaled to a city-wide system with real-time predictions across hundreds of corridors, compute costs would grow substantially. Cloud GPU instances for neural network inference would have measurable energy costs.
- **Recommendation:** Use the simpler GradientBoosting model for deployment. Reserve neural networks for cases where their additional complexity is justified by performance gains — which is not the case for this tabular traffic dataset.
