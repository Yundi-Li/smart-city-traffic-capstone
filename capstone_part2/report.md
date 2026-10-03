# Capstone Part 2: I-94 Interstate Traffic Analysis Report

## Methodology

### Data Pipeline
The pipeline processes 48,204 rows of hourly I-94 traffic data (2012-2018). Data cleaning steps include: standardising inconsistent categorical values in weather columns, parsing and validating timestamps, removing duplicate records (~7,000 duplicates found), and detecting physically implausible outliers (0 Kelvin temperatures and rainfall exceeding 9,000mm). Outliers are imputed using per-month median values rather than global medians to preserve seasonal patterns. All operations are logged with severity-appropriate levels for auditability.

### Feature Engineering
The cleaned dataset is enriched with ML-ready features: cyclical time encodings (sine/cosine transforms of hour and day-of-week) to capture periodic patterns without artificial boundaries, one-hot encoded weather categories, binary rain/snow indicators, min-max normalized numerical features, and a quartile-based congestion classification (Low/Medium/High/Severe).

### Visualizations
Four visualizations explore traffic patterns:
1. **Hourly traffic by day type** reveals distinct weekday bimodal rush-hour peaks (7-8am, 4-5pm) versus a single weekend midday plateau.
2. **Traffic distribution** shows a bimodal pattern with most volume below the 5,500-vehicle congestion threshold.
3. **Temperature vs. traffic scatter** indicates stable traffic across temperature ranges, with drops at extreme cold.
4. **Day-hour heatmap** confirms the weekday commute pattern and lighter weekend traffic across all hours.

### CLI Application
An interactive command-line tool enables ad-hoc queries: daily traffic breakdowns, peak hour identification, weekday-vs-weekend monthly comparisons, and weather-aware travel time recommendations.

## Key Findings

- **Rush-hour dominance**: Weekday traffic peaks at 7-8am and 4-5pm, reaching 5,500-6,000 vehicles/hour, while weekends peak around 12-3pm at roughly 4,000 vehicles/hour.
- **Data quality issues**: The raw dataset contains significant duplicates (~15% of rows) and sensor anomalies (0K temperatures, 9,800mm rainfall spikes), all addressed through the cleaning pipeline.
- **Weather impact**: While severe weather (snow, thunderstorms) correlates with modest traffic reductions, temperature alone has limited predictive power outside extreme cold.
- **Congestion patterns**: Approximately 25% of observations exceed the "Severe" congestion threshold, concentrated during weekday commute hours.

## Conclusions

The I-94 corridor exhibits strong temporal regularity that lends itself well to predictive modelling. The engineered cyclical time features and congestion categories provide a solid foundation for Part 3 classification tasks. The CLI application demonstrates practical utility for traffic planners seeking optimal travel windows.
