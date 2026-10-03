# Part 2 — Methodology and Findings Report

## Data Pipeline (pipeline.py)

The pipeline processes 48,204 raw rows of hourly I-94 traffic data (October 2012 – September 2018). Cleaning steps, each logged individually:

1. **Schema validation:** Confirmed all 9 expected columns are present.
2. **Weather standardisation:** 1,730 `weather_description` entries normalised to consistent casing; 0 `weather_main` entries needed adjustment.
3. **Exact duplicate removal:** 17 rows removed (identical across all columns).
4. **Timestamp aggregation:** 5,430 timestamps appear more than once, accounting for 7,612 extra rows. These arise from multiple weather condition readings logged for the same hour while `traffic_volume` is identical within each hour. Rather than simply dropping duplicates, aggregation keeps the most severe weather reading per hour, preserving information. This reduces the dataset from 48,187 to 40,575 unique hourly records.
5. **Missing-value check:** The raw holiday column contains nulls for all non-holiday hours (48,143 rows); these represent non-holiday hours and were standardised to "No Holiday". No unexpected missing values were found in any other column.
6. **Outlier detection and imputation:** 10 rows with 0 Kelvin temperature and 1 row with rain_1h > 9,000 mm (physically implausible) detected. All imputed using per-month median values rather than a global median, preserving seasonal patterns.

Final cleaned dataset: **40,575 rows × 10 columns**, saved to `cleaned_traffic.csv`.

## Feature Engineering (feature_engineering.py)

Starting from the cleaned 40,575-row dataset, 27 new columns were created:

- **Time features (6):** `hour`, `day_of_week`, `is_weekend`, plus cyclical sine/cosine encodings of hour (period 24) and day_of_week (period 7).
- **Holiday feature (1):** `is_holiday` — flags all 24 hours of each holiday date, not just the midnight row where the label appears. Result: 1,203 rows flagged across 53 holiday dates.
- **Weather features (15):** 11 one-hot encoded `weather_main` categories, binary `is_rainy` and `is_snowy` indicators, `is_low_visibility` (clouds_all > 80 AND weather in [Fog, Mist, Haze, Smoke]), and `weather_severity` ordinal (Clear=0, Clouds=1, Mist/Haze/Drizzle=2, Rain/Fog=3, Snow/Thunderstorm/Squall/Smoke=4).
- **Normalised features (3):** Min-max scaled `temp_norm` (range 243.39–310.07 K), `clouds_all_norm` (0–100), and `rain_1h_norm` (0–55.63 mm). The target variable (`traffic_volume`) is not scaled as a feature.
- **Congestion category (1):** `congestion_category` — quartile-based classification using thresholds Q1=1,248.5, Q2=3,427.0, Q3=4,952.0. Distribution: Low=10,144, Medium=10,144, High=10,145, Severe=10,142 — approximately equal by design.

Final featured dataset: **40,575 rows × 37 columns**, saved to `featured_traffic.csv`.

## Visualisations (visualizations.py)

Four Matplotlib visualisations, each saved to `figures/` and `capstone_part2/figures/`:

1. **Average traffic by hour (weekday vs weekend):** Weekdays show distinct bimodal peaks at 07:00–08:00 (~5,800 vehicles/hour) and 16:00–17:00 (~5,900). Weekends show a single broad plateau from 11:00–17:00 at ~3,500–4,000 vehicles/hour. Overnight hours (00:00–05:00) average below 1,000 for both day types.

2. **Traffic volume distribution:** A bimodal histogram reflecting the day/night split. The 5,500-vehicle congestion threshold line sits between the two modes, with approximately 14.7% of all hours exceeding it.

3. **Temperature vs traffic scatter (by weather):** Traffic is stable across a wide temperature range (260–300 K), with slight drops at extreme cold (<255 K). Weather condition colouring shows no dramatic separation — consistent with the weak r = 0.13 correlation from Part 1.

4. **Day-hour heatmap:** Confirms the weekday commute pattern (Monday–Friday, 07:00–08:00 and 16:00–17:00 are the hottest cells). Saturday and Sunday are uniformly lighter, with a gradual midday rise.

## CLI Application (app.py)

An interactive command-line tool with 5 commands: `traffic <date>` (hourly breakdown), `peak <weekday|weekend>` (top 5 hours), `compare <month>` (weekday vs weekend average), `recommend <day_type> <weather>` (best travel windows), and `help`. All commands log the invocation at INFO level; invalid input triggers an ERROR log with a user-friendly message instead of a raw traceback.

## Key Findings

- **Traffic is overwhelmingly time-driven.** Hour of day and day of week explain most variation, as shown by the bimodal weekday pattern and the flat weekend plateau.
- **Data quality required substantial attention.** Beyond the 17 exact duplicates, 5,430 timestamps appear more than once, accounting for 7,612 extra rows (15.8% of the dataset) that needed aggregation. The 10 zero-Kelvin readings and 1 extreme rain outlier (rain_1h > 9,000 mm) required monthly-median imputation.
- **Congestion is concentrated.** Only 14.7% of hours exceed 5,500 vehicles, clustering in weekday rush windows. The quartile-based congestion category distributes records evenly for balanced ML classification.
- **The pipeline is fully reproducible.** Running `pipeline.py → feature_engineering.py → visualizations.py` regenerates all outputs from the raw CSV. Every step is logged with timestamps, levels, and row counts.
