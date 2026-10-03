# Part 1 — Data Analytics Insights Report

## 1. SQL-Based Traffic Analysis

### Annual Trends (Task 1.2)

The dataset contains 48,204 raw rows spanning October 2012 to September 2018, but coverage is uneven. After deduplicating timestamps (the dataset contains 5,445 duplicate timestamp entries with different weather readings), the unique hourly records per year are: 2012: 2,103 (partial, starts Oct 2); 2013: 7,294; 2014: 4,501; 2015: 3,593; 2016: 7,838; 2017: 8,713. A major sensor gap runs from 2014-08-08 01:00 to 2015-06-11 20:00 (~10 months), which makes raw yearly totals misleading.

When normalised to average volume per hour, traffic demand is stable across years: 3,227 (2012), 3,310 (2013), 3,270 (2014), 3,258 (2015), 3,194 (2016), 3,377 (2017). The largest year-on-year change is +5.73% from 2016 to 2017. This means historical hourly averages from any complete year are a valid planning baseline, as long as the 2014–15 gap is excluded.

### Holiday Temperature Patterns (Task 1.3)

After retrieving all hourly rows on each holiday date (not just the midnight row where the holiday label appears):

- **Labor Day** temperatures are consistently warm: 22.17°C (2015), 21.86°C (2016), 18.14°C (2017). Average traffic ranges from 2,152 to 2,430 vehicles/hour across 24-hour periods.
- **New Year's Day** temperatures are sub-zero: −6.06°C (2016), −1.30°C (2017). Average traffic is 1,832–2,091 vehicles/hour. New Year's Day 2015 has no data due to the sensor gap. In 2017, the federal holiday was observed on 2 January because 1 January fell on a Sunday.

Labor Day consistently shows higher average traffic than New Year's Day, reflecting both warmer weather and the holiday travel pattern.

## 2. Descriptive Statistics (Task 2.1)

Traffic volume statistics across all 48,204 hourly observations:

| Statistic | Value |
|-----------|-------|
| Mean | 3,259.82 vehicles/hour |
| Median | 3,380.00 vehicles/hour |
| Std Dev | 1,986.86 |
| Variance | 3,947,615.32 |
| Range | 7,280 (0 to 7,280) |

The mean (3,260) is lower than the median (3,380), indicating a left-skewed distribution. This occurs because the data is bimodal: daytime hours cluster around 4,000–6,000 vehicles/hour, while overnight hours (roughly 22:00–05:00) regularly drop below 1,000. The large number of low-volume overnight hours pulls the mean below the median. The standard deviation of 1,987 (61% of the mean) reflects this wide day–night spread and is important for capacity planning — average-based models will substantially under-predict peak demand.

## 3. Correlation Analysis (Task 2.2)

The Pearson correlation between temperature and traffic volume is r = 0.1303, a weak positive relationship. The direction (positive) is plausible: warmer temperatures in Minnesota are associated with somewhat higher traffic, likely because severe cold discourages travel.

However, correlation does not imply causation. The relationship is likely confounded by seasonality: summer months bring both warmer temperatures and more travel due to vacations, longer daylight, and outdoor activities. Time of day is another confounder — temperature peaks in early afternoon, which overlaps with higher-traffic daytime hours. A regression model with seasonal and time-of-day controls would be needed to isolate any independent temperature effect.

## 4. Probability Analysis (Task 3)

Defining congestion as traffic volume > 5,500 vehicles/hour:

| Probability | Value |
|-------------|-------|
| P(Congestion) | 0.1473 (14.73%) |
| P(Clear Weather) | 0.2778 (27.78%) |
| P(Congestion ∩ Clear Weather) | 0.0366 (3.66%) |
| P(Clear Weather \| Congestion) | 0.2483 (24.83%) |
| P(High Temp > 292K \| Congestion) | 0.2630 (26.30%) |

**Independence test:** P(Congestion) × P(Clear) = 0.0409, while P(Congestion ∩ Clear) = 0.0366. The difference is 0.0043 — small but not zero. Congestion and clear weather are approximately independent, meaning weather type alone does not strongly predict whether congestion occurs.

**Odds ratio (clear vs cloudy):** In clear weather, 1,763 out of 13,391 hours are congested (odds = 0.152). In cloudy weather, 2,592 out of 15,164 hours are congested (odds = 0.206). The odds ratio is 0.74, meaning congestion is 1.36× more likely in cloudy weather than in clear weather. While statistically detectable, this is a modest effect — time of day and day of week are far stronger congestion drivers.

## 5. Dashboard Analysis (Task 4)

An interactive HTML dashboard (`dashboard.py` / `traffic_dashboard.html`) was built using Plotly as an alternative to Power BI Desktop, which is not available on macOS. The dashboard covers all required analytical views:

- **Daily traffic trends** for 2015, 2016, and 2017 (line chart, one line per year)
- **Hourly traffic patterns** for 2017 (bar chart showing average traffic by hour)
- **Weather impact analysis:** Clouds has the highest average traffic (3,617 vehicles/hour); Squall has the lowest (420). The difference is 3,197 vehicles/hour. Excluding the rare Squall condition (only a handful of observations), Fog has the lowest average among common conditions at 2,724 vehicles/hour.
- **Temperature vs traffic scatter plot** coloured by weather condition
- **KPI cards:** total hours analysed (48,204), average traffic volume, average temperature
- **Weather condition dropdown filter** for interactivity

Note: A Power BI Desktop `.pbix` file was not produced as Power BI Desktop is Windows-only.

## 6. Synthesis and Recommendations

Combining findings from SQL, statistics, probability, and dashboard analysis, the key insights for the Smart City Mobility Analytics Team are:

1. **Traffic demand is time-driven, not weather-driven.** The weak temperature–traffic correlation (r = 0.13) and the near-independence of congestion and weather type (odds ratio 0.74) confirm that hour of day and day of week are the dominant factors. Traffic management systems should prioritise time-based signal optimisation over weather-reactive strategies.

2. **Congestion is concentrated in predictable windows.** Only 14.7% of hours exceed the 5,500-vehicle congestion threshold, and these cluster in weekday morning (07:00–08:00) and afternoon (16:00–17:00) rush periods. Targeted interventions during these 2–3 hour windows would address most congestion.

3. **Data coverage matters for any longitudinal analysis.** The 10-month sensor gap (Aug 2014 – Jun 2015) and uneven yearly coverage mean raw yearly totals are unreliable. Any trend analysis must normalise by hours of data. The coverage-adjusted averages (3,194–3,377 vehicles/hour) show demand was flat across the study period.

4. **Holiday traffic follows predictable patterns.** Labor Day averages 2,152–2,430 vehicles/hour (below normal weekday levels), while New Year's Day averages 1,832–2,091. Holiday-specific traffic management can use these baselines for resource planning.

5. **Cloudy conditions carry slightly elevated congestion risk.** The 1.36× higher odds of congestion during cloudy weather, while modest, could inform a secondary weather-aware layer in traffic management — for example, slightly extending green-light phases on cloudy weekday afternoons.
