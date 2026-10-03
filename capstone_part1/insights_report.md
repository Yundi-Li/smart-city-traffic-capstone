# Smart City Traffic Analysis — Part 1 Insights Report

**Dataset:** Metro Interstate Traffic Volume (I-94 Westbound, Minneapolis-St. Paul)
**Period:** October 2012 – September 2018 | ~48,204 hourly observations
**Prepared for:** Smart City Mobility Team

---

## 1. SQL Analysis Findings

### Annual Traffic Trends (2012–2017)

Average hourly traffic volume shows a general upward trend across the study period, consistent with regional population and economic growth in the Twin Cities metro area. Year-over-year changes are not uniform: early years (2012–2013) may reflect partial data collection windows that skew annual averages, while later years with full 12-month coverage provide more reliable baselines.

**Key takeaway:** Traffic demand on the I-94 corridor is growing. Infrastructure and signal-timing plans should account for continued volume increases rather than assuming stationarity.

### Holiday Temperature Patterns (2015–2017)

New Year's Day observations show temperatures around 260–270 K (-13 to -3 °C) with lower traffic volumes, while Labor Day readings cluster near 295–300 K (22–27 °C) with notably higher volumes. The seasonal contrast in both temperature and traffic is stark, confirming that holiday-period planning must account for weather-driven behavioral differences — not just the holiday itself.

---

## 2. Descriptive Statistics

Traffic volume exhibits high variability (standard deviation exceeding 1,900 vehicles/hour) and a wide range from near-zero overnight readings to over 7,000 during peak hours. The mean exceeds the median, indicating right-skewness: most hours see moderate traffic, but peak-hour surges pull the average upward.

**Implication:** Average-based capacity models will systematically underestimate peak demand. The mobility team should use percentile-based thresholds (e.g., 85th or 95th percentile volumes) for infrastructure sizing and congestion trigger definitions.

---

## 3. Correlation Analysis

The Pearson correlation between temperature and traffic volume is positive but moderate in strength. Warmer conditions are associated with higher volumes, which aligns with the expectation that Minnesota's cold winters suppress discretionary travel.

However, this correlation is largely driven by seasonality — summer brings both warm weather and peak travel demand (vacations, events, longer days). Temperature alone should not be used as a causal predictor of traffic volume without controlling for time-of-year, day-of-week, and event schedules.

---

## 4. Probability Analysis

- **Congestion probability** (volume > 5,500): Roughly one-quarter to one-third of observed hours meet this threshold, indicating that congestion is a frequent rather than exceptional event.
- **Clear weather** accounts for a meaningful share of all observations. The joint probability of congestion during clear weather, compared to the product of the marginal probabilities, indicates that these events are not fully independent — weather conditions do shift congestion risk.
- **Odds ratio** analysis comparing clear versus cloudy conditions shows that congestion likelihood varies by weather type, providing an empirical basis for weather-responsive traffic management.
- **Conditional probability of high temperature given congestion** is elevated, reinforcing the seasonal pattern: congestion clusters in warmer months.

---

## 5. Synthesis and Recommendations

1. **Capacity planning should be percentile-driven.** The high variance and skewness of traffic volume mean that average-based planning will fail during the hours that matter most. Use the 85th–95th percentile as the design target.

2. **Seasonal staffing and signal timing.** The strong seasonal pattern — warmer months drive both higher volumes and higher congestion probability — justifies differentiated traffic management strategies by season, not just by time of day.

3. **Weather-responsive operations have empirical support.** The dependence between weather type and congestion, quantified by the odds ratio, means that integrating real-time weather feeds into adaptive signal control and route-guidance systems can improve congestion prediction accuracy.

4. **Holiday-specific planning.** New Year's Day and Labor Day differ dramatically in both temperature and traffic behavior. Event-based traffic plans should be calibrated to the specific holiday, not applied uniformly.

5. **Data quality note.** The dataset spans partial years at the boundaries (2012 and 2018). Annual comparisons should weight full-year periods more heavily, and any trend extrapolation should acknowledge the uneven observation windows.

---

*This report summarizes Part 1 analysis outputs. Further work (Part 2) will apply visualization dashboards and predictive modeling to deepen these findings.*
