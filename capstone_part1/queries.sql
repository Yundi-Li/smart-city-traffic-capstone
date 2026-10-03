-- 1.1_row_count
SELECT COUNT(*) AS total_rows FROM traffic;

-- 1.1_unique_timestamps
SELECT COUNT(DISTINCT date_time) AS unique_timestamps FROM traffic;

-- 1.1_sample_rows
SELECT * FROM traffic LIMIT 5;

-- 1.1_column_info
PRAGMA table_info(traffic);

-- 1.1_null_check
SELECT
    SUM(CASE WHEN holiday IS NULL OR holiday = '' THEN 1 ELSE 0 END) AS holiday_nulls,
    SUM(CASE WHEN temp IS NULL THEN 1 ELSE 0 END) AS temp_nulls,
    SUM(CASE WHEN rain_1h IS NULL THEN 1 ELSE 0 END) AS rain_nulls,
    SUM(CASE WHEN snow_1h IS NULL THEN 1 ELSE 0 END) AS snow_nulls,
    SUM(CASE WHEN clouds_all IS NULL THEN 1 ELSE 0 END) AS clouds_nulls,
    SUM(CASE WHEN weather_main IS NULL OR weather_main = '' THEN 1 ELSE 0 END) AS weather_main_nulls,
    SUM(CASE WHEN date_time IS NULL OR date_time = '' THEN 1 ELSE 0 END) AS date_time_nulls,
    SUM(CASE WHEN traffic_volume IS NULL THEN 1 ELSE 0 END) AS traffic_volume_nulls
FROM traffic;

-- 1.2_annual_trends_deduped
WITH deduped AS (
    SELECT date_time,
           AVG(temp) AS temp,
           AVG(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)
SELECT
    CAST(strftime('%Y', date_time) AS INTEGER) AS year,
    COUNT(*) AS hours_of_data,
    ROUND(AVG(traffic_volume), 2) AS avg_volume_per_hour,
    CAST(SUM(traffic_volume) AS INTEGER) AS total_volume,
    MIN(date_time) AS first_record,
    MAX(date_time) AS last_record
FROM deduped
WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- 1.2_yoy_change
WITH deduped AS (
    SELECT date_time,
           AVG(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
),
annual AS (
    SELECT
        CAST(strftime('%Y', date_time) AS INTEGER) AS year,
        COUNT(*) AS hours_of_data,
        ROUND(AVG(traffic_volume), 2) AS avg_volume
    FROM deduped
    WHERE strftime('%Y', date_time) BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT
    a.year,
    a.hours_of_data,
    a.avg_volume,
    LAG(a.avg_volume) OVER (ORDER BY a.year) AS prev_avg_volume,
    ROUND(a.avg_volume - LAG(a.avg_volume) OVER (ORDER BY a.year), 2) AS yoy_change,
    ROUND(
        (a.avg_volume - LAG(a.avg_volume) OVER (ORDER BY a.year))
        / LAG(a.avg_volume) OVER (ORDER BY a.year) * 100, 2
    ) AS yoy_pct_change
FROM annual a
ORDER BY a.year;

-- 1.3_holiday_temperature
WITH holiday_dates AS (
    SELECT DISTINCT
        holiday,
        DATE(date_time) AS holiday_date,
        CAST(strftime('%Y', date_time) AS INTEGER) AS year
    FROM traffic
    WHERE holiday IN ('New Years Day', 'Labor Day')
      AND strftime('%Y', date_time) BETWEEN '2015' AND '2017'
),
deduped AS (
    SELECT date_time,
           DATE(date_time) AS record_date,
           AVG(temp) AS temp,
           AVG(traffic_volume) AS traffic_volume
    FROM traffic
    GROUP BY date_time
)
SELECT
    hd.holiday,
    hd.year,
    hd.holiday_date,
    COUNT(*) AS hours_observed,
    ROUND(AVG(d.temp), 2) AS avg_temp_k,
    ROUND(MIN(d.temp), 2) AS min_temp_k,
    ROUND(MAX(d.temp), 2) AS max_temp_k,
    ROUND(AVG(d.temp) - 273.15, 2) AS avg_temp_c,
    ROUND(MIN(d.temp) - 273.15, 2) AS min_temp_c,
    ROUND(MAX(d.temp) - 273.15, 2) AS max_temp_c,
    ROUND(AVG(d.traffic_volume), 0) AS avg_traffic
FROM holiday_dates hd
JOIN deduped d ON d.record_date = hd.holiday_date
GROUP BY hd.holiday, hd.year, hd.holiday_date
ORDER BY hd.holiday, hd.year;

