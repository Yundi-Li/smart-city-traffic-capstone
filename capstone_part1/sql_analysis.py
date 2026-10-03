"""
SQL Analysis for Smart City Traffic Capstone - Part 1
Loads Metro Interstate Traffic Volume data into SQLite and runs analytical queries.
"""

import sqlite3
import csv
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / ".." / "data" / "Metro_Interstate_Traffic_Volume.csv"
DB_PATH = BASE_DIR / "traffic.db"
QUERIES_PATH = BASE_DIR / "queries.sql"

QUERIES = {
    "1.1_row_count": """
SELECT COUNT(*) AS total_rows FROM traffic;
""",
    "1.1_unique_timestamps": """
SELECT COUNT(DISTINCT date_time) AS unique_timestamps FROM traffic;
""",
    "1.1_sample_rows": """
SELECT * FROM traffic LIMIT 5;
""",
    "1.1_column_info": """
PRAGMA table_info(traffic);
""",
    "1.1_null_check": """
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
""",
    "1.2_annual_trends_deduped": """
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
""",
    "1.2_yoy_change": """
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
""",
    "1.3_holiday_temperature": """
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
""",
}


def load_csv_to_sqlite(csv_path: Path, db_path: Path) -> None:
    """Load the CSV file into a SQLite database."""
    logger.info("Loading CSV from %s into SQLite at %s", csv_path, db_path)

    if db_path.exists():
        db_path.unlink()
        logger.info("Removed existing database file")

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE traffic (
            holiday TEXT,
            temp REAL,
            rain_1h REAL,
            snow_1h REAL,
            clouds_all INTEGER,
            weather_main TEXT,
            weather_description TEXT,
            date_time TEXT,
            traffic_volume INTEGER
        )
    """)

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append((
                row["holiday"],
                float(row["temp"]),
                float(row["rain_1h"]),
                float(row["snow_1h"]),
                int(row["clouds_all"]),
                row["weather_main"],
                row["weather_description"],
                row["date_time"],
                int(row["traffic_volume"]),
            ))

    cur.executemany(
        "INSERT INTO traffic VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
    )
    conn.commit()
    logger.info("Inserted %d rows into traffic table", len(rows))
    conn.close()


def save_queries(queries: dict, path: Path) -> None:
    """Save all SQL queries to a .sql file."""
    with open(path, "w", encoding="utf-8") as f:
        for name, sql in queries.items():
            f.write(f"-- {name}\n{sql.strip()}\n\n")
    logger.info("Saved SQL queries to %s", path)


def run_query(conn: sqlite3.Connection, name: str, sql: str) -> list:
    """Execute a query and return results."""
    cur = conn.execute(sql)
    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    return columns, rows


def print_results(name: str, columns: list, rows: list) -> None:
    """Pretty-print query results."""
    col_widths = []
    for i, c in enumerate(columns):
        max_w = len(c)
        for r in rows:
            max_w = max(max_w, len(str(r[i])))
        col_widths.append(min(max_w, 25))

    header = " | ".join(f"{c:<{col_widths[i]}}" for i, c in enumerate(columns))
    print(f"\n{'='*len(header)}")
    print(f"  {name}")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(" | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row)))


def main():
    logger.info("Starting SQL Analysis for Smart City Traffic Capstone")

    csv_path = DATA_PATH.resolve()
    if not csv_path.exists():
        logger.error("CSV file not found at %s", csv_path)
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    load_csv_to_sqlite(csv_path, DB_PATH)
    save_queries(QUERIES, QUERIES_PATH)

    conn = sqlite3.connect(str(DB_PATH))

    # Task 1.1: Verify data
    print("\n" + "#" * 70)
    print("  TASK 1.1: Load and Verify Data")
    print("#" * 70)

    for qname in ["1.1_row_count", "1.1_unique_timestamps", "1.1_sample_rows",
                   "1.1_column_info", "1.1_null_check"]:
        cols, rows = run_query(conn, qname, QUERIES[qname])
        print_results(qname, cols, rows)

    # Task 1.2: Annual traffic trends
    print("\n" + "#" * 70)
    print("  TASK 1.2: Annual Traffic Trends (2012-2017)")
    print("#" * 70)

    cols, rows = run_query(conn, "Annual Trends (deduplicated)",
                           QUERIES["1.2_annual_trends_deduped"])
    print_results("Annual Traffic Volume (deduplicated timestamps)", cols, rows)

    cols, rows = run_query(conn, "YoY Change", QUERIES["1.2_yoy_change"])
    print_results("Year-over-Year Changes (avg volume per hour)", cols, rows)

    print("\n--- Observations ---")
    print(
        "Observation 1: Total yearly volumes vary dramatically, but this is primarily\n"
        "  driven by uneven data coverage rather than real traffic changes. The dataset\n"
        "  has a major sensor gap from 2014-08-08 to 2015-06-11 (~10 months), so 2014\n"
        "  and 2015 have far fewer hours of data. 2012 is also partial (starts Oct 2).\n"
        "  When we normalise by looking at average volume per hour, the figures stay\n"
        "  remarkably stable across years (~3,170–3,340 vehicles/hour), indicating that\n"
        "  underlying traffic demand on this corridor was largely flat over the period."
    )
    print(
        "Observation 2: The year-on-year change in average hourly volume is small in\n"
        "  every pair (<6%), confirming that traffic demand did not materially grow or\n"
        "  decline. 2016 shows the lowest average (~3,169 vehicles/hour) and 2017 the\n"
        "  highest (~3,341), but both are within normal variation. The practical\n"
        "  implication is that historical hourly averages from any complete year are a\n"
        "  reasonable baseline for planning, as long as the sensor gap in 2014-15 is\n"
        "  excluded."
    )

    # Task 1.3: Temperature around holidays
    print("\n" + "#" * 70)
    print("  TASK 1.3: Temperature Around Holidays (2015-2017)")
    print("#" * 70)

    cols, rows = run_query(conn, "Holiday Temps (all hours on holiday date)",
                           QUERIES["1.3_holiday_temperature"])
    print_results("Holiday Temperature & Traffic (full-day, deduplicated)", cols, rows)

    print("\n--- Notes ---")
    print(
        "- New Year's Day 2015 has no data because it falls inside the sensor gap\n"
        "  (2014-08-08 to 2015-06-11).\n"
        "- In 2017 the federal New Year's Day holiday was observed on Monday 2 Jan\n"
        "  because 1 Jan fell on a Sunday. The query captures this correctly via\n"
        "  the holiday column.\n"
        "- Timestamps are deduplicated (averaged where multiple weather readings\n"
        "  exist for the same hour) before computing statistics.\n"
        "- The query retrieves ALL hourly rows on each holiday date, not just the\n"
        "  midnight row where the holiday label appears."
    )

    print("\n--- Interpretation ---")
    # Print actual values from the query
    if rows:
        for r in rows:
            holiday, year, date, hours, avg_k, min_k, max_k, avg_c, min_c, max_c, avg_traf = r
            print(f"  {holiday} {year} ({date}): {hours} hours observed, "
                  f"avg temp {avg_c}°C ({avg_k} K), "
                  f"range {min_c}°C to {max_c}°C, "
                  f"avg traffic {avg_traf:.0f} vehicles/hour")

    print(
        "\nNew Year's Day temperatures are consistently sub-zero Celsius, reflecting\n"
        "  Minnesota's harsh winter. Labor Day temperatures are warm (20-22°C),\n"
        "  typical of early September. Traffic volumes differ by holiday: Labor Day\n"
        "  shows higher average volumes than New Year's Day in years where both are\n"
        "  available, likely due to warmer weather encouraging travel and the holiday\n"
        "  weekend pattern. Year-on-year temperature variation within each holiday\n"
        "  is modest (1-5°C), suggesting weather is fairly consistent at these times."
    )

    conn.close()
    logger.info("SQL analysis complete")


if __name__ == "__main__":
    main()
