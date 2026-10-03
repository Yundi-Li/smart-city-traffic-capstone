# Part 2 — Python: Building a Reproducible Traffic Analytics Pipeline

## Project Structure

```
capstone_part2/
├── pipeline.py              # Main entry point — data loading, validation, cleaning
├── feature_engineering.py   # Feature creation for ML readiness
├── visualizations.py        # Matplotlib chart generation
├── app.py                   # CLI mini-application for traffic queries
├── logging_config.py        # Centralised logging setup
├── cleaned_traffic.csv      # Output of pipeline.py (40,575 rows × 10 cols) — generated output included for grading convenience; reproducible via the pipeline
├── featured_traffic.csv     # Output of feature_engineering.py (40,575 rows × 37 cols) — generated output included for grading convenience; reproducible via the pipeline
├── requirements.txt         # Python dependencies
├── report.md                # Methodology and findings report
├── pipeline.log             # Sample log output (normal run)
├── pipeline_debug_sample.log # Sample log output (debug run)
├── logs/
│   └── pipeline.log         # Runtime log (regenerated on each run)
└── figures/
    ├── traffic_by_hour.png
    ├── traffic_distribution.png
    ├── temp_vs_traffic.png
    └── traffic_heatmap.png
```

## How to Run

Run scripts from the **repository root** in this order:

```bash
# Step 1: Clean the raw data
python capstone_part2/pipeline.py

# Step 2: Engineer features
python capstone_part2/feature_engineering.py

# Step 3: Generate visualisations
python capstone_part2/visualizations.py

# Step 4: Launch the CLI application
python capstone_part2/app.py
```

### CLI Application Commands

| Command | Description | Example |
|---------|-------------|---------|
| `traffic <date>` | Show hourly traffic for a date | `traffic 2017-06-15` |
| `peak <weekday\|weekend>` | Top 5 high-traffic hours | `peak weekday` |
| `compare <month>` | Weekday vs weekend average | `compare 6` |
| `recommend <day_type> <weather>` | Best travel times | `recommend weekday clear` |
| `help` | Show available commands | `help` |
| `quit` | Exit the application | `quit` |

### Using the --debug Flag

All scripts accept `--debug` to enable DEBUG-level logging:

```bash
python capstone_part2/pipeline.py --debug
python capstone_part2/feature_engineering.py --debug
```

## Logging

### Where Logs Are Written

- **Console:** All log messages are printed to stdout during execution.
- **File:** `capstone_part2/logs/pipeline.log` — appended by each script run.

### Log Format

```
2026-10-03 22:37:25 | INFO     | __main__ | Cleaned data saved to capstone_part2/cleaned_traffic.csv
```

Format: `timestamp | level | module | message`

### Log Levels

| Level | Meaning | Example |
|-------|---------|---------|
| `DEBUG` | Fine-grained internal values for troubleshooting. Only visible with `--debug`. | Quartile thresholds: Q1=1248.5, Q2=3427.0, Q3=4952.0 |
| `INFO` | Normal pipeline milestones — data loaded, step completed, file saved. | Loaded 48,204 rows and 9 columns |
| `WARNING` | Unexpected but recoverable events — rows dropped, values imputed, outliers handled. | Removing 7,612 rows with duplicate date_time values |
| `ERROR` | A failure that prevents the pipeline from continuing. Includes traceback via `exc_info=True`. | Dataset not found at path |

### Configuration

Logging is configured in `logging_config.py`. Entry-point scripts call `setup_logging(debug)` once. Library modules use `logger = logging.getLogger(__name__)` and do not configure handlers. No `print()` statements are used for internal status reporting; `print()` is reserved for direct user-facing output in the CLI application.
