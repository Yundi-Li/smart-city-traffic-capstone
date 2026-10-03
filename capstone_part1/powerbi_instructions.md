# Power BI Dashboard — Step-by-Step Build Instructions

## 1. Data Import and Power Query Preparation

### 1.1 Import the CSV

1. Open Power BI Desktop.
2. Click **Home > Get Data > Text/CSV**.
3. Navigate to `data/Metro_Interstate_Traffic_Volume.csv` and click **Open**.
4. In the preview dialog, verify that columns are correctly detected, then click **Transform Data** to open Power Query Editor.

### 1.2 Set Data Types

In Power Query Editor, set column types explicitly:

| Column              | Type              |
|---------------------|-------------------|
| holiday             | Text              |
| temp                | Decimal Number    |
| rain_1h             | Decimal Number    |
| snow_1h             | Decimal Number    |
| clouds_all          | Whole Number      |
| weather_main        | Text              |
| weather_description | Text              |
| date_time           | Date/Time         |
| traffic_volume      | Whole Number      |

Select each column, right-click the header, choose **Change Type**, and select the appropriate type.

### 1.3 Create the Hour Column

1. Select the `date_time` column.
2. Go to **Add Column > Time > Hour > Hour**.
3. This creates a new `Hour` column (Whole Number, 0–23).

### 1.4 Create the Celsius Temperature Column

1. Go to **Add Column > Custom Column**.
2. Name: `temp_celsius`
3. Formula: `[temp] - 273.15`
4. Click **OK**, then set the type to **Decimal Number**.

### 1.5 Create Traffic Category Column

1. Go to **Add Column > Conditional Column**.
2. Name: `Traffic Category`
3. Rules:
   - If `traffic_volume` is **greater than** `5500` → output `High`
   - If `traffic_volume` is **greater than** `3000` → output `Medium`
   - Otherwise → output `Low`
4. Click **OK**.

### 1.6 Create Year Column

1. Select `date_time`.
2. Go to **Add Column > Date > Year > Year**.
3. Rename to `Year` if needed.

### 1.7 Apply and Close

Click **Close & Apply** to load the transformed data into the model.

---

## 2. Chart Specifications

### 2.1 Daily Traffic Trends (2015–2017)

- **Chart type:** Line Chart
- **Axis (X):** `date_time` (set to Date hierarchy: Year > Month or continuous Date)
- **Values (Y):** Average of `traffic_volume`
- **Filter:** `Year` in {2015, 2016, 2017}
- **Formatting:**
  - Title: "Daily Average Traffic Volume (2015–2017)"
  - Enable data labels: Off (too dense for daily)
  - Add trend line: On (linear)
  - Grid lines: Light gray

### 2.2 Hourly Traffic Bar Chart (2017)

- **Chart type:** Clustered Bar Chart (horizontal) or Column Chart (vertical)
- **Axis (X):** `Hour` (0–23)
- **Values (Y):** Average of `traffic_volume`
- **Filter:** `Year` = 2017
- **Formatting:**
  - Title: "Average Hourly Traffic Volume — 2017"
  - Data labels: On
  - Sort by: Hour ascending
  - Color: Use a single color or gradient by volume

### 2.3 Weather Impact Analysis

- **Chart type:** Clustered Column Chart
- **Axis (X):** `weather_main`
- **Values (Y):** Average of `traffic_volume`
- **Formatting:**
  - Title: "Average Traffic Volume by Weather Condition"
  - Sort by: Average traffic_volume descending
  - Data labels: On
  - Optional: Add a secondary line showing count of observations per weather type (to contextualize small-sample categories)

### 2.4 Temperature Scatter Plot

- **Chart type:** Scatter Chart
- **X Axis:** Average or individual `temp_celsius`
- **Y Axis:** `traffic_volume`
- **Details:** Optionally color by `weather_main` or `Traffic Category`
- **Formatting:**
  - Title: "Temperature vs Traffic Volume"
  - Add a trend line (linear)
  - Bubble size: keep uniform (or use `clouds_all` for a third dimension)
  - Axis labels: "Temperature (°C)" and "Traffic Volume (vehicles/hour)"

---

## 3. KPI Cards

Create three **Card** visuals and position them in a row at the top of the report page:

| KPI Card           | Measure                                                   | Format      |
|--------------------|-----------------------------------------------------------|-------------|
| Total Hours        | Count of rows: `COUNTROWS(traffic)`                       | #,##0       |
| Avg Traffic Volume | `AVERAGE(traffic[traffic_volume])`                        | #,##0       |
| Avg Temperature    | `AVERAGE(traffic[temp_celsius])` with unit suffix " °C"   | #,##0.0     |

To create these as explicit measures:
1. Go to **Modeling > New Measure**.
2. Enter the DAX formula (e.g., `Total Hours = COUNTROWS(traffic)`).
3. Assign to a Card visual.

---

## 4. Slicers

Add three slicer visuals along the left side or top of the dashboard:

### 4.1 Hour Range Slicer

- **Field:** `Hour`
- **Type:** Between (range slider)
- Set default range 0–23. Users can narrow to e.g. 7–9 for morning rush.

### 4.2 Weather Condition Slicer

- **Field:** `weather_main`
- **Type:** Dropdown or List
- Enable "Select All" option. Users can filter to specific conditions (Clear, Clouds, Rain, etc.).

### 4.3 Traffic Category Slicer

- **Field:** `Traffic Category`
- **Type:** List (checkboxes)
- Shows: High, Medium, Low

### Slicer Configuration Tips

- Under **Format > Slicer settings**, enable **Single select: Off** to allow multi-select.
- Under **Format > Selection controls**, enable **Show "Select all"**.
- Ensure all slicers interact with all visuals (default behavior unless sync settings are changed).

---

## 5. Layout Suggestions

```
+---------------------------------------------------------------+
|  [Total Hours]    [Avg Traffic Volume]    [Avg Temperature]   |  <- KPI row
+---------------------------------------------------------------+
|         |                                                     |
| Slicers |    Daily Traffic Trends Line Chart (2015-2017)      |
|  Hour   |                                                     |
| Weather |-----------------------------------------------------+
| Category|    Hourly Bar Chart (2017)   |  Weather Impact      |
|         |                              |  Column Chart        |
|         |------------------------------+----------------------|
|         |         Temperature Scatter Plot                    |
+---------+-----------------------------------------------------+
```

- **Page size:** 16:9 (default) or custom 1280 x 720 px.
- **Background:** White or very light gray (#F5F5F5).
- **Font:** Segoe UI throughout, 10–12 pt for labels, 20–24 pt for KPI values.
- **Color palette:** Use a consistent 3–5 color palette. Suggested: blues and grays for a professional, city-planning aesthetic.
- **Interactivity:** All visuals should cross-filter by default. Clicking a bar in the weather chart should filter the line chart and scatter plot.

---

## 6. Final Checks

1. Verify all filters are working by clicking through each slicer combination.
2. Confirm that KPI cards update dynamically with slicer selections.
3. Test the scatter plot trend line remains visible when filtered.
4. Add a text box at the bottom: "Data Source: Metro Interstate Traffic Volume — I-94 Westbound, Minneapolis-St. Paul | Period: 2012–2018".
5. Save as `SmartCityTraffic_Dashboard.pbix`.
