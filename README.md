# Singapore Bunker Sales Forecast

A next-month forecast of Singapore bunker sales, built from public MPA statistics.
It blends five time-series models, weighted by how accurate each was over the last
36 months, with a cross-check based on vessel calls for bunkers.

**September 2026 forecast: 4,683 kt** (80% range 4,364–5,002 kt), using data through August 2026.

## What's in this folder

| File | What it is |
|---|---|
| `app.py` | Streamlit dashboard. Runs the model live and lets you upload newer MPA files |
| `bunker_forecast.py` | The forecasting model. Also runs on its own from the command line |
| `data/` | MPA source data: monthly bunker sales (total and by grade), vessel calls by purpose |
| `tableau/bunker_forecast_tableau.xlsx` | Model results laid out for Tableau, one sheet per chart |
| `requirements.txt` | Python packages |

## Run it on your computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first load takes about a minute while it runs the 36-month accuracy test. After that it's cached.

To run the model without the dashboard:

```bash
python bunker_forecast.py --data-dir data
```

## Publish the Streamlit app (free public link)

1. Create a new **public** GitHub repository and upload everything in this folder, keeping the `data/` and `tableau/` folders.
2. Go to **share.streamlit.io**, sign in with GitHub, and click **Create app**.
3. Pick your repository, branch `main`, and main file `app.py`. Click **Deploy**.
4. After a few minutes you get a link like `your-name-bunker-forecast.streamlit.app`. Anyone can open it without an account.

Apps on the free tier go to sleep after a few days without visitors. The first person to open it then waits about a minute while it wakes up.

**Each month:** replace the files in `data/` with MPA's new releases and push to GitHub. The app redeploys itself and forecasts the next month.

## Build it in Tableau Public

Tableau Public is free (Mac and Windows) and gives you a public link and profile page.

1. Download Tableau Public Desktop from **public.tableau.com** and sign in.
2. **Connect → Microsoft Excel** → open `tableau/bunker_forecast_tableau.xlsx`.
3. Drag the **Sales** sheet onto the canvas. Use **Add new data source** for the other sheets. Keep them as separate data sources; don't join them.

### Sheet 1 – Monthly sales and forecast (main chart)
- Data source: **Sales**. Drag **Month** to Columns, right-click it and choose **Month (continuous, May 2015 format)**, or use discrete **MY(Month)**.
- **Sales_kt** to Rows. Set the mark type to **Bar**.
- **Type** to **Color**. Set Forecast = orange, Same month last year = blue, Actual = grey.
- Range whisker: Analytics pane → drag **Reference Band** onto the chart. Choose **Per Cell**, band from **SUM(Low_80)** to **SUM(High_80)**, and use a thin line with no fill.
- Forecast label: right-click the forecast bar → **Mark Label → Always Show**. Set the label to show `SUM(Sales_kt)` plus " kt".

### Sheet 2 – Headline tiles
- Data source: **Summary**. Make one sheet per number: drag **Forecast_kt** to **Text**, set it to a large font, and format the number with thousands separators and no decimals.
- Repeat for **Chg_vs_prev_pct**, **Chg_vs_last_year_pct** and **Model_avg_error_pct**. Add a "%" suffix in **Format → Numbers → Custom**.

### Sheet 3 – Forecast by model
- Data source: **Models**. **Model** to Rows, sorted by **Order** (right-click → Sort → Field → Order, ascending). **Forecast_kt** to Columns. Mark type **Bar**.
- **Kind** to **Color**. Set Final = orange, Blend and Check = blue, Model = grey.
- Right-click the axis → **Edit Axis** → uncheck *Include zero*, so the differences between models are visible.
- Analytics → **Reference Line** at a constant 4,683 (the final forecast), dashed.

### Sheet 4 – Forecast by fuel grade
- Data source: **Grades**. **Grade** to Rows, **Forecast_kt** to Columns, sorted descending. **Share_pct** to Label.

### Sheet 5 – Forecast miss, last 36 months
- Data source: **Backtest**. **Month** to Columns (continuous month), **Miss_pct** to Rows, mark type **Bar**.
- To colour big misses, create the calculated field `IF ABS([Miss_pct]) > 5 THEN "Over 5%" ELSE "Within 5%" END` and put it on Color.

### Sheet 6 – Tonnes per thousand GT of bunker calls
- Data source: **Bunker_calls**. **Month** to Columns, **Tonnes_per_kGT** to Rows, mark type **Bar**.

### Dashboard and publishing
1. **New Dashboard**. Set size to **Automatic**, or **Fixed 1200 × 900**.
2. Put the headline tiles across the top, the main chart full width below them, then two rows of two smaller charts.
3. Add a **Text** object with your name, a one-line description and the data source: *Maritime and Port Authority of Singapore (MPA)*.
4. **File → Save to Tableau Public**. You get a public link to share.

**Each month:** run `python bunker_forecast.py --data-dir data` on the new MPA files, re-create the Excel file (or ask Claude to), then in Tableau use **Data → Refresh** and save to Tableau Public again.

## Method

1. **Five models** on monthly totals since 1995:
   - last year's month × recent growth
   - last year's month × year-to-date growth
   - last month × the usual seasonal step
   - SARIMA on daily sales rates
   - Holt-Winters smoothing
2. **Accuracy test.** Each model forecasts each of the last 36 months one month ahead, using only data available at the time. Its average miss sets its weight in the blend.
3. **Bunker-call check.** Expected bunker calls (GT) × tonnes sold per GT, from MPA vessel calls by purpose.
4. **Final number.** 70% model blend, 30% cross-check. The 80% range is ±1.28 × the spread of past blend misses.
5. **Grade split.** The total divided by the latest three-month grade mix.

Data: Maritime and Port Authority of Singapore (MPA) monthly port statistics.
