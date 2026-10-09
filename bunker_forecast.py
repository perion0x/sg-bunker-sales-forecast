"""
Singapore bunker sales - next-month forecast model
====================================================

Forecasts total Singapore bunker sales ('000 tonnes) for the month after the
latest month in the data, plus a split by fuel grade.

Method
------
1. Five univariate models on the monthly total (1995 onwards):
     a. Seasonal naive x YoY growth (last 3 months vs same 3 months a year earlier)
     b. Seasonal naive x YTD growth
     c. Last month x median 10-year seasonal step ratio (on a per-day basis)
     d. SARIMA(1,1,1)(0,1,1,12) on log daily sales rate, trained 2010+
     e. Holt-Winters (damped additive trend, additive seasonality) on log sales, 2010+
2. Rolling one-step-ahead backtest over the last 36 months gives each model's MAPE.
3. Models are blended with inverse-MAPE weights  ->  "model ensemble".
4. Cross-check from MPA vessel calls for bunkers (GT):
     next-month bunker-call GT  = last month GT x (same step last year)
     tonnes per kGT              = midpoint of (last 3m average, same month last year)
     cross-check                 = GT forecast x tonnes per kGT
   (Only used if the vessel-purpose file is supplied.)
5. Final = 70% model ensemble + 30% cross-check.
6. 80% band = final x (1 +/- 1.28 x std of ensemble backtest errors).
7. Grade split = final x grade shares over the last 3 months.

Usage
-----
    pip install pandas numpy statsmodels xlrd
    python bunker_forecast.py --data-dir path/to/folder

The folder should contain (filenames are matched by pattern, so MPA's
monthly file names work as-is):
    *Bunker_Sales_Total_Monthly*.csv      (required)  columns: month, bunker_sales
    *Bunker_Sales_Breakdown_Monthly*.csv  (optional)  columns: month, bunker_type, bunker_sales
    *vessel-purpose*.xls                  (optional)  MPA "Vessel calls by purpose"

Each month: drop in the updated files and re-run. Results are printed and
saved to bunker_forecast_<YYYY-MM>.csv.
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

BACKTEST_MONTHS = 36
MODEL_WEIGHT = 0.70          # weight on model ensemble vs bunker-call cross-check
Z80 = 1.2816                 # z-score for an 80% two-sided band
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

GRADE_GROUPS = {
    "VLSFO (incl. bio-blended)": ["Low Sulphur Fuel Oil", "Bio-Blended Low Sulphur Fuel Oil",
                                  "Ultra Low Sulphur Fuel Oil", "Bio-Blended Ultra Low Sulphur Fuel Oil"],
    "HSFO (incl. bio-blended)": ["Marine Fuel Oil", "Bio-Blended Marine Fuel Oil"],
    "LSMGO / MGO / MDO": ["Low Sulphur Marine Gas Oil", "Bio-Blended Low Sulphur Marine Gas Oil",
                          "Marine Gas Oil", "Bio-Blended Marine Gas Oil",
                          "Marine Diesel Oil", "Bio-Blended Marine Diesel Oil"],
    "LNG": ["LNG"],
}


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def find(data_dir, pattern, required=False):
    hits = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not hits and required:
        raise FileNotFoundError(f"No file matching {pattern} in {data_dir}")
    return hits[-1] if hits else None


def load_total(path):
    s = pd.read_csv(path, parse_dates=["month"], index_col="month")["bunker_sales"]
    return s.astype(float).asfreq("MS").dropna()


def parse_mpa_sections(path):
    """Parse an MPA monthly .xls into a list of DataFrames (one per table),
    each indexed by month start with the table's column headers."""
    raw = pd.read_excel(path, header=None)
    sections, cur, header, year = [], None, None, None
    for _, row in raw.iterrows():
        c0 = row.iloc[0]
        if isinstance(c0, str) and c0.isupper() and "(" in c0:       # table title
            if cur:
                sections.append(pd.DataFrame(cur).T.sort_index())
            cur, header, year = {}, None, None
            continue
        if cur is None:
            continue
        if pd.isna(c0) and isinstance(row.iloc[1], str) and header is None:
            header = [str(h).strip() for h in row.iloc[1:]]
            continue
        if isinstance(c0, (int, float)) and not pd.isna(c0) and pd.isna(row.iloc[1]):
            year = int(c0)                                             # year label row
            continue
        if isinstance(c0, str) and year and c0.strip("* ")[:3] in MONTHS:
            vals = pd.to_numeric(row.iloc[1:].astype(str).str.replace(",", ""), errors="coerce")
            if vals.notna().any():
                m = MONTHS.index(c0.strip("* ")[:3]) + 1
                cur[pd.Timestamp(year, m, 1)] = pd.Series(vals.values, index=header)
    if cur:
        sections.append(pd.DataFrame(cur).T.sort_index())
    return sections


def load_bunker_call_gt(path):
    secs = parse_mpa_sections(path)
    gt = secs[1] if len(secs) > 1 else secs[0]                         # 2nd table = by GT
    col = [c for c in gt.columns if "bunker" in c.lower()][0]
    return gt[col].astype(float).dropna()


# --------------------------------------------------------------------------- #
# Models  (each takes history h, returns forecast for the next month)
# --------------------------------------------------------------------------- #
def nxt(h):
    return h.index[-1] + pd.offsets.MonthBegin()


def m_yoy3(h):
    t = nxt(h)
    g = h.iloc[-3:].sum() / h.iloc[-15:-12].sum()
    return h[t - pd.DateOffset(years=1)] * g


def m_ytd(h):
    t = nxt(h)
    n = (t.month - 1) or 12
    g = h.iloc[-n:].sum() / h.iloc[-n - 12:-12].sum()
    return h[t - pd.DateOffset(years=1)] * g


def m_step_ratio(h):
    t = nxt(h)
    d = h / h.index.days_in_month
    r = [d[t - pd.DateOffset(years=k)] / d[t - pd.DateOffset(years=k, months=1)] for k in range(1, 11)]
    return d.iloc[-1] * np.median(r) * t.days_in_month


def m_sarima(h):
    t = nxt(h)
    hh = h["2010":]
    y = np.log(hh / hh.index.days_in_month)
    fit = SARIMAX(y, order=(1, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=0)
    return float(np.exp(fit.forecast(1).iloc[0]) * t.days_in_month)


def m_holtwinters(h):
    y = np.log(h["2010":])
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True,
                               seasonal="add", seasonal_periods=12).fit()
    return float(np.exp(fit.forecast(1).iloc[0]))


MODELS = {
    "Seasonal naive x YoY (3m)": m_yoy3,
    "Seasonal naive x YTD": m_ytd,
    "Seasonal step ratio": m_step_ratio,
    "SARIMA (log daily rate)": m_sarima,
    "Holt-Winters ETS": m_holtwinters,
}


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def backtest(s):
    preds = {k: [] for k in MODELS}
    actual = []
    for end in s.index[-BACKTEST_MONTHS:]:
        h = s[: end - pd.offsets.MonthBegin()]
        actual.append(s[end])
        for k, f in MODELS.items():
            preds[k].append(f(h))
    actual = np.array(actual)
    mape = {k: np.mean(np.abs(np.array(p) / actual - 1)) for k, p in preds.items()}
    w = {k: (1 / v) / sum(1 / x for x in mape.values()) for k, v in mape.items()}
    ens = sum(np.array(preds[k]) * w[k] for k in MODELS)
    err = ens / actual - 1
    return mape, w, float(np.mean(np.abs(err))), float(np.std(err))


def cross_check(s, gt):
    t = nxt(s)
    ly, ly_prev = t - pd.DateOffset(years=1), t - pd.DateOffset(years=1, months=1)
    last = s.index[-1]
    if not {ly, ly_prev, last}.issubset(gt.index) or not {ly}.issubset(s.index):
        return None
    gt_fc = gt[last] * gt[ly] / gt[ly_prev]
    tpk = (s / gt).dropna()
    rate = 0.5 * (tpk.iloc[-3:].mean() + tpk[ly])
    return {"gt_fc": gt_fc, "t_per_kgt_recent": tpk.iloc[-3:].mean() * 1000,
            "t_per_kgt_ly": tpk[ly] * 1000, "forecast": gt_fc * rate}


def grade_split(path, total):
    b = pd.read_csv(path)
    p = b.pivot_table(index="month", columns="bunker_type", values="bunker_sales", aggfunc="sum").sort_index()
    q = pd.DataFrame({g: p[[c for c in cols if c in p.columns]].sum(axis=1) for g, cols in GRADE_GROUPS.items()})
    q["Other (B100, methanol, ammonia)"] = p.sum(axis=1) - q.sum(axis=1)
    share = q.iloc[-3:].sum() / q.iloc[-3:].sum().sum()
    return (share * total).round(0), share


def compute(s, gt=None, breakdown_path=None, model_weight=MODEL_WEIGHT):
    """Run the full pipeline and return every intermediate result (used by the Streamlit app)."""
    preds = {k: [] for k in MODELS}
    actual = []
    months = s.index[-BACKTEST_MONTHS:]
    for end in months:
        h = s[: end - pd.offsets.MonthBegin()]
        actual.append(s[end])
        for k, f in MODELS.items():
            preds[k].append(f(h))
    actual = np.array(actual)
    mape = {k: float(np.mean(np.abs(np.array(p) / actual - 1))) for k, p in preds.items()}
    w = {k: (1 / v) / sum(1 / x for x in mape.values()) for k, v in mape.items()}
    ens_bt = sum(np.array(preds[k]) * w[k] for k in MODELS)
    err = ens_bt / actual - 1
    fc = {k: float(f(s)) for k, f in MODELS.items()}
    ens = sum(fc[k] * w[k] for k in MODELS)
    xc = cross_check(s, gt) if gt is not None else None
    final = model_weight * ens + (1 - model_weight) * xc["forecast"] if xc else ens
    sd = float(np.std(err))
    target = nxt(s)
    out = {
        "target": target, "final": final, "low": final * (1 - Z80 * sd), "high": final * (1 + Z80 * sd),
        "ensemble": ens, "crosscheck": xc, "model_weight": model_weight if xc else 1.0,
        "prev": float(s.iloc[-1]), "last_year": float(s.get(target - pd.DateOffset(years=1), np.nan)),
        "models": pd.DataFrame({"model": list(MODELS), "forecast_kt": [fc[k] for k in MODELS],
                                "mape_pct": [mape[k] * 100 for k in MODELS], "weight_pct": [w[k] * 100 for k in MODELS]}),
        "backtest": pd.DataFrame({"month": months, "actual_kt": actual, "forecast_kt": ens_bt, "error_pct": err * 100}),
        "ens_mape_pct": float(np.mean(np.abs(err)) * 100),
        "grades": None,
    }
    if breakdown_path is not None:
        split, share = grade_split(breakdown_path, final)
        g = pd.DataFrame({"grade": split.index, "share_pct": share.values * 100, "forecast_kt": split.values})
        out["grades"] = g[g.forecast_kt >= 5].reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()

    s = load_total(find(a.data_dir, "*Bunker_Sales_Total_Monthly*.csv", required=True))
    target = nxt(s)
    print(f"Data through {s.index[-1]:%b %Y} ({s.iloc[-1]:,.1f} kt). Forecasting {target:%B %Y}.\n")

    # 1-3. Models, backtest, ensemble
    mape, w, ens_mape, ens_sd = backtest(s)
    fc = {k: f(s) for k, f in MODELS.items()}
    ens = sum(fc[k] * w[k] for k in MODELS)
    print(f"{'Model':30s} {'Forecast':>9s} {'MAPE':>7s} {'Weight':>7s}")
    for k in MODELS:
        print(f"{k:30s} {fc[k]:9,.0f} {mape[k]*100:6.2f}% {w[k]*100:6.1f}%")
    print(f"{'Model ensemble':30s} {ens:9,.0f} {ens_mape*100:6.2f}%   (backtest, last {BACKTEST_MONTHS}m)\n")

    # 4. Bunker-call cross-check
    final, xc = ens, None
    vp = find(a.data_dir, "*vessel-purpose*.xls*")
    if vp:
        xc = cross_check(s, load_bunker_call_gt(vp))
    if xc:
        final = MODEL_WEIGHT * ens + (1 - MODEL_WEIGHT) * xc["forecast"]
        print(f"Bunker-call cross-check: GT {xc['gt_fc']:,.0f} k x "
              f"{(xc['t_per_kgt_recent'] + xc['t_per_kgt_ly'])/2:.1f} t/kGT "
              f"(recent {xc['t_per_kgt_recent']:.1f}, last year {xc['t_per_kgt_ly']:.1f}) "
              f"= {xc['forecast']:,.0f} kt\n")
    else:
        print("Bunker-call cross-check skipped (vessel-purpose file missing or too short).\n")

    # 5-6. Final + band
    lo, hi = final * (1 - Z80 * ens_sd), final * (1 + Z80 * ens_sd)
    prev, ly = s.iloc[-1], s.get(target - pd.DateOffset(years=1))
    print(f"FINAL FORECAST {target:%b %Y}: {final:,.0f} kt   80% band {lo:,.0f} - {hi:,.0f} kt")
    print(f"  vs {s.index[-1]:%b %Y}: {(final/prev-1)*100:+.1f}%", end="")
    if ly is not None:
        print(f"   vs {target - pd.DateOffset(years=1):%b %Y}: {(final/ly-1)*100:+.1f}%", end="")
    print("\n")

    rows = [{"item": "Total", "forecast_kt": round(final, 1), "low_80": round(lo, 1), "high_80": round(hi, 1)}]
    rows += [{"item": f"Model: {k}", "forecast_kt": round(v, 1)} for k, v in fc.items()]
    rows.append({"item": "Model ensemble", "forecast_kt": round(ens, 1)})
    if xc:
        rows.append({"item": "Bunker-call cross-check", "forecast_kt": round(xc["forecast"], 1)})

    # 7. Grade split
    bd = find(a.data_dir, "*Bunker_Sales_Breakdown_Monthly*.csv")
    if bd:
        split, share = grade_split(bd, final)
        print("Grade split (last-3-month mix):")
        for g in split.index:
            print(f"  {g:34s} {share[g]*100:5.1f}%  {split[g]:7,.0f} kt")
            rows.append({"item": f"Grade: {g}", "forecast_kt": split[g]})

    out = os.path.join(a.out_dir, f"bunker_forecast_{target:%Y-%m}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
