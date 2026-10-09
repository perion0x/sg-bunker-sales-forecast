"""Singapore bunker sales forecast - Streamlit app.

Run locally:   streamlit run app.py
"""
import hashlib
import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import bunker_forecast as bf

DATA = Path(__file__).parent / "data"
ACCENT = "#C2410C"      # forecast
BASE = "#2563EB"        # comparison / model blend
MUTED = "#94A3B8"       # history
BAD = "#DC2626"

st.set_page_config(page_title="Singapore Bunker Sales Forecast", page_icon="⛽", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stMetricValue"] {font-size: 2rem;}
.hero {border: 2px solid #C2410C; border-radius: 10px; padding: 1rem 1.25rem;}
.hero .lbl {font-size: .8rem; letter-spacing: .05em; text-transform: uppercase; opacity: .7;}
.hero .num {font-size: 3rem; font-weight: 600; color: #C2410C; line-height: 1.1;}
.hero .sub {font-size: .85rem; opacity: .75;}
</style>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Data")
    st.caption("Uses the bundled MPA data through August 2026. Upload newer MPA files to re-run the forecast.")
    up_total = st.file_uploader("Bunker sales, monthly total (CSV)", type="csv")
    up_break = st.file_uploader("Bunker sales by grade (CSV)", type="csv")
    up_vp = st.file_uploader("Vessel calls by purpose (MPA .xls)", type=["xls", "xlsx"])
    st.header("Settings")
    mw = st.slider("Weight on model blend (rest on bunker-call check)", 0.0, 1.0, bf.MODEL_WEIGHT, 0.05)
    hist_months = st.select_slider("Months of history in main chart", [12, 24, 36, 60], value=24)


def as_bytes(upload, default_path):
    return upload.getvalue() if upload is not None else Path(default_path).read_bytes()


total_b = as_bytes(up_total, DATA / "Bunker_Sales_Total_Monthly.csv")
break_b = as_bytes(up_break, DATA / "Bunker_Sales_Breakdown_Monthly.csv")
vp_b = as_bytes(up_vp, next(DATA.glob("*vessel-purpose*.xls*")))


@st.cache_data(show_spinner="Running the models and the 36-month accuracy test…")
def run(total_b, break_b, vp_b, mw, _key):
    s = bf.load_total(io.BytesIO(total_b))
    try:
        gt = bf.load_bunker_call_gt(io.BytesIO(vp_b))
    except Exception:
        gt = None
    res = bf.compute(s, gt, io.BytesIO(break_b), model_weight=mw)
    calls = None
    if gt is not None:
        idx = gt.index.intersection(s.index)
        calls = pd.DataFrame({"month": idx, "sales_kt": s[idx].values, "bunker_call_kgt": gt[idx].values,
                              "t_per_kgt": (s[idx] / gt[idx] * 1000).values})
    return s, res, calls


key = hashlib.md5(total_b + break_b + vp_b).hexdigest()
s, r, calls = run(total_b, break_b, vp_b, mw, key)
T = r["target"]
fmt = lambda v: f"{v:,.0f}"
pct = lambda v: f"{v:+.1f}%"

# --------------------------------------------------------------------------- #
# Header + KPIs
# --------------------------------------------------------------------------- #
st.title("Singapore bunker sales forecast")
st.caption(f"Thousand tonnes (kt) · MPA data through {s.index[-1]:%B %Y}")

c0, c1, c2, c3 = st.columns([2, 1, 1, 1])
with c0:
    st.markdown(f"""<div class="hero"><div class="lbl">Forecast · {T:%B %Y}</div>
<div class="num">{fmt(r['final'])} <span style="font-size:1rem;opacity:.7">kt</span></div>
<div class="sub">80% range {fmt(r['low'])} – {fmt(r['high'])} kt</div></div>""", unsafe_allow_html=True)
c1.metric(f"vs {s.index[-1]:%b %Y}", pct((r['final'] / r['prev'] - 1) * 100))
c1.caption(f"Actual {fmt(r['prev'])} kt · one day shorter month")
if pd.notna(r["last_year"]):
    c2.metric("vs same month last year", pct((r['final'] / r['last_year'] - 1) * 100))
    c2.caption(f"Actual {fmt(r['last_year'])} kt")
c3.metric("Model avg error", f"{r['ens_mape_pct']:.1f}%")
c3.caption("Last 36 months, one month ahead")

# --------------------------------------------------------------------------- #
# Main bar chart
# --------------------------------------------------------------------------- #
st.subheader("Monthly bunker sales and forecast")
h = s.iloc[-hist_months:]
ly = T - pd.DateOffset(years=1)
x = [d.strftime("%b %Y") for d in h.index] + [T.strftime("%b %Y")]
y = list(h.values) + [r["final"]]
colors = [BASE if d == ly else MUTED for d in h.index] + [ACCENT]
fig = go.Figure()
fig.add_bar(x=x[:-1], y=y[:-1], marker_color=colors[:-1], hovertemplate="%{x}<br>%{y:,.0f} kt<extra></extra>")
fig.add_bar(x=x[-1:], y=y[-1:], marker_color=ACCENT,
            error_y=dict(type="data", symmetric=False, array=[r["high"] - r["final"]], arrayminus=[r["final"] - r["low"]],
                         color="#475569", thickness=1.5, width=8),
            hovertemplate="%{x} forecast<br>%{y:,.0f} kt<extra></extra>")
fig.add_hline(y=r["final"], line_dash="dash", line_color=ACCENT, line_width=1)
fig.add_annotation(x=x[-1], y=r["high"], text=f"<b>{fmt(r['final'])} kt</b><br>{T:%b %Y} forecast",
                   showarrow=False, yshift=28, font=dict(color=ACCENT, size=13), xanchor="right")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="kt", bargap=0.25, barmode="overlay",
                  yaxis=dict(range=[0, max(y + [r['high']]) * 1.12], tickformat=","), showlegend=False)
st.plotly_chart(fig, width="stretch")
st.caption("Grey: actual · Blue: same month last year · Orange: forecast with 80% range")

# --------------------------------------------------------------------------- #
# Models + grades
# --------------------------------------------------------------------------- #
a, b = st.columns(2)
with a:
    st.subheader("Forecast by model")
    m = r["models"].copy()
    names = list(m.model) + ["Model blend"]
    vals = list(m.forecast_kt) + [r["ensemble"]]
    cols = [MUTED] * len(m) + [BASE]
    texts = [f"{v:,.0f} · {w:.0f}%" for v, w in zip(m.forecast_kt, m.weight_pct)] + [fmt(r["ensemble"])]
    if r["crosscheck"]:
        names.append("Bunker-call check"); vals.append(r["crosscheck"]["forecast"]); cols.append(BASE); texts.append(fmt(r["crosscheck"]["forecast"]))
    names.append("Final forecast"); vals.append(r["final"]); cols.append(ACCENT); texts.append(fmt(r["final"]))
    f2 = go.Figure(go.Bar(y=names, x=vals, orientation="h", marker_color=cols, text=texts, textposition="outside",
                          hovertemplate="%{y}: %{x:,.0f} kt<extra></extra>"))
    lo = min(vals) * 0.97
    f2.add_vline(x=r["final"], line_dash="dash", line_color=ACCENT)
    f2.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[lo, max(vals) * 1.03], tickformat=","),
                     yaxis=dict(autorange="reversed"), showlegend=False)
    st.plotly_chart(f2, width="stretch")
    st.caption("Label: forecast · weight in blend. Weights come from each model's accuracy in the 36-month test. "
               "Axis does not start at zero.")
with b:
    st.subheader("Forecast by fuel grade")
    g = r["grades"]
    if g is not None:
        f3 = go.Figure(go.Bar(y=g.grade, x=g.forecast_kt, orientation="h", marker_color=[ACCENT, BASE, "#0D9488", "#7C3AED", MUTED][: len(g)],
                              text=[f"{v:,.0f} kt · {p:.1f}%" for v, p in zip(g.forecast_kt, g.share_pct)], textposition="outside",
                              hovertemplate="%{y}: %{x:,.0f} kt<extra></extra>"))
        f3.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(range=[0, g.forecast_kt.max() * 1.7], tickformat=","),
                         yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(f3, width="stretch")
        st.caption("Final forecast split using the last three months' grade mix.")

# --------------------------------------------------------------------------- #
# Backtest + per-GT
# --------------------------------------------------------------------------- #
a, b = st.columns(2)
with a:
    st.subheader("Forecast miss, last 36 months")
    bt = r["backtest"]
    f4 = go.Figure(go.Bar(x=bt.month, y=bt.error_pct, marker_color=[BAD if abs(e) > 5 else BASE for e in bt.error_pct],
                          customdata=bt[["actual_kt", "forecast_kt"]],
                          hovertemplate="%{x|%b %Y}<br>Miss %{y:+.1f}%<br>Actual %{customdata[0]:,.0f} kt<br>Forecast %{customdata[1]:,.0f} kt<extra></extra>"))
    f4.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="% miss", showlegend=False)
    st.plotly_chart(f4, width="stretch")
    st.caption("Above zero: forecast too high. Red: missed by more than 5%.")
with b:
    st.subheader("Tonnes sold per thousand GT of bunker calls")
    if calls is not None:
        f5 = go.Figure(go.Bar(x=calls.month, y=calls.t_per_kgt,
                              marker_color=[BASE if d.year == calls.month.max().year else MUTED for d in calls.month],
                              hovertemplate="%{x|%b %Y}<br>%{y:.1f} t per kGT<extra></extra>"))
        f5.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(f5, width="stretch")
        st.caption("Ships calling for bunkers have bought less per call since April 2026.")

# --------------------------------------------------------------------------- #
# How it's built
# --------------------------------------------------------------------------- #
st.subheader("How the final number is reached")
if r["crosscheck"]:
    xc = r["crosscheck"]
    st.markdown(f"Model blend **{fmt(r['ensemble'])}** kt × {r['model_weight']:.0%} + bunker-call check "
                f"**{fmt(xc['forecast'])}** kt × {1 - r['model_weight']:.0%} = **{fmt(r['final'])} kt**")
    st.caption(f"Check: {fmt(xc['gt_fc'])} thousand GT of expected bunker calls × midpoint of "
               f"{xc['t_per_kgt_recent']:.1f} (last 3 months) and {xc['t_per_kgt_ly']:.1f} (same month last year) tonnes per thousand GT.")
with st.expander("Method"):
    st.markdown("""
1. **Five models** on monthly totals since 1995: last year's month × recent growth, last year's month × year-to-date growth,
   last month × the usual seasonal step, SARIMA on daily sales rates, and Holt-Winters smoothing.
2. **Accuracy test.** Each model forecasts each of the last 36 months one month ahead; its average miss sets its weight.
3. **Bunker-call check.** Expected bunker calls (GT) × tonnes sold per GT, from MPA vessel calls by purpose.
4. **Final number.** Model blend and cross-check combined (70/30 by default, adjustable in the sidebar).
   The 80% range is ±1.28 × the spread of past blend misses.
5. **Grade split.** The total divided by the latest three-month grade mix.

Data: Maritime and Port Authority of Singapore (MPA) monthly statistics.
""")
    st.dataframe(r["models"].round(2), hide_index=True, width="stretch")

out = pd.DataFrame([{"month": f"{T:%Y-%m}", "forecast_kt": round(r["final"], 1), "low_80": round(r["low"], 1),
                     "high_80": round(r["high"], 1), "model_blend_kt": round(r["ensemble"], 1)}])
st.download_button("Download forecast (CSV)", out.to_csv(index=False), f"bunker_forecast_{T:%Y-%m}.csv", "text/csv")
