"""
Kimberly Market Analytics
Streamlit interface for the fresh-produce price forecasting engine
(commodity_price_forecasting.py).

Pages mirror the desktop app: Dashboard, Price Forecasts, Model Comparison,
Model Insights, Historical Data, Trends & Seasonality, Evaluation Checks,
Monitoring, Settings & Export.
"""
from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from commodity_price_forecasting import (
    COMMODITY_COLOURS,
    LEVELS,
    MODEL_COLOURS,
    MONTH_ABBR,
    TARGETS,
    compare_table,
    draw_compare,
    draw_history,
    draw_results,
    load_data,
    monthly_table,
    prepare_commodity,
    run_pipeline,
)

APP_NAME = "Kimberly Market Analytics"
APP_BUILD = "build-2026-11-05-d"
PAGES = [
    "Dashboard", "Price Forecasts", "Model Comparison", "Model Insights",
    "Historical Data", "Trends & Seasonality", "Evaluation Checks",
    "Monitoring", "Settings & Export",
]
DEFAULT_FILES = ("Cleaned_dataset.csv", "Cleaned dataset.csv")

st.set_page_config(page_title=APP_NAME, page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; padding-bottom: 2rem;}
      [data-testid="stMetric"] {
        background: rgba(82,183,136,.10); border: 1px solid rgba(82,183,136,.35);
        border-top: 4px solid #2d6a4f; padding: 12px 16px; border-radius: 10px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

FRIENDLY = {
    "lag_1": ("Price 1 month ago", "The average price from last month - the strongest clue about this month."),
    "lag_2": ("Price 2 months ago", "The average price from 2 months earlier. Older prices help the model see the trend."),
    "lag_3": ("Price 3 months ago", "The average price from 3 months earlier. Older prices help the model see the trend."),
    "roll_mean_3": ("Average price, last 3 months", "A smoothed price over the previous 3 months."),
    "lag_qty_1": ("Quantity sold 1 mo. ago", "How much was sold last month - a sign of demand."),
    "month_sin": ("Time of year (part 1)", "Month of the year turned into a circle (with month_cos) so the model knows December is next to January - this captures seasonality."),
    "month_cos": ("Time of year (part 2)", "The second half of the seasonal 'circle' - always used together with month_sin."),
    "inflation": ("Inflation", "The monthly inflation figure - general price pressure in the economy."),
    "temperature": ("Temperature", "Average monthly temperature - affects growing conditions and supply."),
    "rainfall": ("Rainfall", "Total monthly rainfall - affects growing conditions and supply."),
    "prev_month_price": ("Previous month's average price", "Last month's average price for this commodity."),
    "time_idx": ("Time index", "Months since the start of the data - captures long-run drift."),
    "QtySold": ("Quantity sold", "Quantity sold in the record."),
    "QtyOnHand": ("Quantity on hand", "Stock on hand at the market."),
    "Inflation": ("Inflation", "The monthly inflation figure."),
    "Temp": ("Temperature", "Average monthly temperature."),
    "Rain": ("Rainfall", "Total monthly rainfall."),
}


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def _decode_bytes(data: bytes) -> str:
    """Decode an uploaded file's bytes to text, trying the encodings Excel/Windows commonly use."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not decode this file. Save it as CSV (UTF-8) and try again.")


@st.cache_data(show_spinner=False)
def parse_csv(data: bytes) -> pd.DataFrame:
    return load_data(io.StringIO(_decode_bytes(data)))


@st.cache_data(show_spinner=False)
def load_default_dataset(path: str) -> pd.DataFrame:
    return load_data(path)


def _normalise_excel(file_obj) -> pd.DataFrame:
    """Convert an Excel sheet to the CSV-like text the original loader expects (in memory only)."""
    workbook = pd.ExcelFile(file_obj)
    sheet = st.sidebar.selectbox("Worksheet", workbook.sheet_names, index=0)
    raw = pd.read_excel(workbook, sheet_name=sheet, header=None, dtype=str).fillna("")
    return load_data(io.StringIO(raw.to_csv(index=False, header=False)))


def get_dataset(uploaded):
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix.lower()
        if suffix == ".csv":
            return parse_csv(uploaded.getvalue()), uploaded.name
        if suffix in {".xlsx", ".xls"}:
            return _normalise_excel(io.BytesIO(uploaded.getvalue())), uploaded.name
        raise ValueError("Please upload a CSV, XLSX or XLS file.")
    for path in (os.getenv("COMMODITY_DATA_PATH"), *DEFAULT_FILES):
        if path and os.path.exists(path):
            return load_default_dataset(path), Path(path).name
    return None, None


@st.cache_data(show_spinner=False)
def all_monthly(df: pd.DataFrame, tcol: str, winsorise: bool) -> dict:
    out = {}
    for c in sorted(df["Commodity"].dropna().unique()):
        sub = prepare_commodity(df, c, tcol, winsorise)
        if not sub.empty:
            out[c] = monthly_table(sub)
    return out


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def date_axis(ax):
    lo, hi = ax.get_xlim()
    if (hi - lo) / 30.4 <= 14:  # short ranges: exactly one tick per month
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax.tick_params(axis="x", rotation=45, labelsize=8)


def short(name: str) -> str:
    return {"Naive (last month)": "Naive", "Linear Regression": "Linear Reg."}.get(name, name)


def pred_by_month(res) -> pd.DataFrame:
    return res["preds"].groupby("Date").mean(numeric_only=True)


def best_ml(res) -> str:
    for m in res["metrics"]["Model"]:
        if not m.startswith("Naive"):
            return m
    return res["best"]


def current_result(ctx):
    res = st.session_state.get("res")
    if res is None:
        st.info("Press **▶ Run all models** in the sidebar to train and compare the models.")
        return None
    if st.session_state.get("res_sig") != ctx["sig"]:
        st.warning("Settings or data changed since the last run. Press **▶ Run all models** in the sidebar to refresh.")
        return None
    return res


def year_labels(df):
    ymap = df.groupby("Year")["Date"].agg(["min", "max"])
    return {int(y): f"{int(y)} ({r['min']:%b}-{r['max']:%b})" for y, r in ymap.iterrows()}


def fig_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def chart_actual_vs_pred(res, zoom):
    pr = pred_by_month(res)
    hist = res["hist"]
    fig, ax = plt.subplots(figsize=(11, 5))
    if not zoom:
        ax.plot(hist.index, hist.values, color="lightgrey", lw=2, label="History")
        ax.axvspan(res["test_start"], hist.index.max() + pd.Timedelta(days=15), color="gold", alpha=.12)
    ax.plot(pr.index, pr["Actual"], color="#1f4e9c", lw=2.8, marker="o", label="Actual", zorder=5)
    for name in pr.columns.drop("Actual"):
        ax.plot(pr.index, pr[name], lw=1.5, marker="o", ms=3, color=MODEL_COLOURS.get(name),
                ls="--" if name.startswith("Naive") else "-", label=name, alpha=.9)
    ax.set_title("Actual vs predicted prices" + ("" if zoom else " (shaded = test period)"))
    ax.set_ylabel(res["unit"])
    date_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.005, .5), fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def chart_best_vs_baseline(res, name, zoom):
    pr = pred_by_month(res)
    hist = res["hist"]
    rmse = float(res["metrics"].set_index("Model").loc[name, "RMSE"])
    col = MODEL_COLOURS.get(name, "#ff7f0e")
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    if not zoom:
        ax.plot(hist.index, hist.values, color="lightgrey", lw=2, label="History")
        ax.axvspan(res["test_start"], hist.index.max() + pd.Timedelta(days=15), color="gold", alpha=.12)
    ax.fill_between(pr.index, pr[name] - 1.96 * rmse, pr[name] + 1.96 * rmse, color=col, alpha=.18,
                    label="approx. 95% interval (±1.96×RMSE)")
    ax.plot(pr.index, pr["Actual"], color="#1f4e9c", lw=2.8, marker="o", label="Actual", zorder=5)
    ax.plot(pr.index, pr[name], color=col, lw=2.2, marker="s", label=name)
    ax.plot(pr.index, pr["Naive (last month)"], color="grey", ls="--", marker="s", ms=4, label="Naive (last month)")
    ax.set_title(f"Best model ({name}) vs baseline")
    ax.set_ylabel(res["unit"])
    date_axis(ax)
    ax.legend(fontsize=8, loc="best")
    lo, hi = min(pr["Actual"].min(), pr[name].min()), max(pr["Actual"].max(), pr[name].max())
    ax2.scatter(pr["Actual"], pr[name], color=col, s=60, zorder=3)
    ax2.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax2.set_xlabel(f"Actual ({res['unit']})")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Predicted vs actual (dashed = perfect)", fontsize=10)
    fig.tight_layout()
    return fig


def chart_comparison(res):
    mt = res["metrics"].copy()
    names = mt["Model"].tolist()
    colours = [MODEL_COLOURS.get(n, "#999999") for n in names]
    x = np.arange(len(names))
    labels = [short(n) for n in names]
    base = mt[mt["Model"].str.startswith("Naive")].iloc[0]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.5))

    a = axs[0, 0]
    b1 = a.bar(x - .2, mt["RMSE"], .4, color="#1f77b4", label="RMSE")
    b2 = a.bar(x + .2, mt["MAE"], .4, color="#ff7f0e", label="MAE")
    a.bar_label(b1, fmt="%.2f", fontsize=7, padding=2)
    a.bar_label(b2, fmt="%.2f", fontsize=7, padding=2)
    a.set_title("MAE & RMSE (lower = better)")
    a.set_ylabel(res["unit"])
    a.legend(fontsize=8)

    b = axs[0, 1]
    b.bar(x, mt["MAPE %"], color=colours)
    for i, (n, v) in enumerate(zip(names, mt["MAPE %"])):
        if n.startswith("Naive"):
            txt, c = f"baseline\n{v:.1f}%", "#555555"
        else:
            d = (v - base["MAPE %"]) / base["MAPE %"] * 100
            txt, c = f"{'▲' if d > 0 else '▼'} {d:+.1f}%\n{v:.1f}%", "#c0392b" if d > 0 else "#1e8449"
        b.text(i, v, txt, ha="center", va="bottom", fontsize=8, color=c, fontweight="bold")
    b.margins(y=.25)
    b.set_title("MAPE % (lower = better)")

    c_ax = axs[1, 0]
    bars = c_ax.bar(x, mt["R2"], color=colours)
    c_ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    c_ax.axhline(0, color="black", lw=.8)
    c_ax.margins(y=.15)
    c_ax.set_title("R² (higher = better)")

    d_ax = axs[1, 1]
    vals = mt["RMSE vs naive %"]
    bars = d_ax.bar(x, vals, color=["#1e8449" if v > 0 else "#c0392b" if v < 0 else "#888888" for v in vals])
    for i, (n, v) in enumerate(zip(names, vals)):
        if n.startswith("Naive"):
            d_ax.text(i, 0, "baseline", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#555555")
        else:
            d_ax.text(i, v, f"{'▲' if v > 0 else '▼'} {v:+.1f}%", ha="center",
                      va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold",
                      color="#1e8449" if v > 0 else "#c0392b")
    d_ax.axhline(0, color="black", lw=.8)
    d_ax.margins(y=.2)
    d_ax.set_title("RMSE improvement vs naive (higher = better)")

    for ax in axs.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    return fig


def chart_insights(res):
    imp = res["importance"].tail(10)
    labels = [FRIENDLY.get(i, (i, ""))[0] for i in imp.index]
    lh = res["lstm_hist"]
    if lh:  # only draw the LSTM panel when the LSTM actually ran
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.1, 1]})
    else:
        fig, a1 = plt.subplots(figsize=(9, 4.8))
        a2 = None
    a1.barh(labels, imp.values, color="#2ca02c")
    for i, v in enumerate(imp.values):
        a1.text(v, i, f" {v * 100:.0f}%", va="center", fontsize=9)
    a1.set_xlabel("Importance (share of the model's decision)")
    a1.set_title("Random Forest - what drives the price?")
    a1.margins(x=.12)
    if a2 is not None:
        a2.plot(lh["loss"], label="Training error")
        if "val_loss" in lh:
            a2.plot(lh["val_loss"], label="Validation error")
        a2.set_yscale("log")
        a2.set_xlabel("epoch (training round)")
        a2.set_ylabel("loss (log scale)")
        a2.set_title("LSTM learning curve (falling = learning)")
        a2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chart_yoy(all_m, commodity, unit, ylab, labels_on):
    s = all_m[commodity]["price"].dropna()
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for yr in sorted(s.index.year.unique()):
        t = s[s.index.year == yr]
        ax.plot(t.index.month, t.values, marker="o", lw=2.2, label=ylab.get(int(yr), str(yr)))
        if labels_on:
            for m, v in zip(t.index.month, t.values):
                ax.annotate(f"{v:.1f}", (m, v), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR)
    ax.set_ylabel(unit)
    ax.set_title(f"{commodity}: the same calendar month compared across years")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def chart_yearly(all_m, df, commodity, unit, ylab, labels_on):
    names = list(all_m)
    years = sorted(df["Year"].dropna().unique().astype(int))
    width = .8 / max(len(years), 1)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for j, yr in enumerate(years):
        vals = []
        for n in names:
            s = all_m[n]["price"].dropna()
            s = s[s.index.year == yr]
            vals.append(s.mean() if len(s) else np.nan)
        bars = ax.bar(np.arange(len(names)) + j * width - .4 + width / 2, vals, width, label=ylab.get(yr, str(yr)))
        if labels_on:
            ax.bar_label(bars, labels=["" if np.isnan(v) else f"{v:.1f}" for v in vals], fontsize=8, padding=2)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    for t in ax.get_xticklabels():
        if t.get_text() == commodity:
            t.set_color("#d62728")
            t.set_fontweight("bold")
    ax.set_ylabel(unit)
    ax.set_title("Average price per year - all commodities")
    ax.legend(fontsize=8)
    fig.text(.5, -.01, "Note: years with fewer months are partial years, so compare with care.",
             ha="center", fontsize=8, color="grey")
    fig.tight_layout()
    return fig


def chart_trends(all_m, selected, commodity, unit, indexed, zoom, labels_on):
    fig, ax = plt.subplots(figsize=(12, 5.2))
    allv = []
    for i, n in enumerate(all_m):
        if n not in selected:
            continue
        s = all_m[n]["price"].dropna()
        if len(s) < 3:
            continue
        sp = s / s.iloc[:3].mean() * 100 if indexed else s
        xnum = ((s.index.year - s.index.min().year) * 12 + s.index.month - s.index.min().month).values
        slope, icpt = np.polyfit(xnum, sp.values, 1)
        col = COMMODITY_COLOURS[i % len(COMMODITY_COLOURS)]
        bold = n == commodity
        ax.plot(sp.index, sp.values, color=col, lw=3 if bold else 1.3, marker="o", ms=3,
                alpha=1 if bold else .75, label=f"{n} (trend {slope:+.2f}/month)")
        ax.plot(sp.index, slope * xnum + icpt, color=col, ls="--", lw=1.5)
        if labels_on and bold:
            for d, v in zip(sp.index, sp.values):
                ax.annotate(f"{v:.1f}", (d, v), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7, color=col)
        allv.extend(sp.values)
    if zoom and allv:
        ax.set_ylim(max(0, min(allv) * .8), np.percentile(allv, 95) * 1.2)
    ax.set_ylabel("Index" if indexed else unit)
    ax.set_title("Monthly price trend lines (dashed = overall trend)")
    date_axis(ax)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def chart_bias_and_error(res, et):
    pr = res["preds"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    names = et["Model"].tolist()
    bias = et["Bias"].values
    bars = a1.bar([short(n) for n in names], bias, color=["#ff7f0e" if b > 0 else "#1f77b4" for b in bias])
    for bar, b in zip(bars, bias):
        a1.text(bar.get_x() + bar.get_width() / 2, b, f"{b:+.2f}", ha="center", va="bottom" if b >= 0 else "top", fontsize=8, fontweight="bold")
    a1.axhline(0, color="black", lw=.8)
    a1.margins(y=.2)
    a1.set_title("Bias by model (▲ orange = too high, ▼ blue = too low)", fontsize=10)
    a1.set_ylabel(res["unit"])
    a1.tick_params(axis="x", rotation=25, labelsize=8)
    for n in names:
        err = (pr[n] - pr["Actual"]).abs().groupby(pr["Date"]).mean()
        a2.plot(err.index, err.values, marker="o", ms=4, lw=1.6, color=MODEL_COLOURS.get(n), label=short(n),
                ls="--" if n.startswith("Naive") else "-")
    a2.set_title("Absolute error by month (look for rising error)", fontsize=10)
    a2.set_ylabel(res["unit"])
    date_axis(a2)
    a2.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------
# Evaluation logic
# ----------------------------------------------------------------------------
def eval_table(res, mae_t, mape_t) -> pd.DataFrame:
    mt, pr = res["metrics"], res["preds"]
    naive_rmse = mt.loc[mt["Model"].str.startswith("Naive"), "RMSE"].iloc[0]
    rows = []
    for _, r in mt.iterrows():
        n = r["Model"]
        err = pr[n] - pr["Actual"]
        by_m = err.abs().groupby(pr["Date"]).mean()
        trend = (by_m.iloc[-1] - by_m.iloc[0]) / by_m.iloc[0] * 100 if len(by_m) > 1 and by_m.iloc[0] > 0 else np.nan
        ok_mae, ok_mape = r["MAE"] <= mae_t, r["MAPE %"] <= mape_t
        beats = (not n.startswith("Naive")) and r["RMSE"] < naive_rmse
        rows.append({"Model": n, "MAE": r["MAE"], "MAE target": ok_mae, "MAPE %": r["MAPE %"],
                     "MAPE target": ok_mape, "Beats naive": beats, "Bias": err.mean(), "Error trend %": trend,
                     "Decision": "Proceed" if (ok_mae and ok_mape and beats) else "Refine model"})
    return pd.DataFrame(rows)


def tint_rows(df, column, colours):
    def fn(row):
        c = colours.get(row[column], "")
        return [f"background-color: {c}" if c else "" for _ in row]
    return df.style.apply(fn, axis=1)


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_dashboard(ctx):
    res = current_result(ctx)
    if not res:
        return
    best = res["metrics"].iloc[0]
    unit = res["unit"]
    r1 = st.columns(3)
    r1[0].metric("Best model", res["best"], help=f"{ctx['commodity']} · {res['level']}")
    r1[1].metric("MAE", f"{best['MAE']:.2f} {unit}")
    r1[2].metric("RMSE", f"{best['RMSE']:.2f} {unit}")
    r2 = st.columns(3)
    r2[0].metric("MAPE", f"{best['MAPE %']:.1f}%", help="Average % error")
    r2[1].metric("R²", f"{best['R2']:.2f}", help="Variation explained")
    r2[2].metric("vs naive baseline", f"{best['RMSE vs naive %']:+.1f}%", help="RMSE vs last month's price")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Actual vs predicted")
        view = st.radio("View", ["Test period (zoomed in)", "Full history"], horizontal=True, key="dash_view")
        show(chart_actual_vs_pred(res, view.startswith("Test")))
    with right:
        st.markdown("#### Model leaderboard")
        st.dataframe(res["metrics"][["Rank", "Model", "MAE", "RMSE", "MAPE %", "R2"]].round(2), hide_index=True)
        if res["best"].startswith("Naive"):
            st.warning("No model beat the naive 'last month' baseline on this test period.")
        else:
            st.success(f"{res['best']} is the best model with an average error of {best['MAE']:.2f} {unit} "
                       f"({best['MAPE %']:.1f}%). It beats the naive last-month baseline by "
                       f"{best['RMSE vs naive %']:.1f}% (RMSE).")
    if res["notes"]:
        with st.expander("Run notes"):
            for n in res["notes"]:
                st.write(f"- {n}")


def page_forecasts(ctx):
    res = current_result(ctx)
    if not res:
        return
    name = best_ml(res)
    zoom = st.radio("View", ["Test period (zoomed in)", "Full history"], horizontal=True, key="fc_view").startswith("Test")
    show(chart_best_vs_baseline(res, name, zoom))
    st.markdown("#### Predictions table (test period)")
    tbl = pred_by_month(res).reset_index()
    tbl["Date"] = tbl["Date"].dt.strftime("%b %Y")
    st.dataframe(tbl.rename(columns={"Date": "Month"}).round(2), hide_index=True)
    if res["level"] == "transaction":
        st.caption("Transaction-level predictions are averaged per month for display. Full records are in the export.")


def page_comparison(ctx):
    res = current_result(ctx)
    if not res:
        return
    show(chart_comparison(res))
    st.markdown("#### Full metrics table")
    st.dataframe(res["metrics"].round(3), hide_index=True)


def page_insights(ctx):
    res = current_result(ctx)
    if not res:
        return
    show(chart_insights(res))
    with st.expander("What the words on these charts mean", expanded=True):
        for key in res["importance"].tail(10).index[::-1]:
            label, text = FRIENDLY.get(key, (key, "A feature used by the model."))
            st.markdown(f"**{label}** `[{key}]`  \n{text}")
        if res["lstm_hist"]:
            st.markdown("**Learning curve** - training error should fall as the LSTM learns; "
                        "a rising validation line means over-fitting.")


def page_history(ctx):
    mt = ctx["all_m"][ctx["commodity"]]
    unit = ctx["unit"]
    p = mt["price"]
    first, last = p.iloc[0], p.iloc[-1]
    c = st.columns(3)
    c[0].metric("Mean price", f"{p.mean():.2f} {unit}")
    c[1].metric("Std deviation", f"{p.std():.2f}", help="Month-to-month spread")
    c[2].metric("Records", f"{int(mt['n'].sum()):,}", help=f"{len(mt)} months with trades")
    c = st.columns(3)
    c[0].metric("Lowest month", f"{p.min():.2f}", help=f"{p.idxmin():%b %Y}")
    c[1].metric("Highest month", f"{p.max():.2f}", help=f"{p.idxmax():%b %Y}")
    c[2].metric("First → last month", f"{(last / first - 1) * 100:+.1f}%", help=f"{first:.2f} → {last:.2f}")
    st.markdown("#### Monthly price history")
    fig = plt.figure(figsize=(12, 6), constrained_layout=True)
    draw_history(fig, mt, ctx["commodity"], unit)
    show(fig)
    st.markdown("#### Monthly data table")
    tbl = mt.reset_index().rename(columns={"index": "Date"})
    tbl["Month"] = tbl["Date"].dt.strftime("%b %Y")
    tbl = tbl[["Month", "price", "min_price", "max_price", "qty", "n"]]
    tbl.columns = ["Month", "Avg price", "Min", "Max", "Qty sold", "Records"]
    st.dataframe(tbl.round(2), hide_index=True)


def page_trends(ctx):
    all_m, df, unit, commodity = ctx["all_m"], ctx["df"], ctx["unit"], ctx["commodity"]
    c1, c2, _ = st.columns([1, 1, 2])
    mode = c1.radio("Prices", ["Actual prices", "Indexed (first 3 months = 100)"], key="tr_mode")
    labels_on = c2.checkbox("Show data labels", value=True, key="tr_labels")
    ylab = year_labels(df)
    t1, t2, t3 = st.tabs(["Year-over-year", "Yearly averages", "Trend lines"])
    with t1:
        st.caption("Each line is one year for the commodity selected in the sidebar - compare the same calendar month across years.")
        show(chart_yoy(all_m, commodity, unit, ylab, labels_on))
    with t2:
        show(chart_yearly(all_m, df, commodity, unit, ylab, labels_on))
    with t3:
        sel = st.multiselect("Show", list(all_m), default=list(all_m), key="tr_sel")
        zoom = st.checkbox("Zoom in (hide extreme spikes)", key="tr_zoom")
        show(chart_trends(all_m, sel, commodity, unit, mode.startswith("Indexed"), zoom, labels_on))
    st.markdown("#### Summary by commodity")
    tbl = compare_table(all_m)
    sty = tint_rows(tbl, "Commodity", {commodity: "rgba(82,183,136,.30)"}).format(precision=2, na_rep="")
    st.dataframe(sty, hide_index=True)


def page_evaluation(ctx):
    res = current_result(ctx)
    if not res:
        return
    unit = res["unit"]
    c1, c2, _ = st.columns([1, 1, 2])
    mae_t = c1.number_input(f"MAE ≤ ({unit})", min_value=0.0, value=2.0, step=0.1, key="ev_mae")
    mape_t = c2.number_input("MAPE ≤ (%)", min_value=0.0, value=15.0, step=0.5, key="ev_mape")
    st.caption("A model is accepted only if it meets both targets and beats the naive baseline.")
    et = eval_table(res, mae_t, mape_t)
    ok = et[et["Decision"] == "Proceed"]
    if len(ok):
        st.success(f"✔ {ok.iloc[0]['Model']} meets the acceptance criteria → proceed to pilot / deployment.")
    else:
        st.warning("No model meets all acceptance criteria yet → refine the models before deployment.")
    st.markdown("#### Acceptance criteria per model")
    disp = et.copy()
    for col in ("MAE target", "MAPE target", "Beats naive"):
        disp[col] = disp[col].map({True: "✔", False: "✖"})
    disp["Error trend"] = disp["Error trend %"].map(lambda v: "" if pd.isna(v) else f"{'▲' if v > 0 else '▼'} {v:+.0f}%")
    disp = disp.drop(columns="Error trend %")
    sty = tint_rows(disp, "Decision", {"Proceed": "rgba(82,183,136,.30)", "Refine model": "rgba(255,193,7,.25)"})
    st.dataframe(sty.format({"MAE": "{:.3f}", "MAPE %": "{:.1f}", "Bias": "{:+.2f}"}), hide_index=True)
    st.caption("Bias: positive = forecasts too high, negative = too low. Error trend compares the last test month's "
               "error with the first (▲ = error rising).")
    st.markdown("#### Bias and error checks")
    show(chart_bias_and_error(res, et))


def page_monitoring(ctx):
    res = current_result(ctx)
    if not res:
        return
    unit, commodity = res["unit"], ctx["commodity"]
    sub = prepare_commodity(ctx["df"], commodity, ctx["tcol"], ctx["winsorise"])
    mt = ctx["all_m"][commodity]
    expected = len(pd.date_range(mt.index.min(), mt.index.max(), freq="MS"))
    missing = expected - len(mt)
    clipped = int((sub["y"] != sub[ctx["tcol"]].astype(float)).sum())
    pr = res["preds"]
    best = res["best"]
    err = (pr[best] - pr["Actual"]).abs().groupby(pr["Date"]).mean()
    avg = float(err.mean())
    c = st.columns(5)
    c[0].metric("Months with trades", len(mt), help=f"of {expected} expected")
    c[1].metric("Months without trades", missing, help="forward-filled in models")
    c[2].metric("Records received", f"{len(sub):,}", help="valid price records")
    c[3].metric("Outliers clipped", clipped, help="winsorised" if ctx["winsorise"] else "winsorising is off")
    c[4].metric("Latest forecast error", f"{err.iloc[-1]:.2f}", help=f"avg {avg:.2f} {unit}")
    st.markdown("#### Data drift and model drift")
    s = res["hist"]
    earlier, recent = s.iloc[:-6], s.iloc[-6:]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    bins = np.histogram_bin_edges(s.values, bins=10)
    if len(earlier):
        a1.hist(earlier, bins=bins, color="#4a90d9", alpha=.85, label="Earlier months")
    a1.hist(recent, bins=bins, color="#e74c3c", alpha=.6, label="Last 6 months")
    a1.set_title("Price distribution - check for data drift", fontsize=10)
    a1.set_xlabel(f"Price ({unit})")
    a1.set_ylabel("Number of months")
    a1.legend(fontsize=8)
    a2.plot(err.index, err.values, marker="o", lw=2, color="#1f77b4", label="Absolute error")
    a2.axhline(avg, color="red", ls="--", label=f"Average ({avg:.2f})")
    a2.set_title("Forecast error over time - check for model drift", fontsize=10)
    a2.set_ylabel(unit)
    date_axis(a2)
    a2.legend(fontsize=8)
    fig.tight_layout()
    show(fig)
    st.markdown("#### Alerts and system monitoring")
    alerts = []
    if clipped:
        lo, hi = sub.attrs.get("bounds") or (np.nan, np.nan)
        alerts.append((mt.index.max().strftime("%d %b %Y"), "Unusual price",
                       f"{clipped} record(s) outside [{lo:.2f}, {hi:.2f}] {unit} clipped", "Resolved"))
    for d, e in err.items():
        if avg > 0 and e > 1.5 * avg:
            alerts.append((d.strftime("%d %b %Y"), "Rising error", f"Error {e:.2f} is >1.5× the average ({avg:.2f})", "Investigate"))
    if missing:
        alerts.append((mt.index.max().strftime("%d %b %Y"), "Missing data", f"{missing} month(s) without trades were forward-filled", "Investigate"))
    alerts.append((pd.Timestamp.now().strftime("%d %b %Y"), "Model run", f"{best} selected for {commodity}", "Success"))
    adf = pd.DataFrame(alerts, columns=["Date", "Type", "Message", "Status"])
    sty = tint_rows(adf, "Status", {"Resolved": "rgba(82,183,136,.25)", "Success": "rgba(82,183,136,.25)",
                                    "Investigate": "rgba(255,193,7,.25)"})
    st.dataframe(sty, hide_index=True)


def build_zip(ctx, res) -> bytes:
    buf = io.BytesIO()
    c = ctx["commodity"].lower().replace(" ", "_")
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{c}_metrics.csv", res["metrics"].to_csv(index=False))
        z.writestr(f"{c}_predictions.csv", res["preds"].to_csv(index=False))
        f1 = plt.figure(figsize=(14, 8), constrained_layout=True)
        draw_results(f1, res)
        z.writestr(f"{c}_model_results.png", fig_png(f1))
        f2 = plt.figure(figsize=(12, 6), constrained_layout=True)
        draw_history(f2, ctx["all_m"][ctx["commodity"]], ctx["commodity"], ctx["unit"])
        z.writestr(f"{c}_history.png", fig_png(f2))
        f3 = plt.figure(figsize=(12, 8), constrained_layout=True)
        draw_compare(f3, ctx["all_m"], ctx["commodity"], ctx["df"], ctx["unit"], False)
        z.writestr(f"{c}_trends.png", fig_png(f3))
    return buf.getvalue()


def page_settings(ctx):
    df = ctx["df"]
    st.markdown("#### Dataset")
    st.write(f"**File:** {ctx['label']}  \n**Rows loaded:** {len(df):,}  \n"
             f"**Commodities:** {', '.join(sorted(df['Commodity'].unique()))}  \n"
             f"**Period:** {df['Date'].min():%b %Y} to {df['Date'].max():%b %Y}")
    st.markdown("#### Export")
    res = st.session_state.get("res")
    if res is not None and st.session_state.get("res_sig") == ctx["sig"]:
        c = ctx["commodity"].lower().replace(" ", "_")
        b1, b2, b3 = st.columns(3)
        b1.download_button("Metrics (CSV)", res["metrics"].to_csv(index=False).encode("utf-8"), f"{c}_metrics.csv", "text/csv")
        b2.download_button("Predictions (CSV)", res["preds"].to_csv(index=False).encode("utf-8"), f"{c}_predictions.csv", "text/csv")
        b3.download_button("Metrics, predictions and charts (ZIP)", build_zip(ctx, res), f"{c}_results.zip", "application/zip")
    else:
        st.info("Run the models first to export metrics, predictions and charts.")
    st.markdown("#### Evaluation checklist")
    st.markdown(
        "- ✔ Split chronologically - never randomly for time series (the last N months are the test set)\n"
        "- ✔ Compare against a simple baseline (last month's price)\n"
        "- ✔ Use several complementary metrics: MAE, RMSE, MAPE and R²\n"
        "- ✔ Evaluate each commodity separately (pick one in the sidebar)\n"
        "- ✔ Check for systematic bias (Evaluation Checks page)\n"
        "- ✔ Assess business usefulness against acceptance targets (Evaluation Checks page)\n"
        "- ✔ Monitor data drift and model drift after deployment (Monitoring page)\n"
        "- ❗ A high R² alone does not prove accurate future forecasts - judge on unseen data and business cost."
    )
    with st.expander("Dataset preview"):
        st.dataframe(df, height=400)
        st.download_button("Download cleaned dataset (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "cleaned_commodity_dataset.csv", "text/csv")


PAGE_FUNCS = {
    "Dashboard": page_dashboard, "Price Forecasts": page_forecasts, "Model Comparison": page_comparison,
    "Model Insights": page_insights, "Historical Data": page_history, "Trends & Seasonality": page_trends,
    "Evaluation Checks": page_evaluation, "Monitoring": page_monitoring, "Settings & Export": page_settings,
}


# ----------------------------------------------------------------------------
# App shell
# ----------------------------------------------------------------------------
def run_models(ctx, status_box):
    counter = {"n": 0}
    bar = status_box.progress(0, text="Preparing dataset…")

    def report(msg):
        counter["n"] += 1
        bar.progress(min(counter["n"] / 6, .95), text=msg)

    try:
        res = run_pipeline(ctx["df"], ctx["commodity"], ctx["level"], ctx["target_key"],
                           ctx["test_months"], ctx["winsorise"], progress=report)
    except Exception as exc:
        status_box.error(f"Forecasting could not be completed: {exc}")
        return
    st.session_state["res"] = res
    st.session_state["res_sig"] = ctx["sig"]
    status_box.success(f"Done - best model: {res['best']}")


def main():
    status_box = st.empty()
    st.sidebar.title(APP_NAME)
    st.sidebar.caption(f"Produce market analytics · {APP_BUILD}")
    page = st.sidebar.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
    st.sidebar.markdown("---")
    uploaded = st.sidebar.file_uploader(
        "Upload your own dataset", type=["csv", "xlsx", "xls"],
        help="Upload a CSV or Excel workbook with the commodity, date, price and market columns.")

    try:
        df, label = get_dataset(uploaded)
        if df is None:
            st.title(page)
            st.info("Upload a CSV or Excel dataset using **Upload your own dataset** in the sidebar to get started.")
            st.markdown("### Expected dataset fields")
            st.write("Product, Year, Month, Variety, Class, Size, Market, Qty Sold, Qty On Hand, Inflation, "
                     "Average Temp, Total Rainfall, Ave Price/Unit and Ave Price/Kg.")
            return
        if df.empty:
            st.error("The dataset contains no usable records after cleaning.")
            return

        st.sidebar.markdown("---")
        st.sidebar.subheader("Settings")
        target_key = st.sidebar.selectbox("Target", list(TARGETS))
        tcol, unit = TARGETS[target_key]
        winsorise = st.sidebar.checkbox("Winsorise extreme prices (1st–99th percentile)", value=True)
        all_m = all_monthly(df, tcol, winsorise)
        if not all_m:
            st.error(f"No usable {tcol} values were found. Check the price columns in your dataset.")
            return
        commodities = list(all_m)
        commodity = st.sidebar.selectbox("Commodity", commodities,
                                         index=commodities.index("Potatoes") if "Potatoes" in commodities else 0)
        level_label = st.sidebar.radio("Data level", list(LEVELS))
        test_months = st.sidebar.slider("Test months", 3, 9, 5)
        run = st.sidebar.button("▶ Run all models", type="primary")

        ctx = {"df": df, "label": label, "all_m": all_m, "commodity": commodity, "target_key": target_key,
               "tcol": tcol, "unit": unit, "level": LEVELS[level_label], "test_months": test_months,
               "winsorise": winsorise,
               "sig": (label, commodity, target_key, LEVELS[level_label], test_months, winsorise)}
        if run:
            run_models(ctx, status_box)

        st.title(page)
        st.caption(f"Dataset: **{label}** · {len(df):,} cleaned records · {commodity}")
        PAGE_FUNCS[page](ctx)
    except Exception as exc:
        st.error(f"Unable to load this dataset: {exc}")
        st.info("Check that your file includes the required columns and that the first "
                "10 rows contain a header row with a Product column.")
        with st.expander("Technical details"):
            st.exception(exc)


main()
