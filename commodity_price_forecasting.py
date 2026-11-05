"""
IFS742 - Group Assignment 2
Fresh-produce price forecasting (CRISP-DM: Modelling phase)

Models compared : Naive baseline, Linear Regression, Random Forest, XGBoost, SVR, LSTM
Commodities     : Apples, Avocado, Onions, Potatoes, Tomatoes  (pick one in the GUI)
Dataset         : Cleaned_dataset.csv  (default folder: C:\\Users\\Admin\\Desktop)

Install once:
    pip install pandas numpy scikit-learn xgboost tensorflow matplotlib

Run the GUI:
    python commodity_price_forecasting.py
Optional:
    python commodity_price_forecasting.py --file "D:\\somewhere\\Cleaned_dataset.csv"
    python commodity_price_forecasting.py --cli Potatoes --level monthly     (no GUI, prints results)

Tabs in the GUI
    1. Model Results      - metrics table, actual vs predicted, error bars, feature importance, LSTM loss
    2. Historical Data    - price history, min-max band, volume, monthly table for the chosen commodity
    3. Trends & Year Comparison - year-over-year lines, all-commodity trend lines, yearly averages
"""

import os
import sys
import queue
import argparse
import threading
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
import matplotlib.dates as mdates
from matplotlib.figure import Figure

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except ImportError:
    HAS_TK = False

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DATA_FOLDER = r"C:\Users\Admin\Desktop"
DATA_FILE = "Cleaned_dataset.csv"
RANDOM_STATE = 42

TARGETS = {"Price per kg (Ave Price/Kg)": ("PriceKg", "R/kg"),
           "Price per unit (Ave Price/Unit)": ("PriceUnit", "R/unit")}
LEVELS = {"Monthly average (time series)": "monthly",
          "Transaction-level (every record)": "transaction"}

MONTH_NUM = {m: i + 1 for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

LSTM_WINDOW = {"monthly": 3, "transaction": 5}
MODEL_COLOURS = {"Naive (last month)": "#7f7f7f", "Linear Regression": "#1f77b4",
                 "Random Forest": "#2ca02c", "XGBoost": "#ff7f0e",
                 "SVR": "#9467bd", "LSTM": "#d62728"}
COMMODITY_COLOURS = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]

# GUI colour palette (green "agriculture" theme)
UI_BG = "#f3f6f4"        # window background
UI_DARK = "#1b4332"      # dark green (header bar)
UI_MAIN = "#2d6a4f"      # main green (buttons, table headers)
UI_LIGHT = "#d8f3dc"     # light green (best row / selected row)
UI_STRIPE = "#f1f8f4"    # alternate table row

matplotlib.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlecolor": UI_DARK,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
    "figure.facecolor": "white", "axes.facecolor": "#fbfdfb",
    "legend.framealpha": 0.9, "legend.edgecolor": "#cccccc",
})


# ----------------------------------------------------------------------------
# 1. DATA LOADING & CLEANING
# ----------------------------------------------------------------------------
def find_dataset(cli_path=None):
    """Look for the CSV in the usual places; return a path or None."""
    candidates = []
    if cli_path:
        candidates.append(cli_path)
    candidates.append(os.path.join(DATA_FOLDER, DATA_FILE))
    candidates.append(os.path.join(os.path.expanduser("~"), "Desktop", DATA_FILE))
    try:
        candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FILE))
    except NameError:
        pass
    candidates.append(os.path.join(os.getcwd(), DATA_FILE))
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _std_name(col):
    c = str(col).strip().lower()
    rules = [("mrk", "Market"), ("year", "Year"), ("month", "Month"), ("product", "Product"),
             ("variety", "Variety"), ("class", "Class"), ("size", "Size"),
             ("qty sold", "QtySold"), ("qty on hand", "QtyOnHand"), ("inflation", "Inflation"),
             ("average temp", "Temp"), ("total rainfall", "Rain"), ("total value", "TotalValue"),
             ("ave price/unit", "PriceUnit"), ("ave price/kg", "PriceKg")]
    for key, name in rules:
        if c.startswith(key):
            return name
    return str(col).strip()


def load_data(path):
    """Read the CSV (handles the blank first row, the degree sign encoding and '1,212.00' strings)."""
    raw, last_err = None, None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            raw = pd.read_csv(path, header=None, dtype=str, encoding=enc, keep_default_na=False)
            break
        except UnicodeDecodeError as e:
            last_err = e
    if raw is None:
        raise last_err

    header_row = None
    for i in range(min(10, len(raw))):
        if "product" in [str(v).strip().lower() for v in raw.iloc[i]]:
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not find the header row (a row containing 'Product').")

    df = raw.iloc[header_row + 1:].copy()
    df.columns = [_std_name(c) for c in raw.iloc[header_row]]
    df = df.reset_index(drop=True)

    for col in ["Year", "QtySold", "QtyOnHand", "Inflation", "Temp", "Rain",
                "TotalValue", "PriceUnit", "PriceKg"]:
        df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True),
                                errors="coerce")

    for col in ["Product", "Variety", "Class", "Size", "Market"]:
        df[col] = df[col].astype(str).str.strip().str.upper()
    df["Commodity"] = df["Product"].str.title()

    df["MonthNum"] = df["Month"].astype(str).str.strip().str[:3].str.lower().map(MONTH_NUM)
    df = df.dropna(subset=["Year", "MonthNum"]).copy()
    df["Date"] = pd.to_datetime(dict(year=df["Year"].astype(int), month=df["MonthNum"].astype(int), day=1))
    df["QtyOnHand"] = df["QtyOnHand"].fillna(df["QtyOnHand"].median())
    return df.sort_values("Date", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. AGGREGATION HELPERS
# ----------------------------------------------------------------------------
def prepare_commodity(df, commodity, target_col, winsorise=True):
    """Rows for one commodity with a clean target column 'y' (optionally clipped to 1st-99th percentile)."""
    sub = df[df["Commodity"] == commodity].dropna(subset=[target_col]).copy()
    sub["y"] = sub[target_col].astype(float)
    bounds = None
    if winsorise and len(sub) >= 20:
        lo, hi = sub["y"].quantile([0.01, 0.99])
        sub["y"] = sub["y"].clip(lo, hi)
        bounds = (lo, hi)
    sub.attrs["bounds"] = bounds
    return sub.sort_values("Date", kind="stable").reset_index(drop=True)


def monthly_table(sub):
    """Volume-weighted monthly price plus min / max / quantity / record count (observed months only)."""
    w = sub["QtySold"].clip(lower=0)
    tmp = sub.assign(_w=w, _pw=sub["y"] * w)
    g = tmp.groupby("Date")
    out = pd.DataFrame({
        "sum_w": g["_w"].sum(), "sum_pw": g["_pw"].sum(), "mean_price": g["y"].mean(),
        "min_price": g["y"].min(), "max_price": g["y"].max(),
        "qty": g["QtySold"].sum(), "n": g.size()})
    out["price"] = (out["sum_pw"] / out["sum_w"].replace(0, np.nan)).fillna(out["mean_price"])
    return out[["price", "min_price", "max_price", "qty", "n"]]


def env_table(df):
    """Inflation / temperature / rainfall per month (same for every commodity)."""
    e = df.groupby("Date")[["Inflation", "Temp", "Rain"]].mean()
    idx = pd.date_range(e.index.min(), e.index.max(), freq="MS")
    return e.reindex(idx).ffill().bfill()


def _month_cyc(dates):
    m = pd.Series(dates).dt.month.values
    return np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)


def build_monthly_dataset(df, sub):
    """One row per month. Features use only information available BEFORE the month (lags),
    plus that month's calendar / weather / inflation values."""
    m = monthly_table(sub)
    idx = pd.date_range(m.index.min(), m.index.max(), freq="MS")
    m = m.reindex(idx)
    n_filled = int(m["price"].isna().sum())
    m["price"] = m["price"].ffill()          # forward fill = no look-ahead
    m["qty"] = m["qty"].fillna(0)
    env = env_table(df).reindex(idx).ffill().bfill()

    X = pd.DataFrame(index=idx)
    for k in (1, 2, 3):
        X[f"lag_{k}"] = m["price"].shift(k)
    X["roll_mean_3"] = m["price"].shift(1).rolling(3).mean()
    X["lag_qty_1"] = m["qty"].shift(1)
    X["month_sin"], X["month_cos"] = _month_cyc(idx)
    X["inflation"], X["temperature"], X["rainfall"] = env["Inflation"], env["Temp"], env["Rain"]
    keep = X.notna().all(axis=1)
    X, y = X[keep], m["price"][keep]
    return {"X": X.reset_index(drop=True), "y": y.values, "dates": pd.Series(X.index),
            "naive": X["lag_1"].values, "note": f"{n_filled} month(s) without trades were forward-filled."}


def build_transaction_dataset(df, sub):
    """One row per record (variety / class / size / month). Includes last month's average price."""
    m = monthly_table(sub)
    idx = pd.date_range(m.index.min(), m.index.max(), freq="MS")
    mprice = m["price"].reindex(idx).ffill()
    prev = mprice.shift(1)

    d = sub.copy()
    d["prev_month_price"] = d["Date"].map(prev)
    d = d.dropna(subset=["prev_month_price"]).reset_index(drop=True)
    d["month_sin"], d["month_cos"] = _month_cyc(d["Date"])
    d["time_idx"] = (d["Date"].dt.year - idx.min().year) * 12 + d["Date"].dt.month - idx.min().month

    X = d[["month_sin", "month_cos", "time_idx", "QtySold", "QtyOnHand",
           "Inflation", "Temp", "Rain", "prev_month_price"]].astype(float).copy()
    cats = pd.get_dummies(d[["Variety", "Class", "Size"]], dtype=float)
    X = pd.concat([X, cats], axis=1)
    return {"X": X, "y": d["y"].values, "dates": d["Date"].reset_index(drop=True),
            "naive": d["prev_month_price"].values, "note": "Rows from the first month dropped (no previous month)."}


def chrono_split(dates, test_months):
    uniq = np.sort(pd.Series(dates).unique())
    if len(uniq) < test_months + 6:
        raise ValueError("Not enough months of data for this test size - reduce 'Test months'.")
    cut = uniq[-test_months]
    test = np.asarray(pd.Series(dates) >= cut)
    return ~test, test


# ----------------------------------------------------------------------------
# 3. MODELS
# ----------------------------------------------------------------------------
def metrics(y_true, y_pred):
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else np.nan
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAPE %": mape,
            "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan}


def _cv(n_train):
    return TimeSeriesSplit(n_splits=max(2, min(5, n_train // 6)))


def _grid(estimator, grid, X, y, n_train):
    gs = GridSearchCV(estimator, grid, cv=_cv(n_train), scoring="neg_root_mean_squared_error", n_jobs=1)
    gs.fit(X, y)
    return gs


def fit_predict_lstm(X, y, train_mask, window, progress):
    import tensorflow as tf
    from tensorflow import keras

    tf.keras.utils.set_random_seed(RANDOM_STATE)
    xs = StandardScaler().fit(X.values[train_mask])
    ys = StandardScaler().fit(y[train_mask].reshape(-1, 1))
    Xs = xs.transform(X.values)
    yt = ys.transform(y.reshape(-1, 1)).ravel()

    padded = np.vstack([np.repeat(Xs[:1], window - 1, axis=0), Xs])
    seqs = np.stack([padded[i:i + window] for i in range(len(Xs))])   # window ends at row i (no future rows)

    n_train = int(train_mask.sum())
    use_val = n_train >= 40
    model = keras.Sequential([
        keras.Input(shape=(window, Xs.shape[1])),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(0.005), loss="mse")
    stop = keras.callbacks.EarlyStopping(monitor="val_loss" if use_val else "loss", patience=20,
                                         restore_best_weights=True)
    hist = model.fit(seqs[train_mask], yt[train_mask], epochs=300,
                     batch_size=8 if n_train < 100 else 32,
                     validation_split=0.15 if use_val else 0.0,
                     callbacks=[stop], verbose=0, shuffle=False)
    pred = model.predict(seqs[~train_mask], verbose=0)
    return ys.inverse_transform(pred).ravel(), hist.history


def run_pipeline(df, commodity, level, target_key, test_months, winsorise, progress=print):
    target_col, unit = TARGETS[target_key]
    progress(f"Preparing {commodity} data ({level})...")
    sub = prepare_commodity(df, commodity, target_col, winsorise)
    if len(sub) < 30:
        raise ValueError(f"Only {len(sub)} records for {commodity} - too few to model reliably.")

    ds = build_monthly_dataset(df, sub) if level == "monthly" else build_transaction_dataset(df, sub)
    X, y, dates, naive = ds["X"], ds["y"], ds["dates"], ds["naive"]
    train_mask, test_mask = chrono_split(dates, test_months)
    n_train, n_test = int(train_mask.sum()), int(test_mask.sum())
    if n_train < 10:
        raise ValueError("Fewer than 10 training rows - reduce 'Test months'.")

    Xtr, Xte, ytr, yte = X[train_mask], X[test_mask], y[train_mask], y[test_mask]
    preds, params, notes = {"Naive (last month)": naive[test_mask]}, {}, [ds["note"]]
    importance = None
    lstm_hist = None

    progress("Training Linear Regression...")
    lin = Pipeline([("sc", StandardScaler()), ("lr", LinearRegression())]).fit(Xtr, ytr)
    preds["Linear Regression"] = lin.predict(Xte)

    progress("Tuning Random Forest (time-series CV)...")
    rf = _grid(RandomForestRegressor(random_state=RANDOM_STATE),
               {"n_estimators": [100, 300], "max_depth": [3, 5, None], "min_samples_leaf": [1, 3]},
               Xtr, ytr, n_train)
    preds["Random Forest"] = rf.predict(Xte)
    params["Random Forest"] = rf.best_params_
    importance = pd.Series(rf.best_estimator_.feature_importances_, index=X.columns).sort_values()

    if HAS_XGB:
        progress("Tuning XGBoost (time-series CV)...")
        xgb = _grid(XGBRegressor(random_state=RANDOM_STATE, n_jobs=1, verbosity=0, subsample=0.8),
                    {"n_estimators": [100, 300], "max_depth": [2, 3, 5], "learning_rate": [0.03, 0.1]},
                    Xtr, ytr, n_train)
        preds["XGBoost"] = xgb.predict(Xte)
        params["XGBoost"] = xgb.best_params_
    else:
        notes.append("xgboost not installed - XGBoost skipped (pip install xgboost).")

    progress("Tuning SVR (time-series CV)...")
    svr_pipe = TransformedTargetRegressor(
        regressor=Pipeline([("sc", StandardScaler()), ("svr", SVR(kernel="rbf"))]),
        transformer=StandardScaler())
    svr = _grid(svr_pipe, {"regressor__svr__C": [1, 10, 100], "regressor__svr__epsilon": [0.01, 0.1],
                           "regressor__svr__gamma": ["scale", "auto"]}, Xtr, ytr, n_train)
    preds["SVR"] = svr.predict(Xte)
    params["SVR"] = {k.replace("regressor__svr__", ""): v for k, v in svr.best_params_.items()}

    progress("Training LSTM (this can take a minute)...")
    try:
        p, lstm_hist = fit_predict_lstm(X, y, train_mask, LSTM_WINDOW[level], progress)
        preds["LSTM"] = p
        params["LSTM"] = {"window": LSTM_WINDOW[level], "units": 32, "dropout": 0.2,
                          "epochs_run": len(lstm_hist["loss"])}
    except ImportError:
        notes.append("tensorflow not installed - LSTM skipped (pip install tensorflow).")

    rows = []
    for name, p in preds.items():
        r = metrics(yte, np.asarray(p))
        r["Model"] = name
        rows.append(r)
    mt = pd.DataFrame(rows)[["Model", "MAE", "RMSE", "MAPE %", "R2"]]
    naive_rmse = mt.loc[mt["Model"] == "Naive (last month)", "RMSE"].iloc[0]
    mt["RMSE vs naive %"] = (naive_rmse - mt["RMSE"]) / naive_rmse * 100
    mt = mt.sort_values("RMSE").reset_index(drop=True)
    mt.insert(0, "Rank", range(1, len(mt) + 1))

    pred_df = pd.DataFrame({"Date": dates[test_mask].values, "Actual": yte})
    for name, p in preds.items():
        pred_df[name] = np.asarray(p)

    best = mt.iloc[0]["Model"]
    if best.startswith("Naive"):
        notes.append("WARNING: no machine-learning model beat the naive 'last month' baseline on this test period.")
    if sub.attrs.get("bounds"):
        notes.append(f"Prices winsorised to [{sub.attrs['bounds'][0]:.2f}, {sub.attrs['bounds'][1]:.2f}] {unit}.")
    if n_test < 15:
        notes.append(f"Only {n_test} test rows - metrics are indicative, not conclusive.")

    return {"commodity": commodity, "level": level, "unit": unit, "target": target_col,
            "metrics": mt, "preds": pred_df, "hist": monthly_table(sub)["price"],
            "importance": importance, "lstm_hist": lstm_hist, "params": params, "notes": notes,
            "n_train": n_train, "n_test": n_test, "n_features": X.shape[1],
            "test_start": pd.Series(dates[test_mask]).min(), "best": best}


# ----------------------------------------------------------------------------
# 4. PLOTTING (all functions draw into a matplotlib Figure)
# ----------------------------------------------------------------------------
def draw_results(fig, res):
    fig.clear()
    # Top row: actual vs predicted across the full width. Bottom row: three smaller charts.
    gs = fig.add_gridspec(2, 3, height_ratios=[1.7, 1])
    ax1 = fig.add_subplot(gs[0, :])
    ax2, ax3, ax4 = (fig.add_subplot(gs[1, i]) for i in range(3))
    unit, mt, pr = res["unit"], res["metrics"], res["preds"]

    hist = res["hist"]
    ax1.plot(hist.index, hist.values, color="lightgrey", lw=2, label="History", zorder=1)
    agg = pr.groupby("Date").mean(numeric_only=True)
    ax1.plot(agg.index, agg["Actual"], color="black", lw=2.2, marker="o", ms=3,
             label="Actual (test)", zorder=5)
    for name in [c for c in agg.columns if c != "Actual"]:
        # lines only (no dots) so the datapoints stay readable
        ax1.plot(agg.index, agg[name], lw=1.3, color=MODEL_COLOURS.get(name), label=name,
                 ls="--" if name.startswith("Naive") else "-", alpha=.9)
    ax1.axvspan(res["test_start"], hist.index.max() + pd.Timedelta(days=15), color="gold", alpha=.12)
    ttl = "Actual vs predicted (test period)"
    if res["level"] == "transaction":
        ttl += " - monthly mean of records"
    ax1.set_title(ttl, fontsize=10)
    ax1.set_ylabel(unit)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    ax1.tick_params(axis="x", rotation=45, labelsize=8)
    # legend outside the plot, on the right
    ax1.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)

    order = mt.sort_values("RMSE")
    x = np.arange(len(order))
    ax2.bar(x - 0.2, order["RMSE"], 0.4, label="RMSE", color="#1f77b4")
    ax2.bar(x + 0.2, order["MAE"], 0.4, label="MAE", color="#ff7f0e")
    ax2.set_xticks(x)
    ax2.set_xticklabels(order["Model"], rotation=30, ha="right", fontsize=8)
    ax2.set_title("Test error by model (lower = better)", fontsize=10)
    ax2.set_ylabel(unit)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3, axis="y")

    imp = res["importance"].tail(8)
    ax3.barh(imp.index, imp.values, color="#2ca02c")
    ax3.set_title("Random Forest - top feature importances", fontsize=10)
    ax3.tick_params(labelsize=8)

    lh = res["lstm_hist"]
    if lh:
        ax4.plot(lh["loss"], label="train loss")
        if "val_loss" in lh:
            ax4.plot(lh["val_loss"], label="validation loss")
        ax4.set_yscale("log")
        ax4.legend(fontsize=8)
        ax4.set_xlabel("epoch", fontsize=8)
    else:
        ax4.text(.5, .5, "LSTM not run", ha="center", va="center")
    ax4.set_title("LSTM training loss (scaled MSE)", fontsize=10)
    ax4.grid(alpha=.3)


def draw_history(fig, mt, commodity, unit):
    fig.clear()
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1])
    ax, axq = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    ax.fill_between(mt.index, mt["min_price"], mt["max_price"], color="#1f77b4", alpha=.15,
                    label="Min-max range in month")
    ax.plot(mt.index, mt["price"], marker="o", ms=3, color="#1f77b4", lw=2,
            label="Volume-weighted avg price")
    ax.plot(mt.index, mt["price"].rolling(3, min_periods=1).mean(), color="#d62728", ls="--",
            label="3-month rolling mean")
    ax.set_title(f"{commodity} - monthly price history", fontsize=11)
    ax.set_ylabel(unit)
    # legend outside the plot, on the right
    ax.legend(loc="center left", bbox_to_anchor=(1.005, 0.5), fontsize=8, frameon=False)
    ax.grid(alpha=.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axq.bar(mt.index, mt["qty"], width=20, color="#7f7f7f")
    axq.set_ylabel("Qty sold")
    axq.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axq.grid(alpha=.3, axis="y")
    for a in (ax, axq):
        a.tick_params(axis="x", rotation=45, labelsize=8)


def _index_series(s, base_n=3):
    s = s.dropna()
    return s / s.iloc[:base_n].mean() * 100


def draw_compare(fig, all_m, commodity, df, unit, indexed):
    fig.clear()
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    axa, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axb = fig.add_subplot(gs[1, :])

    ymap = df.groupby("Year")["Date"].agg(["min", "max"])
    ylabel = {int(y): f"{int(y)} ({r['min']:%b}-{r['max']:%b})" for y, r in ymap.iterrows()}

    sel = all_m[commodity]["price"].dropna()
    for i, yr in enumerate(sorted(sel.index.year.unique())):
        s = sel[sel.index.year == yr]
        axa.plot(s.index.month, s.values, marker="o", lw=2, label=ylabel.get(yr, str(yr)))
    axa.set_xticks(range(1, 13))
    axa.set_xticklabels(MONTH_ABBR, fontsize=8)
    axa.set_title(f"{commodity}: year-over-year by calendar month", fontsize=10)
    axa.set_ylabel(unit)
    axa.legend(fontsize=8)
    axa.grid(alpha=.3)

    names = list(all_m.keys())
    years = sorted(df["Year"].unique().astype(int))
    width = 0.8 / max(len(years), 1)
    for j, yr in enumerate(years):
        vals = []
        for n in names:
            s = all_m[n]["price"].dropna()
            s = s[s.index.year == yr]
            vals.append(s.mean() if len(s) else np.nan)
        axc.bar(np.arange(len(names)) + j * width - 0.4 + width / 2, vals, width, label=ylabel.get(yr, str(yr)))
    axc.set_xticks(range(len(names)))
    axc.set_xticklabels(names, fontsize=8)
    axc.set_title("Average price per year - all commodities", fontsize=10)
    axc.set_ylabel(unit)
    axc.legend(fontsize=7)
    axc.grid(alpha=.3, axis="y")

    for i, n in enumerate(names):
        s = all_m[n]["price"].dropna()
        if len(s) < 3:
            continue
        s_plot = _index_series(s) if indexed else s
        xnum = ((s.index.year - s.index.min().year) * 12 + s.index.month - s.index.min().month).values
        slope, icpt = np.polyfit(xnum, s_plot.values, 1)
        col = COMMODITY_COLOURS[i % len(COMMODITY_COLOURS)]
        bold = n == commodity
        axb.plot(s_plot.index, s_plot.values, color=col, lw=2.6 if bold else 1.2, alpha=1 if bold else .7,
                 marker="o", ms=3, label=f"{n} (trend {slope:+.2f}/month)")
        axb.plot(s_plot.index, slope * xnum + icpt, color=col, ls="--", lw=1.6)
    axb.set_title("Monthly price trend lines - all commodities" +
                  (" (indexed, first 3 months = 100)" if indexed else ""), fontsize=10)
    axb.set_ylabel("Index" if indexed else unit)
    axb.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    axb.tick_params(axis="x", rotation=45, labelsize=8)
    axb.legend(fontsize=8, ncol=3)
    axb.grid(alpha=.3)


def compare_table(all_m):
    rows = []
    for n, m in all_m.items():
        s = m["price"].dropna()
        xnum = ((s.index.year - s.index.min().year) * 12 + s.index.month - s.index.min().month).values
        slope = np.polyfit(xnum, s.values, 1)[0] if len(s) >= 3 else np.nan
        row = {"Commodity": n}
        for yr in sorted(set(s.index.year)):
            row[str(yr)] = s[s.index.year == yr].mean()
        row["Overall avg"] = s.mean()
        row["Trend /month"] = slope
        row["Volatility CV %"] = s.std() / s.mean() * 100
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 5. GUI
# ----------------------------------------------------------------------------
if HAS_TK:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

    class App(tk.Tk):
        def __init__(self, df):
            super().__init__()
            self.title("IFS742 - Fresh Produce Price Forecasting")
            self.geometry("1320x900")
            self.df = df
            self.res = None
            self.running = False
            self.q = queue.Queue()
            self._cache = {}
            self.commodities = sorted(df["Commodity"].unique())

            self._style()
            self._controls()
            self.nb = ttk.Notebook(self)
            self.nb.pack(fill="both", expand=True, padx=6, pady=(0, 6))
            self._tab_results()
            self._tab_history()
            self._tab_compare()
            self.refresh_views()
            # bring the window to the front (otherwise it can open hidden behind the IDE)
            self.update_idletasks()
            self.lift()
            self.attributes("-topmost", True)
            self.after(600, lambda: self.attributes("-topmost", False))
            self.focus_force()

        # ---------- look & feel ----------
        def _style(self):
            self.configure(bg=UI_BG)
            self.minsize(1100, 760)
            st = ttk.Style(self)
            st.theme_use("clam")
            st.configure(".", background=UI_BG, font=("Segoe UI", 9))
            st.configure("TFrame", background=UI_BG)
            st.configure("TLabel", background=UI_BG)
            st.configure("TCheckbutton", background=UI_BG)
            st.configure("TRadiobutton", background=UI_BG)
            st.configure("Bar.TFrame", background=UI_DARK)
            st.configure("Bar.TLabel", background=UI_DARK, foreground="white")
            st.configure("Bar.TCheckbutton", background=UI_DARK, foreground="white")
            st.map("Bar.TCheckbutton", background=[("active", UI_DARK)], foreground=[("active", "white")])
            st.configure("Title.TLabel", background=UI_DARK, foreground="white", font=("Segoe UI", 13, "bold"))
            st.configure("Accent.TButton", background="#52b788", foreground="white",
                         font=("Segoe UI", 9, "bold"), padding=(14, 5), borderwidth=0)
            st.map("Accent.TButton", background=[("active", "#40916c"), ("disabled", "#8aa89a")])
            st.configure("Soft.TButton", background="#e9f5ec", foreground=UI_DARK, padding=(12, 5))
            st.map("Soft.TButton", background=[("active", UI_LIGHT), ("disabled", "#d9dedb")])
            st.configure("TNotebook", background=UI_BG, borderwidth=0)
            st.configure("TNotebook.Tab", background="#cfe3d6", foreground=UI_DARK,
                         padding=(18, 7), font=("Segoe UI", 10, "bold"))
            st.map("TNotebook.Tab", background=[("selected", UI_MAIN), ("active", "#a9d4b8")],
                   foreground=[("selected", "white")])
            st.configure("Treeview", rowheight=24, background="white", fieldbackground="white", borderwidth=0)
            st.map("Treeview", background=[("selected", "#95d5b2")], foreground=[("selected", "black")])
            st.configure("Treeview.Heading", background=UI_MAIN, foreground="white",
                         font=("Segoe UI", 9, "bold"), padding=5, relief="flat")
            st.map("Treeview.Heading", background=[("active", UI_DARK)])
            st.configure("TLabelframe", background=UI_BG)
            st.configure("TLabelframe.Label", background=UI_BG, foreground=UI_DARK, font=("Segoe UI", 9, "bold"))
            st.configure("Horizontal.TProgressbar", background="#52b788", troughcolor="#2b5444")

        def _stripe(self, tree):
            tree.tag_configure("odd", background=UI_STRIPE)
            tree.tag_configure("even", background="white")

        # ---------- controls ----------
        def _controls(self):
            bar = ttk.Frame(self, padding=(10, 8), style="Bar.TFrame")
            bar.pack(fill="x")
            ttk.Label(bar, text="Commodity:", style="Bar.TLabel").grid(row=0, column=0, sticky="w")
            self.var_c = tk.StringVar(value="Potatoes" if "Potatoes" in self.commodities else self.commodities[0])
            cb = ttk.Combobox(bar, textvariable=self.var_c, values=self.commodities, state="readonly", width=12)
            cb.grid(row=0, column=1, padx=(4, 14))
            cb.bind("<<ComboboxSelected>>", lambda e: self.refresh_views())

            ttk.Label(bar, text="Data level:", style="Bar.TLabel").grid(row=0, column=2, sticky="w")
            self.var_l = tk.StringVar(value=list(LEVELS)[0])
            ttk.Combobox(bar, textvariable=self.var_l, values=list(LEVELS), state="readonly", width=30).grid(
                row=0, column=3, padx=(4, 14))

            ttk.Label(bar, text="Target:", style="Bar.TLabel").grid(row=0, column=4, sticky="w")
            self.var_t = tk.StringVar(value=list(TARGETS)[0])
            tcb = ttk.Combobox(bar, textvariable=self.var_t, values=list(TARGETS), state="readonly", width=28)
            tcb.grid(row=0, column=5, padx=(4, 14))
            tcb.bind("<<ComboboxSelected>>", lambda e: self.refresh_views())

            ttk.Label(bar, text="Test months:", style="Bar.TLabel").grid(row=1, column=0, sticky="w", pady=(6, 0))
            self.var_n = tk.IntVar(value=5)
            ttk.Spinbox(bar, from_=3, to=9, textvariable=self.var_n, width=5).grid(
                row=1, column=1, sticky="w", padx=4, pady=(6, 0))
            self.var_w = tk.BooleanVar(value=True)
            wcb = ttk.Checkbutton(bar, text="Winsorise extreme prices (1%-99%)", variable=self.var_w,
                                  command=self.refresh_views, style="Bar.TCheckbutton")
            wcb.grid(row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))

            self.btn_run = ttk.Button(bar, text="Run all models", command=self.on_run, style="Accent.TButton")
            self.btn_run.grid(row=1, column=4, sticky="e", pady=(6, 0))
            self.btn_exp = ttk.Button(bar, text="Export results...", command=self.on_export, state="disabled", style="Soft.TButton")
            self.btn_exp.grid(row=1, column=5, sticky="w", padx=4, pady=(6, 0))

            self.prog = ttk.Progressbar(bar, mode="indeterminate", length=160)
            self.prog.grid(row=0, column=6, padx=10)
            self.status = tk.StringVar(value="Ready - choose a commodity, then Run.")
            ttk.Label(bar, textvariable=self.status, style="Bar.TLabel").grid(row=1, column=6, padx=10, sticky="w")

        def _target(self):
            return TARGETS[self.var_t.get()]

        # ---------- tab 1 ----------
        def _tab_results(self):
            f = ttk.Frame(self.nb)
            self.nb.add(f, text="  Model Results  ")
            top = ttk.Frame(f)
            top.pack(fill="x", padx=6, pady=6)
            cols = ("Rank", "Model", "MAE", "RMSE", "MAPE %", "R2", "RMSE vs naive %")
            # metrics table now fills the full width (run-details box removed)
            self.tree_m = ttk.Treeview(top, columns=cols, show="headings", height=7)
            for c in cols:
                self.tree_m.heading(c, text=c)
                self.tree_m.column(c, width=60 if c == "Rank" else 140, anchor="center", stretch=True)
            self.tree_m.column("Model", width=220, anchor="w", stretch=True)
            self._stripe(self.tree_m)
            self.tree_m.tag_configure("best", background=UI_LIGHT, font=("Segoe UI", 9, "bold"))
            self.tree_m.pack(side="left", fill="x", expand=True)
            self.fig1 = Figure(figsize=(11, 6), constrained_layout=True)
            self.cv1 = FigureCanvasTkAgg(self.fig1, master=f)
            self.cv1.get_tk_widget().pack(fill="both", expand=True)
            NavigationToolbar2Tk(self.cv1, f)

        # ---------- tab 2 ----------
        def _tab_history(self):
            f = ttk.Frame(self.nb)
            self.nb.add(f, text="  Historical Data  ")
            self.lbl_stats = ttk.Label(f, font=("Segoe UI", 10, "bold"), foreground=UI_DARK)
            self.lbl_stats.pack(anchor="w", padx=10, pady=(8, 2))
            # table first (side=bottom) so it always keeps its space; chart fills the rest
            box = ttk.LabelFrame(f, text="Monthly data table (scroll to see every month)")
            box.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
            cols = ("Month", "Avg price", "Min", "Max", "Qty sold", "Records")
            self.tree_h = ttk.Treeview(box, columns=cols, show="headings", height=6)
            for c in cols:
                self.tree_h.heading(c, text=c)
                self.tree_h.column(c, width=110, anchor="center")
            self._stripe(self.tree_h)
            sb = ttk.Scrollbar(box, command=self.tree_h.yview)
            self.tree_h.configure(yscrollcommand=sb.set)
            self.tree_h.pack(side="left", fill="x", expand=True)
            sb.pack(side="left", fill="y")
            self.fig2 = Figure(figsize=(11, 4.5), constrained_layout=True)
            self.cv2 = FigureCanvasTkAgg(self.fig2, master=f)
            self.cv2.get_tk_widget().pack(fill="both", expand=True)

        # ---------- tab 3 ----------
        def _tab_compare(self):
            f = ttk.Frame(self.nb)
            self.nb.add(f, text="  Trends & Year Comparison  ")
            r = ttk.Frame(f)
            r.pack(fill="x", padx=10, pady=(8, 2))
            self.var_idx = tk.BooleanVar(value=False)
            ttk.Radiobutton(r, text="Actual prices", variable=self.var_idx, value=False,
                            command=self.draw_compare_tab).pack(side="left")
            ttk.Radiobutton(r, text="Indexed (first 3 months = 100) - compares growth regardless of price level",
                            variable=self.var_idx, value=True, command=self.draw_compare_tab).pack(side="left", padx=12)
            box = ttk.LabelFrame(f, text="Summary by commodity (average price per year, trend per month, volatility)")
            box.pack(side="bottom", fill="x", padx=8, pady=(0, 8))
            # taller table so all the summary info is visible
            self.tree_c = ttk.Treeview(box, show="headings", height=6)
            self.tree_c.pack(fill="x", padx=2, pady=2)
            self.fig3 = Figure(figsize=(11, 5.5), constrained_layout=True)
            self.cv3 = FigureCanvasTkAgg(self.fig3, master=f)
            self.cv3.get_tk_widget().pack(fill="both", expand=True)

        # ---------- views ----------
        def _all_monthly(self):
            tcol, _ = self._target()
            key = (tcol, self.var_w.get())
            if key not in self._cache:
                self._cache[key] = {c: monthly_table(prepare_commodity(self.df, c, tcol, self.var_w.get()))
                                    for c in self.commodities}
            return self._cache[key]

        def refresh_views(self):
            self._clear_results()
            c = self.var_c.get()
            _, unit = self._target()
            mt = self._all_monthly()[c]
            draw_history(self.fig2, mt, c, unit)
            self.cv2.draw()
            first, last = mt["price"].iloc[0], mt["price"].iloc[-1]
            self.lbl_stats.config(text=(
                f"{c}:  {len(mt)} months with trades  |  mean {mt['price'].mean():.2f} {unit}  |  "
                f"std {mt['price'].std():.2f}  |  min {mt['price'].min():.2f} ({mt['price'].idxmin():%b %Y})  |  "
                f"max {mt['price'].max():.2f} ({mt['price'].idxmax():%b %Y})  |  "
                f"first -> last month: {first:.2f} -> {last:.2f} ({(last / first - 1) * 100:+.1f}%)  |  "
                f"{int(mt['n'].sum())} records"))
            self.tree_h.delete(*self.tree_h.get_children())
            for i, (d, r) in enumerate(mt.iterrows()):
                self.tree_h.insert("", "end", tags=("odd" if i % 2 else "even",), values=(f"{d:%b %Y}", f"{r['price']:.2f}", f"{r['min_price']:.2f}",
                                                      f"{r['max_price']:.2f}", f"{r['qty']:.0f}", int(r["n"])))
            self.draw_compare_tab()

        def draw_compare_tab(self):
            _, unit = self._target()
            all_m = self._all_monthly()
            draw_compare(self.fig3, all_m, self.var_c.get(), self.df, unit, self.var_idx.get())
            self.cv3.draw()
            tbl = compare_table(all_m)
            self.tree_c.delete(*self.tree_c.get_children())
            self.tree_c["columns"] = list(tbl.columns)
            for c in tbl.columns:
                self.tree_c.heading(c, text=c)
                self.tree_c.column(c, width=140, minwidth=110, anchor="center", stretch=True)
            self.tree_c.tag_configure("sel", background=UI_LIGHT, font=("Segoe UI", 9, "bold"))
            for _, r in tbl.iterrows():
                vals = [r["Commodity"]] + ["" if pd.isna(v) else f"{v:.2f}" for v in r.values[1:]]
                self.tree_c.insert("", "end", values=vals, tags=("sel",) if r["Commodity"] == self.var_c.get() else ())

        def _clear_results(self):
            self.res = None
            self.btn_exp.config(state="disabled")
            self.tree_m.delete(*self.tree_m.get_children())
            self.fig1.clear()
            self.fig1.text(0.5, 0.5, "Press 'Run all models' to train and compare the models.",
                           ha="center", va="center", fontsize=12, color="grey")
            self.cv1.draw()

        # ---------- run ----------
        def on_run(self):
            if self.running:
                return
            try:
                n = int(self.var_n.get())
            except (tk.TclError, ValueError):
                messagebox.showerror("Input", "Test months must be a whole number.")
                return
            args = (self.var_c.get(), LEVELS[self.var_l.get()], self.var_t.get(), n, self.var_w.get())
            self.running = True
            self.btn_run.config(state="disabled")
            self.prog.start(12)
            self.nb.select(0)
            threading.Thread(target=self._worker, args=args, daemon=True).start()
            self.after(150, self._poll)

        def _worker(self, *args):
            try:
                res = run_pipeline(self.df, *args, progress=lambda m: self.q.put(("status", m)))
                self.q.put(("done", res))
            except Exception:
                import traceback
                self.q.put(("error", traceback.format_exc()))

        def _poll(self):
            try:
                while True:
                    kind, payload = self.q.get_nowait()
                    if kind == "status":
                        self.status.set(payload)
                    elif kind == "done":
                        self._finish(payload)
                        return
                    else:
                        self._fail(payload)
                        return
            except queue.Empty:
                pass
            self.after(150, self._poll)

        def _finish(self, res):
            self.running = False
            self.prog.stop()
            self.btn_run.config(state="normal")
            self.btn_exp.config(state="normal")
            self.res = res
            self.status.set(f"Done - best: {res['best']}")
            self.tree_m.delete(*self.tree_m.get_children())
            for i, (_, r) in enumerate(res["metrics"].iterrows()):
                self.tree_m.insert("", "end", tags=("best",) if r["Rank"] == 1 else ("odd" if i % 2 else "even",), values=(
                    int(r["Rank"]), r["Model"], f"{r['MAE']:.3f}", f"{r['RMSE']:.3f}", f"{r['MAPE %']:.1f}",
                    f"{r['R2']:.3f}", f"{r['RMSE vs naive %']:+.1f}"))
            draw_results(self.fig1, res)
            self.cv1.draw()

        def _fail(self, tb):
            self.running = False
            self.prog.stop()
            self.btn_run.config(state="normal")
            self.status.set("Error - see message.")
            messagebox.showerror("Model run failed", tb[-1500:])

        # ---------- export ----------
        def on_export(self):
            if not self.res:
                return
            folder = filedialog.askdirectory(title="Choose a folder for the results")
            if not folder:
                return
            c = self.res["commodity"].lower()
            self.res["metrics"].to_csv(os.path.join(folder, f"{c}_metrics.csv"), index=False)
            self.res["preds"].to_csv(os.path.join(folder, f"{c}_predictions.csv"), index=False)
            self.fig1.savefig(os.path.join(folder, f"{c}_model_results.png"), dpi=200)
            self.fig2.savefig(os.path.join(folder, f"{c}_history.png"), dpi=200)
            self.fig3.savefig(os.path.join(folder, f"{c}_trends.png"), dpi=200)
            messagebox.showinfo("Export", f"Saved metrics, predictions and 3 charts to:\n{folder}")


# ----------------------------------------------------------------------------
# 6. ENTRY POINT
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="full path to Cleaned_dataset.csv")
    ap.add_argument("--cli", metavar="COMMODITY", help="run without the GUI, e.g. --cli Potatoes")
    ap.add_argument("--level", default="monthly", choices=["monthly", "transaction"])
    ap.add_argument("--test-months", type=int, default=5)
    args = ap.parse_args()

    path = find_dataset(args.file)
    if path is None and HAS_TK and not args.cli:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=f"Locate {DATA_FILE}", filetypes=[("CSV", "*.csv")])
        root.destroy()
    if not path:
        sys.exit(f"Could not find {DATA_FILE}. Put it in {DATA_FOLDER} or use --file.")

    df = load_data(path)
    print(f"Loaded {len(df)} rows from {path}", flush=True)

    if args.cli or not HAS_TK:
        name = (args.cli or "Potatoes").strip().title()
        name = {"Avocadoes": "Avocado", "Avocados": "Avocado"}.get(name, name)
        res = run_pipeline(df, name, args.level, list(TARGETS)[0], args.test_months, True)
        print(res["metrics"].round(3).to_string(index=False))
        print("\nBest params:", res["params"])
        print("Notes:", *res["notes"], sep="\n - ")
        return

    App(df).mainloop()


if __name__ == "__main__":
    main()
