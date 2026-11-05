
"""
Commodity Price Forecaster
Streamlit interface for the existing fresh-produce price forecasting engine.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from commodity_price_forecasting import (
    TARGETS,
    MONTH_NUM,
    _std_name,
    draw_history,
    draw_results,
    load_data,
    monthly_table,
    prepare_commodity,
    run_pipeline,
)


APP_NAME = "Commodity Price Forecaster"


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
      .block-container {
          padding-top: 1.6rem;
          padding-bottom: 2rem;
      }

      [data-testid="stMetric"] {
          background: #f3f7f4;
          border: 1px solid #e2ebe5;
          padding: 14px 16px;
          border-radius: 10px;
      }

      h1, h2, h3 {
          color: #1b4332;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# DATA UPLOAD AND CLEANING
# ---------------------------------------------------------

def _normalise_excel(uploaded_file) -> pd.DataFrame:
    """
    Read and clean an Excel worksheet directly.
    Does not convert the Excel workbook into CSV.
    """

    # Read the uploaded Excel workbook
    workbook = pd.ExcelFile(
        io.BytesIO(uploaded_file.getvalue())
    )

    if not workbook.sheet_names:
        raise ValueError(
            "The uploaded Excel workbook does not contain any worksheets."
        )

    # Let the user choose the worksheet
    sheet = st.sidebar.selectbox(
        "Worksheet",
        workbook.sheet_names,
        index=0,
        key="excel_worksheet",
    )

    # Read worksheet without assuming the first row is the header
    raw = pd.read_excel(
        workbook,
        sheet_name=sheet,
        header=None,
        dtype=str,
    ).fillna("")

    if raw.empty:
        raise ValueError(
            "The selected worksheet is empty. Please select another worksheet."
        )

    # Locate the header row within the first 10 rows
    header_row = None

    for idx in range(min(10, len(raw))):

        values = [
            str(value).strip().lower()
            for value in raw.iloc[idx].tolist()
        ]

        if "product" in values:
            header_row = idx
            break

    if header_row is None:
        raise ValueError(
            "Could not find a header row containing a 'Product' column "
            "in the first 10 rows of the selected worksheet."
        )

    # Extract records below the header
    df = raw.iloc[header_row + 1:].copy()

    # Standardise column names using the original forecasting engine
    df.columns = [
        _std_name(column)
        for column in raw.iloc[header_row].tolist()
    ]

    df = df.reset_index(drop=True)

    # Remove completely empty rows
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    df = df.dropna(how="all").copy()

    # Required dataset columns
    required_columns = [
        "Market",
        "Year",
        "Month",
        "Product",
        "Variety",
        "Class",
        "Size",
        "QtySold",
        "QtyOnHand",
        "Inflation",
        "Temp",
        "Rain",
        "TotalValue",
        "PriceUnit",
        "PriceKg",
    ]

    # Check for missing columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Your Excel file is missing these required columns: "
            + ", ".join(missing_columns)
        )

    # Convert numerical columns
    numeric_columns = [
        "Year",
        "QtySold",
        "QtyOnHand",
        "Inflation",
        "Temp",
        "Rain",
        "TotalValue",
        "PriceUnit",
        "PriceKg",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column]
            .astype(str)
            .str.replace(
                r"[^0-9.\-]",
                "",
                regex=True,
            ),
            errors="coerce",
        )

    # Standardise text columns
    text_columns = [
        "Product",
        "Variety",
        "Class",
        "Size",
        "Market",
    ]

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    # Create commodity names
    df["Commodity"] = df["Product"].str.title()

    # Convert month names into month numbers
    df["MonthNum"] = (
        df["Month"]
        .astype(str)
        .str.strip()
        .str[:3]
        .str.lower()
        .map(MONTH_NUM)
    )

    # Remove records without a valid year or month
    df = df.dropna(
        subset=["Year", "MonthNum"]
    ).copy()

    if df.empty:
        raise ValueError(
            "No usable records were found. Please check the Year and Month columns."
        )

    # Create the date column
    df["Date"] = pd.to_datetime(
        dict(
            year=df["Year"].astype(int),
            month=df["MonthNum"].astype(int),
            day=1,
        )
    )

    # Fill missing stock quantities
    df["QtyOnHand"] = df["QtyOnHand"].fillna(
        df["QtyOnHand"].median()
    )

    return (
        df.sort_values(
            "Date",
            kind="stable",
        )
        .reset_index(drop=True)
    )


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load CSV, XLSX or XLS datasets.
    """

    suffix = Path(uploaded_file.name).suffix.lower()

    # CSV upload
    if suffix == ".csv":

        return load_data(
            io.BytesIO(
                uploaded_file.getvalue()
            )
        )

    # Excel upload
    elif suffix in {".xlsx", ".xls"}:

        return _normalise_excel(uploaded_file)

    else:

        raise ValueError(
            "Please upload a CSV, XLSX or XLS file."
        )


# ---------------------------------------------------------
# DEFAULT DATASET
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_default_dataset(path: str) -> pd.DataFrame:

    return load_data(path)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

def render_sidebar():

    st.sidebar.title("Commodity Price Forecaster")

    st.sidebar.caption(
        "Fresh-produce price forecasting"
    )

    uploaded = st.sidebar.file_uploader(
        "Upload your own dataset",
        type=["csv", "xlsx", "xls"],
        help=(
            "Upload a CSV or Excel workbook. "
            "The dataset must contain the required commodity, "
            "date, price and market-related columns."
        ),
    )

    st.sidebar.markdown("---")

    st.sidebar.subheader("Forecast settings")

    target_key = st.sidebar.selectbox(
        "Price measure",
        list(TARGETS.keys()),
    )

    level_label = st.sidebar.radio(
        "Modelling level",
        [
            "Monthly average (time series)",
            "Transaction-level (every record)",
        ],
    )

    level = (
        "monthly"
        if level_label.startswith("Monthly")
        else "transaction"
    )

    test_months = st.sidebar.slider(
        "Test months",
        min_value=2,
        max_value=12,
        value=5,
    )

    winsorise = st.sidebar.checkbox(
        "Limit extreme prices (1st–99th percentile)",
        value=True,
    )

    return (
        uploaded,
        target_key,
        level,
        test_months,
        winsorise,
    )


# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

def main():

    st.title("📈 Commodity Price Forecaster")

    st.write(
        "Explore historical fresh-produce prices, compare forecasting models, "
        "and evaluate predictions using your own dataset."
    )

    (
        uploaded,
        target_key,
        level,
        test_months,
        winsorise,
    ) = render_sidebar()

    try:

        # -------------------------------------------------
        # LOAD UPLOADED DATASET
        # -------------------------------------------------

        if uploaded is not None:

            with st.spinner("Loading your dataset..."):

                df = load_uploaded_dataset(uploaded)

            data_label = uploaded.name

        # -------------------------------------------------
        # LOAD DEFAULT DATASET
        # -------------------------------------------------

        else:

            default_path = os.getenv(
                "COMMODITY_DATA_PATH",
                "Cleaned_dataset.csv",
            )

            if not os.path.exists(default_path):

                st.info(
                    "Upload a CSV or Excel dataset using "
                    "**Upload your own dataset** in the sidebar "
                    "to get started. You can also place "
                    "`Cleaned_dataset.csv` beside the app."
                )

                st.markdown("### Expected dataset fields")

                st.write(
                    "The forecasting engine expects columns for "
                    "Product, Year, Month, Variety, Class, Size, "
                    "Qty Sold, Qty On Hand, Inflation, Average Temp, "
                    "Total Rainfall, Ave Price/Unit and Ave Price/Kg. "
                    "The loader recognises the original assignment "
                    "dataset's column naming."
                )

                return

            df = load_default_dataset(default_path)

            data_label = Path(default_path).name

        # -------------------------------------------------
        # VALIDATE DATASET
        # -------------------------------------------------

        if df.empty:

            st.error(
                "The uploaded dataset contains no usable records after cleaning."
            )

            return

        commodities = sorted(
            df["Commodity"].dropna().unique().tolist()
        )

        if not commodities:

            st.error(
                "No commodities were found. Check the Product column in your dataset."
            )

            return

        # -------------------------------------------------
        # DATASET SUMMARY
        # -------------------------------------------------

        st.caption(
            f"Dataset: **{data_label}** · {len(df):,} cleaned records"
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Records",
            f"{len(df):,}",
        )

        c2.metric(
            "Commodities",
            len(commodities),
        )

        c3.metric(
            "Markets",
            df["Market"].nunique() if "Market" in df else "—",
        )

        c4.metric(
            "Date range",
            f"{df['Date'].min():%Y}–{df['Date'].max():%Y}",
        )

        st.divider()

        # -------------------------------------------------
        # APPLICATION TABS
        # -------------------------------------------------

        (
            tab_dashboard,
            tab_forecast,
            tab_history,
            tab_data,
        ) = st.tabs(
            [
                "Dashboard",
                "Forecast & model comparison",
                "Historical prices",
                "Dataset preview",
            ]
        )

        # =================================================
        # TAB 1: DASHBOARD
        # =================================================

        with tab_dashboard:

            st.subheader("Commodity overview")

            selected = st.selectbox(
                "Select commodity",
                commodities,
                key="overview_commodity",
            )

            target_col, unit = TARGETS[target_key]

            sub = prepare_commodity(
                df,
                selected,
                target_col,
                winsorise,
            )

            if sub.empty:

                st.warning(
                    f"No usable {target_col} values were found for {selected}."
                )

            else:

                monthly = monthly_table(sub)

                a, b, c = st.columns(3)

                a.metric(
                    "Latest monthly price",
                    f"{monthly['price'].iloc[-1]:,.2f} {unit}",
                )

                b.metric(
                    "Average price",
                    f"{monthly['price'].mean():,.2f} {unit}",
                )

                c.metric(
                    "Monthly observations",
                    f"{len(monthly):,}",
                )

                fig, ax = plt.subplots(
                    figsize=(12, 4)
                )

                ax.plot(
                    monthly.index,
                    monthly["price"],
                    marker="o",
                    linewidth=2,
                )

                ax.fill_between(
                    monthly.index,
                    monthly["min_price"],
                    monthly["max_price"],
                    alpha=.15,
                )

                ax.set_title(
                    f"{selected} – monthly price history"
                )

                ax.set_ylabel(unit)

                ax.grid(alpha=.25)

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

        # =================================================
        # TAB 2: FORECASTING AND MODEL COMPARISON
        # =================================================

        with tab_forecast:

            st.subheader("Forecasting models")

            selected = st.selectbox(
                "Commodity to forecast",
                commodities,
                key="forecast_commodity",
            )

            st.write(
                "The app compares the naive baseline, Linear Regression, "
                "Random Forest, XGBoost (if installed), SVR and "
                "LSTM (if TensorFlow is installed)."
            )

            if st.button(
                "Run forecasting models",
                type="primary",
                use_container_width=False,
            ):

                progress_area = st.empty()

                bar = st.progress(0)

                stages = [
                    "Preparing dataset…",
                    "Training Linear Regression…",
                    "Tuning Random Forest…",
                    "Tuning XGBoost…",
                    "Tuning SVR…",
                    "Training LSTM…",
                ]

                counter = {"n": 0}

                def report(message):

                    counter["n"] += 1

                    progress_area.info(message)

                    bar.progress(
                        min(
                            counter["n"] / len(stages),
                            0.95,
                        )
                    )

                try:

                    with st.spinner(
                        "Training and evaluating models. "
                        "This may take several minutes…"
                    ):

                        result = run_pipeline(
                            df,
                            selected,
                            level,
                            target_key,
                            test_months,
                            winsorise,
                            progress=report,
                        )

                    st.session_state["forecast_result"] = result

                    st.session_state["forecast_dataset_name"] = data_label

                    bar.progress(1.0)

                    progress_area.success(
                        "Model training completed."
                    )

                except Exception as exc:

                    progress_area.empty()

                    bar.empty()

                    st.error(
                        f"Forecasting could not be completed: {exc}"
                    )

            result = st.session_state.get(
                "forecast_result"
            )

            if (
                result is not None
                and result["commodity"] == selected
            ):

                st.markdown("#### Model performance")

                best = result["metrics"].iloc[0]

                m1, m2, m3, m4 = st.columns(4)

                m1.metric(
                    "Best model",
                    result["best"],
                )

                m2.metric(
                    "RMSE",
                    f"{best['RMSE']:.3f}",
                )

                m3.metric(
                    "MAE",
                    f"{best['MAE']:.3f}",
                )

                m4.metric(
                    "Test observations",
                    f"{result['n_test']:,}",
                )

                st.dataframe(
                    result["metrics"].round(3),
                    use_container_width=True,
                    hide_index=True,
                )

                fig, ax = plt.subplots(
                    figsize=(14, 8)
                )

                draw_results(
                    fig,
                    result,
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

                if result["notes"]:

                    st.markdown("#### Model notes")

                    for note in result["notes"]:

                        st.write(f"- {note}")

                st.download_button(
                    "Download prediction results (CSV)",
                    data=result["preds"].to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=(
                        f"{selected.lower().replace(' ', '_')}"
                        "_predictions.csv"
                    ),
                    mime="text/csv",
                )

                st.download_button(
                    "Download model metrics (CSV)",
                    data=result["metrics"].to_csv(
                        index=False
                    ).encode("utf-8"),
                    file_name=(
                        f"{selected.lower().replace(' ', '_')}"
                        "_model_metrics.csv"
                    ),
                    mime="text/csv",
                )

            else:

                st.info(
                    "Choose a commodity and click "
                    "**Run forecasting models** to generate results."
                )

        # =================================================
        # TAB 3: HISTORICAL PRICES
        # =================================================

        with tab_history:

            st.subheader("Historical data")

            selected = st.selectbox(
                "Commodity",
                commodities,
                key="history_commodity",
            )

            target_col, unit = TARGETS[target_key]

            sub = prepare_commodity(
                df,
                selected,
                target_col,
                winsorise,
            )

            if not sub.empty:

                hist = monthly_table(sub)

                fig, ax = plt.subplots(
                    figsize=(13, 6)
                )

                draw_history(
                    fig,
                    hist,
                    selected,
                    unit,
                )

                fig.tight_layout()

                st.pyplot(
                    fig,
                    use_container_width=True,
                )

                plt.close(fig)

                st.dataframe(
                    hist.reset_index().rename(
                        columns={"Date": "Month"}
                    ),
                    use_container_width=True,
                )

        # =================================================
        # TAB 4: DATASET PREVIEW
        # =================================================

        with tab_data:

            st.subheader("Cleaned dataset")

            st.dataframe(
                df,
                use_container_width=True,
                height=450,
            )

            st.download_button(
                "Download cleaned dataset (CSV)",
                data=df.to_csv(
                    index=False
                ).encode("utf-8"),
                file_name="cleaned_commodity_dataset.csv",
                mime="text/csv",
            )

    # -----------------------------------------------------
    # ERROR HANDLING
    # -----------------------------------------------------

    except Exception as exc:

        st.error(
            f"Unable to load this dataset: {exc}"
        )

        st.info(
            "Check that your file includes the required columns "
            "and that the first 10 rows contain a header row "
            "with a Product column."
        )


# ---------------------------------------------------------
# RUN APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
