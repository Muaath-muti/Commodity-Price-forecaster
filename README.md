# Commodity Price Forecaster

A Streamlit web application for exploring fresh-produce prices and comparing time-series forecasting models.

## Features

- Upload a personal dataset in **CSV, XLSX or XLS** format from the sidebar.
- Explore cleaned data and historical monthly prices.
- Select a commodity and price measure (price per kg or price per unit).
- Choose monthly or transaction-level modelling.
- Compare a naive baseline, Linear Regression, Random Forest, XGBoost (when available), SVR and LSTM (when TensorFlow is installed).
- Download prediction results, model metrics and the cleaned dataset.

## Run locally

Python 3.10–3.12 is recommended.

```bash
git clone https://github.com/YOUR-USERNAME/commodity-price-forecaster.git
cd commodity-price-forecaster

python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Dataset upload

Use **Upload your own dataset** in the left sidebar. CSV and Excel files are accepted.

The current forecasting engine expects the same core fields as the original project dataset, including:

- Product, Year, Month
- Variety, Class, Size, Market
- Qty Sold, Qty On Hand
- Inflation, Average Temp, Total Rainfall
- Ave Price/Unit and Ave Price/Kg

The loader searches the first 10 rows for a header row containing `Product`, and applies the original project's column-name normalisation and cleaning logic. A file can upload successfully but still be unsuitable for forecasting if required columns are missing or its column names cannot be recognised.

## Optional models

XGBoost is included in `requirements.txt`. TensorFlow is optional because it is a comparatively large dependency. To enable the LSTM model, install a compatible TensorFlow version for your Python/platform:

```bash
pip install tensorflow
```

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Select **Create app** and choose this GitHub repository.
4. Set the branch to `main` and the main file path to `app.py`.
5. Deploy.

Do not commit private datasets, credentials, or `.streamlit/secrets.toml`. The app can accept user datasets at runtime through its upload control.

## Keep the app available

A Streamlit deployment is not guaranteed to remain continuously awake on every hosting plan. The hosting provider controls sleep and availability behaviour. Once deployed, we can discuss a suitable health-check/ping approach if the host's terms and deployment model permit it.

## Project structure

```text
commodity-price-forecaster/
├── app.py
├── commodity_price_forecasting.py
├── requirements.txt
├── Procfile
├── .gitignore
└── README.md
```

## Notes

This repository contains the original forecasting engine in `commodity_price_forecasting.py` and a new Streamlit interface in `app.py`. Model results depend on the quality, coverage and structure of the dataset supplied.
