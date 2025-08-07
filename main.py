# Import necessary libraries
import io
import os
import webbrowser
import threading
import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# Libraries for time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Supress any warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# 1. Initialize FastAPI app
app = FastAPI(
    title="AI Sales Forecaster API",
    description="An API to upload sales data and get AI-powered forecasts.",
    version="1.1"
)

# --- Helper Functions ---

async def _load_and_validate_data(file: UploadFile, date_column: str, value_column: str) -> pd.DataFrame:
    """Loads data from an uploaded file and performs robust validation and cleaning."""
    try:
        contents = await file.read()
        buffer = io.BytesIO(contents)
        if file.filename.endswith('.csv'):
            data = pd.read_csv(buffer, low_memory=False)
        elif file.filename.endswith(('.xls', '.xlsx')):
            data = pd.read_excel(buffer)
        else:
            raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV or XLSX file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading the uploaded file. Details: {e}")

    # 1. Validate that specified columns exist
    if date_column not in data.columns or value_column not in data.columns:
        raise HTTPException(status_code=400, detail=f"Error: The file must contain the columns '{date_column}' and '{value_column}'.")

    # 2. Rename columns to standard names
    data.rename(columns={value_column: 'sales', date_column: 'date'}, inplace=True)

    # 3. Parse and validate the date column
    try:
        # Using dayfirst=True helps correctly parse DD/MM/YY formats, common outside the US.
        data['date'] = pd.to_datetime(data['date'], errors='coerce', dayfirst=True)
        data.dropna(subset=['date'], inplace=True) # Drop rows where date could not be parsed
        if data.empty:
            raise ValueError("No valid dates found after parsing. Please check the date column format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse the date column. Ensure it contains valid dates. Error: {e}")

    # 4. Parse and validate the value column, converting it to a numeric type
    data['sales'] = pd.to_numeric(data['sales'], errors='coerce')
    if data['sales'].isnull().any():
        num_nulls = data['sales'].isnull().sum()
        print(f"Warning: Found and removed {num_nulls} non-numeric or empty rows in the value column ('{value_column}').")
        data.dropna(subset=['sales'], inplace=True)

    if data.empty:
        raise HTTPException(status_code=400, detail="The dataset is empty after cleaning. Check for valid date and numeric value entries.")

    # 5. Set index, sort (crucial for time series), and clean up
    data.set_index('date', inplace=True)
    data.sort_index(inplace=True)
    data.drop([col for col in ['frequency', 'horizon'] if col in data.columns], axis=1, inplace=True)
    return data

def _get_seasonal_period(freq: str) -> int:
    """Maps frequency string to a seasonal period for SARIMAX."""
    return {"D": 7, "W": 52, "M": 12}.get(freq, 7)

def _create_supervised_dataset(data: pd.DataFrame, target_col: str, exog_features: list, n_lags: int = 3) -> (pd.DataFrame, pd.Series):
    """Creates a supervised learning dataset from time series data with lags."""
    df = data.copy()
    lag_cols = []
    for i in range(1, n_lags + 1):
        lag_col = f'lag_{i}'
        df[lag_col] = df[target_col].shift(i)
        lag_cols.append(lag_col)

    df.dropna(inplace=True)
    x = df[exog_features + lag_cols]
    y = df[target_col]
    return x, y

def _predict_rf_iteratively(model: RandomForestRegressor, history: list, n_lags: int, future_exog: pd.DataFrame) -> list:
    """Helper to iteratively predict future values using a trained RandomForest model."""
    predictions = []
    current_history = history.copy()
    for i in range(len(future_exog)):
        lags = list(reversed(current_history[-n_lags:]))
        current_exog_values = future_exog.iloc[i].values
        model_input = np.concatenate([current_exog_values, lags]).reshape(1, -1)

        pred = model.predict(model_input)[0]
        predictions.append(pred)
        current_history.append(pred)
    return predictions

def _train_and_evaluate_models(data: pd.DataFrame, horizon: int, exog_features: list, seasonal_period: int) -> (str, float):
    """Trains and evaluates models, returning the best model's name and its RMSE."""
    if len(data) <= horizon:
        raise HTTPException(status_code=400, detail=f"Horizon ({horizon}) is too large for the dataset size ({len(data)}).")

    train_data = data.iloc[:-horizon]
    test_data = data.iloc[-horizon:]
    metrics = {}

    # --- SARIMAX Model ---
    try:
        model = SARIMAX(train_data['sales'], exog=train_data[exog_features], order=(1, 1, 1),
                        seasonal_order=(1, 1, 1, seasonal_period)).fit(disp=False)
        pred = model.get_forecast(steps=len(test_data), exog=test_data[exog_features]).predicted_mean
        metrics['SARIMAX'] = {'RMSE': np.sqrt(mean_squared_error(test_data['sales'], pred))}
        print(f"SARIMAX RMSE: {metrics['SARIMAX']['RMSE']:.2f}")
    except Exception as e:
        print(f"SARIMAX failed: {e}")

    # --- Prophet Model ---
    try:
        prophet_df = train_data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
        model = Prophet()
        for feature in exog_features:
            model.add_regressor(feature)
        model.fit(prophet_df)
        future_df = test_data.reset_index().rename(columns={'date': 'ds'})
        pred = model.predict(future_df)['yhat']
        metrics['Prophet'] = {'RMSE': np.sqrt(mean_squared_error(test_data['sales'], pred))}
        print(f"Prophet RMSE: {metrics['Prophet']['RMSE']:.2f}")
    except Exception as e:
        print(f"Prophet failed: {e}")

    # --- RandomForest Model ---
    try:
        # FIX: To avoid data leakage, evaluation must be iterative, like the final forecast.
        n_lags = 7
        # 1. Create supervised dataset from training data ONLY
        x_train, y_train = _create_supervised_dataset(train_data, 'sales', exog_features, n_lags=n_lags)

        if not x_train.empty:
            # 2. Train the model
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(x_train, y_train)

            # 3. Iteratively predict on the test set horizon to simulate real forecasting
            history = train_data['sales'].tolist()
            test_exog = test_data[exog_features]
            predictions = _predict_rf_iteratively(model, history, n_lags, test_exog)

            metrics['RandomForest'] = {'RMSE': np.sqrt(mean_squared_error(test_data['sales'], predictions))}
            print(f"RandomForest RMSE: {metrics['RandomForest']['RMSE']:.2f}")

    except Exception as e:
        print(f"RandomForest failed: {e}")

    if not metrics:
        raise HTTPException(status_code=500, detail="All models failed to train.")

    best_model_name = min(metrics, key=lambda k: metrics[k]['RMSE'])
    min_rmse = metrics[best_model_name]['RMSE']
    print(f"Best model: {best_model_name} with RMSE: {min_rmse:.2f}")
    return best_model_name, min_rmse


def _generate_final_forecast(best_model_name: str, data: pd.DataFrame, horizon: int, exog_features: list, seasonal_period: int, freq: str) -> pd.Series:
    """Retrains the best model on the full dataset and returns the forecast."""
    future_dates = pd.date_range(start=data.index[-1], periods=horizon + 1, freq=freq)[1:]

    # Create future exogenous features
    future_exog = pd.DataFrame(index=future_dates)
    future_exog['month'] = future_exog.index.month
    future_exog['day_of_week'] = future_exog.index.dayofweek
    future_exog['day_of_year'] = future_exog.index.dayofyear
    future_exog['quarter'] = future_exog.index.quarter

    # Add future values for any other exogenous features found in the original data
    # (e.g., user-uploaded features like 'marketing_spend') by using their historical mean.
    user_exog_features = [
        feat for feat in exog_features
        if feat not in ['month', 'day_of_week', 'day_of_year', 'quarter']
    ]
    for feature in user_exog_features:
        if feature in data.columns:
            # Use the mean of the historical data as the future value
            future_exog[feature] = data[feature].mean()

    if best_model_name == 'SARIMAX':
        print("Retraining SARIMAX on full dataset for final forecast.")
        final_model = SARIMAX(data['sales'], exog=data[exog_features], order=(1, 1, 1),
                              seasonal_order=(1, 1, 1, seasonal_period)).fit(disp=False)
        return final_model.get_forecast(steps=horizon, exog=future_exog[exog_features]).predicted_mean

    elif best_model_name == 'Prophet':
        print("Retraining Prophet on full dataset for final forecast.")
        prophet_df = data.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
        final_model = Prophet()
        for feature in exog_features:
            final_model.add_regressor(feature)
        final_model.fit(prophet_df)

        future_df = pd.DataFrame(future_dates, columns=['ds'])
        for feature in exog_features:
            future_df[feature] = future_exog[feature].values

        forecast = final_model.predict(future_df)
        final_forecast_mean = forecast['yhat']
        final_forecast_mean.index = future_dates
        return final_forecast_mean

    elif best_model_name == 'RandomForest':
        print("Retraining RandomForest on full dataset for final forecast.")
        n_lags = 7
        x_full, y_full = _create_supervised_dataset(data, 'sales', exog_features, n_lags=n_lags)
        final_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        final_model.fit(x_full, y_full)

        # Iteratively forecast
        history = data['sales'].tolist()
        predictions = _predict_rf_iteratively(final_model, history, n_lags, future_exog[exog_features])

        return pd.Series(predictions, index=future_dates)
    
    raise ValueError(f"Unknown best_model_name: '{best_model_name}'")

# --- API Endpoints ---

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=FileResponse, include_in_schema=False)
async def serve_frontend():
    """Serves the main HTML file for the user interface."""
    # This path assumes your 'frontend' and 'backend' folders are siblings
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    if not os.path.exists(frontend_path):
        raise HTTPException(status_code=404, detail="index.html not found. Ensure the folder structure is correct.")
    return FileResponse(frontend_path)


@app.post("/api/forecast")
async def create_forecast(
        file: UploadFile = File(..., description="The CSV or XLSX file with historical data."),
        date_column: str = Form(..., description="Name of the column containing dates."),
        value_column: str = Form(..., description="Name of the column containing the values to forecast."),
        horizon: int = Form(..., description="Number of periods to forecast into the future."),
        frequency: str = Form(..., description="Frequency of the data ('Daily', 'Weekly', 'Monthly').")
):
    """
    Receives a data file and parameters, runs a forecasting pipeline,
    and returns the forecast in a table-friendly format.
    """
    # 1. Load and Validate Data
    data = await _load_and_validate_data(file, date_column, value_column)

    # 2. Determine Frequency and Seasonality
    freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
    freq = freq_map.get(frequency, "D")
    seasonal_period = _get_seasonal_period(freq)

    # 3. Dynamic Feature Engineering
    # Identify user-provided exogenous features (any numeric columns other than 'sales')
    user_exog_features = [
        col for col in data.columns
        if pd.api.types.is_numeric_dtype(data[col]) and col != 'sales'
    ]
    print(f"Found user-provided exogenous features: {user_exog_features}")

    # Create time-based features from the date index
    data['month'] = data.index.month
    data['day_of_week'] = data.index.dayofweek
    data['day_of_year'] = data.index.dayofyear
    data['quarter'] = data.index.quarter
    time_based_features = ['month', 'day_of_week', 'day_of_year', 'quarter']
    exogenous_features = user_exog_features + time_based_features

    # 4. Train, Evaluate, and Select Best Model
    best_model_name, min_rmse = _train_and_evaluate_models(data, horizon, exogenous_features, seasonal_period)

    # 5. Generate Final Forecast with Best Model
    final_forecast_mean = _generate_final_forecast(best_model_name, data, horizon, exogenous_features, seasonal_period, freq)

    # 6. Format and Return Output
    forecast_table = [{"date": date.strftime('%Y-%m-%d'), "forecast": round(value, 2)} for date, value in
                      final_forecast_mean.items()]

    return {
        "message": "Forecast generated successfully!",
        "best_model": best_model_name,
        "rmse_on_test_set": round(min_rmse, 2),
        "forecast_horizon": horizon,
        "forecast_frequency": frequency,
        "forecast_table": forecast_table
    }


if __name__ == "__main__":
    # Define the URL for the frontend
    port = 8000
    url = f"http://127.0.0.1:{port}"

    # Define a function to open the browser after a short delay
    def open_browser():
        webbrowser.open_new_tab(url)

    print(f"Starting AI Forecaster API server at {url}")
    print("Your browser will open automatically to the UI.")

    # Use a timer to run the open_browser function after 1 second
    threading.Timer(1, open_browser).start()

    # Run the Uvicorn server
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)
