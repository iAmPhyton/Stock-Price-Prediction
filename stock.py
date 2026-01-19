import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

#loading data
#ticker: AAPL (Apple), GOOGL (Google), TSLA (Tesla), BTC-USD (Bitcoin)
ticker =  "AAPL" #using data from Apple stocks
print (f"Downloading data for {ticker}...")

#getting data fro Jan 2020 to today
data = yf.download(ticker, start="2020-01-01", end="2026-01-01") 

#inspecting data
print(f"Total Trading Days: {data.shape[0]}")
print(data.head())

#visualizing closing price
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Closing Price')
plt.title(f'{ticker} Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show() 

from statsmodels.tsa.stattools import adfuller

#forcing the data to be a 1D Series
#this ensures it's a flat list, not a 1-column table
price_series = data['Close'].squeeze()

#defining the Test Function
def check_stationarity(series):
    #dropping NaN values immediately before passing to the test
    clean_series = series.dropna()
    
    #safety check: Is there data left?
    if len(clean_series) == 0:
        print("Error: Data is empty after dropping NaNs.")
        return

    try:
        result = adfuller(clean_series)
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        if result[1] < 0.05:
            print("Data is STATIONARY (Ready for Modeling)")
        else:
            print("Data is NOT Stationary (Needs Differencing)")
    except Exception as e:
        print(f"Error running ADF test: {e}")

#checking original price
print("--- Checking Original Price ---")
check_stationarity(price_series)

#creating and checking differenced data
#using the squeezed series to calculate the difference
diff_series = price_series.diff()

print("\n--- Checking Differenced Data ---")
check_stationarity(diff_series)

#plotting visuals
plt.figure(figsize=(12, 4))
plt.plot(diff_series)
plt.title('Daily Changes (Differenced Data)')
plt.axhline(0, color='red', linestyle='--')
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

#creating figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

#plotting ACF (which determines 'q')
#using the differenced data
plot_acf(diff_series.dropna(), lags=20, ax=ax1)
ax1.set_title('Autocorrelation (ACF) - determining "q"')

#ploting PACF (which determines 'p')
plot_pacf(diff_series.dropna(), lags=20, ax=ax2)
ax2.set_title('Partial Autocorrelation (PACF) - determining "p"')

plt.tight_layout()
plt.show() 

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

#splitting the data (keeping the last 60 days for testing)
train_data = price_series[:-60]
test_data = price_series[-60:]

print(f"Training on {len(train_data)} days")
print(f"Testing on {len(test_data)} days") 

#training the model
#order =(p,d,q) => (1,1,1)
model = ARIMA(train_data, order=(1, 1, 1))
fitted_model = model.fit() 

print(fitted_model.summary()) 

#forecasting test set
#start=Index of first test day, end=Index of last test day
forecast_result = fitted_model.get_forecast(steps=len(test_data))
forecast_series = forecast_result.predicted_mean
conf_int = forecast_result.conf_int() #confidence intervals

#aligning the index for plotting
forecast_series.index = test_data.index
conf_int.index = test_data.index 

#calculating RMSE
rmse =  np.sqrt(mean_squared_error(test_data, forecast_series))
print(f"RMSE: ${rmse:.2f}") 

#visualizing RMSE
plt.figure(figsize=(12, 6))
#plotting training data (using only rthe end for clear details)
plt.plot(train_data.index[-100:], train_data[-100:], label='Training Data')
#plotting actual test data
plt.plot(test_data.index, test_data, label='Actual Price', color='blue')
#plotting forecast
plt.plot(forecast_series.index, forecast_series, label='Predicted Price', color='orange', linestyle='--')
#plotting confidence interval
plt.fill_between(conf_int.index,
                 conf_int.iloc[:, 0],
                 conf_int.iloc[:, 1],
                 color='grey', alpha=0.2, label='Confidence Interval')
plt.title('ARIMA Forecast vs Actual (Apple Stock)')
plt.legend()
plt.show() 

from prophet import Prophet
#preparing data for prophet
stock_prophet = data.reset_index()[['Date', 'Close']]
stock_prophet.columns = ['ds', 'y']

#removing timezone info (prophet can be picky about timezones)
stock_prophet['ds'] = stock_prophet['ds'].dt.tz_localize(None) 

#splitting data
train_prophet = stock_prophet.iloc[:-60]
test_prophet = stock_prophet.iloc[-60:] 

#training the model
#daily_seasonalit=True tells Prophet to look for daily patterns
m = Prophet(daily_seasonality=True)
m.fit(train_prophet)

#making a forecast
#making a prediction for next 60 days
future_stock = m.make_future_dataframe(periods=60)
forecast = m.predict(future_stock)

#visual check using Prophet
fig1 = m.plot(forecast)
plt.title('Prophet Forecast')
plt.show()

#checking the components
fig2 = m.plot_components(forecast)
plt.show() 

#calculating the final scoreboard (rmse)
#getting predictions for specific dates tested
prediction_cols = forecast.iloc[-60:]['yhat']
actual_cols = test_prophet['y']

#calculating rmse
rmse_prophet = np.sqrt(mean_squared_error(actual_cols, prediction_cols))
print(f"ARIMA RMSE: $15.90") #from previous step
print(f"Prophet RMSE: ${rmse_prophet:.2f}") 

if rmse_prophet < 15.90:
    print("Prophet is most accurate")
else:
    print("ARIMA (Simpler was better)") 