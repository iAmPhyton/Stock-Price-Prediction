Stock Price Prediction (Time Series)

Project Overview:
- Predicting stock market movements is notoriously difficult due to volatility. This project compares a statistical approach (ARIMA) vs. a machine learning approach (Facebook Prophet) to forecast Apple (AAPL) stock prices.

The Data
- Source: Yahoo Finance API (`yfinance`).
- Range: Jan 2020 - Jan 2026.
- Target: Daily Closing Price.

Results:
- ARIMA RMSE: ~$15.90 (Conservative, flat-line prediction).
- Prophet RMSE: ~43.37 (Captures trends, but may overfit volatility).
- Insight: The "Weekly Component" chart revealed specific days of the week with consistently higher returns.

Author:
- Name: Chukwuemeka Eugene Obiyo
- Email: praise609@gmail.com
- LinkedIn: https://www.linkedin.com/in/chukwuemekao
