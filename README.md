# Stock-Price-Predictor_Codec
# Stock Price Predictor using Linear Regression with RSI and MACD Indicators

This project predicts the next day closing stock price of a given ticker (default: AAPL) using a Linear Regression model. It incorporates popular technical indicators — Relative Strength Index (RSI) and Moving Average Convergence Divergence (MACD) — as features to enhance prediction accuracy.

---

## Features

- Downloads historical stock data for the past 2 years using Yahoo Finance API
- Computes RSI and MACD technical indicators
- Uses previous day closing price and indicators as features
- Predicts next day closing price using Linear Regression
- Visualizes actual vs predicted stock prices with matplotlib
- Outputs Root Mean Squared Error (RMSE) to evaluate model performance
- Prints the predicted next day closing price

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   cd stock-price-predictor
