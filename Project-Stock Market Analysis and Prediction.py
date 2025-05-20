# Stock Price Predictor using Linear Regression with RSI and MACD indicators

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --------------------------
# 1. Download historical data
# --------------------------
ticker = 'AAPL'
end_date = datetime.now()
start_date = datetime(end_date.year - 2, end_date.month, end_date.day)  # Last 2 years

df = pdr.get_data_yahoo(ticker, start=start_date, end=end_date)
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

# --------------------------
# 2. Add Technical Indicators: RSI and MACD
# --------------------------

# RSI function
def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# MACD function
def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Calculate RSI and MACD
df['RSI'] = compute_rsi(df['Close'], 14)
df['MACD'], df['Signal_Line'] = compute_macd(df['Close'])

# Drop NaN rows caused by rolling indicators
df.dropna(inplace=True)

# --------------------------
# 3. Feature Engineering
# --------------------------
# Target: next day close price
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Feature set
features = ['Close', 'RSI', 'MACD', 'Signal_Line']
X = df[features]
y = df['Target']

# --------------------------
# 4. Train/Test Split and Linear Regression Model
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# --------------------------
# 5. Visualization of Prediction vs Actual
# --------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.index, y_test, label='Actual Price', marker='o')
plt.plot(y_test.index, y_pred, label='Predicted Price', marker='x')
plt.xlabel('Date')
plt.ylabel('Stock Price (USD)')
plt.title(f'{ticker} - Actual vs Predicted Closing Price')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --------------------------
# 6. Predict the next day price
# --------------------------
latest_input = df[features].iloc[-1].values.reshape(1, -1)
next_day_prediction = model.predict(latest_input)
print(f"Predicted next closing price for {ticker}: ${next_day_prediction[0]:.2f}")
