import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Fetch Apple stock data
stock = yf.Ticker("AAPL")
data = stock.history(start="2023-01-01", end="2025-03-16")
data.to_csv("aapl_data.csv")
print(data.head())

# Features and target
data["MA_10"] = data["Close"].rolling(window=10).mean()
data["MA_50"] = data["Close"].rolling(window=50).mean()
data["Target"] = data["Close"].shift(-1)
data = data.dropna()

# Prepare data for modeling
X = data[["Close", "MA_10", "MA_50"]]  # Features
y = data["Target"]  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, predictions)
print(f"\nMean Squared Error: {mse:.2f}")

# Plot predictions
plt.plot(y_test.index, y_test, label="Actual Price")
plt.plot(y_test.index, predictions, label="Predicted Price", color="red")
plt.title("AAPL Price Prediction")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("aapl_prediction_plot.png")
plt.show()

# Preview prepared data
print("\nPrepared Data (first 5 rows):")
print(data[["Close", "MA_10", "MA_50", "Target"]].head())