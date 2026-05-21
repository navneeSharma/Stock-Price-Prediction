# 📈 Stock Price Prediction with Deep Learning

Predicting short-term stock prices using an LSTM-based Recurrent Neural Network trained on historical market data from 37 companies — helping investors and fund managers make more informed decisions.

---

## 📌 Problem Statement

Stock markets are volatile and difficult to interpret manually at scale. This project explores:

> *Can we use deep learning on historical price patterns to generate reliable short-term stock price forecasts?*

**Business impact:** Accurate short-term forecasts can support fund managers, retail investors, and trading platforms in reducing losses and timing decisions more effectively.

---

## 📊 Dataset

- **Source:** Kaggle
- **Coverage:** 37 companies, historical data from April 2020 onwards
- **Features:** Date, Open, High, Low, Close, Adjusted Close, Volume, Stock ticker

---

## 🔧 Methodology

### 1. Exploratory Data Analysis
- Candlestick chart visualizations for OHLC price patterns
- Trend analysis across multiple stocks

### 2. Preprocessing
- Parsed dates as datetime objects for time-series handling
- Normalized price data using **MinMaxScaler**
- Randomly sampled one company per run (reusable & reproducible pipeline)

### 3. Model Architecture

Built a **Sequential LSTM model** using TensorFlow/Keras:
- LSTM layers with Dropout regularization to reduce overfitting
- Dense output layer for price regression
- Callbacks: `EarlyStopping` and `ModelCheckpoint` for efficient training

### 4. Experiments
- Ran **10 experiments** varying hyperparameters (layers, units, dropout rates, epochs)
- Forecasted stock prices for the next **15 days**

---

## 🛠️ Tech Stack

`Python` `TensorFlow` `Keras` `Pandas` `NumPy` `Matplotlib` `scikit-learn` `mplfinance`

---

## 📁 Project Structure

```
Stock-Price-Prediction/
│
└── Stock Price Prediction.ipynb   # EDA → preprocessing → LSTM model → experiments → forecast
```

---

## 💡 Key Takeaways

- LSTM is well-suited for time-series forecasting due to its ability to learn long-range sequential patterns
- Hyperparameter tuning (dropout rate, number of LSTM units, epochs) significantly impacts prediction quality
- The randomized company selection makes the notebook reusable across all 37 stocks with no code changes

---

## ⚠️ Limitations & Future Work

- Only one company was fully modelled due to computational constraints (10 experiments took 1+ hour)
- [ ] Scale to all 37 companies with cloud compute (Google Colab Pro / AWS)
- [ ] Incorporate external signals: news sentiment, trading volume spikes
- [ ] Compare LSTM against ARIMA and Facebook Prophet for benchmark

---

## 👤 Author

**Navnee Sharma** — [LinkedIn](https://linkedin.com/in/navneesharma) | [navnee.sharma30@gmail.com](mailto:navnee.sharma30@gmail.com)
