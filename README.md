# 📈 Stock Price Forecasting – Apple (AAPL)

This project demonstrates **time series forecasting** using Apple (AAPL) stock prices from **2015–2023**.  
Three different forecasting approaches are implemented and compared: **ARIMA, Prophet, and LSTM**.

---

## 🚀 Project Overview

**Objective:**  
Predict future Apple stock prices using historical data from Yahoo Finance, and compare the performance of **statistical** and **deep learning** models.

**Models Used:**
- **ARIMA (AutoRegressive Integrated Moving Average):** Classical statistical model.  
- **Prophet:** Facebook’s model for trend and seasonality forecasting.  
- **LSTM (Long Short-Term Memory):** Deep learning model designed for sequential data.  

---

## 🗂 Data

- **Source:** Yahoo Finance (`yfinance` library)  
- **Time Period:** 2015-01-01 to 2023-12-31  
- **Features Used:** Adjusted Closing Price (Close)  
- **Train-Test Split:** 80% train, 20% test (time-ordered split to preserve sequence)  

---

## 🔹 Data Exploration & Preprocessing

- Plotted stock prices to visualize trends and volatility.  
- Calculated rolling mean (30-day) for trend smoothing.  
- Tested for stationarity using **Augmented Dickey-Fuller (ADF) test**.  
- Examined **Autocorrelation (ACF)** and **Partial Autocorrelation (PACF)** for ARIMA.  
- Normalized data for LSTM using **Min-Max scaling**.  

---

## 🔹 Modeling

### 1️⃣ ARIMA
- Parameters: (p=5, d=1, q=0)  
- First-order differencing used to make series stationary.  
- Forecasted test set and future **30 trading days**.  

### 2️⃣ Prophet
- Captures trend and seasonality.  
- Forecasted test set and future **30 trading days**.  
- Handles business-day alignment with stock market calendar.  

### 3️⃣ LSTM
- Input sequences of **60 previous days** to predict next day.  
- Two stacked LSTM layers with **50 neurons each**.  
- Trained for **10 epochs** with batch size **64**.  
- Forecasted test set and future **30 trading days** using recursive prediction.  

---

## 📊 Model Evaluation

**Metric:** Root Mean Square Error (RMSE)

| Model   | RMSE  | Notes |
|---------|-------|-------|
| ARIMA   | 20.60 | Good for short-term trends; struggles with non-linear patterns |
| Prophet | 37.27 | Captures trend & seasonality; less accurate for volatile stock data |
| LSTM    | 5.90  | Best performance; captures non-linear sequential dependencies |

**Key Insight:**  
➡️ **LSTM outperforms ARIMA and Prophet**, showing the strength of deep learning in modeling complex sequential financial data.  

---

## 📈 Future Forecasting

- All three models forecasted **30 trading days** into the future.  
- **LSTM** provides the most accurate and smooth predictions.  
- **ARIMA & Prophet** show larger deviations but remain interpretable.  

⚠️ These forecasts can assist in short-term decision-making, but should be combined with financial analysis.  

---

## 🔧 Libraries & Tools Used

- Python 3  
- `yfinance` – Stock data retrieval  
- `pandas`, `numpy`, `matplotlib` – Data handling & visualization  
- `statsmodels` – ARIMA modeling  
- `prophet` – Trend & seasonality forecasting  
- `tensorflow.keras` – LSTM deep learning  
- `scikit-learn` – Min-Max scaling, RMSE calculation  

---

## 📌 Conclusion

- Stock price forecasting is challenging due to volatility and noise.  
- **LSTM proved to be the most accurate model** for Apple stock, handling non-linear patterns effectively.  
- **ARIMA and Prophet** provide interpretable results, useful for understanding trends and seasonality.  
- Combining **statistical + deep learning** models could be a robust approach for financial forecasting.  

---

👤 **Author:** Kapil Dev Bhandari  
📍 *AI & Computer Science Student | Specializing in Machine Learning and Deep Learning*
