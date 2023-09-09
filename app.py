import streamlit as st
from pycaret.regression import *
import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objs as go

st.title("Stock Price Prediction App")

# Sidebar
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Enter Stock Symbol (e.g., TSLA):", "TSLA")
num_previous_data = st.sidebar.slider("Number of Previous Data Points:", 1, 1000, 10)
timeframe = st.sidebar.selectbox("Select Timeframe:", ["1d","1wk", "1mo"])
forecast = st.sidebar.slider("Future Days you want to predict:", 1, 30, 10)
# Download stock data
end_date = datetime.datetime.today()
start_date = end_date - datetime.timedelta(days=num_previous_data)
data = yf.download(symbol, start=start_date, end=end_date, interval=timeframe)
data = data.reset_index(drop=False)
data1 = data.copy()
data.drop(['Adj Close', 'Volume'], axis=1, inplace=True)

data.index = pd.to_datetime(data['Date'])
data.drop(['Date'], axis=1, inplace=True)

# Calculate EMA and predict
ema_period = forecast  # You can adjust this as needed
for i in range(ema_period):
    next_date = data.index[-1] + pd.DateOffset(minutes=15)
    next_data = data.iloc[-ema_period:]
    next_sma_open = next_data['Open'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    next_sma_high = next_data['High'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    next_sma_low = next_data['Low'].ewm(span=ema_period, adjust=False).mean().iloc[-1]
    new_row = pd.Series({'Open': next_sma_open, 'High': next_sma_high, 'Low': next_sma_low}, name=next_date)
    data = data.append(new_row)

# Show the last 15 rows of data
st.subheader("Last 15 Rows of Data")
st.write(data)

# Calculate Exponential Moving Average (EMA)
data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()

# Candlestick Chart with EMA
st.subheader("Candlestick Chart with EMA")
fig = go.Figure(data=[go.Candlestick(x=data.index,
                                    open=data['Open'],
                                    high=data['High'],
                                    low=data['Low'],
                                    close=data['Close'],
                                    name="Candlestick"),
                     go.Scatter(x=data.index, y=data['EMA'], mode='lines', name='EMA', line=dict(color='orange'))])

fig.update_layout(title=f"{symbol} Candlestick Chart with EMA",
                  xaxis_title="Date",
                  yaxis_title="Price (USD)")
st.plotly_chart(fig)

# Volume Chart
st.subheader("Trading Volume Chart")
volume_chart = go.Figure(data=[go.Bar(x=data.index, y=data1['Volume'], name="Volume")])
volume_chart.update_layout(title=f"{symbol} Trading Volume Over Time", xaxis_title="Date", yaxis_title="Volume")
st.plotly_chart(volume_chart)

# Display historical stock price plot
st.subheader("Historical Stock Price Plot")
fig, ax = plt.subplots()
ax.plot(data.index, data['Close'], label='Close Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.set_title(f"{symbol} Stock Price Over Time")
ax.legend()
st.pyplot(fig)

# preparing the data for modeling
test = data.tail(ema_period)
test = test.reset_index(drop=False)
test1 = test.copy()
test = test.drop('Close', axis=1)
train = data[:-ema_period]
train = train.reset_index(drop=False)

# initialize setup
Close = setup(data = train, target = 'Close',train_size = 0.99,
              numeric_features = ['Date','Open','High','Low'], session_id = 123,n_jobs=1)
Close = compare_models(sort = 'MAE')
st.subheader("Trained Models")
results = pull()
st.write(results)

# # creating a model
huber = automl(optimize='MAE')

st.write(huber)

close_pred = predict_model(Close, data=test)

# Predict prices for all the days for which EMA is calculated and display the table
st.subheader("Predicted Stock Prices for Future Days (Best Model)")
prediction_table = pd.DataFrame(columns=["Date", "Predicted Price"])

st.write(close_pred)

# Price Prediction Line Chart
st.subheader("Price Prediction Line Chart")
price_prediction_chart = go.Figure()
price_prediction_chart.add_trace(go.Scatter(x=data.index[-ema_period:], y=test1['Close'], mode='lines', name='Actual Close Price'))
price_prediction_chart.add_trace(go.Scatter(x=data.index[-ema_period:], y=close_pred['prediction_label'], mode='lines', name='Predicted Close Price'))
price_prediction_chart.update_layout(title=f"Actual vs. Predicted {symbol} Stock Price", xaxis_title="Date", yaxis_title="Price (USD)")
st.plotly_chart(price_prediction_chart)
