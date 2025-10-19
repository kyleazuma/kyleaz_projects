#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

KYLE AZUMA – PROJECT 1

Created on Fri Oct 17 19:41:37 2025

@author: kyleazuma
"""



# import libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
pio.renderers.default = "browser"



# defaults for consistent formatting
plt.rcParams['figure.figsize'] = (12,6)
sns.set_style("darkgrid")



# add company tickers
tickers = ["GOOGL", "AMZN", "AAPL", "META", "NVDA", "TSLA"]



# specify time frame
start_date = "2020-01-01"
end_date = "2025-10-17"



# download daily data
data = yf.download(tickers, start=start_date, end=end_date, progress=False, group_by='ticker', auto_adjust=True)



# clean up formatting of DataFrame
# if yf returns multi-level columns, select 'Close'
if isinstance(data.columns, pd.MultiIndex):
    # create a DataFrame with columns = tickers and rows = dates
    price = pd.DataFrame({t: data[t]["Close"] for t in tickers})
else: 
    price = data["Close"].to_frame() if "Close" in data.columns else data.copy()
    # if single ticker then ensure price is DataFrame; if many tickers, price will already be correct



# sort index and drop rows with all NaNs to remove missing data
price = price.sort_index().dropna(how="all")
price.head()



# basic inspection
print("Date range:", price.index.min(), "to", price.index.max())
print("Columns:", price.columns.tolist())
print("Number of trading days:", len(price))

price.describe().T #summary stats for each ticker – describe() gives mean/std/min/max



"""
DAILY RETURNS & ANNUAL RETURNS
Daily return measures the percentage change in price from one day to the next.
Annual return converts total return into a yearly rate.
"""
returns = price.pct_change().dropna() #fractional returns
returns.head()

# average daily return and daily volatility
daily_mean = returns.mean()
daily_vol = returns.std()

#annualize (approximate trading days = 252)
annual_return = (1 + daily_mean) ** 252 - 1
annual_vol = daily_vol * np.sqrt(252)

pd.DataFrame({
    "Daily mean": daily_mean,
    "Daily vol": daily_vol,
    "Annual return": annual_return,
    "Annual vol": annual_vol
}).round(4)



"""
CUMULATIVE RETURNS
Stock performance over the entire given period.
"""
cum_returns = (1 + returns).cumprod()-1 # percent since start
cum_index = (1 + returns).cumprod() * 100 # index starting at 100

# plot cumulative index (matplotlib)
ax = cum_index.plot(title="Cumultative index (start=100)")
ax.set_ylabel("Index (start = 100)")
plt.show()

# plot culumartive index (plotly)
fig = px.line(
    cum_index,
    title="Cumulative Index (start = 100)",
    labels={"value": "Index (start=100)", "index": "Date"}
)
fig.update_layout(legend_title_text="Ticker")
fig.show()


"""
MOVING AVERAGES
Smooths short-term fluctuations, showing the trend.
"""
ticker = "AAPL"
price[ticker].plot(title=f"{ticker} Price")
plt.ylabel("Price (USD)")
plt.show()

ma_short = price[ticker].rolling(window=20).mean() # 20-day MA
ma_long = price[ticker].rolling(window=50).mean() # 50-day MA

#matplotlib version
plt.plot(price[ticker], label="Price")
plt.plot(ma_short, label="MA20")
plt.plot(ma_long, label="MA50")
plt.title(f"{ticker} Price with Moving Averages")
plt.legend()
plt.show()

# plotly version
df_ma = pd.DataFrame({"Price": price[ticker], "MA20": ma_short, "MA50": ma_long})
fig = px.line(
    df_ma,
    title=f"{ticker} Price with Moving Averages",
    labels={"value": "Price (USD)", "index": "Date"}
)
fig.update_layout(legend_title_text="Metric")
fig.show()



"""
ROLLING VOLATILITY
Shows extent of fluctuation via standard deviation of returns.
"""
# matplotlib version
rolling_vol_30 = returns[tickers].rolling(window=30).std() * np.sqrt(252)
rolling_vol_30.plot(title="30-day rolling annualized volatility")
plt.ylabel("Volatility (annualized)")
plt.show()

# plotly version
fig = px.line(
    rolling_vol_30,
    title="30-Day Rolling Annualized Volatility",
    labels={"value": "Volatility (annualized)", "index": "Date"}
)
fig.update_layout(legend_title_text="Ticker")
fig.show()


"""
DRAWDOWNS
Measures the percentage drop from peak to trough.
"""
wealth = (1 + returns).cumprod() # wealth index for each ticker (starting value = 1)

# running_max = wealth.cummax()
running_max = wealth.cummax()
drawdown = (wealth - running_max) / running_max # negative or zero

# example plot for NVDA (matplotlib)
drawdown["NVDA"].plot(title="NVDA Drawdown")
plt.ylabel("Drawdown (fraction)")
plt.show()

# example plot for NVDA (plotly)
fig = px.area(
    drawdown,
    y="NVDA",
    title="NVDA Drawdown",
    labels={"index": "Date", "NVDA": "Drawdown (fraction)"}
)
fig.update_traces(fillcolor="red", line_color="darkred")
fig.show()



"""
CORRELATION MATRIX & HEATMAP
Shows relationship between stocks and how they co-move.
"""
corr = returns.corr()
print(corr.round(2))

# heatmap (seaborn)
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0)
plt.title("Correlation matrix (daily returns)")
plt.show()
# correlations near 1 = move together; negative = move opposite

# heatmap (plotly)
fig = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.index,
    colorscale="RdBu",
    zmid=0,
    text=corr.round(2),
    texttemplate="%{text}"
))
fig.update_layout(
    title="Correlation Matrix (Daily Returns)",
    xaxis_nticks=36
)
fig.show()



"""
EQUAL-WEIGHT PORTFOLIO BACKTEST
Simulates how portfolio would have performed historically.
"""
# equal weights
n = len(tickers)
weights = np.array([1/n] * n)

#portfolio daily returns
port_returns = returns.dot(weights)

# portfolio cumulative index starting at 100
port_cum = (1 + port_returns).cumprod() * 100

# portfolio metrics
port_annual_return = (1 + port_returns.mean())**252 - 1
port_annual_vol = port_returns.std() * np.sqrt(252)
sharpe_ratio = (port_annual_return - 0.02) / port_annual_vol #example risk-free 2%

print("Portfolio annual return:", round(port_annual_return, 4))
print("Portfolio annual vol:", round(port_annual_vol, 4))
print("Portfolio Sharpe (rf=2%):", round(sharpe_ratio, 4))

#plot portfolio vs each ticker cumulative index (matplotlib)
plt.plot(port_cum, label="Equal-weight portfolio")
for t in tickers:
    plt.plot((1 + returns[t]).cumprod() * 100, alpha=0.4)
plt.title("Portfolio vs individual stocks (index start = 100")
plt.legend()
plt.show()

#plot portfolio vs each ticker cumulative index (plotly)
combined = pd.DataFrame({"Portfolio": port_cum})
for t in tickers:
    combined[t] = (1 + returns[t]).cumprod() * 100

fig = px.line(
    combined,
    title="Portfolio vs Individual Stocks (Index Start = 100)",
    labels={"value": "Index (start=100)", "index": "Date"}
)
fig.update_layout(legend_title_text="Series")
fig.show()
