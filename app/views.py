from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

import pandas as pd
import numpy as np
import json

import yfinance as yf
import datetime as dt

from .forecast_model import forecast_data




# The Home page when Server loads up
def index(request):
    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
    data = yf.download(
        
        # passes the ticker
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],
        
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Adj Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['NVDA']['Adj Close'], name="NVDA")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JPM']['Adj Close'], name="JPM")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')


    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df4 = yf.download(tickers = 'UBER', period='1d', interval='1d')
    df5 = yf.download(tickers = 'TSLA', period='1d', interval='1d')

    df1.insert(0, "Ticker", "AAPL")
    df2.insert(0, "Ticker", "AMZN")
    df3.insert(0, "Ticker", "GOOGL")
    df4.insert(0, "Ticker", "UBER")
    df5.insert(0, "Ticker", "TSLA")

    df = pd.concat([df1, df2, df3, df4, df5], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    # ========================================== Page Render section =====================================================

    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks
    })

def search(request):
    return render(request, 'search.html', {})

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })


# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})

    print(df)
    if df.size == 0:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days < 0:
        return render(request, 'Invalid_Days_Format.html', {})
    
    if number_of_days > 365:
        return render(request, 'Invalid_Days_Format.html', {})


    try:
        df_ml_all = yf.download(tickers = ticker_value, period='3mo', interval='1h')
    except:
        ticker_value = 'AAPL'
        df_ml_all = yf.download(tickers = ticker_value, period='3mo', interval='1m')

    
    
    df_ml = df_ml_all[['Adj Close']] 
    forecast_out = int(number_of_days)
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)    
    
    import pandas_ta as ta

    forecast_close = np.append(np.array(df_ml_all['Adj Close']), forecast_data(df_ml_all['Adj Close'], forecast_out))
    forecast_high = np.append(np.array(df_ml_all['High']), forecast_data(df_ml_all['High'], forecast_out))
    forecast_low = np.append(np.array(df_ml_all['Low']), forecast_data(df_ml_all['Low'], forecast_out))
    forecast_open = np.append(np.array(df_ml_all['Open']), forecast_data(df_ml_all['Open'], forecast_out))

    OHLC_dict = {"Open":[], "High":[], "Low":[], "Close":[]}
    for i in range(len(forecast_open)):
        OHLC_dict["Open"].append(forecast_open[i])
        OHLC_dict["High"].append(forecast_high[i])
        OHLC_dict["Low"].append(forecast_low[i])
        OHLC_dict["Close"].append(forecast_close[i])
    OHLC_df = pd.DataFrame(OHLC_dict)
    sti = ta.supertrend(OHLC_df['High'], OHLC_df['Low'], OHLC_df['Close'], length=7, multiplier=3)

    forecast_supertrend = np.array(sti.iloc[:, 0]).reshape(-1, 1)
    # ========================================== Plotting predicted data ======================================

    sold = True
    pred_dict = {"Date": [], "Prediction": [], "Supertrend": [], "Buy": [], "Sell": []}
    actual_dict = {"Date": [], "Prediction": [], "Supertrend": [], "Buy": [], "Sell": []}
    for i in range(0, forecast_out):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast_close[-forecast_out + i])
        pred_dict["Supertrend"].append(forecast_supertrend[-forecast_out + i][0] if forecast_supertrend[-forecast_out + i] != 0 else forecast_close[-forecast_out + i])
        
        if sold and (forecast_close[-forecast_out + i] > (forecast_supertrend[-forecast_out + i])):
            sold = False
            pred_dict["Buy"].append(forecast_close[-forecast_out + i])
        else:
            pred_dict["Buy"].append(np.nan)

        if not sold and (forecast_close[-forecast_out + i] < (forecast_supertrend[-forecast_out + i])):
            sold = True
            pred_dict["Sell"].append(forecast_close[-forecast_out + i])
        else:
            pred_dict["Sell"].append(np.nan)

    for i in range(0, len(forecast_close) - forecast_out):
        actual_dict["Date"].append(dt.datetime.today() - dt.timedelta(days=(len(forecast_close) - i)))
        actual_dict["Prediction"].append(forecast_close[i])
        actual_dict["Supertrend"].append(forecast_supertrend[i][0] if forecast_supertrend[i] != 0 else forecast_close[i])
        
        if sold and (forecast_close[i] > (forecast_supertrend[i])):
            sold = False
            actual_dict["Buy"].append(forecast_close[i])
        else:
            actual_dict["Buy"].append(np.nan)

        if not sold and (forecast_close[i] < (forecast_supertrend[i])):
            sold = True
            actual_dict["Sell"].append(forecast_close[i])
        else:
            actual_dict["Sell"].append(np.nan)
    
    pred_df = pd.DataFrame(pred_dict)
    actual_df = pd.DataFrame(actual_dict)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual_df['Date'], y=actual_df['Prediction'], name="Forecast", marker=dict(color="blue", size=5) ))

    fig.add_trace(go.Scatter(x=actual_df["Date"], y=actual_df["Buy"], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
    
    fig.add_trace(go.Scatter(x=actual_df["Date"], y=actual_df["Sell"], mode='markers', marker=dict(color='green', size=10, symbol='triangle-down'), name='Sell'))

    fig.add_trace(go.Scatter(x=actual_df["Date"], y=actual_df["Supertrend"], name='Supertrend', marker=dict(color="red", size=5)))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')

    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'], name="Forecast")])

    pred_fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Buy"], mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'), name='Buy'))
    
    pred_fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Sell"], mode='markers', marker=dict(color='green', size=10, symbol='triangle-down'), name='Sell'))

    pred_fig.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Supertrend"], name='Supertrend', marker=dict(color="red")))

    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')


    cfig = go.Figure()
    cfig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    cfig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    cfig.update_xaxes(
    rangeslider_visible=True
    )
    cfig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_candle = plot(cfig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('app/Data/Tickers.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    found = False
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            found = True
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    if not found:
        return render(request, 'Invalid_Ticker.html', {})

    # ========================================== Page Render section ==========================================
    
    confidence = 0.0
    return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast_close,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'plot_div_candle':plot_div_candle,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'Last_Sale':Last_Sale,
                                                    'Net_Change':Net_Change,
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    })
