import pandas as pd

SECRET_KEY = "GIU6EXCSZPBSVJEB"

TICKER = "NDAQ"

INTERVAL = "1min"

SLICE = "year1month10"

ADJUSTED = "true"




datapath = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={}&interval={}&slice={}&adjusted={}&apikey={}".format(TICKER,INTERVAL,SLICE,ADJUSTED,SECRET_KEY)


DATA = pd.read_csv(datapath)
filepath = "STOCK_DATA/{}_STOCK_DATA_{}{}_INTERVALS_{}.csv".format(
    TICKER,
    "ADJUSTED_" if ADJUSTED else "",
    INTERVAL,SLICE[-1])
DATA.to_csv(filepath)