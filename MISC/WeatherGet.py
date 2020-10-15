
import pandas as pd

import json 

from pandas import json_normalize


FORMAT = 'json'
APIKEY = "0a5c8268606b412cbf6141240201110"
DATE ="2020-09-30"
ZIP = "10006"

path = "http://api.worldweatheronline.com/premium/v1/past-weather.ashx?q={}&date={}&format={}&key={}".format(ZIP,DATE,FORMAT,APIKEY)


with open("weather.json") as f:
    df = json.load(f)

# with open(path) as file:
#     df = json.load(file)

WeatherDat = json_normalize(df['data'])
print(WeatherDat.head(200))

works_data = json_normalize(data=df['data'], record_path='weather', 
                            meta=['astronomy'],errors = 'ignore',record_prefix='_')
print(works_data.head(3))


