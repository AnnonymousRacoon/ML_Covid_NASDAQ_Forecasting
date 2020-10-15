# Using Quantifying Global COVID Policy to Better Forecast NASDAQ Stock Price

 **IN DEVELOPMENT**

This project uses machine learning to better predict NASDAQ stocks. The model is fed intraday data on historic Open,Close,High and Low, as well as COVID related data. The COVID data includes US infection and and death rate data, aswell as quantified Global COVID policy data.

This project investigates a variety of model architectures. Most do well at identifying local peaks and troughs, but generally struggle on local gradient. At present, the best performer is an auto-regressive LSTM with residual learning. It is also the only model that doesn't suffer from over-fitting. The Baseline (Flatline from the end of the input array) still gives the best validation lost which is concerning.
 
 <img src = "https://github.com/AnnonymousRacoon/-Users-Ali-Documents-ML_Covid_NASDAQ_Forecasting/blob/main/DENSE.png"><br>
 
#### Dense Peformance


### Validation Loss

BASELINE    : 0.0419
DENSE       : 0.1125
DENSE_16    : 0.1765
DENSE_16_8_16: 0.2931
FEEDBACK LSTM: 0.0708

