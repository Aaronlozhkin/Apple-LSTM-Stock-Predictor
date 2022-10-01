# Apple LSTM Stock Predictor
 
### Project Description

Takes intraday stock data of APPL from 2010-2019. Resamples to 10-minute intervals using pandas and adds columns for MACD and RSI technical indicators based on 'Open' market prices. Trains a Long Short-Term Memory neural network on 'Open' price market data from 2010-mid2018 and then predicts 'Closed' price data from mid2018-2019. Data is min-max scaled using Sci-Kit learn and visualized using MatPlotLib. Last Prediction Accurary: 95.6%.

Note: Algorithm does not forcast data into the future, but uses batches of 64 representing the previous 10 minutes to predict the closing price of the next 10 minutes.
