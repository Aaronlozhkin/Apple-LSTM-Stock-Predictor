# Apple LSTM Stock Predictor
 
### Project Description

Takes intraday stock data of APPL from 2010-2019. Resamples to 10-minute intervals using pandas and adds columns for MACD and RSI technical indicators based on 'Open' market prices. Trains a Long Short-Term Memory neural network on 'Open' price market data from 2010-mid2018 and then predicts 'Closed' price data from mid2018-2019. Data is min-max scaled using Sci-Kit learn and visualized using MatPlotLib. Last Prediction Accurary: 95.6%.

Developer Update: The model seems to be heavily relying on the previous time step to predict the future. With some baseline tests, a recursive use of this model which utilizes the outputs of the model as the inputs in the next iteration seem to all be very similar (the model predicts a straight line). Further development of the model will need to be done to train the algorithm to predict multiple steps into the future.
