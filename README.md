# Apple LSTM Stock Predictor

<a target="_blank" href="https://colab.research.google.com/github/Aaronlozhkin/Apple-LSTM-Stock-Predictor/blob/main/predictingAPPL.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
 
Takes intraday stock data of APPL from 2010-2019. Resamples to 10-minute intervals using pandas and adds columns for MACD and RSI technical indicators based on 'Open' market prices. Trains a Long Short-Term Memory neural network on 'Open' price market data from 2010-2017 and then predicts 'Closed' price data from 2018-2019. Data is min-max scaled using Sci-Kit learn and visualized using MatPlotLib.

Includes features for the amount of data used to predict along with the **amount of data desired to be forecasted**. The example below shows 100 intervals of previous data used to predict 50 intervals of future data. These parameters can be adjusted to train for longer and longer forecasting.

![image](https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor/assets/23532191/d1d6c1a7-46ac-4322-be58-77347fdf2b20)
![image](https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor/assets/23532191/6de6c23d-28d7-45ad-9f09-09fa0f3e15bb)



## Library Usage
- [**Pandas**](https://pandas.pydata.org/) 
   - Data Acquisition and Manipulation
   - Resampling to Consistent Time Scale with Mean and Backfill
   - Removing Holidays and Closed Market Dates
- [**Scikit-learn**](https://scikit-learn.org/stable/)
   - Min-Max Scaling for Quicker Convergence
   - Train-Test Split for Tensor Dataset
- [**PyTorch**](https://pytorch.org/)
   - Custom Stock Dataset Class Based on Prior and Forecast Sequence Length
   - DataLoader Class for (input, output) Pairing
   - Long-Short-Term-Memory Neural Network Implementation
   - Training Loop Utilizing Mean Squared Error Loss
