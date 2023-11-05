# Apple LSTM Stock Predictor

Apple LSTM Stock Predictor is a deep learning financial model that predicts the future price of a stock. It is a sequence-to-sequence price foreacasting system which utilizes stock price and technical indicators to make predictions into the future. This project was trained on Apple inc. stock data (APPL), but can be adapted to fit any other stock with enough intraday prices.

## Installation

For a local download, utilize a [conda](conda.io) environment with [Torch](https://pytorch.org/) and install using pip

```
pip install git+https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor
pip install pandas_ta
pip install sklearn
```

Alternatively, open the notebook in Colab

<a target="_blank" href="https://colab.research.google.com/github/Aaronlozhkin/Apple-LSTM-Stock-Predictor/blob/main/predictingAPPL.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Data Processing

Intraday sotck data of Apple Inc. from January 1, 2010 to December 31, 2019 is imported into a [Pandas](https://pandas.pydata.org/)  dataframe.
Market data is extracted for when the market was open by omitting late hours, weekends, and all national holidays from 2010 to 2019.

### Technical Indicator Derivation

This project opted to use the Moving Average Convergence/Divergence ([MACD](https://www.investopedia.com/terms/m/macd.asp)) and Relative Strength Index ([RSI](https://www.investopedia.com/terms/r/rsi.asp)) technical indicators.
```
import pandas_ta as pta

ema_12 = df['Open'].ewm(span=12, adjust=False).mean()
ema_26 = df['Open'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26

df['RSI14'] = pta.rsi(df['Open'], length=14)
```

These indicators along with 'Open' prices were utilized as features for the model.

### Custom Stock Dataset

The custom stock dataset class takes in a formatted intraday stock dataset with the relevant times, applies the technical indicator derivation, and resamples the data to be of 10 minute intervals using a mean backfill approach. 

As input it requires a dataframe, the **sequence_length**, and the number of **prediction_steps**. The model will learn to use the amount of data specified by sequence_length to predict the 'Close' price every 10 minutes for as long as specified by prediction_steps.

By default, the model uses 100 time steps (1000 minutes) of previous data to predict 50 time steps (500 minutes) in the future.

```
sequence_length = 100
prediction_steps = 50 
```

### Output

We train a Long Short-Term Memory (LSTM) neural network on 'Open' price market data from 2010-2017 and then predict 'Closed' price data from 2018-2019. Data is min-max scaled using Sci-Kit learn and visualized using MatPlotLib. 
The example below shows the default using 100 intervals of previous data to predict 50 intervals of future data. These parameters can be adjusted to train for longer forecasting or larger inference.

<p align='center'>
  <img src= https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor/assets/23532191/d1d6c1a7-46ac-4322-be58-77347fdf2b20>
  Model results on learning from the 'Open' price of APPL from 2010 to 2017
</p>

<p align='center'>
  <img src= https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor/assets/23532191/6de6c23d-28d7-45ad-9f09-09fa0f3e15bb>
  Model results on forecasting the "Close" price of APPL from 2018-2019
</p>

## Model Architecture
<p align='center'>
  <img src= https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor/assets/23532191/63c73902-61d9-4ac6-9c53-4633d67b639c>
  A sequence of LSTM gates where $X_t$ represents the input data and $h_t$ represents the hidden state at time $t$
</p>

<p align='center'>
  <img src= https://github.com/Aaronlozhkin/Apple-LSTM-Stock-Predictor/assets/23532191/58de1046-72f0-4501-8953-eb1e480ae101>
  The internal structure of a standard LSTM gate
</p>

The model takes a sequence of prices and technical indicators over time, and trains weights to output a specified amount of data assumed to be a future forecast. The hidden states are fed through a linear network which then produce a sequence of numbers equal to the desired forecasting step. Mean squared error loss was used to train and adjust the corresponding weights.

## Library Usage
- [**Pandas**](https://pandas.pydata.org/)
   - [pandas_ta](https://pypi.org/project/pandas-ta/) for technical indicator derivation
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
