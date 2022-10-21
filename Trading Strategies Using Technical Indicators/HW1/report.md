# Report - FinTech HW1
## The TIs you adopt and how do you use them.
* TIs
    * MA - Moving Average
    * SMA - Simple Moving Average
    * WMA - Weighted Moving Average
    * EMA - Exponential Moving Average
    * TRIMA - Triangular Moving Average
    * EMA_DIFF - EMA intersection
    * RSI - Relative Strength Index
    * StochRSI - Stochastic Relative Strength Index
    * Linear Regression
    * Time Series Forecast
    * MACD - Moving Average Convergence/Divergence
* Usage
    * Use the function in package ```talib``` [https://mrjbq7.github.io/ta-lib/doc_index.html]
        * MA 
            * defined in sample code
            * buy: price - ma > alpha
            * sell: price - ma < -beta 
        * SMA 
            * SMA(close, timeperiod)
            * buy: price - sma > alpha
            * sell: price - sma < -beta 
        * WMA 
            * WMA(close, timeperiod)
            * buy: price - wma > alpha
            * sell: price - wma < -beta 
        * EMA 
            * EMA(close, timeperiod)
            * buy: price - ema > alpha
            * sell: price - ema < -beta 
        * TRIMA 
            * TRIMA(close, timeperiod)
            * buy: price - trima > alpha
            * sell: price - trima < -beta 
        * EMA_DIFF 
            * EMA(close, Shorttimeperiod) - EMA(close, Longtimeperiod)
            * buy: fast_ema - slow_ema > threshlod
            * sell: slow_ema - fast_ema > threshlod
        * RSI 
            * RSI(close, timeperiod)
            * buy: rsi > alpha
            * sell: rsi < beta 
        * StochRSI 
            * STOCHRSI(close, timeperiod, fastk_period, fastd_period, fastd_matype)
            * buy: D > alpha or K > D
            * sell: D < beta or K < D
        * Linear Regression
            * LINEARREG(close, timeperiod)
            * buy: (prediction - currentPrice) / prediction > threshold
            * sell: (currentPrice - prediction) / prediction > threshold
        * TSF
            * TSF(close, timeperiod)
            * buy: (prediction - currentPrice) / prediction > threshold
            * sell: (currentPrice - prediction) / prediction > threshold
        * MACD 
            * MACD(close, fastperiod, slowperiod, signalperiod)
            * buy: macdhist > threshold
            * sell: macdhist < threshold 
            
    * each technical indicator will indicate the action should be buy, hold, or sell, and giving each strategy some weights to determine their impact on the final action. Each stratrgy votes on the final action.
    
## What are the modified parameters of your strategy, and how do you fine tune the parameters.
* parameters
    * MA 
        * windowSize = 14
        * alpha = 0
        * beta = -5
    * SMA 
        * windowSize = 15
        * alpha = 0
        * beta = -5
    * WMA 
        * windowSize = 15
        * alpha = 0
        * beta = -5
    * EMA 
        * windowSize = 14
        * alpha = 0
        * beta = -5
    * TRIMA 
        * windowSize = 15
        * alpha = 0
        * beta = -5
    * EMA_DIFF 
        * fastperiod = 20 
        * slowperiod = 60
        * threshlod = 3
    * RSI 
        * windowSize = 21
        * alpha = 70
        * beta = 30
    * StochRSI 
        * windowSize = 14
        * fastk_period = 5
        * fastd_period = 3
        * alpha = 70
        * beta = 30
        * threshold = 2
    * Linear Regression
        * windowSize = 21
        * threshold = 0.02
    * TSF
        * windowSize = 21
        * threshold = 0.02
    * MACD 
        * fastperiod = 12
        * slowperiod = 26
        * signalperiod = 9
        * threshold = 0.03
    * Final action
        * threshold = 0
* fine tune
    * For some technical indicators like MA series and others, I find the better parameters with the approach in bestParamByExhaustiveSearch.py. Given the min and max of the parameters and using them iteratively in the the range to calculate the return rate in history data. Finally, keeping the best result with its parameters.
    * For some technical indicators like RSI, MACD, etc., using the intuition parameters in the market and try their nearby values. Ex: alpha for about 80 and beta for around 20.
    * Getting better parameters in single strategy by backtesting on only one strategy. After collecting all the parameters, combine strategies with some weights. Then try all weight combinations to find the best weights for strategies.
## Any other things you have done to optimize your strategy.
* If the length of data is shorter than the window size or time period, adjust them to fit the dataand enable the function in package ```talib``` to run without bugs.
* Backtesting on other stock price dataset to adjust the parameters. 
* The value of each vote is not the same, and the trend of the action needs to be greater than the threshold or holding the stock.
* different ma different voting values.