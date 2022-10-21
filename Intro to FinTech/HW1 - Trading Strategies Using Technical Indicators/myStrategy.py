import numpy as np
import talib

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_ma(pastPriceVec, currentPrice, windowSize=14, alpha=0, beta=-5):
	# Explanation of my approach:
	# 1. Technical indicator used: MA
	# 2. if price-ma>alpha ==> buy
	#    if price-ma<-beta ==> sell
	# 3. Modifiable parameters: alpha, beta, and window size for MA
	# 4. Use exhaustive search to obtain these parameter values (as shown in bestParamByExhaustiveSearch.py)

    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action
    # Compute ma
    if dataLen < windowSize:
        ma = np.mean(pastPriceVec)  # If given price vector is small than windowSize, compute MA by taking the average
    else:
        windowedData = pastPriceVec[-windowSize:]  # Compute the normal MA using windowSize
        ma = np.mean(windowedData)
    # Determine action
    if (currentPrice - ma) > alpha:  # If price - ma > alpha ==> buy
        action = 1
    elif (currentPrice - ma) < -beta:  # If price - ma < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_ema(pastPriceVec, currentPrice, windowSize=14, alpha=0, beta=-5):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action
    # Compute ema
    elif dataLen == 1 or windowSize < 2:
        ema = pastPriceVec[0]
    else:
        if dataLen < windowSize:
            ema = talib.EMA(pastPriceVec, timeperiod=dataLen)[-1]  # If given price vector is small than windowSize, compute EMA by taking the average
        else:
            ema = talib.EMA(pastPriceVec[-windowSize:], timeperiod=windowSize)[-1]  # Compute the normal EMA using windowSize

    if np.isnan(ema):
        return action
    
    # Determine action
    if (currentPrice - ema) > alpha:  # If price - ema > alpha ==> buy
        action = 1
    elif (currentPrice - ema) < -beta:  # If price - ema < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_sma(pastPriceVec, currentPrice, windowSize=15, alpha=0, beta=-5):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action
    # Compute ema
    elif dataLen == 1 or windowSize < 2:
        sma = pastPriceVec[0]
    else:
        if dataLen < windowSize:
            sma = talib.SMA(pastPriceVec, timeperiod=dataLen)[-1]  # If given price vector is small than windowSize, compute SMA by taking the average
        else:
            sma = talib.SMA(pastPriceVec[-windowSize:], timeperiod=windowSize)[-1]  # Compute the normal SMA using windowSize

    if np.isnan(sma):
        return action
    
    # Determine action
    if (currentPrice - sma) > alpha:  # If price - sma > alpha ==> buy
        action = 1
    elif (currentPrice - sma) < -beta:  # If price - sma < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_trima(pastPriceVec, currentPrice, windowSize=15, alpha=0, beta=-5):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action
    # Compute ema
    elif dataLen == 1 or windowSize < 2:
        trima = pastPriceVec[0]
    else:
        if dataLen < windowSize:
            trima = talib.TRIMA(pastPriceVec, timeperiod=dataLen)[-1]  # If given price vector is small than windowSize, compute TRIMA by taking the average
        else:
            trima = talib.TRIMA(pastPriceVec[-windowSize:], timeperiod=windowSize)[-1]  # Compute the normal TRIMA using windowSize

    if np.isnan(trima):
        return action
    
    # Determine action
    if (currentPrice - trima) > alpha:  # If price - trima > alpha ==> buy
        action = 1
    elif (currentPrice - trima) < -beta:  # If price - trima < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_wma(pastPriceVec, currentPrice, windowSize=15, alpha=0, beta=-5):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen == 0:
        return action
    # Compute ema
    elif dataLen == 1 or windowSize < 2:
        wma = pastPriceVec[0]
    else:
        if dataLen < windowSize:
            wma = talib.WMA(pastPriceVec, timeperiod=dataLen)[-1]  # If given price vector is small than windowSize, compute WMA by taking the average
        else:
            wma = talib.WMA(pastPriceVec[-windowSize:], timeperiod=windowSize)[-1]  # Compute the normal WMA using windowSize

    if np.isnan(wma):
        return action
    
    # Determine action
    if (currentPrice - wma) > alpha:  # If price - wma > alpha ==> buy
        action = 1
    elif (currentPrice - wma) < -beta:  # If price - wma < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_ema_diff(pastPriceVec, currentPrice, fastperiod=20, slowperiod=60, threshlod=3):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    pastPriceVec = np.append(pastPriceVec, currentPrice)
    dataLen = len(pastPriceVec)  # Length of the data vector
    if dataLen < slowperiod:
        return action

    # Compute the normal EMA using windowSize
    fast_ema = talib.EMA(pastPriceVec[-fastperiod:], timeperiod=fastperiod)[-1] 
    slow_ema = talib.EMA(pastPriceVec[-slowperiod:], timeperiod=slowperiod)[-1]  

    if np.isnan(slow_ema):
        return action
    
    # Determine action
    if (fast_ema - slow_ema) > threshlod:  # If fast_ema - slow_ema > threshlod ==> buy
        action = 1
    elif (slow_ema - fast_ema) > threshlod:  # If slow_ema - fast_ema < threshlod ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_rsi(pastPriceVec, currentPrice, windowSize=21, alpha=70, beta=30):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    pastPriceVec = np.append(pastPriceVec, currentPrice)
    dataLen = len(pastPriceVec)  # Length of the data vector

    if dataLen == 0 or windowSize < 2:
        return action
    while dataLen < 3:
        pastPriceVec = np.append(pastPriceVec, pastPriceVec[-1])
        dataLen = len(pastPriceVec)  # Length of the data vector
    # Compute rsi using windowSize
    if dataLen >= windowSize + 1:
        rsi = talib.RSI(pastPriceVec[-(windowSize + 1):], timeperiod=windowSize)[-1]
    else:
        rsi = talib.RSI(pastPriceVec, timeperiod=dataLen-1)[-1]

    if np.isnan(rsi):
        return action

    # Determine action
    if rsi > alpha:  # If rsi > alpha ==> buy
        action = 1

    elif rsi < beta:  # If rsi < beta ==> sell
        action = -1

    return action

# Decision of the current day by the current price, with 6 modifiable parameters
def myStrategy_StochRSI(pastPriceVec, currentPrice, windowSize=14, fastk_period=5, fastd_period=3, alpha=70, beta=30, threshold=2):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    pastPriceVec = np.append(pastPriceVec, currentPrice)
    dataLen = len(pastPriceVec)  # Length of the data vector

    if dataLen == 0 or windowSize < 2:
        return action

    # Compute rsi using windowSize
    fastk, fastd = talib.STOCHRSI(pastPriceVec, timeperiod=windowSize, fastk_period=fastk_period, fastd_period=fastd_period, fastd_matype=0)
    fastk = fastk[-1]
    fastd = fastd[-1]
    if np.isnan(fastk) or np.isnan(fastd):
        return action

    # Determine action
    if (fastk - fastd) > threshold:		# K > D ==> buy
        action += 1
    elif (fastd - fastk) > threshold:	# K < D ==> sell
        action += -1
    if fastd > alpha:  	# D > alpha ==> buy
        action += 1
    elif fastd < beta:  # D < beta ==> buy
        action -= -1
    
    if action >= 1:
        action = 1
    elif action <= -1:
        action = -1
    else:
        action = 0

    return action

# Decision of the current day by the current price, with 2 modifiable parameters
def myStrategy_linear_regression(pastPriceVec, currentPrice, windowSize=21, threshold=0.02):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector

    if dataLen < 2 or windowSize < 2:
        return action

    # Compute linear regression
    if dataLen < windowSize:
        prediction = talib.LINEARREG(pastPriceVec, timeperiod=dataLen)[-1]  
    else:
        prediction = talib.LINEARREG(pastPriceVec[-windowSize:], timeperiod=windowSize)[-1] # Compute the linear regression using windowSize

    if np.isnan(prediction):
        
        return action

    # Determine action
    if (prediction - currentPrice) / prediction > threshold:  # If stock is too cheap ==> buy
        action = 1

    elif (currentPrice - prediction) / prediction > threshold:  # If stock is too cheap ==> sell
        action = -1

    return action


# Decision of the current day by the current price, with 2 modifiable parameters
def myStrategy_TSF(pastPriceVec, currentPrice, windowSize=21, threshold=0.02):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector

    if dataLen < 2 or windowSize < 2:
        return action

    # Compute Time Series Forecast
    if dataLen < windowSize:
        prediction = talib.LINEARREG(pastPriceVec, timeperiod=dataLen)[-1]  
    else:
        prediction = talib.LINEARREG(pastPriceVec[-windowSize:], timeperiod=windowSize)[-1] # Compute the Time Series Forecast using windowSize

    if np.isnan(prediction):
        return action

    # Determine action
    if (prediction - currentPrice) / prediction > threshold:  # If stock is too cheap ==> buy
        action = 1

    elif (currentPrice - prediction) / prediction > threshold:  # If stock is too cheap ==> sell
        action = -1

    return action

# Decision of the current day by the current price, with 4 modifiable parameters
def myStrategy_MACD(pastPriceVec, currentPrice, fastperiod=12, slowperiod=26, signalperiod=9, threshold=0.03):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    pastPriceVec = np.append(pastPriceVec, currentPrice)
    dataLen = len(pastPriceVec)  # Length of the data vector

    if fastperiod < 2 or slowperiod < 2:
        return action

    # Compute MACD
    macd, macdsignal, macdhist = talib.MACD(pastPriceVec, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
    macdhist = macdhist[-1]
    
    if np.isnan(macdhist):
        return action

    # Determine action
    if macdhist > threshold:  # If stock is too cheap ==> buy
        action = 1

    elif macdhist < -threshold:  # If stock is too cheap ==> sell
        action = -1

    return action

def myStrategy(pastPriceVec, currentPrice):
	
	action = {}
	action["ma"] = myStrategy_ma(pastPriceVec, currentPrice) * 0.1
	action["ema"] = myStrategy_ema(pastPriceVec, currentPrice) * 0.5
	action["sma"] = myStrategy_sma(pastPriceVec, currentPrice) * 0.2
	action["trima"] = myStrategy_trima(pastPriceVec, currentPrice) * 0.4
	action["wma"] = myStrategy_wma(pastPriceVec, currentPrice) * 0.3
	action["ema_diff"] = myStrategy_ema_diff(pastPriceVec, currentPrice)
	action["rsi"] = myStrategy_rsi(pastPriceVec, currentPrice)
	action["stochrsi"] = myStrategy_StochRSI(pastPriceVec, currentPrice)
	action["lr"] = myStrategy_linear_regression(pastPriceVec, currentPrice)
	action["tsf"] = myStrategy_TSF(pastPriceVec, currentPrice)
	action["macd"] = myStrategy_MACD(pastPriceVec, currentPrice)
	action_aggr = sum(action.values())

	action = 0
	threshold = 0
	if action_aggr > threshold:
		action = 1
	elif action_aggr < -threshold:
		action = -1
	return action