import sys
import numpy as np
import pandas as pd
import talib

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_ma(pastPriceVec, currentPrice, windowSize, alpha, beta):
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
    if (currentPrice - ma) > alpha:  # If price-ma > alpha ==> buy
        action = 1
    elif (currentPrice - ma) < -beta:  # If price-ma < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_ema(pastPriceVec, currentPrice, windowSize, alpha, beta):
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
    if (currentPrice - ema) > alpha:  # If price-ema > alpha ==> buy
        action = 1
    elif (currentPrice - ema) < -beta:  # If price-ema < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_sma(pastPriceVec, currentPrice, windowSize, alpha, beta):
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
    if (currentPrice - sma) > alpha:  # If price-sma > alpha ==> buy
        action = 1
    elif (currentPrice - sma) < -beta:  # If price-sma < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_trima(pastPriceVec, currentPrice, windowSize, alpha, beta):
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
    if (currentPrice - trima) > alpha:  # If price-trima > alpha ==> buy
        action = 1
    elif (currentPrice - trima) < -beta:  # If price-trima < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_wma(pastPriceVec, currentPrice, windowSize, alpha, beta):
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
    if (currentPrice - wma) > alpha:  # If price-wma > alpha ==> buy
        action = 1
    elif (currentPrice - wma) < -beta:  # If price-wma < -beta ==> sell
        action = -1
    return action

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_rsi(pastPriceVec, currentPrice, windowSize=14, alpha=80, beta=20):
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

    elif rsi < beta:  # If rsi < -beta ==> sell
        action = -1

    return action


# Decision of the current day by the current price, with 2 modifiable parameters
def myStrategy_linear_regression(pastPriceVec, currentPrice, windowSize=14, threshold=0.02):
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


# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_TSF(pastPriceVec, currentPrice, windowSize=14, threshold=0.02):
    action = 0  # action=1(buy), -1(sell), 0(hold), with 0 as the default action
    dataLen = len(pastPriceVec)  # Length of the data vector

    if dataLen < 2 or windowSize < 2:
        return action

    # Compute linear regression
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

# Decision of the current day by the current price, with 3 modifiable parameters
def myStrategy_MACD(pastPriceVec, currentPrice, fastperiod=12, slowperiod=26, signalperiod=9, threshold=0.02):
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

# Compute return rate over a given price vector, with 3 modifiable parameters
def computeReturnRate(priceVec, windowSize, alpha, beta):
    capital = 1000  # Initial available capital
    capitalOrig = capital  # original capital
    dataCount = len(priceVec)  # day size
    suggestedAction = np.zeros((dataCount, 1))  # Vec of suggested actions
    stockHolding = np.zeros((dataCount, 1))  # Vec of stock holdings
    total = np.zeros((dataCount, 1))  # Vec of total asset
    realAction = np.zeros(
        (dataCount, 1)
    )  # Real action, which might be different from suggested action. For instance, when the suggested action is 1 (buy) but you don't have any capital, then the real action is 0 (hold, or do nothing).
    # Run through each day
    for ic in range(dataCount):
        currentPrice = priceVec[ic]  # current price
        suggestedAction[ic] = myStrategy_MACD(
            priceVec[0:ic], currentPrice, windowSize, threshold=alpha*0.01
        )  # Obtain the suggested action
        # get real action by suggested action
        if ic > 0:
            stockHolding[ic] = stockHolding[ic - 1]  # The stock holding from the previous day
        if suggestedAction[ic] == 1:  # Suggested action is "buy"
            if stockHolding[ic] == 0:  # "buy" only if you don't have stock holding
                stockHolding[ic] = capital / currentPrice  # Buy stock using cash
                capital = 0  # Cash
                realAction[ic] = 1
        elif suggestedAction[ic] == -1:  # Suggested action is "sell"
            if stockHolding[ic] > 0:  # "sell" only if you have stock holding
                capital = stockHolding[ic] * currentPrice  # Sell stock to have cash
                stockHolding[ic] = 0  # Stocking holding
                realAction[ic] = -1
        elif suggestedAction[ic] == 0:  # No action
            realAction[ic] = 0
        else:
            assert False
        total[ic] = capital + stockHolding[ic] * currentPrice  # Total asset, including stock holding and cash
    returnRate = (total[-1] - capitalOrig) / capitalOrig  # Return rate of this run
    return returnRate


if __name__ == "__main__":
    returnRateBest = -1.00  # Initial best return rate
    df = pd.read_csv(sys.argv[1])  # read stock file
    adjClose = df["Adj Close"].values  # get adj close as the price vector
    windowSizeMin = 11
    windowSizeMax = 20
    # Range of windowSize to explore
    alphaMin = -5
    alphaMax = 5
    # Range of alpha to explore
    betaMin = -5
    betaMax = 5  # Range of beta to explore
    # Start exhaustive search
    for windowSize in range(windowSizeMin, windowSizeMax + 1):  # For-loop for windowSize
        # print("windowSize=%d" % (windowSize))
        for alpha in range(alphaMin, alphaMax + 1):  # For-loop for alpha
            # print("\talpha=%d" % (alpha))
            for beta in range(betaMin, betaMax + 1):  # For-loop for beta
                # print("\t\tbeta=%d" % (beta), end="")  # No newline
                returnRate = computeReturnRate(
                    adjClose, windowSize, alpha, beta
                )  # Start the whole run with the given parameters
                # print(" ==> returnRate=%f " % (returnRate))
                if returnRate > returnRateBest:  # Keep the best parameters
                    windowSizeBest = windowSize
                    alphaBest = alpha
                    betaBest = beta
                    returnRateBest = returnRate
    print(
        "Best settings: windowSize=%d, alpha=%d, beta=%d ==> returnRate=%f"
        % (windowSizeBest, alphaBest, betaBest, returnRateBest)
    )  # Print the best result
