def myStrategy(pastPriceVec, currentPrice):
	# Explanation of my approach:
	# 1. Technical indicator used: MA
	# 2. if price-ma>alpha ==> buy
	#    if price-ma<-beta ==> sell
	# 3. Modifiable parameters: alpha, beta, and window size for MA
	# 4. Use exhaustive search to obtain these parameter values (as shown in bestParamByExhaustiveSearch.py)
	
	import numpy as np
	action = 0
	# Set best parameters
	windowSize=14
	alpha=0
	beta=-5
	dataLen=len(pastPriceVec)		# Length of the data vector
	if dataLen==0: 
		return action
	# Compute MA
	if dataLen<windowSize:
		ma=np.mean(pastPriceVec)	# If given price vector is small than windowSize, compute MA by taking the average
	else:
		windowedData=pastPriceVec[-windowSize:]		# Compute the normal MA using windowSize 
		ma=np.mean(windowedData)
	# Determine action
	action=0		# action=1(buy), -1(sell), 0(hold), with 0 as the default action
	if (currentPrice-ma)>alpha:		# If price-ma > alpha ==> buy
		action=1
	elif (currentPrice-ma)<-beta:	# If price-ma < -beta ==> sell
		action=-1

	return action
