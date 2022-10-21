This is the readme.txt file in the example code for HW of "Trading using technical indicators".

myStrategy.py:
	The only script you need to submit, which returns the action of "buy" or "sell".
	The parameters of this function are optimized by "bestParamByExhaustiveSearch.py".

bestParamByExhaustiveSearch.py:
	This script obtains the best parameters by exhaustive search.
	You can then insert the best parameters into myStrategy.py for evaluation.
	To run it:
		python bestParamByExhaustiveSearch.py 0050.TW-short.csv

rrEstimate.py:
	This script calls myStrategy.py to obtain RR (return rate) for a given price vectors.
	Our judge will use a similiar script to evaluate your submission.
	To run it:
		python rrEstimate.py 0050.TW-short.csv

*.csv
	0050.TW-short.csv: Price of 0050 over the past 5 years