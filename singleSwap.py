'''
An Implementation of the single swap improvement of the GreedyKCenters Algorithm for clustering
usage: python singleSwap.py [input_csv_filename.csv]
'''

import csv,pylab
import numpy as np
import pylab
import cluster
from numpy.linalg import norm
import sys


# Preparing the Data for Analysis
datafile = sys.argv[1]                      # The input mxn matrix given in the form of a CSV file
X = cluster.readCsv(datafile)               # The input matrix X
Q = np.ones([1,X.shape[1]])
nearestPoint = cluster.selectKRandom(X,1)
Q = np.append(Q, nearestPoint, axis=0)
Q = np.delete(Q,0,0)
objVal = []

if __name__ == '__main__':
	Q, D, x = cluster.greedyKcenter(X, Q, 4)
	print("greedyKcenters completed, Objective Value = ", D)
	print("Single Swap Algorithm beginning execution now")
	stop = False
	i = 0
	while not stop:
		(xBest, QBest, errorFlag, cost) = cluster.swap(x, Q)
		x = xBest
		q = QBest
		if (errorFlag == True):
			stop = True
		objVal.append(cost)
		if (i > 0):
			if (objVal[i] <= 0.95 * objVal[i-1]):
				stop = True
		print('Swap: ',i,' Cost = ',cost)
		i += 1

	print('Single Swap Algorithm Completed')
	C = cluster.findNearestCentroids(x, Q)
	cluster.plotSingleSwap(x, Q, C)

	