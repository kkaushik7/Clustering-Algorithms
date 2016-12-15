'''
An Implementation of the GreedyKCenters Algorithm for clustering
usage: python greedyKcenters.py [input_csv_filename.csv]
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

# The greedy K centers algorithm
def greedyKcenter(x,q,k):
	objVals = []
	for i in range(k-1):
		indexToBeDeleted, rowTobeInserted, cost = cluster.findFarthestPoint(x,q)
		print('Iteration : ',i,' cost = ', cost)
		q = np.append(q, rowTobeInserted, axis=0)
		x = np.delete(x, indexToBeDeleted, axis = 0)
		objVals.append(cost)
	return q, objVals[-1]


if __name__ == '__main__':
	Qfinal, D = greedyKcenter(X, Q, 5)
	print("Algorithm Completed, Optimal Objective Value = ", D)
	print("Q = ",Qfinal)
	C = cluster.findNearestCentroids(X,Qfinal)
	cluster.plotKGreedy(X,Qfinal,C)