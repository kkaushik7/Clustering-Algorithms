'''
An Implementation of the Lloyd's Algorithm for clustering
usage: python lloyd.py [input_csv_filename.csv]
'''

import csv,pylab
import numpy as np
import pylab
import cluster
from numpy.linalg import norm
import sys

def kMeans(X,k):
	# select 10 random initializations and select the best out of them
	(yBest, cBest, dBest) = (0, 0, 999999)
	for init in range(10):
		print ('Initialization ',init+1,' begins...')
		Y = cluster.selectKRandom(X,k)  # Select k Random Initial Centroids as starting points
		converged = False
		D = []
		i = 0
		tol = 0.00001
		while not converged:
			C = cluster.findNearestCentroids(X,Y)   # Find the nearest centroid to every data point
			objVal = cluster.LloydObjective(X,Y,C)  # The objective cost function of the k-means algorithm
			D.append(objVal)
			print(' Iteration: ',i,", Cost = ",objVal)
			if (i > 0):
				if (D[i-1] - D[i] <= tol):
					converged = True
					break
			Y = cluster.LloydRecomputeCentroids(X,C,Y,k)  # Recompute the centroids based on the revised centroid allocation
			i += 1
		if D[-1] <= dBest:
			dBest = D[-1]
			(yBest, cBest, dBest) = (Y, C, D[-1])
	return yBest, cBest, dBest


if __name__ == '__main__':
	datafile = sys.argv[1]                      # The input mxn matrix given in the form of a CSV file
	X = cluster.readCsv(datafile)               # The input matrix X
	(Y, C, D) = kMeans(X,2)
	print("Lloyd's algorithm completed")
	print("Final Objective Function Value = ", D)
	cluster.plotKMeans(X,Y,C)