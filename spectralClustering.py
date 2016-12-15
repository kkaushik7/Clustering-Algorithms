'''
An Implementation of the Spectral Clustering Algorithm for clustering
usage: python spectralClustering.py [input_csv_filename.csv]
'''

import csv,pylab
import numpy as np
import pylab
import cluster
from numpy.linalg import norm
import sys
from scipy.linalg import eigh


# Preparing the Data for Analysis
datafile = sys.argv[1]                      # The input mxn matrix given in the form of a CSV file
X = cluster.readCsv(datafile)               # The input matrix X


if __name__ == '__main__':
	A = cluster.adjacencyMatrix(X)
	D = cluster.diagonalMatrix(A)
	L = D - A
	N = X.shape[0]
	k = 4
	trash, U = np.asarray(eigh(L,eigvals = (1,k)))
	print(type(U))
	print(U)
	(Y, C, D) = cluster.kMeans(U,k)
	print("Spectral Clustering completed")
	print("Final Objective Function Value = ", D)
	Y = cluster.LloydRecomputeCentroids(X,C,np.zeros([k,2]),k) 
	cluster.plotSpectral(X,Y,C)