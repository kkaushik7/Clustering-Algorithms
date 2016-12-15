'''
IE529 - Computational Assignment 2
Clustering Techniques
Kaushik Krishnan
'''

'''
This is a file with all the helper functions I will call in other codes
The algorithms are written in a general implementation: Will work for any d-dimensional dataset
'''

import csv,pylab
import numpy as np
import pylab
from numpy.linalg import norm
import math
import matplotlib.pyplot as plt

# This function reads a CSV file and returns the data matrix x
def readCsv (fileName):
	ls = []
	with open(fileName) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			a = [float(i) for i in row]
			ls.append(np.asarray(a))
	return np.asarray(ls)

# This function calculates the distance between two points x1 and x2 based on the L-2 norm
def l2dist(x1,x2):
	d = norm((x1-x2),2)
	return math.pow(d,2)

# This function selects k-random centers from the given matrix X
def selectKRandom(x,k):
	y = x[np.random.randint(x.shape[0], size=k), :]
	return y

# This function computes the closest centroid for every point in the input vector X
def findNearestCentroids(x,y):
	c = np.zeros(x.shape[0]) 
	for i in range(x.shape[0]):
		assignment = 0
		minD = 99999
		for j in range(y.shape[0]):
			d = l2dist(x[i], y[j])
			if d < minD:
				assignment = j
				minD = d
		c[i] = assignment
	return c

# This function is used to find the objective function of the Lloyd's algorithm
def LloydObjective(x,y,c):
	sum = 0
	for i in range(x.shape[0]):
		j = int(c[i])
		d = l2dist(x[i], y[j])
		sum += d
	return sum

# This function is used to recompute the centroids in the Lloyd's algorithm
def LloydRecomputeCentroids(x,c,y,k):
	for k in range(k):
		sumVector = np.zeros(x.shape[1])
		numOccurences = 0
		for i in range(x.shape[0]):
			if (c[i] == k):
				sumVector = sumVector + x[i]
				numOccurences += 1
		meanVector = sumVector / numOccurences
		y[k] = meanVector
	return y

# This function prints the results of the clustering in a 2-D Graph
def plotKMeans(x,y,c):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(x[:,0],x[:,1],c=c,s=50)
	for i,j in y:
		ax.scatter(i,j,s=150,c='red',marker='+')
	ax.set_xlabel('X[1]')
	ax.set_ylabel('X[2]')
	plt.title('Lloyds Algorithm Results \n (Colored by Clusters)')
	plt.savefig('voronoi.png')

# GreedyKCenters - This function finds the farthest point with respect to all the points in the set Q
def findFarthestPoint(x,q):
	maxDist = 0
	maxIndex = 0
	z = np.zeros(q.shape[1])
	for j in range(q.shape[0]):
		for i in range(x.shape[0]):
			d =  l2dist(x[i],q[j])
			if (d >= maxDist):
				maxDist = d
				maxIndex = i
	return maxIndex, x[[maxIndex]], maxDist

# This function prints the results of the clustering in a 2-D Graph
def plotKGreedy(x,y,c):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(x[:,0],x[:,1],c=c,s=50)
	for i,j in y:
		ax.scatter(i,j,s=150,c='red',marker='+')
	ax.set_xlabel('X[1]')
	ax.set_ylabel('X[2]')
	plt.title('Lloyds Algorithm Results \n (Colored by Clusters)')
	plt.savefig('greedyKcenters.png')

# The k-Means clustering algorithm
def kMeans(X,k):
	# select 10 random initializations and select the best out of them
	(yBest, cBest, dBest) = (0, 0, 999999)
	for init in range(1):
		print ('Initialization ',init+1,' begins...')
		Y = selectKRandom(X,k)  # Select k Random Initial Centroids as starting points
		converged = False
		D = []
		i = 0
		tol = 0.00001
		while not converged:
			C = findNearestCentroids(X,Y)   # Find the nearest centroid to every data point
			objVal = LloydObjective(X,Y,C)  # The objective cost function of the k-means algorithm
			D.append(objVal)
			print(' Iteration: ',i,", Cost = ",objVal)
			if (i > 0):
				if (D[i-1] - D[i] <= tol):
					converged = True
					break
			Y = LloydRecomputeCentroids(X,C,Y,k)  # Recompute the centroids based on the revised centroid allocation
			i += 1
		if D[-1] <= dBest:
			dBest = D[-1]
			(yBest, cBest, dBest) = (Y, C, D[-1])
	return yBest, cBest, dBest

# The Greedy k- Centers Clustering Algorithm
def greedyKcenter(x,q,k):
	objVals = []
	for i in range(k-1):
		indexToBeDeleted, rowTobeInserted, cost = findFarthestPoint(x,q)
		print('Iteration : ',i,' cost = ', cost)
		q = np.append(q, rowTobeInserted, axis=0)
		x = np.delete(x, indexToBeDeleted, axis = 0)
		objVals.append(cost)
	return q, objVals[-1], x

# For the single sway algorithm, this step performs a swap
def swap(x, q):
	waste1, waste2, curCost = findFarthestPoint(x, q)
	errorFlag = True
	stop = False
	while not stop:
		i = np.random.randint(0,x.shape[0])
		j = np.random.randint(0,q.shape[0])
		rowToBeInserted = x[i]
		q = np.append(q, [rowToBeInserted], axis=0)
		x = np.delete(x, i, axis = 0)
		q = np.delete(q, j, axis = 0)
		waste1, waste2, newCost = findFarthestPoint(x, q)
		if newCost < curCost:
			stop = True
			errorFlag = False
			break
	return x, q, errorFlag, newCost

# This function prints the results of the clustering in a 2-D Graph
def plotSingleSwap(x,y,c):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(x[:,0],x[:,1],c=c,s=50)
	for i,j in y:
		ax.scatter(i,j,s=150,c='red',marker='+')
	ax.set_xlabel('X[1]')
	ax.set_ylabel('X[2]')
	plt.title('Lloyds Algorithm Results \n (Colored by Clusters)')
	plt.savefig('singleSwap.png')

# This function returns the adjacency matrix for the spectral clustering algorithm
def adjacencyMatrix(x):
	n = x.shape[0]
	a = np.zeros([n,n])
	for i in range(n):
		for j in range(n):
			a[i,j] = math.exp(-l2dist(x[i],x[j]))
	return a

def diagonalMatrix(a):
	n = a.shape[0]
	d = np.sum(a, axis = 0)
	return np.diagflat(d)

# Plot the results of the spectral clustering algorithm
def plotSpectral(x,y,c):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(x[:,0],x[:,1],c=c,s=50)
	for i,j in y:
		ax.scatter(i,j,s=150,c='red',marker='+')
	ax.set_xlabel('X[1]')
	ax.set_ylabel('X[2]')
	plt.title('Lloyds Algorithm Results \n (Colored by Clusters)')
	plt.savefig('spectral.png')

# Compute the variance vector
def emVariance(x,y,c,k):
	for k in range(k):
		sumVector = np.zeros(x.shape[1])
		numOccurences = 0
		for i in range(x.shape[0]):
			if (c[i] == k):
				sumVector = sumVector + np.square(x[i]-y[k])
				numOccurences += 1
		meanVector = sumVector / numOccurences
	return y

# Compute the alpha vector for the EM Algorithm
def emAlpha(x, c, k):
	cVec = np.zeros(k)
	N = x.shape[0]
	for k in range(k):
		j = int(c[k])
		cVec[j] += 1
	return cVec / N

