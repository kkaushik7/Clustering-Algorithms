'''
Expectation Maximization for clustering using Gaussian Mixture Models
usage: python em.py [input_csv_filename.csv]
'''

import csv,pylab
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import cluster
from sklearn import mixture as a
import sys

# Initialization of the EM algorithm
def emInit(X, k):
	(mu, C, D) = cluster.kMeans(X,k)					# The mean vector
	sigma = cluster.emVariance(X, mu, C, k)
	alpha = cluster.emAlpha(X, C, k)
	return mu, sigma, alpha

# Plot the results of the Expectation - Maximization algorithm
def plotEm(x,y,c):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	scatter = ax.scatter(x[:,0],x[:,1],c=c,s=50)
	for i,j in y:
		ax.scatter(i,j,s=150,c='red',marker='+')
	ax.set_xlabel('X[1]')
	ax.set_ylabel('X[2]')
	plt.title('Lloyds Algorithm Results \n (Colored by Clusters)')
	plt.savefig('em.png')

def emAlg(X, k):
	mu, sigma, init = emInit(X, k)
	emAlg = a.GaussianMixture(n_components=k, covariance_type='full')
	clustering = emAlg.fit(X)
	print('clustering means = ',clustering.means_)
	return clustering.means_

if __name__ == '__main__':
	datafile = sys.argv[1]                      
	X = cluster.readCsv(datafile)               # The input matrix X
	k = 3
	Y = emAlg(X, k)
	C = cluster.findNearestCentroids(X, Y)
	plotEm(X, Y, C)