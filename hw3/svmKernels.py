"""
Custom SVM Kernels

Author: Eric Eaton, 2014

"""

import numpy as np


_polyDegree = 2
_gaussSigma = 1


def myPolynomialKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    return (np.dot(X1, X2.T)+1)**_polyDegree


def myGaussianKernel(X1, X2):
    """
        Arguments:
            X1 - an n1-by-d numpy array of instances
            X2 - an n2-by-d numpy array of instances
        Returns:
            An n1-by-n2 numpy array representing the Kernel (Gram) matrix
    """
    #return np.exp(-np.linalg.norm(X1-X2)**2/(2*(_gaussSigma**2)))
    matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i in range(X1.shape[0]):
        matrix[i] = np.exp(-(np.sum(np.subtract(X2, X1[i])**2, axis = 1))/(2*_gaussSigma**2))    
    return matrix