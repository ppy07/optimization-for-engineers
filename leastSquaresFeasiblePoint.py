# Optimization for Engineers - Dr.Johannes Hild
# Least squares feasible point

# Purpose: Provides .residual() and .jacobian() of the least squares mapping x -> 0.5*sum_k (p_k*h_k(x))**2

# Input Definition:
# hArray: N-dimensional array with objective classes mapping R**n->R with methods .objective() and .gradient(), equality constraints
# p: column vector in R**N, weights for the constraints

# Output Definition:
# residual(): column vector in R**N, the k-th entry is p[k]*h[k](x)
# jacobian(): matrix in R**Nxm, the [k,j]-th entry returns: partial derivative with respect to x_j of (p[k]*h[k](x))

# Required files:
# <none>

# Test cases:
# p0 = np.array([[2],[-1]], dtype=float)
# myObjectives =  np.array([simpleValleyObjective(p0)], dtype=object)
# myWeights = np.array([1], dtype=float)
# myErrorVector = leastSquaresFeasiblePoint(myObjectives, myWeights)
# x0 = np.array([[0],[4]], dtype=float)
# should return
# myErrorVector.residual(x0) close to [[18]]
# myErrorVector.jacobian(x0) = [[0, 12]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23186144
    return matrnr


class leastSquaresFeasiblePoint:

    def __init__(self, hArray:np.array, p: np.array):
        self.hArray = hArray # array storing all constraints
        self.p = p # weights for the constraints
        self.N = hArray.shape[0] # number of constraints

    def residual(self, x: np.array):
        myResidual = np.zeros((self.N, 1)) # initialize residual vector as zero vector

        # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE

        for k in range(self.N):  # Iterate through each constraint
            
            modelobj = self.p[k] *self.hArray[k].objective(x)        # Calculate the objective value for the k-th constraint and multiply by the weight p[k]    
            myResidual[k] =   myResidual[k] +  modelobj # Add the weighted objective value to the corresponding entry in the residual vector

        # INCOMPLETE CODE ENDS

        return myResidual # Return the final residual vector

    def jacobian(self, x: np.array):
        myJacobian = np.zeros((self.N, x.shape[0])) # initialize jacobian matrix as zero matrix

        # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
        for k in range(self.N):  # Iterate through each constraint
             
            grad = self.p[k] * self.hArray[k].gradient(x).T  # Calculate the gradient of the k-th objective, multiply by weight p[k], and transpose it
            myJacobian[k,:] =  grad # Assign the gradient (as a row vector) to the k-th row of the Jacobian matrix

        # INCOMPLETE CODE ENDS

        return myJacobian
