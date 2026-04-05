# Optimization for Engineers - Dr.Johannes Hild
# Levenberg-Marquardt descent

# Purpose: Find pmin to satisfy norm(jacobian_R.T @ R(pmin))<=eps

# Input Definition:
# R: error vector class with methods .residual() and .jacobian()
# p0: column vector in R**n (parameter point), starting point.
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# alpha0: positive value, starting value for damping. Default value: 1.0e-3.
# beta: positive value bigger than 1, scaling factor for alpha. Default value: 100.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# pmin: column vector in R**n (parameter point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# p0 = np.array([[2], [-1]], dtype=float)
# myObjectives = np.array([simpleValleyObjective(p0)], dtype=object)
# myWeights = np.array([1], dtype=float)
# myErrorVector = leastSquaresFeasiblePoint(myObjectives, myWeights)
# x0 = np.array([[0], [4]], dtype=float)
# eps = 1.0e-6
# alpha0 = 1.0e-8
# beta = 1000
# xFeasible = levenbergMarquardtDescent(myErrorVector, x0, eps, alpha0, beta, 1)
# feasibleErrorVector = myErrorVector.residual(xFeasible)
# should return feasibleErrorVector close to zeros

import numpy as np
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 23186144
    return matrnr


def levenbergMarquardtDescent(R, p0: np.array, eps=1.0e-4, alpha0=1.0e-3, beta=100, verbose=0):
    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if alpha0 <= 0: # check for positive alpha0
        raise TypeError('range of alpha0 is wrong!')

    if beta <= 1: # check for sufficiently large beta
        raise TypeError('range of beta is wrong!')

    if verbose: # print information
        print('Start levenbergMarquardtDescent...') # print start

    countIter = 0 # counter for loop iterations

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    p = p0 # Initialize p = p0
    alpha = alpha0 # alpha = alpha0
    jac = R.jacobian(p)     # Compute the Jacobian at the initial point
    res = R.residual(p) # Compute the  residual at the initial point
    gradp = jac.T @ res # Compute gradient
    
    while(np.linalg.norm(gradp) > eps): # Compute gradient
        
        A = jac.T @ jac + alpha * np.eye(jac.shape[1])  # Compute A = J^T * J + alpha * I
        b = -gradp  # Right-hand side vector for the conjugate gradient solver
        d = PCG.PrecCGSolver(A, b)  # Solve for the step d_k using PCG
        respd = 0.5 * R.residual(p+d).T @ R.residual(p+d) # Residual at p + d
        resp = 0.5 * R.residual(p).T @ R.residual(p)       # Residual at p         
        if (respd < resp):   # If the residual decreases
            p = p+d  # Accept the step: p_{k+1} = p_k + d_k
            alpha = alpha0  # Reset alpha to its initial value
            
        else: # If the residual does not decrease
            alpha = alpha * beta # Increase alpha by a factor of beta

        jac = R.jacobian(p)  # Recompute the Jacobian at the new point
        res = R.residual(p) # Recompute the residual at the new point
        gradp = jac.T @ res# Recompute the gradient at the new point
            
        countIter += 1  # Increment iteration counter
 

    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        gradp = R.jacobian(p).T @ R.residual(p) # store final gradient
        print('levenbergMarquardtDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradp)) # print termination and gradient information

    return p  # Return the final parameter vector p
