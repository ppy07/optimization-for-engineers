# Optimization for Engineers - Dr.Johannes Hild
# projected BFGS descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k is the reduced BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the BFGS matrix is reset.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedBFGSDescent(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def projectedBFGSDescent(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start projectedBFGSDescent...') # print start

    countIter = 0 # counter for number of loop iterations
    xp = P.project(x0) #initialize with projected starting point
    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    H_k = np.diag(np.ones(xp.shape[0]))  # Initialize BFGS Hessian approximation as identity matrix
    A_k= P.activeIndexSet(xp)  # Store the initial active set (boundary constraints)
    # a) Main optimization loop — check stationarity condition
    while np.linalg.norm(xp-P.project(xp-(f.gradient(xp))))>eps:
        grad = f.gradient(xp) # compute gradient at current point ∇f(xp)
        d = PCG.PrecCGSolver(H_k, -grad, eps, 0) # solve H_k d = -grad using CG solver to get search direction
        # b) if not a descent direction, fall back to steepest descent
        if (grad.T @ d) >= 0:
            d = -grad.copy() # Use -grad as descent direction
            H_k = np.diag(np.ones(xp.shape[0])) # Reset Hessian approximation

        # Perform projected backtracking line search to compute step length t
        t = PB.projectedBacktrackingSearch(
            f, P, xp, d,
            sigma=1e-3, rho=1e-2,
            verbose=verbose
            )

        #Compute new iterate and project it back to the feasible box
        x_new = P.project(xp+t*d) # new iterate
        active_set= P.activeIndexSet(x_new) #Get the active constraint set after the step
   
        if active_set!=A_k:
            H_k[active_set, :] = (np.eye(xp.shape[0])[active_set, :]) # reset the rows in A_plus to the corresponding rows of Iₙ
            H_k[:, active_set] = (np.eye(xp.shape[0])[:, active_set]) # reset the columns in A_plus to the corresponding columns of Iₙ

        else:
            delta_g_k= f.gradient(P.project(xp+t*d))-f.gradient(xp) # gradient difference
            deltax= x_new-xp # step difference
            if (delta_g_k.T @ deltax) <=eps**2:
                H_k= np.diag(np.ones(xp.shape[0])) # reset if curvature fails
            else:
                # Compute BFGS update terms: H ← H + (Δg Δgᵀ)/(Δgᵀ Δx) − (H Δx Δxᵀ H)/(Δxᵀ H Δx)
                den1=delta_g_k.T @ deltax # Δgᵀ Δx Denominator for first term
                term1 = (delta_g_k @ delta_g_k.T) / den1 # First rank-2 update term
                Hx = H_k @ deltax # H·Δx
                den2 = deltax.T @ Hx # Δxᵀ H Δx Denominator for second term
                term2 = (Hx @ Hx.T) / den2 # Second rank-1 update term
                H_k = H_k + term1 - term2 #  Perform BFGS update

                
                # Re-apply active set structure by resetting parts of Hessian
                H_k[active_set, :] = (np.eye(xp.shape[0])[active_set, :]) 
                H_k[:, active_set] = (np.eye(xp.shape[0])[:, active_set])
        xp= x_new # accept new iterate
        A_k= active_set # accept new active‐set
        countIter += 1 # increment counter

    if verbose: # print information
        gradx = f.gradient(xp) # get gradient
        stationarity = np.linalg.norm(xp - P.project(xp - gradx)) # get stationarity
        print('projectedBFGSDescent terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity)) # print termination

    return xp
