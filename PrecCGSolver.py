# Optimization for Engineers - Dr.Johannes Hild
# Preconditioned Conjugate Gradient Solver

# Purpose: PregCGSolver finds y such that norm(A * y - b) <= delta using incompleteCholesky as preconditioner

# Input Definition:
# A: real valued matrix nxn
# b: column vector in R ** n
# delta: positive value, tolerance for termination. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# x: column vector in R ^ n(solution in domain space)

# Required files:
# L = incompleteCholesky(A, 1.0e-3, delta) from IncompleteCholesky.py
# y = LLTSolver(L, r) from LLTSolver.py

# Test cases:
# A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
# b = np.array([[5], [8], [3]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return x = [[1], [1], [1]]

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return approx x = [[1], [0], [2], [0], [3]]


import numpy as np
import incompleteCholesky as IC
import LLTSolver as LLT


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose: # print information
        print('Start PrecCGSolver...') # print start

    countIter = 0 # counter for number of loop iterations

    L = IC.incompleteCholesky(A) # initialize L as incomplete Cholesky decomposition of A
    x = LLT.LLTSolver(L, b) # store solution of L x = b
    r = A @ x - b # residual of solving the linear system
    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    d= -LLT.LLTSolver(L,r) #initial search precondition, (-ve preconditioned residual) 

    while np.linalg.norm(r)> delta: #iterate until residual norm deops below the tolerance delta
        d_tilde = A @ d   # ˜dj ← Adj(used in step length and residual update)

        rho = d.T @ d_tilde   #ρj ← dj⊤ ˜ dj(denominator for optimal step length) 

        t = (r.T@LLT.LLTSolver(L,r))/rho # tⱼ = rj⊤LLTSolver(L,rj )/ρj  — optimal step length along dⱼ

        x += t*d #update solution xj ← xj + tjdj .

        r_old = r #save ocurrent residual for beta calculation

        r = r_old + t * d_tilde #update residuals

        beta = ((r.T @ LLT.LLTSolver(L,r))/(r_old.T@LLT.LLTSolver(L,r_old))) # βj ← rj⊤LLTSolver(L,rj )/rold⊤LLTSolver(L,rold)
        d =-(LLT.LLTSolver(L,r)) + beta*d # new direction

        countIter +=1 #increment iteration counter
   
    
    # INCOMPLETE CODE ENDS

    if verbose: # print information
        print('precCGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(r)) # print termination

    return x