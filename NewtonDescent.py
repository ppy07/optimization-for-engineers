# Optimization for Engineers - Dr.Johannes Hild
# Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + d_k
# d_k is the Newton direction

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# myObjective = bananaValleyObjective()
# x0 = np.array([[0], [1]])
# xmin = NewtonDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[1],[1]]

import numpy as np
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def NewtonDescent(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for correct range of eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start NewtonDescent...') # print start

    countIter = 0 # counter for number of loop iterations
    x = x0 # initialize with starting value

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    while np.linalg.norm(f.gradient(x))>eps:  #repeat until gradient norm is larger than eps

        g = (f.gradient(x))    #−∇f(xk) -current gradient
        H = f.hessian(x)           #∇2f(xₖ) – current Hessian
        d = PCG.PrecCGSolver(H, -g) #Newton direction: solve with the preconditioned CG solver
        t = 1.0                    #Unit step length
        x=x+ t*d                     #xk ← xk + tkdk update iterate
        countIter +=1              #Icounting the iteration we did


    # INCOMPLETE CODE ENDS

    if verbose: # print information
        gradx = f.gradient(x) # get gradient at solution
        print('NewtonDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx)) # print termination and gradient norm information

    return x
