# Optimization for Engineers - Dr.Johannes Hild
# implicit Filtering

# Purpose: Inner and outer loop with projected steepest descent update to find the LMP at all scales of a noisy objective.

# Input Definition:
# f: objective class with method .objective(), can have noise
# P: projection class with method .project()
# x0: column vector in R ** n (domain point), starting point
# h: column vector in R ** m, scales for filtering
# eps: positive value, tolerance for termination. Default value: 1.0e-4.
# verbose: bool, if set to true, verbose information is displayed.

# Output Definition:
# xmin: column vector in R**n (LMP at all scales)

# Required files:
# grad_f_h = SUCSGradient(f, x, h) from SUCSGradient.py
# isStencilFailure = SUCSStencilFailure(f, x, h) from SUCSGradient.py

# Test cases:
# myObjective = noisyObjective()
# x0 = np.array([[1],[1],[1],[1],[1],[1],[1],[1]], dtype=float)
# h = np.array([[1], [0.1], [0.01], [0.001], [0.0001], [0.00001]], dtype=float)
# xmin = implicitFiltering(myObjective, x0)
# should return xmin close to [[1.027],[0],[0],[0],[0],[0],[0],[0]]

import numpy as np
import SUCSGradient as SUC


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def implicitFiltering(f, P, x0: np.array, h: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start implicitFiltering...') # print start

    def SUCSProjectedSteepestDescent(xk: np.array, hk: float, epsk=1.0e-3, sigma=1.0e-4, verbose=0): # subroutine
        if epsk <= 0: # check for positive epsk
            raise TypeError('range of eps is wrong!')

        if verbose: # print information
            print('Start SUCSProjectedSteepestDescent...') # print start of subroutine

        n = xk.shape[0] # get dimension of vector
        xp = P.project(xk) # set starting iteration to projected starting point
        grad_f_h = SUC.SUCSGradient(f, xp, hk) # build simplex gradient

        isStencilFailure = SUC.SUCSStencilFailure(f, xp, hk) # check for stencil failure
        loopCounter = 0 # initialize counter of loops
        linesearchFail = 0 # initialize linesearchFail as false

        if isStencilFailure or np.linalg.norm(xp - P.project(xp - grad_f_h)) <= epsk * hk or loopCounter > 10 * n or linesearchFail: # check all sources of stencil failure
            satisfiesTermination = 1 # set termination criterion to true
        else:
            satisfiesTermination = 0 # set termination criterion to false

        while not satisfiesTermination: # while not terminating
            beta = np.min([[1.0], 10 * hk / np.linalg.norm(grad_f_h)]) # set scaling factor to either 1 or 10 times scale divided by gradient norm
            d = - beta * grad_f_h # scaled steepest descent
            t = 1 # starting guess for step size
            linesearchCounter = 0 # counts number of linesearch loops
            while f.objective(xp + t * d) > f.objective(xp) - sigma / t * np.linalg.norm(xp - P.project(xp - t * grad_f_h)) ** 2: # sufficient decrease condition
                t = 0.5 * t # update t
                linesearchCounter += 1 # update counter
                if linesearchCounter > 10: # terminate after 10 loops
                    linesearchFail = 1 # flag for fail of line search
                    break

            xp = P.project(xp + t * d) # project with found t
            loopCounter += 1 # update loop counter
            isStencilFailure = SUC.SUCSStencilFailure(f, xp, hk) # check for stencil failure
            if isStencilFailure or np.linalg.norm(xp - P.project(xp - grad_f_h)) <= epsk * hk or loopCounter > 10 * n or linesearchFail: # check termination criteria
                satisfiesTermination = 1 # set termination criterion to true
            else:
                satisfiesTermination = 0 # set termination criterion to false

        if verbose: # print information
            print('SUCSProjectedSteepestDescent terminated after ', loopCounter, ' steps with stationarity =', np.linalg.norm(xp - P.project(xp - grad_f_h))) # print termination of inner loop

        return xp # end of subroutine

    countIter = 0 # counter for outer loops
    xk = x0 # start value of outer loop
    xb = xk.copy()
    fb = f.objective(xb)
    m = h.shape[0]
    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    for j in range(m):
        hk = h[j, 0]
        x_new = SUCSProjectedSteepestDescent(xb, hk, eps)
        f_new = f.objective(x_new)
        if f_new < fb:
            xb = x_new.copy()
            fb = f_new
        countIter += 1

    xk = xb.copy()


    # INCOMPLETE CODE ENDS

    if verbose: # print information
        print('implicitFiltering terminated after ', countIter, ' outer loops with LMP at all scales = ', xk) # print termination of outer loop
    return xk



