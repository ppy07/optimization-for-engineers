# Optimization for Engineers - Dr.Johannes Hild
# projected Wolfe-Powell line search

# Purpose: Find t to satisfy f(P(x+t*d))<f(x) + sigma*gradf(x).T@(P(x+t*d)-x) with P(x+t*d)-x being a descent direction
# and in addition but only if x+t*d is inside the feasible set gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set to the biggest 2**m, such that 2**m satisfies the projected sufficient decrease condition
# and in addition if x+t*d is inside the feasible set, the second Wolfe-Powell condition holds

# Required files:
# <none>

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[-2], [1]])
# b = np.array([[2], [2]])
# eps = 1.0e-6
# myBox = projectionInBox(a, b, eps)
# x = np.array([[1], [1]])
# d = np.array([[-1.99], [0]])
# sigma = 0.5
# rho = 0.75
# t = projectedBacktrackingSearch(myObjective, myBox, x, d, sigma, rho, 1)
# should return t = 0.5

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def projectedBacktrackingSearch(f, P, x: np.array, d: np.array, sigma=1.0e-4, rho=1.0e-2, verbose=0):
    xp = P.project(x) # initialize with projected starting point
    fx = f.objective(xp) # get current objective
    gradx = f.gradient(xp) # get current gradient
    descent = gradx.T @ d # descent direction check value

    if descent >= 0: # if not a descent direction
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5: # if sigma is out of range
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1: # if rho does not fit to sigma
        raise TypeError('range of rho is wrong!')

    if verbose: # print information
        print('Start projectedBacktracking...') # print start
    # 6) stationarity check: if no movement under projection, x is already stationary

    # starting guess for t
    t = 1 # start with full step

    if np.allclose(P.project(x+d),xp):    # If projection has no effect (stationary point)
        raise TypeError("detected stationary point") # failsafe for no movement

     # Wolfe-Powell condition 1 (sufficient decrease with projection)
    def WP1(t):
        xpt = P.project(x + t*d) #project the trial point x + t d
        dir_proj = xpt - xp # Compute the projected search direction
        dir_der = float(gradx.T @ dir_proj) #scalar directional derivative
        f_xpt = float(f.objective(xpt)) #scalar f at projected trial
        # need descent AND sufficient decrease
        return (dir_der < 0) and (f_xpt <= fx + sigma * dir_der)  # Check Armijo condition

    # Wolfe-Powell condition 2 (curvature condition, only if trial point is inside box)
    def WP2(t):
        trial = x + t*d # raw trial point
        xpt = P.project(trial) # project into box
        # if projection moved us off the line, accept immediately
        if not np.allclose(trial, xpt, atol=1e-12):
            return True
        # otherwise curvature check
        dir_der2 = float(f.gradient(xpt).T @ d)  # Directional derivative at new point
        return dir_der2 >= rho * descent # curvature condition

    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE
    #Begin backtracking or fronttracking to bracket acceptable t
    if not WP1(t):   # If full step fails WP1, begin backtracking
        t=t/2 # half the step length
        while not WP1(t):  # Continue halving until WP1 satisfied
            t=t/2 # halve again
        t_low=t # Store lower bound
        t_high=2*t # Store upper bound just beyond WP1 failure
    elif WP2(t): # If full step passes both WP1 and WP2
        return t # Accept and return full step
    else: # WP1 holds but WP2 fails → fronttracking
        t=2*t # Try a larger step
        while WP1(t) and np.allclose(P.project(x+t*d),x+t*d): # while Armijo still holds and no projection move
            t=2*t # Keep doubling while WP1 passes and inside box
        t_low=t/2 # set lower bracket endpoint
        t_high=t # set upper bracket endpoint


    t=t_low # initialize t for refinement

    while not WP2(t): # refine until curvature holds
        t_mean=((t_low+t_high)/2) # midpoint of current bracket
        if WP1(t_mean): # If midpoint satisfies WP1
            t_low=t_mean # raise lower bracket
        else:
            t_high=t_mean # lower upper bracket

        t=t_low # use updated lower bracket as current t


 # INCOMPLETE CODE ENDS

    if verbose: # print verbose information
        xt = P.project(x + t * d) # get x+td for found step size t
        fxt = f.objective(xt) # get objective value at this point
        print('projectedBacktracking terminated with t=', t) # print termination
        print('Sufficient decrease: ', fxt, '<=', fx+t*sigma*descent) # print result of sufficient decrease check

    return t
