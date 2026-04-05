# Optimization for Engineers - Dr.Johannes Hild
# inexact Newton CG

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dA = directionalHessApprox(f, x, d) from directionalHessApprox.py
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = inexactNewtonCG(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]]

import numpy as np
import WolfePowellSearch as WP
import directionalHessApprox as DHA

def matrnr():
    # set your matriculation number here
    matrnr = 23393224
    return matrnr


def inexactNewtonCG(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0: # check for positive eps
        raise TypeError('range of eps is wrong!')

    if verbose: # print information
        print('Start inexactNewtonCG...') # print start

    countIter = 0 #counter for number of loop iterations
    xk = x0 #initialize starting iteration
    grad_fk= f.gradient(xk)  #Gradient at current point
    norm_grad_fk = np.linalg.norm(grad_fk) #Norm of the gradient
    eta_k = np.min([0.5, np.sqrt(norm_grad_fk)])* norm_grad_fk #For CG tolerance


    # INCOMPLETE CODE STARTS, DO NOT FORGET TO WRITE A COMMENT FOR EACH LINE YOU WRITE

    while norm_grad_fk>eps:
        xj=xk.copy()   # inner CG loop point
        rj= grad_fk.copy() # Initial residual = gradient
        dj= -rj  # Initial search direction = steepest descent
        norm_rj= np.linalg.norm(rj) # Norm of residual

        #CG loop
        while norm_rj>eta_k:
            da =DHA.directionalHessApprox(f,xk,dj)  # Approximate Hessian * dj
            rho= dj.T@da    # Compute curvature term

            # CURVATURE CONDITION: stop CG if too flat
            if rho<=eps*(dj.T@dj):
                break
            
            # Compute optimal step size in CG direction
            tj = (rj.T@rj)/rho
            xj_new = xj + tj*dj            # Move along CG direction

            r_old = rj.copy()    # Save old residual
            rj = r_old + tj*da   # Update residual
            beta = (rj.T@rj)/ (r_old.T@r_old)  # Compute CG beta term
            dj = -rj + beta*dj   # Update search direction

            xj = xj_new    # Update xj to new point
            norm_rj = np.linalg.norm(rj)  # Recompute residual norm
        
        #descent direction dk
        if np.linalg.norm(xj -xk) < 1.0e-12:
            dk = -grad_fk      # Fall back to steepest descent if CG did nothing
        else:
            dk = xj-xk   #Use CG direction

        #Compute step length using Wolfe-Powell line search
        tk = WP.WolfePowellSearch(f, xk, dk)

        # Update iterate
        xk = xk + tk*dk
        # Update gradient and ηₖCG tolerance for next iteration  
        grad_fk = f.gradient(xk) # Compute new gradient
        norm_grad_fk = np.linalg.norm(grad_fk) # update gradient norm
        eta_k = np.min([0.5, np.sqrt(norm_grad_fk)]) * norm_grad_fk # update CG tolerance

        countIter += 1  # Increment iteration counter

    # INCOMPLETE CODE ENDS
    
    if verbose: # print information
        stationarity = np.linalg.norm(f.gradient(xk)) # store stationarity value
        print('inexactNewtonCG terminated after ', countIter, ' steps with norm of gradient =', stationarity) # print termination with stationarity value

    return xk
