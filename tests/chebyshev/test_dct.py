import sys
from os.path import join
from numpy import sin, exp, abs
from scipy.linalg import solve

sys.path.insert(0, join('..', '..'))
from orthopoly.chebyshev import *

#-------------------------------------------------------------------------------
#INPUTS

#arbitrary test function
f = lambda x: sin(6*x*x) - exp(x) + x

#number of points
N = 64

#-------------------------------------------------------------------------------
#MAIN

#get the chebyshev grid and other stuff
xhat, theta, x, A, scale = cheby_coef_setup(-1, 1, N)

#evaluate the test function
y = f(xhat)

#find expansion coefficents directly using a matrix
a_mat = solve(A, y)

#find expansion coefficents using a discrete cosine transform
a_dct = cheby_dct(y)

#compare
err = (a_mat - a_dct).max()
print('Maximum absolute error of coefficients: %g' % err)

err = (y - cheby_idct(a_dct)).max()
print('Maximum absolute error of idct values: %g' % err)
