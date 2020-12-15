import sys
from os.path import join
from numpy import pi, sqrt, sin, cos, abs, linspace, meshgrid
from scipy.linalg import solve
import matplotlib.pyplot as plt

sys.path.insert(0, join('..', '..'))
from orthopoly.spherical_harmonic import *

#plt.style.use('dark_background')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

#-------------------------------------------------------------------------------
# INPUT

#degree of truncation (should be an even number)
T = 20

#number of grid points for plotting
nphi = 500

#function to test
f_z = lambda t, p: sqrt(2*((sin(t)**3)*cos(2*p) + 2))

#-------------------------------------------------------------------------------
#MAIN

#get number of points
assert T % 2 == 0
N = T2nharm(T)

#generate a fibonacci grid
t, p = grid_fibonacci(N)

#generate the matrix
Y, yn, ym = sph_har_T_matrix(t, p, T)

#evaluate the test function
z = f_z(t, p)

#solve linear system for expansion coefficients
a = solve(Y, z)

#create an Expansion object
ex = Expansion(a, yn, ym)

#make a grid for sampling
ntheta = nphi//2
pg = linspace(0, 2*pi, nphi)
tg = linspace(0, pi, ntheta)
Pg, Tg = meshgrid(pg, tg)

#plot results
fig, (axa, axb, axc) = plt.subplots(3,1)

r = axa.pcolormesh(
    linspace(0, 2*pi, nphi+1),
    linspace(0, pi, ntheta+1),
    f_z(Tg, Pg))
plt.colorbar(r, ax=axa)
axa.scatter(p, t, c='k', s=1)
axa.set_title('Exact Function')

r = axb.pcolormesh(
    linspace(0, 2*pi, nphi+1),
    linspace(0, pi, ntheta+1),
    ex(Tg, Pg))
plt.colorbar(r, ax=axb)
axb.scatter(p, t, c='k', s=1)
axb.set_title('Reproduced Function')

err = ex(Tg, Pg) - f_z(Tg, Pg)
r = axc.pcolormesh(
    linspace(0, 2*pi, nphi+1),
    linspace(0, pi, ntheta+1),
    err,
    vmin=-abs(err).max(),
    vmax=abs(err).max(),
    cmap='RdBu')
plt.colorbar(r, ax=axc)
axc.scatter(p, t, c='k', s=1)
axc.set_title('Error with Dense Sampling')

fig.tight_layout()
plt.show()
