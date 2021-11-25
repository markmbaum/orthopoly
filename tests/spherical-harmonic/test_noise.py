import sys
from os.path import join
from numpy import *
import matplotlib.pyplot as plt

sys.path.insert(0, join('..', '..'))
from orthopoly.spherical_harmonic import *

plt.style.use('dark_background')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

#-------------------------------------------------------------------------------
#FUNCTIONS

def test_noise(N, ntheta=201, nphi=401, cmap='RdBu'):
    """Generate noise, from red to blue, and plot the spectra along with
    the expansions
    args:
        N - maximum degree of noise expansions
    optional args:
        ntheta - number of colatitude rows to plot
        nphi - number of longitude columns to plot
        cmap - colormap name"""

    figa = plt.figure(figsize=(8,4))
    figb, axb = plt.subplots(1,1)
    t = linspace(0, pi, ntheta)
    p = linspace(0, 2*pi, nphi)
    P, T = meshgrid(p, t)
    M = sph_har_matrix(T.flatten(), P.flatten(), *Tnm(N))
    p = -3.0
    c = 1
    for i in range(2):
        for j in range(2):

            axa = figa.add_subplot(2, 2, c, projection='mollweide')
            ex = noise(N, p)
            Y = M.dot(ex.a).reshape(ntheta, nphi)
            v = abs(Y).max()
            r = axa.pcolormesh(#P - pi, -T + pi/2, Y,
                    linspace(-pi, pi, nphi+1),
                    linspace(-pi/2, pi/2, ntheta+1),
                    Y,
                    cmap='RdBu',
                    vmin=-v,
                    vmax=v)
            axa.grid(False)
            axa.set_xticks([])
            axa.set_yticks([])
            axa.set_title('$p=%g$' % p)

            ns, ps = ex.spectrum
            axb.loglog(ns[ps != 0], ps[ps != 0], ':.', label='$p=%g$' % p)
            p += 1.0
            c += 1

    figa.tight_layout()

    axb.set_xlabel('Spherical Harmonic Degree')
    axb.set_ylabel('Power Density')
    axb.set_title('Power Density vs. Degree')
    axb.legend()
    figb.tight_layout()

def plot_noise(N, p='red', ntheta=151, nphi=301, ni=4, nj=5, cmap='RdBu'):
    """Generate noise, from red to blue, and plot the spectra along with
    the expansions
    args:
        N - maximum degree of noise expansions
    optional args:
        p - type of noise
        ntheta - number of colatitude rows to plot
        nphi - number of longitude columns to plot
        ni - number of rows in plotted grid
        nj - number of columns in plotted grid
        cmap - colormap name"""

    fig = plt.figure()
    theta = linspace(0, pi, ntheta)
    phi = linspace(0, 2*pi, nphi)
    P, T = meshgrid(phi, theta)
    M = sph_har_matrix(T.flatten(), P.flatten(), *Tnm(N))
    c = 1
    for i in range(ni):
        for j in range(nj):
            ax = fig.add_subplot(ni, nj, c, projection='mollweide')
            ex = noise(N, p, seed=i*j)
            Y = M.dot(ex.a).reshape(ntheta, nphi)
            v = abs(Y).max()
            ax.pcolormesh(#P - pi, T - pi/2, Y,
                    linspace(-pi, pi, nphi+1),
                    linspace(-pi/2, pi/2, ntheta+1),
                    Y,
                    cmap=cmap,
                    vmin=-v,
                    vmax=v)
            ax.set_xticks([])
            ax.set_yticks([])

            c += 1

def noise_means(N, npts, neval, p='red', cmap='RdBu'):

    #create a grid
    theta, phi = grid_fibonacci(npts)
    #get spherical harmoinics at the grid points
    yn, ym = Tnm(N)
    Y = sph_har_matrix(theta, phi, yn, ym)
    #take an average of many noises
    m = zeros((len(theta),))
    ma = zeros((neval,))
    for i in range(neval):
        ex = noise(N, p)
        y = Y.dot(ex.a)
        m += y/float(neval)
        ma[i] = max(abs(m))

    fig, ax = plt.subplots(1,1)
    r = ax.tricontourf(phi, theta, m, cmap=cmap)
    cb = plt.colorbar(r, ax=ax)
    cb.set_label('Mean of %d Random Expansions (p = %s)' % (neval, str(p)))
    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'$\theta$')
    fig, ax = plt.subplots(1,1)
    ax.plot(ma)
    ax.set_xlabel('Number of Random Expansions')
    ax.set_ylabel('Maximum Magnitude in Running Mean')
    plt.show()

#-------------------------------------------------------------------------------
#MAIN

print('TESTING SPHERICAL HARMONIC NOISE')

print('generating different types of noise for inspection')
test_noise(25)
plt.show()

print('generating a bunch of different noisy spectra')
plot_noise(10)
plt.show()

#print('inspecting mean value of many instances of noise')
#noise_means(10, 201, 10000)
