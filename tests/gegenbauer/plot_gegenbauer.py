import sys
from os.path import join
from numpy import *
import matplotlib.pyplot as plt

sys.path.insert(0, join('..', '..'))
from orthopoly.gegenbauer import *

plt.style.use('dark_background')
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')

xhat = linspace(-1, 1, 1000)
ylims = [(-3,4), (-8,12),(-20,25)]

for m,ylim in zip([1,2,3],ylims):
    fig, ax = plt.subplots(1,1)
    for n in range(6):
        ax.plot(xhat, gegen(xhat, n, m), label='$C_{%d}^{%d}$' % (n,m))
    ax.set_ylim(ylim)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$C_n^m(x)$')
    ax.set_title('$m=%d$' % m)

plt.show()
