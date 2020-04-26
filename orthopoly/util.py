"""
General use functions
"""

from numpy import *

__all__ = [
    'isnum',
    'xhat2x',
    'x2xhat',
    'xhat2theta',
    'theta2xhat',
]

def isnum(x):
    """Tests if an object is float-able (a number)

    :param x: some object

    :return: True or False"""

    try:
        x = float(x)
    except ValueError:
        return(False)
    else:
        return(True)

#-------------------------------------------------------------------------------
# coordinate mappings/conversions

xhat2x = lambda xhat, a, b: (xhat + 1.0)*((b - a)/2.0) + a
xhat2x.__doc__ = r'Converts :math:`\hat{x}` coordinates in :math:`[-1,1]` to :math:`x` coordinates in :math:`[x_a,x_b]`'

x2xhat = lambda x, a, b: 2.0*(x - a)/(b - a) - 1.0
x2xhat.__doc__ = r'Converts :math:`x` coordinates in :math:`[x_a,x_b]` to :math:`\hat{x}` coordinates in :math:`[-1,1]`'

xhat2theta = lambda xhat: arccos(xhat)
xhat2theta.__doc__ = r'Converts :math:`\hat{x}` coordinates in :math:`[-1,1]` to :math:`\theta` coordinates :math:`[\pi,0]`'

theta2xhat = lambda xhat: arccos(xhat)
theta2xhat.__doc__ = r'Converts :math:`\theta` coordinates in :math:`[\pi,0]` to :math:`\hat{x}` coordinates in :math:`[-1,1]`'
