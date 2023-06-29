"""
General use functions
"""

from numpy import arccos

__all__ = [
    "isnum",
    "xhat2x",
    "x2xhat",
    "xhat2theta",
    "theta2xhat",
]


def isnum(x):
    """Tests if an object is float-able (a number)

    :param x: some object

    :return: True or False"""

    try:
        x = float(x)
    except ValueError:
        return False
    else:
        return True


# -------------------------------------------------------------------------------
# coordinate mappings/conversions


def xhat2x(xhat, a, b):
    r"Converts :math:`\hat{x}` coordinates in :math:`[-1,1]` to :math:`x` coordinates in :math:`[x_a,x_b]`"
    return (xhat + 1.0) * ((b - a) / 2.0) + a


def x2xhat(x, a, b):
    r"Converts :math:`x` coordinates in :math:`[x_a,x_b]` to :math:`\hat{x}` coordinates in :math:`[-1,1]`"
    return 2.0 * (x - a) / (b - a) - 1.0


def xhat2theta(xhat):
    r"Converts :math:`\hat{x}` coordinates in :math:`[-1,1]` to :math:`\theta` coordinates :math:`[\pi,0]`"
    return arccos(xhat)


def theta2xhat(xhat):
    r"Converts :math:`\theta` coordinates in :math:`[\pi,0]` to :math:`\hat{x}` coordinates in :math:`[-1,1]`"
    return arccos(xhat)
