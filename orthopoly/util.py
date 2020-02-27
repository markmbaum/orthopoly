def isnum(x):
    """Test if an object is float-able (a number)
    args:
        x - some object
    returns:
        b - True or False"""

    try:
        x = float(x)
    except ValueError:
        return(False)
    else:
        return(True)
