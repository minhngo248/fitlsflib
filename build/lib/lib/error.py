import numpy as np

def rms_error(xdata, xappro) -> float:
    """
    Calculate summary of square error of an approximation

    Parameters
    ---------------
    xdata : an 1-dim array of float
            real data
    xappro : an 1-dim array of float having same size as xdata
            approximate data of real data

    Returns
    ---------------
    sq_error : float
    """
    sq_error = np.sum((xdata - xappro)**2)
    return np.sqrt(sq_error/len(xdata))

def max_relative_error(xdata, xappro) -> float:
    """
    Calculate maximum percentage of relative error

    Parameters
    ---------------
    xdata : an 1-dim array of float
            real data
    xappro : an 1-dim array of float having same size as xdata
            approximate data of real data

    Returns
    ---------------
    err_max : float    
    """
    err_max = np.max(abs(xdata-xappro)) / np.max(xdata)
    return err_max