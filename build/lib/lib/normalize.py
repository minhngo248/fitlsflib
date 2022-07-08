import numpy as np

def normalize(image_cut):
    """
    Normalize an image by uniform distribution

    Parameters
    ----------
    image_cut : array 2 dim

    Returns
    -----------
    image_uni : array 1 dim

    """  
    image_uni = np.ravel(image_cut)
    max = np.max(image_uni)
    min = np.min(image_uni)
    image_uni = (image_uni - min)/(max - min)
    return image_uni