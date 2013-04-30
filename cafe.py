"""CAFE: Centromere-Associated Fluorescence Estimator

Quantify fluorescence around fluorescently-labelled centromeres in
another channel.
"""

#import numpy as np
from scipy import ndimage as nd
#from skimage import io
from skimage import filter as imfilter
from skimage.morphology import selem

def get_centromere_neighbourhood(im, dilation_size=3, threshold=None,
                                 threshold_function=imfilter.threshold_otsu):
    """Obtain the locations near centromeres in an image.

    Parameters
    ----------
    im : np.ndarray, shape (M, N)
        The input image, containing fluorescently-labelled centromeres.
    dilation_size : int (optional, default 3)
        Size in pixels of neighbourhood around actual centromere locations.
    threshold : float (optional, default None)
        Use this threshold instead of one computed by `threshold_function`.
    threshold_function : function, im -> int or im -> im
                         (optional, default `skimage.filter.threshold_otsu`)
        Use this function to find a suitable threshold for the input image.

    Returns
    -------
    centro : np.ndarray of bool, shape (M, N)
        The locations around centromeres marked as `True`.
    """
    if threshold is None:
        threshold = threshold_function(im)
    centro = im > threshold
    strel = selem.disk(dilation_size)
    centro = nd.binary_dilation(centro, structure=strel)
    return centro
