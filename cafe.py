"""CAFE: Centromere-Associated Fluorescence Estimator

Quantify fluorescence around fluorescently-labelled centromeres in
another channel.
"""

import numpy as np
from scipy import ndimage as nd
#from skimage import io
from skimage import filter as imfilter
from skimage.morphology import selem, remove_small_objects


def otsu(image, offset=0.0, factor=1.0):
    """Threshold an image using Otsu's method with an optional offset.

    Parameters
    ----------
    image : array
        The image to be thresholded.
    offset : float, optional
        Change the computed threshold by subtracting a fixed amount.
    factor : float, optional
        Change the computed threshold by multiplying by a constant
        factor.

    Returns
    -------
    image_t : array of bool
        The thresholded image.
    """
    t = imfilter.threshold_otsu(image)
    t -= offset
    t *= factor
    image_t = image > t
    return image_t


def encode_centro_telomeres(image_centro, image_telo,
                            centro_offset=0.0, centro_factor=1.0,
                            centro_min_size=36, centro_radius=10,
                            telo_offset=0.0, telo_adapt_radius=49,
                            telo_open_radius=4):
    """Find centromeres, telomeres, and their overlap.

    Parameters
    ----------
    image_centro : array, shape (M, N)
        The grayscale channel for centromeres.
    image_telo : array, shape (M, N)
        The grayscale channel for telomeres.
    centro_offset : float, optional
        Offset Otsu's threshold by this amount (i.e. be less stringent
        about what image intensity constitutes a centromere)
    centro_factor : float, optional
        Offset Otsu's threshold by a multiplicative constant.
    centro_min_size : int, optional
        Remove objects smaller than this, as they would be too small to
        be a centromere.
    centro_radius : int, optional
        Consider anything within this radius to be "near" a centromere.
    telo_offset : float, optional
        Offset the telomere image threshold by this amount.
    telo_adapt_radius : int, optional
        Use this radius to threshold telomere image adaptively.
    telo_open_radius : int, optional
        Use this radius for a binary opening of thresholded telomeres
        (removes noise).

    Returns
    -------
    encoded_regions : array of int, shape (M, N)
        A uint8 image with the following values:
         - 0: background
         - 1: telomeres
         - 2: centromeres
         - 3: centromere/telomere overlap
    """
    centros = otsu(image_centro, centro_offset, centro_factor)
    centros = remove_small_objects(centros, centro_min_size)
    centro_strel = selem.disk(centro_radius)
    centros = nd.binary_dilation(centros, structure=centro_strel)
    telos = imfilter.threshold_adaptive(image_telo, telo_adapt_radius,
                                        offset=telo_offset)
    telo_strel = selem.disk(telo_open_radius)
    telos = nd.binary_opening(telos, structure=telo_strel)
    encoded_regions = 2 * centros.astype(np.uint8) + telos
    return encoded_regions


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


def get_chromatin(im, background_diameter=51, opening_size=2, opening_iter=2,
                  size_filter=256):
    """Find the chromatin in an unevenly illuminated image.

    Parameters
    ----------
    im : np.ndarray, shape (M, N)
        The chromatin grayscale image.
    background_diameter : int, optional
        The diameter of the block size in which to find the background. (This
        is used by the scikit-image function `threshold_adaptive`.)
        (default: 51)
    opening_size : int, optional
        Perform a binary opening with a disk of this radius. (default: 2)
    opening_iter : int, optional
        Perform this many opening iterations. (default: 2)
    size_filter : int, optional
        After the morphological opening, filter out segments smaller than this
        size. (default: 256)

    Returns
    -------
    chrs : np.ndarray, shape (M, N)
        A thresholded image of the chromatin regions.
    """
    if im.ndim == 3 and im.shape[2] == 3:
        im = im[..., 0]
    fg = imfilter.threshold_adaptive(im, background_diameter)
    # on an unevenly lit image, `fg` will have all sorts of muck lying around,
    # in addition to the chromatin. Thankfully, the muck is noisy and full of
    # holes, whereas the chromatin is solid. An opening followed by a size
    # filtering removes it quite effectively.
    strel = selem.disk(opening_size)
    fg_open = nd.binary_opening(fg, strel, iterations=opening_iter)
    chrs = remove_small_objects(fg_open, size_filter)
    return chrs


def rnapii_centromere_vs_chromatin(rgb_im, channels=(0, 1, 2),
                                   normalise_to_1=True,
                                   centromere_dilation_size=3,
                                   centromere_threshold=None,
                                   centromere_threshold_function=
                                                    imfilter.threshold_otsu,
                                   chromatin_background_diameter=51,
                                   chromatin_opening_size=2,
                                   chromatin_opening_iter=2,
                                   chromatin_size_filter=256):
    """Find intensity differences in the RNA-Pol-II channel.

    Parameters
    ----------
    rgb_im : np.ndarray, shape (M, N, 3)
        The 3-channel image containing an RNA-Pol-II signal, a centromere
        signal, and a chromatin signal.
    channels : tuple of int, optional
        The channels corresponding to RNA-Pol-II, centromere, and chromatin
        signals, respectively. Default: (0, 1, 2).
    normalize_to_1: bool, optional
        Make the maximum red channel value 1.
    centromere_* : various types, optional
        Parameters passed through to `get_centromere_neighbourhood`.
    chromatin_* : various types, optional
        Parameters passed through to `get_chromatin`.

    Returns
    -------
    rnapii_centro, rnapii_chrom : 1D np.ndarray
        The RNA Pol II channel intensity in centromere-adjacent regions and
        in other chromatin regions.
    """
    rnapii, centromeres, chromatin = [rgb_im[..., i] for i in channels]
    centromeric_regions = get_centromere_neighbourhood(centromeres,
                                centromere_dilation_size, centromere_threshold,
                                centromere_threshold_function)
    chromatin_regions = get_chromatin(chromatin, chromatin_background_diameter,
                                      chromatin_opening_size,
                                      chromatin_opening_iter,
                                      chromatin_size_filter)
    chromatin_regions *= True - centromeric_regions
    if normalise_to_1:
        rnapii = rnapii.astype(float) / rnapii.max()
    rnapii_centro = rnapii[centromeric_regions]
    rnapii_chrom = rnapii[chromatin_regions]
    return rnapii_centro, rnapii_chrom

