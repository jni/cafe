import numpy as np
from scipy import ndimage as nd
from matplotlib import pyplot as plt
from skimage import filters, measure
import toolz as tz

def threshold(im):
    return im > filters.threshold_otsu(im)


@tz.curry
def trf_quantify(im, prop='mean_intensity', normalise=None):
    # normalise can be None, 'pre', 'post'
    trf = im[..., 0]
    chrom = im[..., 2]
    objs = nd.label(threshold(trf))[0]
    if normalise == 'pre':
        trf = trf.astype(float) / (chrom + 1)
    if normalise is None or normalise == 'pre':
        chrom = np.ones_like(trf)
    props = [getattr(p, prop) / getattr(q, prop)
             for p, q in zip(measure.regionprops(objs, trf),
                             measure.regionprops(objs, chrom))]
    return props


def scatter(kd, control, colors=['orange', 'blue'], **kwargs):
    xs = list(tz.concat([i + 0.2 * np.random.randn(n)
                         for i, n in enumerate(map(len, kd + control))]))
    color_vector = ([colors[0]] * sum(map(len, kd)) +
                    [colors[1]] * sum(map(len, control)))
    ys = list(tz.concat(kd + control))
    fig = plt.scatter(xs, ys, c=color_vector, **kwargs)
    plt.xlim(0, max(xs) + 1)
    plt.ylim(0, max(ys) + 1)
    return fig
