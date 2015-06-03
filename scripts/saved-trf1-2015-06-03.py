from matplotlib import pyplot as plt
from skimage import io
import trf1 as tr
import matplotlib as mpl
kd_ims = io.imread_collection('20150417-aukb-kd/*.tif')
con_ims = io.imread_collection('20150417-control/*.tif')
mpl.style.use('ggplot')
import itertools as it
trf_args = list(it.product(['mean_intensity', 'max_intensity'],
                           [None, 'pre', 'post']))
kwargs = [{'prop': p, 'normalise': n} for p, n in trf_args]
for kwarg in kwargs:
    title = kwarg['prop'] + '-normalise=' + str(kwarg['normalise'])
    kd = list(map(tr.trf_quantify(**kwarg), kd_ims))
    co = list(map(tr.trf_quantify(**kwarg), con_ims))
    tr.scatter(kd, co)
    plt.xlabel('image number')
    plt.ylabel('intensity'); plt.title(title)
    plt.savefig(title + '.png')
    plt.clf()
