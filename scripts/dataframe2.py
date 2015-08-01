import pandas as pd
from skimage import io
import trf1 as tr
import matplotlib as mpl
direct = '/Volumes/BLACK HOLE/201507 aukbgfp files/'
dnkd_ims = io.imread_collection(direct + '*huAUKBDNKD*.tif')
pegfp_ims = io.imread_collection(direct + '*pegfp*.tif')
wt_ims = io.imread_collection(direct + '*huAUKBDWT*.tif')
mpl.style.use('ggplot')
colnames = ['filename', 'file-number', 'aukb-kd', 'size(pixels)',
            'raw-mean', 'raw-total', 'raw-max', 'pre-mean', 'pre-total',
            'pre-max', 'post-mean', 'post-total', 'post-max', 'eccentricity']
kds = map(tr.trf_quantify, dnkd_ims)
cos = map(tr.trf_quantify, pegfp_ims)
wts = map(tr.trf_quantify, wt_ims)
result = []
for i, (kd_fn, kd) in enumerate(zip(dnkd_ims.files, kds)):
    for blob_data in kd:
        result.append([kd_fn, i, 'kd'] + list(blob_data))
for j, (con_fn, co) in enumerate(zip(pegfp_ims.files, cos),
                                 start=len(dnkd_ims)):
    for blob_data in co:
        result.append([con_fn, j, 'con'] + list(blob_data))
for k, (wt_fn, wt) in enumerate(zip(wt_ims.files, wts),
                                start=len(dnkd_ims) + len(pegfp_ims)):
    for blob_data in wt:
        result.append([wt_fn, k, 'wt'] + list(blob_data))
df = pd.DataFrame(result, columns=colnames)
df.to_csv('full-dataset201507gfp.csv')
