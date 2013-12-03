import os
import numpy as np
import mahotas as mh
from skimage import viewer
from skimage.viewer.plugins.overlayplugin import OverlayPlugin
from skimage.viewer.widgets import Slider
from skimage import segmentation as seg
import cafe


def encode(im, **kwargs):
    if kwargs.has_key('alpha'):
        a = kwargs['alpha']
        del kwargs['alpha']
    else:
        a = 100
    ov = cafe.encode_centro_telomeres_multichannel(im, **kwargs)
    return (a * (ov > 0) + int(a * 0.42) * ov).astype(np.uint8)


class CentroPlugin(OverlayPlugin):
    def __init__(self, *args, **kwargs):
        super(CentroPlugin, self).__init__(image_filter=encode, **kwargs)
        
    def attach(self, image_viewer):
        self.add_widget(Slider('alpha', 0, 100, value=100, value_type='int'))
        self.add_widget(Slider('centro_min_size', 0, 100,
                                value=10, value_type='int'))
        self.add_widget(Slider('centro_radius', 0, 100,
                                value=10, value_type='int'))
        self.add_widget(Slider('telo_offset', -10, 10, value=0))
        self.add_widget(Slider('telo_adapt_radius', 0, 101,
                                value=49, value_type='int'))
        self.add_widget(Slider('telo_open_radius', 0, 20,
                                value=4, value_type='int'))
        super(CentroPlugin, self).attach(image_viewer)


def compute_spot_stats(image, target, directory):
    v = viewer.ImageViewer(image)
    v += CentroPlugin()
    overlay = v.show()[0][0]
    overlay = seg.relabel_sequential(overlay)[0]
    mask = (overlay == 1)
    target_measurement = target[mask]
    fout_txt = os.path.join(directory, 'measure.txt')
    np.savetxt(fout_txt, target_measurement)
    fout_im = os.path.join(directory, 'mask.png')
    mh.imsave(fout_im, 64 * overlay.astype(np.uint8))
