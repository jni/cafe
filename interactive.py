import numpy as np
from skimage.viewer.plugins.overlayplugin import OverlayPlugin
from skimage.viewer.widgets import Slider
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
                                value=14, value_type='int'))
        self.add_widget(Slider('centro_radius', 0, 100,
                                value=10, value_type='int'))
        self.add_widget(Slider('telo_offset', -256, 255, value=0))
        self.add_widget(Slider('telo_adapt_radius', 0, 101,
                                value=49, value_type='int'))
        self.add_widget(Slider('telo_open_radius', 0, 20,
                                value=4, value_type='int'))
        super(CentroPlugin, self).attach(image_viewer)


