#!/usr/bin/env python

# standard library imports
import os
import sys
import argparse
import itertools as it

# dependency library imports
import numpy as np
import mahotas as mh
from mahotas import io
from matplotlib import pyplot as plt
from skimage import viewer

# local imports
import cafe


def strip_extension(fn):
    """Remove the file extension from the input filename.

    Parameters
    ----------
    fn : string
        The input filename.

    Returns
    -------
    fn_strip : string
        The input filename without the extension.
    """
    fn_strip = os.path.splitext(fn)[0]
    return fn_strip


parser = argparse.ArgumentParser(description=
                                 "Chromatin-associated fluorescence estimator")
subpar = parser.add_subparsers()

centro = subpar.add_parser('centro', help="Compare centromeric intensity "
                           "against chromatin background.")
centro.add_argument('-t', '--test-cases', nargs='+', required=True,
                    help="Filenames of test images.")
centro.add_argument('-c', '--controls', nargs='+', required=True,
                    help="Filenames of control images.")
centro.add_argument('-H', '--save-chromatin', action='store_true',
                    default=False, help='Store segmented chromatin regions.')
centro.add_argument('-C', '--save-centromeres', action='store_true',
                    default=False, help='Store segmented centromere regions.')
centro.add_argument('-o', '--output-file', default='boxplot.pdf',
                    help='The name of the output file.')


telo = subpar.add_parser('interactive', help="Quantify fluorescence on "
                         "telomeres but not centromeres.")
telo.add_argument('directories', nargs='+', metavar='DIR', help=
                  "Input directories, containing four .tif files each: "
                  "target (AuKB), telomere, centromere, and DAPI.")


def get_command(argv):
    """Return the command name used in the command line call.

    Parameters
    ----------
    argv : list of string
        The argument vector.

    Returns
    -------
    cmd : string
        The command name.
    """
    return argv[1]


def main():
    """Run the command-line interface."""
    args = parser.parse_args()
    cmd = get_command(sys.argv)
    if cmd == 'centro':
        run_centro(args)
    elif cmd == 'interactive':
        run_interactive(args)


def run_centro(args):
    """Run the program on some input images and produce statistics and plots.

    Use `cafe -h` or `cafe --help` for options.
    """
    test_images = [io.imread(fn) for fn in args.test_cases]
    test_rnapii = it.chain(*[cafe.rnapii_centromere_vs_chromatin(im)
                             for im in test_images])
    control_images = [io.imread(fn) for fn in args.controls]
    control_rnapii = it.chain(*[cafe.rnapii_centromere_vs_chromatin(im)
                                for im in control_images])
    if args.save_chromatin:
        chromatin_image_fns = map(strip_extension,
                                  args.test_cases + args.controls)
        chromatin_images = [cafe.get_chromatin(im[..., 2]) for
                            im in test_images + control_images]
        for fn, im in zip(chromatin_image_fns, chromatin_images):
            io.imsave(fn + '_chromatin.tif', 255 * im.astype(np.uint8))
    if args.save_centromeres:
        centromere_image_fns = map(strip_extension,
                                   args.test_cases + args.controls)
        centromere_images = [cafe.get_centromere_neighbourhood(im[..., 1]) for
                             im in test_images + control_images]
        for fn, im in zip(centromere_image_fns, centromere_images):
            io.imsave(fn + '_centromere.tif', 255 * im.astype(np.uint8))

    plt.boxplot(list(test_rnapii) + list(control_rnapii))
    plt.savefig(args.output_file, bbox_inches='tight')


def run_interactive(args):
    from interactive import CentroPlugin
    from skimage import segmentation as seg
    image_files_list = [filter(lambda x: x.lower().endswith('.tif'),
                               os.listdir(d)) for d in args.directories]
    images = [map(mh.imread, image_files) for image_files in image_files_list]
    targets = [im[0] for im in images]
    rgbs = [np.dstack(ims[1:]) for ims in images]
    for s, target, rgb in zip(args.directories, targets, rgbs):
        if rgb.shape[-1] == 9:
            # some single channel images are written out as RGB...
            rgb = rgb[:, :, ::3]
        v = viewer.ImageViewer(rgb)
        v += CentroPlugin()
        overlay = v.show()[0][0]
        overlay = seg.relabel_sequential(overlay)[0]
        mask = (overlay == 1)
        target_measurement = target[mask]
        fout_txt = os.path.join(s, 'measure.txt')
        np.savetxt(fout_txt, target_measurement)
        fout_im = os.path.join(s, 'mask.png')
        mh.imsave(fout_im, 64 * overlay.astype(np.uint8))


if __name__ == '__main__':
    main()
