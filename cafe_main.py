#!/usr/bin/env python

# standard library imports
import os
import argparse
import itertools as it

# dependency library imports
import numpy as np
from mahotas import io
from matplotlib import pyplot as plt

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
                            "Centromere-associated fluorescence estimator.")
parser.add_argument('-t', '--test-cases', nargs='+', required=True,
                    help="Filenames of test images.")
parser.add_argument('-c', '--controls', nargs='+', required=True,
                    help="Filenames of control images.")
parser.add_argument('-H', '--save-chromatin', action='store_true',
                    default=False, help='Store segmented chromatin regions.')
parser.add_argument('-C', '--save-centromeres', action='store_true',
                    default=False, help='Store segmented centromere regions.')
parser.add_argument('-o', '--output-file', default='boxplot.pdf',
                    help='The name of the output file.')


def main():
    """Run the program on some input images and produce statistics and plots.

    Use `cafe -h` or `cafe --help` for options.
    """
    args = parser.parse_args()
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
            io.imsave(fn + '_chromatin.tif', im.astype(np.uint8))
    if args.save_centromeres:
        centromere_image_fns = map(strip_extension,
                                   args.test_cases + args.controls)
        centromere_images = [cafe.get_centromere_neighbourhood(im[..., 1]) for
                             im in test_images + control_images]
        for fn, im in zip(centromere_image_fns, centromere_images):
            io.imsave(fn + '_centromere.tif', im.astype(np.uint8))

    plt.boxplot(list(test_rnapii) + list(control_rnapii))
    plt.savefig(args.output_file, bbox_inches='tight')


if __name__ == '__main__':
    main()
