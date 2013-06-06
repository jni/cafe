#!/usr/bin/env python

# standard library imports
import argparse
import itertools as it

# dependency library imports
from mahotas import io
from matplotlib import pyplot as plt

# local imports
import cafe

parser = argparse.ArgumentParser(description=
                            "Centromere-associated fluorescence estimator.")
parser.add_argument('-t', '--test-cases', nargs='+', required=True,
                    help="Filenames of test images.")
parser.add_argument('-c', '--controls', nargs='+', required=True,
                    help="Filenames of control images.")


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

    plt.boxplot(list(test_rnapii) + list(control_rnapii))
    plt.savefig('boxplot.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
