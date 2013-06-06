#!/usr/bin/env python

# standard library imports
import sys
import argparse

# dependency library imports
import mahotas

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
    parser.parse_args()
    pass
