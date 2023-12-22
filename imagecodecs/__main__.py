# imagecodecs/__main__.py

# Copyright (c) 2019-2024, Christoph Gohlke
# This source code is distributed under the BSD 3-Clause license.

"""Imagecodecs package command line script."""

from __future__ import annotations

import sys

from matplotlib.pyplot import show
from tifffile import Timer, askopenfilename, imshow

from .imagecodecs import imread


def main(argv=None, verbose: bool = True) -> int:
    """Imagecodecs command line usage main function."""
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        filename = askopenfilename(title='Select an image file')
        if not filename:
            print('No file selected')
            return -1
    elif len(argv) == 2:
        filename = argv[1]
    else:
        print('Usage: imagecodecs filename')
        return -1

    message = ''
    timer = Timer()
    try:
        timer.start('Reading image')
        image, codec = imread(filename, return_codec=True, numthreads=0)
        print(timer)
    except ValueError as exception:
        print('failed')
        image = None
        message = str(exception)

    if verbose:
        print()
    if image is None:
        print('Could not decode the file\n')
        if verbose:
            print(message)
        return -1
    if verbose:
        print(f'{codec.__name__.upper()}: {image.shape} {image.dtype}')

    imshow(image, title=filename, interpolation='none')
    show()
    return 0


sys.exit(main())
