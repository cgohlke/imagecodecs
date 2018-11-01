# -*- coding: utf-8 -*-
# imagecodecs/__main__.py

"""Command line script for the imagecodecs package."""

import sys

from matplotlib.pyplot import show

from tifffile import imshow

import imagecodecs


def main(argv=None, verbose=True, decoders=None):
    """Imagecodecs command line usage main function."""
    if argv is None:
        argv = sys.argv

    if len(argv) != 2:
        print('Usage: imagecodecs filename')
        return -1

    fname = argv[1]

    with open(fname, 'rb') as fh:
        data = fh.read()

    if decoders is None:
        decoders = [
            imagecodecs.png_decode,
            imagecodecs.jpeg8_decode,
            imagecodecs.jpeg12_decode,
            imagecodecs.jpegsof3_decode,
            imagecodecs.jpegls_decode,
            imagecodecs.j2k_decode,
            imagecodecs.jxr_decode,
            imagecodecs.webp_decode,
        ]

    image = None
    for decode in decoders:
        if verbose:
            print()
            print(decode.__name__)
        try:
            image = decode(data)
            if image.dtype == 'object':
                image = None
                raise ValueError('failed')
        except Exception as exception:
            # raise(exception)
            if verbose:
                print(' ', exception)
            continue
        break

    if image is None:
        if verbose:
            print()
            print('Could not decode the file')
        return -1
    if verbose:
        print(' ', image.shape, image.dtype)

    imshow(image, title=fname)
    show()
    return 0


sys.exit(main())
