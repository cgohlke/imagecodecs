# -*- coding: utf-8 -*-
# imagecodecs/__main__.py

# Copyright (c) 2019, Christoph Gohlke
# This source code is distributed under the 3-clause BSD license.

"""Imagecodecs package command line script."""

import sys

from matplotlib.pyplot import show

from tifffile import imshow

from ._utils import imread


def askopenfilename(**kwargs):
    """Return file name(s) from Tkinter's file open dialog."""
    try:
        from Tkinter import Tk
        import tkFileDialog as filedialog
    except ImportError:
        from tkinter import Tk, filedialog
    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main(argv=None, verbose=True, codec=None):
    """Imagecodecs command line usage main function."""
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        filename = askopenfilename(title='Select a image file')
        if not filename:
            print('No file selected')
            return -1
    elif len(argv) == 2:
        filename = argv[1]
    else:
        print('Usage: imagecodecs filename')
        return -1

    try:
        image, codec = imread(filename, return_codec=True)
    except ValueError as exception:
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
        print("%s: %s %s" % (codec.__name__.upper(), image.shape, image.dtype))

    imshow(image, title=filename)
    show()
    return 0


sys.exit(main())
