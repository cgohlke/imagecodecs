# -*- coding: utf-8 -*-
# imagecodecs/__main__.py

"""Imagecodecs package command line script."""

import sys

from matplotlib.pyplot import show

from tifffile import imshow

import imagecodecs


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


def main(argv=None, verbose=True, decoders=None):
    """Imagecodecs command line usage main function."""
    if argv is None:
        argv = sys.argv

    if len(argv) < 2:
        fname = askopenfilename(title='Select a. image file')
        if not fname:
            print('No file selected')
            return -1
    elif len(argv) == 2:
        fname = argv[1]
    else:
        print('Usage: imagecodecs filename')
        return -1

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
            imagecodecs.zfp_decode,
            imagecodecs.numpy_decode,
        ]

    messages = []
    image = None
    for decode in decoders:
        try:
            image = decode(data)
            if image.dtype == 'object':
                image = None
                raise ValueError('failed')
        except Exception as exception:
            # raise(exception)
            messages.append('%s: %s' % (decode.__name__.upper(), exception))
            continue
        break

    if verbose:
        print()
    if image is None:
        print('Could not decode the file\n')
        if verbose:
            for message in messages:
                print(message)
        return -1
    if verbose:
        print("%s: %s %s" % (decode.__name__.upper(),
                             image.shape, image.dtype))

    imshow(image, title=fname)
    show()
    return 0


sys.exit(main())
