# imagecodecs/__main__.py

# Copyright (c) Christoph Gohlke
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE file in the project root for details.

"""Imagecodecs package command line script."""

from __future__ import annotations

import sys
from typing import Any

from matplotlib.pyplot import show
from tifffile import Timer, askopenfilename, imshow

from .imagecodecs import imread


def main(argv: list[str] | None = None, *, verbose: bool = True) -> int:
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
        image, info = imread(filename, info=True, numthreads=0)
        print(timer)
    except ValueError as exception:
        print('failed')
        image = None
        message = str(exception)

    if image is None:
        if verbose:
            print(message)
        return -1
    if verbose:
        import numpy

        print_: Any
        try:
            from rich import print as print_
        except ImportError:
            from pprint import pprint as print_

        for key, value in info.items():
            if isinstance(value, bytes) and len(value) > 32:
                info[key] = value[:28] + b' ...'
            elif isinstance(value, str) and len(value) > 32:
                info[key] = value[:28] + ' ...'
            elif isinstance(value, numpy.ndarray):
                info[key] = (value.shape, value.dtype)

        print_(info)  # noqa: T203

    imshow(image, title=filename, interpolation='none')
    show()
    return 0


sys.exit(main())
