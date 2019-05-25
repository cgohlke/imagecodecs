# -*- coding: utf-8 -*-
# imagecodecs/_utils.py

"""Utilities for the imagecodecs package."""

from __future__ import division, print_function

__all__ = 'imread', 'imwrite'

import os

try:
    from . import _imagecodecs as imagecodecs
except ImportError:
    from . import imagecodecs


# map codec names to file name extensions or alternate names
NAMES = {
    'numpy': ('npy', 'npz'),
    'zfp': ('zfp', ),
    'png': ('png', ),
    'webp': ('webp', ),
    'jpeg': ('jpg', 'jpeg', 'jpe', 'jfif', 'jif'),
    'jpegls': ('jls', 'jpgls', 'jpegls'),
    'jxr': ('jxr', 'jpegxr', 'hdp', 'wdp'),
    'j2k': ('j2k', 'jp2', 'j2c', 'jpc', 'jpx'),
    'jpeg8': ('jpg8', 'jpeg8'),
    'jpeg12': ('jpg12', 'jpeg12'),
    'jpegsof3': ('jsof3', 'jpegsof3', 'jpeg0xc3'),
    # 'tiff': ('tif', 'tiff', 'tf8', 'tf2', 'btf')
    }

# map file extensions to codec names
CODECS = {ext: codec for codec, extens in NAMES.items() for ext in extens}


def imread(filename, codec=None, return_codec=False, **kwargs):
    """Return image data from file as numpy array."""
    codecs = []
    if codec is None:
        # find codec based on file extension
        ext = os.path.splitext(filename)[-1].lower()[1:]
        if ext in CODECS:
            codec = CODECS[ext]
            try:
                codec = getattr(imagecodecs, codec + '_decode')
                codecs.append(codec)
            except AttributeError:
                raise ValueError('invalid codec %s' % str(codec))
        # try all other codecs
        for c in (imagecodecs.png_decode,
                  imagecodecs.jpeg8_decode,
                  imagecodecs.jpeg12_decode,
                  imagecodecs.jpegsof3_decode,
                  imagecodecs.jpegls_decode,
                  imagecodecs.j2k_decode,
                  imagecodecs.jxr_decode,
                  imagecodecs.webp_decode,
                  imagecodecs.zfp_decode,
                  imagecodecs.numpy_decode):
            if c is not codec:
                codecs.append(c)
    else:
        # use provided codecs
        if not isinstance(codec, (list, tuple)):
            codec = [codec]
        for c in codec:
            if isinstance(c, str):
                c = CODECS.get(c, c)
                try:
                    c = getattr(imagecodecs, c.lower() + '_decode')
                except AttributeError:
                    raise ValueError('invalid codec %s' % str(c))
            elif not callable(c):
                raise ValueError('invalid codec %s' % str(c))
            codecs.append(c)

    with open(filename, 'rb') as fh:
        data = fh.read()

    exceptions = []
    image = None
    for codec in codecs:
        try:
            image = codec(data, **kwargs)
            if image.dtype == 'object':
                image = None
                raise ValueError('failed')
            break
        except Exception as exception:
            # raise(exception)
            exceptions.append('%s: %s' % (codec.__name__.upper(), exception))

    if image is None:
        raise ValueError('\n'.join(exceptions))

    if return_codec:
        return image, codec
    return image


def imwrite(filename, data, codec=None, **kwargs):
    """Write numpy array to image file."""
    if codec is None:
        # find codec based on file extension
        ext = os.path.splitext(filename)[-1].lower()[1:]
        if ext in CODECS:
            codec = CODECS[ext]
            try:
                codec = getattr(imagecodecs, codec + '_encode')
            except AttributeError:
                raise ValueError('invalid codec %s' % str(codec))
        else:
            raise ValueError('no codec found')
    elif isinstance(codec, str):
        try:
            codec = getattr(imagecodecs, codec.lower() + '_encode')
        except AttributeError:
            raise ValueError('invalid codec %s' % str(codec))
    elif not callable(codec):
        raise ValueError('invalid codec %s' % str(codec))

    data = codec(data, **kwargs)
    with open(filename, 'wb') as fh:
        fh.write(data)
