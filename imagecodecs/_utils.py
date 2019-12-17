# -*- coding: utf-8 -*-
# imagecodecs/_utils.py

# Copyright (c) 2018-2019, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Utilities for the imagecodecs package."""

from __future__ import division, print_function

__all__ = ('imread', 'imwrite')

import os
import pathlib

try:
    from . import _imagecodecs as imagecodecs
except ImportError:
    from . import imagecodecs


def _names(_names_={}):
    """Return map of codec names to file name extensions or alternate names."""
    # defer creation of dict until it is needed
    if not _names_:
        _names_.update({
            'numpy': ('numpy', 'npy', 'npz'),
            'zfp': ('zfp', ),
            'png': ('png', ),
            'webp': ('webp', ),
            'jpeg': ('jpg', 'jpeg', 'jpe', 'jfif', 'jif'),
            'jpegls': ('jls', 'jpgls', 'jpegls'),
            'jpegxl': ('jxl', 'brn', 'jpgxl', 'jpegxl'),
            'jpegxr': ('jxr', 'hdp', 'wdp', 'jpegxr'),
            'jpeg2k': ('j2k', 'jp2', 'j2c', 'jpc', 'jpx', 'jpeg2k'),
            'jpeg8': ('jpg8', 'jpeg8'),
            'jpeg12': ('jpg12', 'jpeg12'),
            'jpegsof3': ('jsof3', 'jpegsof3', 'jpeg0xc3'),
            # 'tiff': ('tif', 'tiff', 'tf8', 'tf2', 'btf')
        })
    return _names_


def _codecs(_codecs_={}):
    """Return map of file extensions to codec names."""
    if not _codecs_:
        _codecs_.update(
            {ext: codec for codec, exts in _names().items() for ext in exts}
        )
    return _codecs_


def imread(fileobj, codec=None, return_codec=False, **kwargs):
    """Return image data from file as numpy array."""
    codecs = []
    if codec is None:
        # find codec based on file extension
        if isinstance(fileobj, (str, pathlib.Path)):
            ext = os.path.splitext(str(fileobj))[-1].lower()[1:]
        else:
            ext = None
        if ext in _codecs():
            codec = _codecs()[ext]
            try:
                codec = getattr(imagecodecs, codec + '_decode')
                codecs.append(codec)
            except AttributeError:
                raise ValueError('invalid codec %s' % str(codec))
        # try all other codecs
        for c in (
            imagecodecs.png_decode,
            imagecodecs.jpeg8_decode,
            imagecodecs.jpeg12_decode,
            imagecodecs.jpegsof3_decode,
            imagecodecs.jpegls_decode,
            imagecodecs.jpegxl_decode,
            imagecodecs.jpeg2k_decode,
            imagecodecs.jpegxr_decode,
            imagecodecs.webp_decode,
            imagecodecs.zfp_decode,
            imagecodecs.numpy_decode
        ):
            if c is not codec:
                codecs.append(c)
    else:
        # use provided codecs
        if not isinstance(codec, (list, tuple)):
            codec = [codec]
        for c in codec:
            if isinstance(c, str):
                c = _codecs().get(c, c)
                try:
                    c = getattr(imagecodecs, c.lower() + '_decode')
                except AttributeError:
                    raise ValueError('invalid codec %s' % str(c))
            elif not callable(c):
                raise ValueError('invalid codec %s' % str(c))
            codecs.append(c)

    if hasattr(fileobj, 'read'):
        # binary stream: open file, BytesIO
        data = fileobj.read()
    elif isinstance(fileobj, (str, pathlib.Path)):
        # file name
        fileobj = str(fileobj)
        with open(fileobj, 'rb') as fh:
            data = fh.read()
        # TODO: support urllib.request.urlopen ?
    else:
        # binary data
        data = fileobj

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


def imwrite(fileobj, data, codec=None, **kwargs):
    """Write numpy array to image file."""
    if codec is None:
        # find codec based on file extension
        if isinstance(fileobj, (str, pathlib.Path)):
            ext = os.path.splitext(str(fileobj))[-1].lower()[1:]
        else:
            ext = None
        if ext in _codecs():
            codec = _codecs()[ext]
            try:
                codec = getattr(imagecodecs, codec + '_encode')
            except AttributeError:
                raise ValueError('invalid codec %s' % str(codec))
        else:
            raise ValueError('no codec specified')
    elif isinstance(codec, str):
        codec = _codecs().get(codec, codec)
        try:
            codec = getattr(imagecodecs, codec.lower() + '_encode')
        except AttributeError:
            raise ValueError('invalid codec %s' % str(codec))
    elif not callable(codec):
        raise ValueError('invalid codec %s' % str(codec))

    data = codec(data, **kwargs)
    if hasattr(fileobj, 'write'):
        # binary stream: open file, BytesIO
        fileobj.write(data)
    else:
        # file name
        with open(str(fileobj), 'wb') as fh:
            fh.write(data)
