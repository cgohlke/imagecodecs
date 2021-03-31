# imagecodecs/_zopfli.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""Zopfli codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from zopfli cimport *


class ZOPFLI:
    """Zopfli Constants."""

    FORMAT_GZIP = ZOPFLI_FORMAT_GZIP
    FORMAT_ZLIB = ZOPFLI_FORMAT_ZLIB
    FORMAT_DEFLATE = ZOPFLI_FORMAT_DEFLATE


class ZopfliError(RuntimeError):
    """Zopfli Exceptions."""


def zopfli_version():
    """Return Zopfli library version string."""
    return 'zopfli 1.0.3'


# zopfli_check = zlib_check
# zopfli_decode = zlib_decode


def zopfli_encode(data, level=None, out=None, **kwargs):
    """Compress Zlib format using Zopfli.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t outsize = 0
        ZopfliOptions options
        ZopfliFormat format = ZOPFLI_FORMAT_ZLIB
        unsigned char* buffer = NULL

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    ZopfliInitOptions(&options)
    if kwargs:
        if 'format' in kwargs:
            format = <ZopfliFormat> <int> (
                _default_value(kwargs['format'], 1, 0, 2)
            )
        if 'verbose' in kwargs:
            options.verbose = bool(kwargs['verbose'])
        if 'verbose_more' in kwargs:
            options.verbose_more = bool(kwargs['verbose_more'])
        if 'numiterations' in kwargs:
            options.numiterations = _default_value(
                kwargs['numiterations'], 15, 1, 255
            )
        if 'blocksplitting' in kwargs:
            options.blocksplitting = bool(kwargs['blocksplitting'])
        if 'blocksplittingmax' in kwargs:
            options.blocksplittingmax = _default_value(
                kwargs['blocksplittingmax'], 15, 0, 2 ** 15 - 1
            )

    with nogil:
        ZopfliCompress(
            &options,
            format,
            <const unsigned char*> &src[0],
            <size_t> srcsize,
            &buffer,
            &outsize
        )
    if buffer == NULL:
        raise ZopfliError('ZopfliCompress returned NULL')

    try:
        if out is None:
            if dstsize >= 0 and dstsize < <ssize_t> outsize:
                raise RuntimeError('output too small')
            dstsize = <ssize_t> outsize
            out = _create_output(outtype, dstsize, <const char*> buffer)
        else:
            dst = out
            dstsize = dst.size
            if dstsize < <ssize_t> outsize:
                raise RuntimeError('output too small')
            memcpy(<void*> &dst[0], <const void*> buffer, outsize)
            del dst
    finally:
        free(buffer)

    return _return_output(out, dstsize, outsize, outgiven)
