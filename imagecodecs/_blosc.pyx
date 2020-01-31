# _blosc.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2020, Christoph Gohlke
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

"""Blosc codec for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.1.31

"""

__version__ = '2020.1.31'

include '_shared.pxi'

from blosc cimport *


class BLOSC:
    """Blosc Constants."""
    SHUFFLE = BLOSC_SHUFFLE
    NOSHUFFLE = BLOSC_NOSHUFFLE


class BloscError(RuntimeError):
    """Blosc Exceptions."""


def blosc_version():
    """Return Blosc library version string."""
    return 'blosc ' + BLOSC_VERSION_STRING.decode()


def blosc_check(data):
    """Return True if data likely contains Blosc data."""


def blosc_encode(data, level=None, compressor='blosclz', typesize=8,
                 blocksize=0, shuffle=None, numthreads=1, out=None):
    """Encode Blosc.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t blocksize_ = blocksize
        size_t typesize_ = typesize
        char* compressor_ = NULL
        int clevel = _default_value(level, 9, 0, 9)
        int doshuffle = BLOSC_SHUFFLE
        int numinternalthreads = numthreads
        int ret

    if data is out:
        raise ValueError('cannot encode in-place')

    compressor = compressor.encode()
    compressor_ = compressor

    if shuffle is not None:
        if shuffle == 'noshuffle' or shuffle == BLOSC_NOSHUFFLE:
            doshuffle = BLOSC_NOSHUFFLE
        elif shuffle == 'bitshuffle' or shuffle == BLOSC_BITSHUFFLE:
            doshuffle = BLOSC_BITSHUFFLE
        else:
            doshuffle = BLOSC_SHUFFLE

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + BLOSC_MAX_OVERHEAD
        if dstsize < 17:
            dstsize = 17
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = blosc_compress_ctx(
            clevel,
            doshuffle,
            typesize_,
            <size_t>srcsize,
            <const void*>&src[0],
            <void*>&dst[0],
            <size_t>dstsize,
            <const char*>compressor_,
            blocksize_,
            numinternalthreads
        )
    if ret <= 0:
        raise BloscError(f'blosc_compress_ctx returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc_decode(data, numthreads=1, out=None):
    """Decode Blosc.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        size_t nbytes, cbytes, blocksize
        int numinternalthreads = numthreads
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            blosc_cbuffer_sizes(
                <const void*>&src[0],
                &nbytes,
                &cbytes,
                &blocksize
            )
            if nbytes == 0 and blocksize == 0:
                raise BloscError('invalid blosc data')
            dstsize = <ssize_t>nbytes
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = blosc_decompress_ctx(
            <const void*>&src[0],
            <void*>&dst[0],
            dstsize,
            numinternalthreads
        )
    if ret < 0:
        raise BloscError(f'blosc_decompress_ctx returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)
