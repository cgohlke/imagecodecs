# imagecodecs/_bitshuffle.pyx
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

"""Bitshuffle codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from bitshuffle cimport *


class BITSHUFFLE:
    """Bitshuffle Constants."""


class BitshuffleError(RuntimeError):
    """Bitshuffle Exceptions."""

    def __init__(self, func, err):
        msg = {
            0: 'No Error',
            -1: 'Failed to allocate memory',
            -11: 'Missing SSE',
            -12: 'Missing AVX',
            -80: 'Input size not a multiple of 8',
            -81: 'Block size not a multiple of 8',
            -91: 'Decompression error, wrong number of bytes processed',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def bitshuffle_version():
    """Return Bitshuffle library version string."""
    return 'bitshuffle {}.{}.{}'.format(
        BSHUF_VERSION_MAJOR, BSHUF_VERSION_MINOR, BSHUF_VERSION_POINT
    )


def bitshuffle_check(data):
    """Return True if data likely contains Bitshuffle data."""


def bitshuffle_encode(data, level=None, itemsize=1, blocksize=0, out=None):
    """Bitshuffle.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        numpy.ndarray ndarr
        ssize_t srcsize
        ssize_t dstsize
        size_t elem_size
        size_t block_size = blocksize
        int64_t ret

    if data is out:
        raise ValueError('cannot encode in-place')

    if isinstance(data, numpy.ndarray):
        out = _create_array(out, data.shape, data.dtype)
        ndarr = out
        srcsize = data.size
        elem_size = <size_t> data.itemsize
        with nogil:
            ret = bshuf_bitshuffle(
                <void*> &src[0],
                <void*> ndarr.data,
                <size_t> srcsize,
                elem_size,
                block_size
            )
        if ret < 0:
            raise BitshuffleError('bshuf_bitshuffle', ret)
        return out

    srcsize = src.size
    elem_size = itemsize

    if elem_size != 1 and elem_size != 2 and elem_size != 4 and elem_size != 8:
        raise ValueError('invalid item size')
    if srcsize % elem_size != 0:
        raise ValueError('data size not a multiple of item size')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dstsize < srcsize:
        raise RuntimeError('output too small')

    with nogil:
        ret = bshuf_bitshuffle(
            <void*> &src[0],
            <void*> &dst[0],
            <size_t> srcsize / elem_size,
            elem_size,
            block_size
        )
    if ret < 0:
        raise BitshuffleError('bshuf_bitshuffle', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> ret, outgiven)


def bitshuffle_decode(data, itemsize=1, blocksize=0, out=None):
    """Un-Bitshuffle.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        numpy.ndarray ndarr
        ssize_t srcsize
        ssize_t dstsize
        size_t elem_size
        size_t block_size = blocksize
        int64_t ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if isinstance(data, numpy.ndarray):
        out = _create_array(out, data.shape, data.dtype)
        ndarr = out
        srcsize = data.size
        elem_size = <size_t> data.itemsize
        with nogil:
            ret = bshuf_bitunshuffle(
                <void*> &src[0],
                <void*> ndarr.data,
                <size_t> srcsize,
                elem_size,
                block_size
            )
        if ret < 0:
            raise BitshuffleError('bshuf_bitunshuffle', ret)
        return out

    srcsize = src.size
    elem_size = itemsize

    if elem_size != 1 and elem_size != 2 and elem_size != 4 and elem_size != 8:
        raise ValueError('invalid item size')
    if srcsize % elem_size != 0:
        raise ValueError('data size not a multiple of item size')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dstsize < srcsize:
        raise RuntimeError('output too small')

    with nogil:
        ret = bshuf_bitunshuffle(
            <void*> &src[0],
            <void*> &dst[0],
            <size_t> srcsize / elem_size,
            elem_size,
            block_size
        )
    if ret < 0:
        raise BitshuffleError('bshuf_bitunshuffle', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> ret, outgiven)
