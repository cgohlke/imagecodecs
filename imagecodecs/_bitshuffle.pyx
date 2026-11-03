# imagecodecs/_bitshuffle.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2019-2026, Christoph Gohlke
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

"""BITSHUFFLE codec for the imagecodecs package."""

include '_shared.pxi'

from bitshuffle cimport *


class BITSHUFFLE:
    """BITSHUFFLE codec constants."""

    available = True


class BitshuffleError(RuntimeError):
    """BITSHUFFLE codec exceptions."""

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
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def bitshuffle_version():
    """Return Bitshuffle library version string."""
    return 'bitshuffle {}.{}.{}'.format(
        BSHUF_VERSION_MAJOR, BSHUF_VERSION_MINOR, BSHUF_VERSION_POINT
    )


def bitshuffle_check(const uint8_t[::1] data not None, /):
    """Return whether data is BITSHUFFLE encoded or None if unknown."""
    return None


def bitshuffle_encode(
    data,
    /,
    *,
    itemsize=None,
    blocksize=0,
    out=None,
):
    """Return BITSHUFFLE encoded data.

    By default, itemsize is the item size of numpy arrays, else 1.

    """
    return _bitshuffle(data, itemsize, blocksize, out, False)


def bitshuffle_decode(
    data,
    /,
    *,
    itemsize=None,
    blocksize=0,
    out=None,
):
    """Return decoded BITSHUFFLE data.

    By default, itemsize is the item size of numpy arrays, else 1.

    """
    return _bitshuffle(data, itemsize, blocksize, out, True)


cdef _bitshuffle(data, itemsize, blocksize, out, bint decode):
    """Return bit shuffled or unshuffled data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        numpy.ndarray ndarr
        void* dstptr
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t elem_size
        ssize_t block_size = blocksize
        bint isarray = isinstance(data, numpy.ndarray)
        int64_t ret

    if itemsize is None:
        elem_size = data.itemsize if isarray else 1
    else:
        elem_size = itemsize
    if elem_size < 1:
        raise ValueError(f'invalid {itemsize=}')
    if block_size < 0:
        raise ValueError(f'invalid {blocksize=}')
    if srcsize % elem_size != 0:
        raise ValueError('data size not a multiple of item size')

    if isarray:
        out = _create_array(out, data.shape, data.dtype)
        ndarr = out
        dstptr = <void*> ndarr.data
    else:
        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            if dstsize < 0:
                dstsize = srcsize
            out = _create_output(outtype, dstsize)
        dst = _inplace_input(out) if outgiven else out
        dstsize = dst.shape[0]
        if dstsize < srcsize:
            raise RuntimeError('output too small')
        dstptr = <void*> dst._data

    with nogil:
        if decode:
            ret = bshuf_bitunshuffle(
                <const void*> src._data,
                dstptr,
                <size_t> (srcsize // elem_size),
                <size_t> elem_size,
                <size_t> block_size
            )
        else:
            ret = bshuf_bitshuffle(
                <const void*> src._data,
                dstptr,
                <size_t> (srcsize // elem_size),
                <size_t> elem_size,
                <size_t> block_size
            )
    if ret < 0:
        raise BitshuffleError(
            'bshuf_bitunshuffle' if decode else 'bshuf_bitshuffle', ret
        )

    if isarray:
        return out
    del dst
    return _return_output(out, dstsize, <ssize_t> ret, outgiven)
