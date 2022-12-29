# imagecodecs/_rcomp.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021-2023, Christoph Gohlke
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

"""Rcomp codec for the imagecodecs package.

Rcomp is an implementation of the Rice algorithm:

Robert Rice, Pen-Shu Yeh, and Warner Miller. Algorithms for high-speed
universal noiseless coding. Proc. of the 9th AIAA Computing in Aerospace.
Conf. AIAA-93-4541-CP, 1993. https://doi.org/10.2514/6.1993-4541

"""

__version__ = '2022.2.22'

include '_shared.pxi'

from cfitsio cimport *


class RCOMP:
    """Rcomp Constants."""


class RcompError(RuntimeError):
    """Rcomp Exceptions."""

    def __init__(self, func):
        cdef:
            char[80] err_message

        msg = [f'{func} failed']
        while ffgmsg(err_message) > 0:
            pstr = <bytes> err_message
            msg.append(pstr.strip())
        super().__init__('\n  '.join(msg))


def rcomp_version():
    """Return cfitsio library version string."""
    return f'cfitsio {CFITSIO_MAJOR}.{CFITSIO_MINOR}'


def rcomp_check(data):
    """Return True if data likely contains Rcomp compressed data."""
    return False


def rcomp_encode(data, level=None, nblock=None, numthreads=None, out=None):
    """Compress Rcomp.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        numpy.dtype dtype = data.dtype
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        int nblock_ = 32 if nblock is None else nblock
        int ret = 0

    if not (
        srcsize < 2147483648
        and dtype.kind in (b'i', b'u')
        and dtype.itemsize in (1, 2, 4)
    ):
        raise ValueError(
            'data is not a numpy integers array of size < 2*31'
        )

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = max(1024, <ssize_t> (<double> src.nbytes * 1.11))
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    if dstsize > 2147483647:
        raise ValueError('output too large')

    if dtype.itemsize == 1:
        with nogil:
            ret = fits_rcomp_byte(
                <signed char*> src.data,
                <int> srcsize,
                <unsigned char*> &dst[0],
                <int> dstsize,
                nblock_
            )
        if ret < 0:
            raise RcompError('fits_rcomp_byte')

    elif dtype.itemsize == 2:
        with nogil:
            ret = fits_rcomp_short(
                <signed short*> src.data,
                <int> srcsize,
                <unsigned char*> &dst[0],
                <int> dstsize,
                nblock_
            )
        if ret < 0:
            raise RcompError('fits_rcomp_short')

    elif dtype.itemsize == 4:
        with nogil:
            ret = fits_rcomp(
                <signed int*> src.data,
                <int> srcsize,
                <unsigned char*> &dst[0],
                <int> dstsize,
                nblock_
            )
        if ret < 0:
            raise RcompError('fits_rcomp')

    else:
        raise RuntimeError

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def rcomp_decode(
    data, shape=None, dtype=None, nblock=None, numthreads=None, out=None
):
    """Decompress Rcomp.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = <size_t> src.size
        ssize_t dstsize
        int ret = 0
        int nblock_ = 32 if nblock is None else nblock

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > 2147483647:
        raise ValueError('input buffer too large')

    if out is not None and isinstance(out, numpy.ndarray):
        if shape is None:
            shape = out.shape
        if dtype is None:
            dtype = out.dtype
    elif dtype is None or shape is None:
        raise ValueError('missing shape or dtype')
    else:
        dtype = numpy.dtype(dtype)
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (int(shape), )

    if not (dtype.kind in 'iu' and dtype.itemsize in (1, 2, 4)):
        raise ValueError('invalid dtype')

    out = _create_array(out, shape, dtype)
    dst = out
    dstsize = dst.size
    if dstsize > 2147483647:
        raise ValueError('output array too large')

    if dtype.itemsize == 1:
        with nogil:
            ret = fits_rdecomp_byte(
                <unsigned char*> &src[0],
                <int> srcsize,
                <unsigned char*> dst.data,
                <int> dstsize,
                nblock_
            )
        if ret != 0:
            raise RcompError('fits_rdecomp_byte')

    elif dtype.itemsize == 2:
        with nogil:
            ret = fits_rdecomp_short(
                <unsigned char*> &src[0],
                <int> srcsize,
                <unsigned short*> dst.data,
                <int> dstsize,
                nblock_
            )
        if ret != 0:
            raise RcompError('fits_rdecomp_short')

    elif dtype.itemsize == 4:
        with nogil:
            ret = fits_rdecomp(
                <unsigned char*> &src[0],
                <int> srcsize,
                <unsigned int*> dst.data,
                <int> dstsize,
                nblock_
            )
        if ret != 0:
            raise RcompError('fits_rdecomp')

    else:
        raise RuntimeError

    del dst
    return out
