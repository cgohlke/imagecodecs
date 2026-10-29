# imagecodecs/_pixarlog.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""PixarLog codec for the imagecodecs package.

PixarLog is a lossless (for 8-bit) / slightly lossy (for 16-bit and float)
logarithmic compression scheme developed by Pixar for TIFF images.

"""

include '_shared.pxi'

from pixarlog cimport *

from zlib import Z_BEST_COMPRESSION, Z_DEFAULT_COMPRESSION

pixarlog_init()


class PIXARLOG:
    """PixarLog codec constants."""

    available = True


class PixarlogError(RuntimeError):
    """PixarLog codec exceptions."""

    def __init__(self, func, err):
        msg = {
            PIXARLOG_OK: 'PIXARLOG_OK',
            PIXARLOG_ERROR: 'error',
            PIXARLOG_MEMORY_ERROR: 'insufficient memory',
            PIXARLOG_VALUE_ERROR: 'invalid value',
            PIXARLOG_OUTPUT_TOO_SMALL: 'output too small',
            PIXARLOG_ZLIB_ERROR: 'zlib error',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def pixarlog_version():
    """Return pixarlog library version string."""
    return 'pixarlog ' + PIXARLOG_VERSION.decode()


def pixarlog_check(const uint8_t[::1] data not None, /):
    """Return whether data is PixarLog encoded or None if unknown."""
    return None if data.shape[0] > 1 else False


def pixarlog_encode(
    data,
    /,
    level=None,
    *,
    deflate=True,
    byteorder=None,
    out=None,
):
    """Return PixarLog encoded image.

    The 11-bit log codes are stored as uint16 in `byteorder`
    (default native).

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.nbytes
        ssize_t dstsize, width, stride, ret
        int datafmt
        int byteswap = _byteswap(byteorder)
        int clevel = _default_value(
            level, Z_DEFAULT_COMPRESSION, -1, Z_BEST_COMPRESSION
        )

    if not (src.ndim == 2 or src.ndim == 3):
        raise ValueError('data must be a 2D or 3D array')

    if src.ndim == 3:
        width = <ssize_t> src.shape[1]
        stride = <ssize_t> src.shape[2]
    else:
        width = <ssize_t> src.shape[1]
        stride = 1

    if src.dtype == numpy.uint8:
        datafmt = PIXARLOG_FMT_8BIT
    elif src.dtype == numpy.uint16:
        datafmt = PIXARLOG_FMT_16BIT
    elif src.dtype == numpy.float32:
        datafmt = PIXARLOG_FMT_FLOAT
    else:
        raise ValueError(
            f'dtype {src.dtype!r} not in {{uint8, uint16, float32}}'
        )

    if not numpy.PyArray_ISALIGNED(src):
        src = src.copy()

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 2 // src.itemsize
            if deflate:
                dstsize = _compress_bound(dstsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    if deflate:
        with nogil:
            ret = pixarlog_encode_(
                <const uint8_t*> src.data,
                srcsize,
                <uint8_t*> dst._data,
                dstsize,
                width,
                stride,
                datafmt,
                clevel,
                byteswap,
            )
    else:
        with nogil:
            ret = pixarlog_encode_raw_(
                <const uint8_t*> src.data,
                srcsize,
                <uint8_t*> dst._data,
                dstsize,
                width,
                stride,
                datafmt,
                byteswap,
            )

    if ret < 0:
        raise PixarlogError(
            'pixarlog_encode' if deflate else 'pixarlog_encode_raw',
            <int> ret,
        )

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def pixarlog_decode(
    data,
    /,
    *,
    shape=None,
    dtype=None,
    deflate=True,
    byteorder=None,
    out=None,
):
    """Return decoded PixarLog image.

    Rows missing from the encoded data are zero-filled.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, width, stride, ret
        int datafmt
        int byteswap = _byteswap(byteorder)
        numpy.dtype outdtype

    if data is out:
        raise ValueError('cannot transform in-place')

    if out is not None and isinstance(out, numpy.ndarray):
        if shape is None:
            shape = out.shape
        if dtype is None:
            dtype = out.dtype

    if dtype is not None:
        outdtype = numpy.dtype(dtype)
        if outdtype == numpy.uint8:
            datafmt = PIXARLOG_FMT_8BIT
        elif outdtype == numpy.float32:
            datafmt = PIXARLOG_FMT_FLOAT
        elif outdtype == numpy.uint16:
            datafmt = PIXARLOG_FMT_16BIT
        else:
            raise ValueError(
                f'dtype={outdtype!r} not in {{uint8, uint16, float32}}'
            )
    else:
        datafmt = PIXARLOG_FMT_16BIT
        outdtype = numpy.dtype(numpy.uint16)

    if shape is None:
        raise TypeError('shape is required')
    shape = tuple(shape)
    if len(shape) == 3:
        width = <ssize_t> shape[1]
        stride = <ssize_t> shape[2]
    elif len(shape) == 2:
        width = <ssize_t> shape[1]
        stride = 1
    else:
        raise ValueError(
            'shape must be 2D (height, width) or 3D (height, width, channels)'
        )

    out = _create_array(out, shape, outdtype)
    dst = out
    if dst.dtype != outdtype:
        raise ValueError(f'invalid output dtype {dst.dtype}')
    if not numpy.PyArray_ISALIGNED(dst):
        raise ValueError('output array is not aligned')
    dstsize = dst.nbytes

    if deflate:
        with nogil:
            ret = pixarlog_decode_(
                <const uint8_t*> src._data,
                srcsize,
                <uint8_t*> dst.data,
                dstsize,
                width,
                stride,
                datafmt,
                byteswap,
            )
    else:
        with nogil:
            ret = pixarlog_decode_raw_(
                <const uint8_t*> src._data,
                srcsize,
                <uint8_t*> dst.data,
                dstsize,
                width,
                stride,
                datafmt,
                byteswap,
            )

    if ret < 0:
        raise PixarlogError(
            'pixarlog_decode' if deflate else 'pixarlog_decode_raw',
            <int> ret,
        )

    if ret < dstsize:
        memset(dst.data + ret, 0, dstsize - ret)

    return out


cdef int _byteswap(byteorder) except -1:
    # return whether uint16 codes must be byte-swapped
    if byteorder is None or byteorder == '=':
        return 0
    if byteorder not in {'<', '>'}:
        raise ValueError(f'invalid {byteorder=!r}')
    return not numpy.dtype(byteorder + 'u2').isnative


cdef ssize_t _compress_bound(const ssize_t srcsize) noexcept nogil:
    # replacement for zlib compressBound
    return srcsize + (srcsize >> 12) + (srcsize >> 14) + (srcsize >> 25) + 13
