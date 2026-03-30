# imagecodecs/_hcomp.pyx
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

"""HCOMP codec for the imagecodecs package.

Hcomp is an implementation of the H-compress algorithm:

    R. L. White.  Compression of astronomical images.
    Bulletin of the Astronomical Society, 24:1135, 1992.

"""

include '_shared.pxi'

from hcompress cimport *


class HCOMP:
    """HCOMP codec constants."""

    available = True


class HcompError(RuntimeError):
    """HCOMP codec exceptions."""

    def __init__(self, func, err):
        msg = {
            HCOMP_OK: 'HCOMP_OK',
            HCOMP_ERROR_MEMORY: 'insufficient memory',
            HCOMP_ERROR_OVERFLOW: 'output buffer overflow',
            HCOMP_ERROR_FORMAT: 'invalid data format',
            HCOMP_ERROR: 'compression/decompression error',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def hcomp_version():
    """Return hcompress library version string."""
    return 'hcompress ' + HCOMP_VERSION.decode()


def hcomp_check(const uint8_t[::1] data, /):
    """Return whether data is HCOMP encoded or None if unknown."""
    if data.size < 25:
        return False
    return data[0] == 0xDD and data[1] == 0x99


def hcomp_encode(
    data,
    /,
    level=0,  # scale
    *,
    out=None,
):
    """Return HCOMP encoded data."""
    cdef:
        numpy.ndarray src = numpy.asarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        long nbytes
        int nx, ny
        int scale = _default_value(level, 0, 0, None)
        int status = 0
        int ret = 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.ndim != 2:
        raise ValueError('data must be 2-dimensional')

    if src.dtype.kind not in {'i', 'u'} or src.dtype.itemsize > 4:
        raise ValueError('data dtype must be integer with itemsize <= 4')

    nx = <int> src.shape[0]
    ny = <int> src.shape[1]

    if nx < 4 or ny < 4:
        raise ValueError('dimensions must be >= 4')

    # hcomp_compress modifies input in-place (htrans)
    # use int32 path only when safe: b + ilog2n(nmax) <= 29
    # (each htrans step grows the DC by ~2x; sum of 4 must fit in int32)
    ilog2n = (max(nx, ny) - 1).bit_length()
    dtype = (
        numpy.int32 if src.dtype.itemsize * 8 + ilog2n <= 29 else numpy.int64
    )
    src = numpy.array(src, dtype=dtype, order='C', copy=True)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # worst case: ~10% larger than input plus 26 bytes header
            dstsize = _align_ssize_t(
                (<ssize_t> nx * ny * 4 * 11) // 10 + 26
            )
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes
    nbytes = <long> dstsize

    if src.dtype.itemsize == 4:
        with nogil:
            ret = hcomp_compress(
                <int*> src.data,
                ny,
                nx,
                scale,
                <char*> &dst[0],
                &nbytes,
                &status
            )
    else:
        with nogil:
            ret = hcomp_compress64(
                <long long*> src.data,
                ny,
                nx,
                scale,
                <char*> &dst[0],
                &nbytes,
                &status
            )

    if ret != HCOMP_OK:
        raise HcompError('hcomp_compress', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> nbytes, outgiven)


def hcomp_decode(
    data,
    /,
    *,
    smooth=0,
    safe32=None,
    out=None,
):
    """Return decoded HCOMP data.

    safe32:
        Use the faster 32-bit decode path (no intermediate int64 array).
        Safe when b + ceil(log2(max(ny, nx))) <= 29,
        e.g. int16 with max dimension <= 8192,
        or int15 with max dimension <= 16384.

    """
    cdef:
        numpy.ndarray dst
        numpy.ndarray tmp
        const uint8_t[::1] src = data
        ssize_t srcsize = <ssize_t> src.nbytes
        int nx = 0
        int ny = 0
        int scale = 0
        int smooth_ = smooth
        int status = 0
        int ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize < 25:
        raise ValueError('data too short')

    # parse nx, ny from header (bytes 2-5 = nx, bytes 6-9 = ny)
    nx = (
        (<int> src[2] << 24) |
        (<int> src[3] << 16) |
        (<int> src[4] << 8) |
        <int> src[5]
    )
    ny = (
        (<int> src[6] << 24) |
        (<int> src[7] << 16) |
        (<int> src[8] << 8) |
        <int> src[9]
    )

    if nx < 1 or ny < 1:
        raise ValueError('invalid dimensions in stream')

    if nx > 2147483647 // ny:
        raise ValueError('dimensions too large')

    if safe32:
        # 32-bit path: decode directly into the output array (no extra copy)
        dst = _create_array(out, (nx, ny), numpy.int32)

        with nogil:
            ret = hcomp_decompress(
                <unsigned char*> &src[0],
                <int> srcsize,
                smooth_,
                <int*> dst.data,
                nx * ny,
                &ny,
                &nx,
                &scale,
                &status
            )
        if ret != HCOMP_OK:
            raise HcompError('hcomp_decompress', ret)
        return dst

    # 64-bit path: use int64 intermediate to avoid overflow in hinv
    tmp = numpy.empty((nx, ny), dtype=numpy.int64)
    with nogil:
        ret = hcomp_decompress64(
            <unsigned char*> &src[0],
            <int> srcsize,
            smooth_,
            <long long*> tmp.data,
            nx * ny,
            &ny,
            &nx,
            &scale,
            &status
        )
    if ret != HCOMP_OK:
        raise HcompError('hcomp_decompress', ret)

    if out is None:
        out = tmp.astype(numpy.int32)
    else:
        out = _create_array(out, (nx, ny), numpy.int32)
        dst = out
        dst[:] = tmp

    return out
