# imagecodecs/_cfitsio.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2021-2026, Christoph Gohlke
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

"""HCOMP, PLIO, and RCOMP codecs for the imagecodecs package.

Hcomp is an implementation of the H-compress algorithm:

    R. L. White.  Compression of astronomical images.
    Bulletin of the Astronomical Society, 24:1135, 1992.

PLIO is an implementation of the IRAF pixel list I/O compression:

    Doug Tody, National Optical Astronomy Observatories (NOAO).
    Line-list encoding for sparse non-negative integer images.
    Used in IRAF and FITS (cfitsio).

Rcomp is an implementation of the Rice algorithm:

    Robert Rice, Pen-Shu Yeh, and Warner Miller. Algorithms for high-speed
    universal noiseless coding. Proc. of the 9th AIAA Computing in Aerospace.
    Conf. AIAA-93-4541-CP, 1993. https://doi.org/10.2514/6.1993-4541

"""

include '_shared.pxi'

from hcompress cimport *
from libc.limits cimport LONG_MAX
from pliocomp cimport *
from ricecomp cimport *


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


def hcomp_check(const uint8_t[::1] data not None, /):
    """Return whether data is HCOMP encoded or None if unknown."""
    return data.shape[0] >= 25 and data[0] == 0xDD and data[1] == 0x99


def hcomp_encode(
    data,
    /,
    level=0,  # scale
    *,
    out=None,
):
    """Return HCOMP encoded data.

    Decoded data are int32. View the result as uint32 for uint32 input.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        long nbytes
        int nx, ny
        int scale = _default_value(level, 0, 0, None)
        int status = 0
        int ret = 0

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
    dstsize = dst.shape[0]
    # long is 32-bit on Windows
    nbytes = <long> min(dstsize, <ssize_t> LONG_MAX)

    if src.dtype.itemsize == 4:
        with nogil:
            ret = hcomp_compress(
                <int*> src.data,
                ny,
                nx,
                scale,
                <char*> dst._data,
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
                <char*> dst._data,
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
        ssize_t srcsize = src.shape[0]
        int nx = 0
        int ny = 0
        int scale = 0
        int smooth_ = smooth
        int status = 0
        int ret = 0

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize < 25:
        raise ValueError('data too short')
    if srcsize > INT32_MAX:
        raise ValueError('input too large')

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
        if dst.dtype != numpy.int32:
            raise ValueError(f'invalid output dtype {dst.dtype}')

        with nogil:
            ret = hcomp_decompress(
                <unsigned char*> src._data,
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
            <unsigned char*> src._data,
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
        if dst.dtype != numpy.int32:
            raise ValueError(f'invalid output dtype {dst.dtype}')
        dst[:] = tmp

    return out


# PLIO ########################################################################


class PLIO:
    """PLIO codec constants."""

    available = True


class PlioError(RuntimeError):
    """PLIO codec exceptions."""

    def __init__(self, func, err):
        msg = {
            PLIO_OK: 'PLIO_OK',
            PLIO_ERROR_MEMORY: 'insufficient memory',
            PLIO_ERROR_OVERFLOW: 'output buffer overflow',
            PLIO_ERROR_FORMAT: 'invalid data format',
            PLIO_ERROR: 'compression/decompression error',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def plio_version():
    """Return pliocomp library version string."""
    return 'pliocomp ' + PLIO_VERSION.decode()


def plio_check(const uint8_t[::1] data not None, /):
    """Return whether data is PLIO encoded or None if unknown."""
    cdef:
        int16_t value

    if data.shape[0] < PLIO_HEADER_SIZE * 2:
        return False
    memcpy(&value, &data[4], 2)  # may be unaligned
    return value == -100


def plio_encode(
    data,
    /,
    *,
    out=None,
):
    """Return PLIO encoded data.

    Data must be 1-dimensional integers less than 2**24.
    Negative values are encoded as 0, like cfitsio.
    Data are cast to int32 without range checks and 64-bit values outside
    the int32 range wrap.
    The encoded data are 16-bit integers in native byte order.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        int nout = 0
        int ret = 0

    if src.ndim != 1:
        raise ValueError('data must be 1-dimensional')

    if srcsize <= 0 or srcsize > INT32_MAX:
        raise ValueError('invalid data size')

    if src.dtype.kind not in {'i', 'u'}:
        raise ValueError(f'data must be integer, not {src.dtype}')

    src = numpy.ascontiguousarray(src, dtype=numpy.int32)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # worst case: each pixel can produce 2 shorts (high-value)
            # plus header, times 2 bytes per short
            dstsize = _align_ssize_t(
                (PLIO_HEADER_SIZE + <ssize_t> srcsize * 2) * 2
            )
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]
    if (<size_t> dst._data) % 2:
        raise ValueError('output buffer is not 2-byte aligned')

    with nogil:
        ret = plio_encode_(
            <const int*> src.data,
            <int> srcsize,
            <short*> dst._data,
            <int> min(dstsize // 2, <ssize_t> INT32_MAX),
            &nout
        )

    if ret != PLIO_OK:
        raise PlioError('plio_encode', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> nout * 2, outgiven)


def plio_decode(
    data,
    /,
    npix=None,
    *,
    out=None,
):
    """Return decoded PLIO data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        ssize_t npix_
        int ret = 0

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize < PLIO_HEADER_SIZE * 2:
        raise ValueError('data too short')
    if srcsize // 2 > INT32_MAX:
        raise ValueError('input too large')

    if (<size_t> src._data) % 2:
        src = bytearray(src)

    if npix is not None:
        npix_ = npix
    elif out is not None and isinstance(out, numpy.ndarray):
        npix_ = out.size
    else:
        raise TypeError('npix is required for PLIO decoding')

    if npix_ <= 0 or npix_ > INT32_MAX:
        raise ValueError(f'invalid npix={npix_}')

    if isinstance(out, numpy.ndarray):
        out = _create_array(out, out.shape, numpy.int32)
        if out.size != npix_:
            raise ValueError(f'output size {out.size} != {npix_=}')
        dst = out.reshape(-1)
    else:
        dst = _create_array(out, (npix_,), numpy.int32)
    if dst.dtype != numpy.int32:
        raise ValueError(f'invalid output dtype {dst.dtype}')

    with nogil:
        ret = plio_decode_(
            <const short*> src._data,
            <int> (srcsize // 2),
            <int*> dst.data,
            <int> npix_
        )

    if ret != PLIO_OK:
        raise PlioError('plio_decode', ret)

    if isinstance(out, numpy.ndarray):
        return out
    return dst


# RCOMP #######################################################################


class RCOMP:
    """RCOMP codec constants."""

    available = True


class RcompError(RuntimeError):
    """RCOMP codec exceptions."""

    def __init__(self, func, err):
        msg = {
            RCOMP_OK: 'RCOMP_OK',
            RCOMP_ERROR_MEMORY: 'insufficient memory',
            RCOMP_ERROR_EOB: 'end of buffer',
            RCOMP_ERROR_EOS: 'reached end of compressed byte stream',
            RCOMP_WARN_UNUSED: 'unused bytes at end of compressed buffer',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def rcomp_version():
    """Return ricecomp library version string."""
    return 'ricecomp ' + RCOMP_VERSION.decode()


def rcomp_check(const uint8_t[::1] data not None, /):
    """Return whether data is RCOMP encoded or None if unknown."""
    return None if data.shape[0] > 2 else False


def rcomp_encode(
    data,
    /,
    *,
    nblock=None,
    out=None,
):
    """Return RCOMP encoded data."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        numpy.dtype dtype = src.dtype
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t nblocks
        int nblock_ = 32 if nblock is None else nblock
        int ret = 0

    if nblock_ < 1:
        raise ValueError(f'invalid {nblock=}')

    if not (
        srcsize <= INT32_MAX
        and dtype.kind in {'i', 'u'}
        and dtype.itemsize in {1, 2, 4}
    ):
        raise ValueError('data is not an integer array of size < 2**31')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # worst case: first value, and per block a FS code (3, 4, or 5
            # bits) followed by nblock uncompressed values
            nblocks = (srcsize + nblock_ - 1) // nblock_
            dstsize = nblocks * (
                dtype.itemsize + 2 + <ssize_t> nblock_ * dtype.itemsize * 8
            )
            dstsize = _align_ssize_t(dtype.itemsize + (dstsize + 7) // 8 + 16)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]
    if dstsize > INT32_MAX:
        raise ValueError('output too large')

    if dtype.itemsize == 1:
        with nogil:
            ret = rcomp_byte(
                <signed char*> src.data,
                <int> srcsize,
                <unsigned char*> dst._data,
                <int> dstsize,
                nblock_
            )
        if ret < 0:
            raise RcompError('rcomp_byte', ret)

    elif dtype.itemsize == 2:
        with nogil:
            ret = rcomp_short(
                <signed short*> src.data,
                <int> srcsize,
                <unsigned char*> dst._data,
                <int> dstsize,
                nblock_
            )
        if ret < 0:
            raise RcompError('rcomp_short', ret)

    elif dtype.itemsize == 4:
        with nogil:
            ret = rcomp_int(
                <signed int*> src.data,
                <int> srcsize,
                <unsigned char*> dst._data,
                <int> dstsize,
                nblock_
            )
        if ret < 0:
            raise RcompError('rcomp_int', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def rcomp_decode(
    data,
    /,
    shape=None,
    dtype=None,
    *,
    nblock=None,
    out=None,
):
    """Return decoded RCOMP data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        int ret = 0
        int nblock_ = 32 if nblock is None else nblock

    if data is out:
        raise ValueError('cannot transform in-place')

    if nblock_ < 1:
        raise ValueError(f'invalid {nblock=}')

    if srcsize > INT32_MAX:
        raise ValueError('input buffer too large')

    if out is not None and isinstance(out, numpy.ndarray):
        if shape is None:
            shape = out.shape
        if dtype is None:
            dtype = out.dtype
        else:
            dtype = numpy.dtype(dtype)
    elif dtype is None or shape is None:
        raise TypeError('missing shape or dtype')
    else:
        dtype = numpy.dtype(dtype)
        try:
            shape = tuple(shape)
        except TypeError:
            shape = (int(shape), )

    if not (dtype.kind in 'iu' and dtype.itemsize in {1, 2, 4}):
        raise ValueError('invalid dtype')

    out = _create_array(out, shape, dtype)
    dst = out
    if dst.dtype.kind not in 'iu':
        raise ValueError(f'invalid output dtype {dst.dtype}')
    dstsize = dst.size
    if dstsize > INT32_MAX:
        raise ValueError('output array too large')

    if dtype.itemsize == 1:
        with nogil:
            ret = rdecomp_byte(
                <unsigned char*> src._data,
                <int> srcsize,
                <unsigned char*> dst.data,
                <int> dstsize,
                nblock_
            )
        if ret != RCOMP_OK and ret != RCOMP_WARN_UNUSED:
            raise RcompError('rdecomp_byte', ret)

    elif dtype.itemsize == 2:
        with nogil:
            ret = rdecomp_short(
                <unsigned char*> src._data,
                <int> srcsize,
                <unsigned short*> dst.data,
                <int> dstsize,
                nblock_
            )
        if ret != RCOMP_OK and ret != RCOMP_WARN_UNUSED:
            raise RcompError('rdecomp_short', ret)

    elif dtype.itemsize == 4:
        with nogil:
            ret = rdecomp_int(
                <unsigned char*> src._data,
                <int> srcsize,
                <unsigned int*> dst.data,
                <int> dstsize,
                nblock_
            )
        if ret != RCOMP_OK and ret != RCOMP_WARN_UNUSED:
            raise RcompError('rdecomp_int', ret)

    del dst
    return out
