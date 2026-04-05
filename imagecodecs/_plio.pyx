# imagecodecs/_plio.pyx
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

"""PLIO codec for the imagecodecs package.

PLIO is an implementation of the IRAF pixel list I/O compression:

    Doug Tody, National Optical Astronomy Observatories (NOAO).
    Line-list encoding for sparse non-negative integer images.
    Used in IRAF and FITS (cfitsio).

"""

include '_shared.pxi'

from pliocomp cimport *


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


def plio_check(const uint8_t[::1] data, /):
    """Return whether data is PLIO encoded or None if unknown."""
    cdef:
        const short* src = <const short*> &data[0]

    return data.size >= PLIO_HEADER_SIZE * 2 and src[2] == -100


def plio_encode(
    data,
    /,
    *,
    out=None,
):
    """Return PLIO encoded data."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data, dtype=numpy.int32)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        int nout = 0
        int ret = 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.ndim != 1:
        raise ValueError('data must be 1-dimensional')

    if srcsize <= 0 or srcsize > INT32_MAX:
        raise ValueError('invalid data size')

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
    dstsize = dst.nbytes

    with nogil:
        ret = plio_encode_(
            <const int*> src.data,
            <int> srcsize,
            <short*> &dst[0],
            <int> (dstsize // 2),  # number of shorts that fit
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
        ssize_t srcsize = src.nbytes
        ssize_t npix_
        int ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize < PLIO_HEADER_SIZE * 2:
        raise ValueError('data too short')

    if npix is not None:
        npix_ = npix
    elif out is not None and isinstance(out, numpy.ndarray):
        npix_ = out.size
    else:
        raise TypeError('npix is required for PLIO decoding')

    if npix_ <= 0 or npix_ > INT32_MAX:
        raise ValueError(f'invalid npix={npix_}')

    dst = _create_array(out, (npix_,), numpy.int32)

    with nogil:
        ret = plio_decode_(
            <const short*> &src[0],
            <int> (srcsize // 2),
            <int*> dst.data,
            <int> npix_
        )

    if ret != PLIO_OK:
        raise PlioError('plio_decode', ret)

    return dst
