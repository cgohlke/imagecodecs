# imagecodecs/_sz3.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2024-2025, Christoph Gohlke
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

"""SZ3 codec for the imagecodecs package."""

include '_shared.pxi'

from sz3c cimport *


class SZ3:
    """SZ3 codec constants."""

    available = True

    class MODE(enum.IntEnum):
        """SZ3 codec error bound modes."""

        ABS = 0
        REL = 1
        ABS_AND_REL = 2
        ABS_OR_REL = 3
        # PSNR = 4
        # NORM = 5
        # PW_REL = 10
        # ABS_AND_PW_REL = 11
        # ABS_OR_PW_REL = 12
        # REL_AND_PW_REL = 13
        # REL_OR_PW_REL = 14


class Sz3Error(RuntimeError):
    """SZ3 codec exceptions."""


def sz3_version():
    """Return sz3 library version string."""
    return 'sz3 3.1.8'


def sz3_check(const uint8_t[::1] data):
    """Return whether data is SZ3 encoded."""
    return None


def sz3_encode(data, mode=None, abs=None, rel=None, out=None):
    """Return SZ3 encoded data."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        unsigned char* buffer = NULL
        size_t r1, r2, r3, r4, r5
        size_t outSize = 0
        double absErrBound = 0.0 if abs is None else abs
        double relBoundRatio = 0.0 if rel is None else rel
        double pwrBoundRatio = 0.0  # if pwr is None else pwr
        int errBoundMode, dataType

    if mode is None:
        errBoundMode = ABS
    elif mode in {ABS, 'abs'}:
        errBoundMode = ABS
    elif mode in {REL, 'rel'}:
        errBoundMode = REL
    elif mode in {ABS_AND_REL, 'abs_and_rel'}:
        errBoundMode = ABS_AND_REL
    elif mode in {ABS_OR_REL, 'abs_or_rel'}:
        errBoundMode = ABS_OR_REL
    # elif mode in {PSNR, 'psnr'}:
    #     errBoundMode = PSNR
    # elif mode in {NORM, 'norm'}:
    #     errBoundMode = NORM
    # elif mode in {PW_REL, 'pw_rel'}:
    #     errBoundMode = PW_REL
    # elif mode in {ABS_AND_PW_REL, 'abs_and_pw_rel'}:
    #     errBoundMode = ABS_AND_PW_REL
    # elif mode in {ABS_OR_PW_REL, 'abs_or_pw_rel'}:
    #     errBoundMode = ABS_OR_PW_REL
    # elif mode in {REL_AND_PW_REL, 'rel_and_pw_rel'}:
    #     errBoundMode = REL_AND_PW_REL
    # elif mode in {REL_OR_PW_REL, 'rel_or_pw_rel'}:
    #     errBoundMode = REL_OR_PW_REL
    else:
        raise ValueError(f'invalid SZ3 error bound {mode=!r}')

    if src.dtype == numpy.float32:
        dataType = SZ_FLOAT
    elif src.dtype == numpy.float64:
        dataType = SZ_DOUBLE
    # elif src.dtype == numpy.uint8:
    #     dataType = SZ_UINT8
    # elif src.dtype == numpy.int8:
    #     dataType = SZ_INT8
    # elif src.dtype == numpy.uint16:
    #     dataType = SZ_UINT16
    # elif src.dtype == numpy.int16:
    #     dataType = SZ_INT16
    # elif src.dtype == numpy.uint32:
    #     dataType = SZ_UINT32
    # elif src.dtype == numpy.int32:
    #     dataType = SZ_INT32
    # elif src.dtype == numpy.uint64:
    #     dataType = SZ_UINT64
    # elif src.dtype == numpy.int64:
    #     dataType = SZ_INT64
    else:
        raise ValueError(f'data dtype={src.dtype} not supported')

    r1 = r2 = r3 = r4 = r5 = 0
    if src.ndim == 1:
        r1 = src.shape[0]
    elif src.ndim == 2:
        r1 = src.shape[1]
        r2 = src.shape[0]
    elif src.ndim == 3:
        r1 = src.shape[2]
        r2 = src.shape[1]
        r3 = src.shape[0]
    elif src.ndim == 4:
        r1 = src.shape[3]
        r2 = src.shape[2]
        r3 = src.shape[1]
        r4 = src.shape[0]
    elif src.ndim == 5:
        r1 = src.shape[4]
        r2 = src.shape[3]
        r3 = src.shape[2]
        r4 = src.shape[1]
        r5 = src.shape[0]
    else:
        raise ValueError(f'data ndim={src.ndim} not supported')

    with nogil:
        buffer = SZ_compress_args(
            dataType,
            <void *> src.data,
            &outSize,
            errBoundMode,
            absErrBound,
            relBoundRatio,
            pwrBoundRatio,
            r5,
            r4,
            r3,
            r2,
            r1,
        )

    if buffer == NULL:
        raise Sz3Error('SZ_compress_args returned NULL')

    try:
        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            out = _create_output(outtype, outSize)

        dst = out
        dstsize = dst.nbytes
        if dstsize < <ssize_t> outSize:
            raise ValueError('output too small')

        memcpy(<void*> &dst[0], <const void*> buffer, <size_t> outSize)
    finally:
        free(buffer)

    del dst
    return _return_output(out, dstsize, outSize, outgiven)


def sz3_decode(data, shape, dtype, out=None):
    """Return decoded SZ3 data."""
    cdef:
        const uint8_t[::1] src = data
        numpy.ndarray dst
        void* buffer = NULL
        size_t r1, r2, r3, r4, r5
        size_t byteLength = <size_t> src.nbytes
        ssize_t ndim = len(shape)
        int dataType

    if data is out:
        raise ValueError('cannot decode in-place')

    if dtype is None:
        raise ValueError(f'{dtype=!r} not supported')
    dtype = numpy.dtype(dtype)
    if dtype == numpy.float32:
        dataType = SZ_FLOAT
    elif dtype == numpy.float64:
        dataType = SZ_DOUBLE
    # elif dtype == numpy.uint8:
    #     dataType = SZ_UINT8
    # elif dtype == numpy.int8:
    #     dataType = SZ_INT8
    # elif dtype == numpy.uint16:
    #     dataType = SZ_UINT16
    # elif dtype == numpy.int16:
    #     dataType = SZ_INT16
    # elif dtype == numpy.uint32:
    #     dataType = SZ_UINT32
    # elif dtype == numpy.int32:
    #     dataType = SZ_INT32
    # elif dtype == numpy.uint64:
    #     dataType = SZ_UINT64
    # elif dtype == numpy.int64:
    #     dataType = SZ_INT64
    else:
        raise ValueError(f'{dtype=!r} not supported')

    r1 = r2 = r3 = r4 = r5 = 0
    if ndim == 1:
        r1 = shape[0]
    elif ndim == 2:
        r1 = shape[1]
        r2 = shape[0]
    elif ndim == 3:
        r1 = shape[2]
        r2 = shape[1]
        r3 = shape[0]
    elif ndim == 4:
        r1 = shape[3]
        r2 = shape[2]
        r3 = shape[1]
        r4 = shape[0]
    elif ndim == 5:
        r1 = shape[4]
        r2 = shape[3]
        r3 = shape[2]
        r4 = shape[1]
        r5 = shape[0]
    else:
        raise ValueError(f'{len(shape)=} not supported')

    with nogil:
        buffer = SZ_decompress(
            dataType,
            <unsigned char*> &src[0],
            byteLength,
            r5,
            r4,
            r3,
            r2,
            r1,
        )

    if buffer == NULL:
        raise Sz3Error('SZ_decompress returned NULL')

    try:
        out = _create_array(out, shape, dtype)
        dst = out

        memcpy(<void*> dst.data, <const void*> buffer, dst.nbytes)
    finally:
        free(buffer)

    return out
