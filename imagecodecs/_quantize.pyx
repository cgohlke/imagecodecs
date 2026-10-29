# imagecodecs/_quantize.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2023-2026, Christoph Gohlke
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

"""QUANTIZE codec for the imagecodecs package."""

include '_shared.pxi'

from libc.math cimport NAN, ceil, fabs, log2, pow, round
from nc4var cimport *


class QUANTIZE:
    """QUANTIZE codec constants."""

    available = True

    class MODE(enum.IntEnum):
        """QUANTIZE mode."""

        NOQUANTIZE = NC_NOQUANTIZE
        BITGROOM = NC_QUANTIZE_BITGROOM
        GRANULARBR = NC_QUANTIZE_GRANULARBR
        GBR = NC_QUANTIZE_GRANULARBR  # alias
        BITROUND = NC_QUANTIZE_BITROUND
        SCALE = 100


class QuantizeError(RuntimeError):
    """QUANTIZE codec exceptions."""


def quantize_version():
    """Return nc4var library version string."""
    return f'nc4var {NC_VERSION_MAJOR}.{NC_VERSION_MINOR}.{NC_VERSION_PATCH}'


def quantize_check(const uint8_t[::1] data not None, /):
    """Return whether data is QUANTIZE encoded or None if unknown."""
    return None


def quantize_encode(
    data,
    /,
    mode,
    int nsd,
    *,
    out=None,
):
    """Return quantized floating point data.

    ``nsd`` ("number of significant digits") is interpreted differently
    by the modes:

    - BitGroom, Granular BitRound, and Scale: number of significant
      decimal digits.
    - BitRound: number of significant binary digits (bits).

    Non-finite values are not quantized.

    """
    cdef:
        numpy.ndarray src
        numpy.ndarray dst
        size_t src_size
        nc_type src_type
        int ret = 0
        int range_error = 0
        int quantize_mode = NC_NOQUANTIZE
        int nsd_min = 1
        int nsd_max = 0
        float fill_flt = NAN
        double fill_dbl = NAN
        const void* fill_value

    data = _array_input(data, out, None, True, True)
    src = data
    src_size = <size_t> src.size

    if src.dtype.kind != 'f' or src.itemsize not in {4, 8}:
        raise ValueError('not a floating-point array')

    if src.itemsize == 4:
        src_type = NC_FLOAT
        # NaN fill value disables fill value masking
        fill_value = <const void*> &fill_flt
    else:
        src_type = NC_DOUBLE
        fill_value = <const void*> &fill_dbl

    quantize_mode = _enum_value(mode, QUANTIZE.MODE)

    # largest nsd that does not exceed the significand bits
    if quantize_mode == NC_QUANTIZE_BITGROOM:
        nsd_max = 6 if src_type == NC_FLOAT else 15
    elif quantize_mode == NC_QUANTIZE_GRANULARBR:
        nsd_max = 7 if src_type == NC_FLOAT else 15
    elif quantize_mode == NC_QUANTIZE_BITROUND:
        nsd_max = 23 if src_type == NC_FLOAT else 52
    elif quantize_mode == 100:
        nsd_min = 0
        nsd_max = 7 if src_type == NC_FLOAT else 15
    if quantize_mode != NC_NOQUANTIZE and not nsd_min <= nsd <= nsd_max:
        raise ValueError(
            f'number of significant digits {nsd} out of range '
            f'[{nsd_min}, {nsd_max}] for {mode=} and {src.dtype}'
        )

    out = _create_array(out, data.shape, data.dtype)
    dst = out

    with nogil:
        if quantize_mode == 100:
            if src_type == NC_FLOAT:
                _quantize_scale(
                    <float*> dst.data, <float*> src.data, src_size, nsd
                )
            else:
                _quantize_scale(
                    <double*> dst.data, <double*> src.data, src_size, nsd
                )
        else:
            ret = nc4_convert_type(
                <const void*> src.data,
                <void*> dst.data,
                src_type,
                src_type,
                src_size,
                &range_error,
                fill_value,
                1,  # strict_nc3
                quantize_mode,
                nsd
            )
            if ret < 0:
                raise QuantizeError(f'nc4_convert_type returned {ret!r}')

    return out


def quantize_decode(
    data,
    /,
    mode,
    nsd,
    *,
    out=None,
):
    """Return de-quantized data. Raise QuantizeError if lossy.

    Out is not used.

    """
    if _enum_value(mode, QUANTIZE.MODE) != NC_NOQUANTIZE:
        raise QuantizeError(f'Quantize {mode=} is lossy.')
    return data


# quantize_scale
# Data is quantized using round(data*scale)/scale, where scale is 2**bits,
# and bits is determined from the nsd. For example, if nsd=1, bits will be 4.
# https://github.com/Blosc/bcolz utils.py

ctypedef fused data_t:
    float
    double


cdef void _quantize_scale(
    data_t* dst,
    const data_t* src,
    const ssize_t size,
    const int nsd,
) noexcept nogil:
    cdef:
        double scale = pow(2.0, ceil(log2(pow(10.0, nsd))))
        double value
        ssize_t i

    for i in range(size):
        value = src[i] * scale
        # skip Inf, NaN, and values without fractional bits at this scale
        if fabs(value) < 4503599627370496.0:  # 2**52
            dst[i] = <data_t> (round(value) / scale)
        else:
            dst[i] = src[i]
