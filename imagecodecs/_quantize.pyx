# imagecodecs/_quantize.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2023-2025, Christoph Gohlke
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

"""Quantize codec for the imagecodecs package."""

include '_shared.pxi'

from nc4var cimport *

from libc.math cimport log10, log2, floor, ceil, round, pow


class QUANTIZE:
    """Quantize codec constants."""

    available = True

    class MODE(enum.IntEnum):
        """Quantize mode."""

        NOQUANTIZE = NC_NOQUANTIZE
        BITGROOM = NC_QUANTIZE_BITGROOM
        GRANULARBR = NC_QUANTIZE_GRANULARBR
        BITROUND = NC_QUANTIZE_BITROUND
        SCALE = 100


class QuantizeError(RuntimeError):
    """Quantize codec exceptions."""


def quantize_version():
    """Return nc4var library version string."""
    return f'nc4var {NC_VERSION_MAJOR}.{NC_VERSION_MINOR}.{NC_VERSION_PATCH}'


def quantize_encode(data, mode, int nsd, out=None):
    """Return quantized floating point data.

    ``nsd`` ("number of significant digits") is interpreted differently
    by the modes:

    - BitGroom and Scale: number of significant decimal digits
    - BitRound: number of significant binary digits (bits).
    - Granular BitRound: ?

    """
    cdef:
        numpy.ndarray src
        numpy.ndarray dst
        size_t src_size
        nc_type src_type
        int ret = 0
        int range_error = 0
        int quantize_mode = NC_NOQUANTIZE
        # uint64_t fill_value = 0

    data = numpy.ascontiguousarray(data)
    src = data
    src_size = <size_t>src.size

    if src.dtype.kind != b'f' and src.itemsize not in {4, 8}:
        raise ValueError('not a floating-point array')

    if src.itemsize == 4:
        src_type = NC_FLOAT
    else:
        src_type = NC_DOUBLE

    if mode in {
        NC_NOQUANTIZE,
        NC_QUANTIZE_BITGROOM,
        NC_QUANTIZE_GRANULARBR,
        NC_QUANTIZE_BITROUND,
        100
    }:
        quantize_mode = mode
    elif mode == 'bitround':
        quantize_mode = NC_QUANTIZE_BITROUND
    elif mode == 'bitgroom':
        quantize_mode = NC_QUANTIZE_BITGROOM
    elif mode in {'granularbr', 'gbr'}:
        quantize_mode = NC_QUANTIZE_GRANULARBR
    elif mode == 'scale':
        quantize_mode = 100
    elif mode == 'noquantize':
        quantize_mode = NC_NOQUANTIZE
    else:
        raise ValueError(f'invalid quantize {mode=!r}')

    if not 0 <= nsd < 64:
        raise ValueError(f'invalid number of significant digits {nsd!r}')

    out = _create_array(out, data.shape, data.dtype)
    dst = out

    with nogil:
        if quantize_mode == 100:
            if src_type == NC_FLOAT:
                quantize_scale_f(
                    <float*>src.data,
                    <float*>dst.data,
                    src_size,
                    nsd
                )
            else:
                quantize_scale_d(
                    <double*>src.data,
                    <double*>dst.data,
                    src_size,
                    nsd
                )
        else:
            ret = nc4_convert_type(
                <const void *>src.data,
                <void *>dst.data,
                src_type,
                src_type,
                src_size,
                &range_error,
                NULL,  # fill_value
                1,  # strict_nc3
                quantize_mode,
                nsd
            )
            if ret < 0:
                raise QuantizeError(f'nc4_convert_type returned {ret!r}')

    return out


def quantize_decode(data, mode, nsd, out=None):
    """Return de-quantized data. Raise QuantizeError if lossy."""
    if mode != NC_NOQUANTIZE:
        raise QuantizeError(f'Quantize {mode=} is lossy.')
    return data

###############################################################################

# quantize_scale
# Data is quantized using round(scale*data)/scale, where scale is 2**bits,
# and bits is determined from the nsd. For example, if nsd=1, bits will be 4.
# https://github.com/Blosc/bcolz utils.py


cdef void quantize_scale_f(
    const float* data,
    float* out,
    ssize_t size,
    int nsb
) noexcept nogil:
    cdef:
        float scale
        double exp
        ssize_t i

    exp = log10(pow(10.0, -nsb))
    exp = floor(exp) if exp < 0.0 else ceil(exp)
    scale = <float> pow(2.0, ceil(log2(pow(10.0, -exp))))

    for i in range(size):
        out[i] = <float> round(data[i] * scale) / scale


cdef void quantize_scale_d(
    const double* data,
    double* out,
    ssize_t size,
    int nsb
) noexcept nogil:
    cdef:
        double scale
        double exp
        ssize_t i

    exp = log10(pow(10.0, -nsb))
    exp = floor(exp) if exp < 0.0 else ceil(exp)
    scale = <double> pow(2.0, ceil(log2(pow(10.0, -exp))))

    for i in range(size):
        out[i] = round(data[i] * scale) / scale
