# imagecodecs/_qoi.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2022-2025, Christoph Gohlke
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

"""QOI codec for the imagecodecs package."""

include '_shared.pxi'

cimport qoi


class QOI:
    """QOI codec constants."""

    available = True

    class COLORSPACE(enum.IntEnum):
        """QOI codec color spaces."""

        SRGB = qoi.QOI_SRGB
        LINEAR = qoi.QOI_LINEAR


class QoiError(RuntimeError):
    """QOI codec exceptions."""


def qoi_version():
    """Return QOI library version string."""
    return 'qoi 36190eb'


def qoi_check(const uint8_t[::1] data):
    """Return whether data is QOI encoded image."""
    cdef:
        bytes sig = bytes(data[:4])

    return sig == b'qoif'


def qoi_encode(data, out=None):
    """Return QOI encoded image."""
    cdef:
        numpy.ndarray src = numpy.asarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        void* buffer = NULL
        qoi.qoi_desc desc
        int out_len
        int samples = <int> src.shape[2] if src.ndim == 3 else 1

    if not (
        src.dtype == numpy.uint8
        and src.ndim == 3
        and src.shape[0] <= 2147483647
        and src.shape[1] <= 2147483647
        and (samples == 3 or samples == 4)
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    desc.width = <unsigned int> src.shape[1]
    desc.height = <unsigned int> src.shape[0]
    desc.channels = <unsigned int> samples
    desc.colorspace = qoi.QOI_LINEAR if samples < 3 else qoi.QOI_SRGB

    with nogil:
        buffer = qoi.qoi_encode(
            <const void*> src.data,
            &desc,
            &out_len
        )

    if buffer == NULL:
        raise QoiError('qoi_encode returned NULL')

    try:
        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            out = _create_output(outtype, out_len)
        dst = out
        dstsize = dst.nbytes
        if dstsize < <ssize_t> out_len:
            raise ValueError('output too small')

        # with nogil: ?
        memcpy(<void*> &dst[0], <const void*> buffer, <size_t> out_len)
    finally:
        free(buffer)

    del dst
    return _return_output(out, dstsize, out_len, outgiven)


def qoi_decode(data, out=None):
    """Return decoded QOI image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        void* buffer = NULL
        qoi.qoi_desc desc

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > 2147483647:
        raise ValueError('input too large')

    with nogil:
        buffer = qoi.qoi_decode(
            <const void*> &src[0],
            <int> srcsize,
            &desc,
            0
        )
    if buffer == NULL:
        raise QoiError('qoi_decode returned NULL')

    try:
        if desc.channels > 1:
            shape = int(desc.height), int(desc.width), int(desc.channels)
        else:
            shape = int(desc.height), int(desc.width)

        out = _create_array(out, shape, numpy.uint8)
        dst = out

        # with nogil: ?
        memcpy(<void*> dst.data, <const void*> buffer, dst.size)
    finally:
        free(buffer)

    return out
