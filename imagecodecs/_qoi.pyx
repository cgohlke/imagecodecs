# imagecodecs/_qoi.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2022-2026, Christoph Gohlke
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

"""QOI (Quite OK Image Format) codec for the imagecodecs package."""

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
    return 'qoi 4461cc3'


def qoi_check(const uint8_t[::1] data, /):
    """Return whether data is QOI encoded image or None if unknown."""
    cdef:
        bytes sig = bytes(data[:4])

    return sig == b'qoif'


def qoi_encode(
    data,
    /,
    *,
    out=None,
):
    """Return QOI encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        void* buffer = NULL
        qoi.qoi_desc desc
        int out_len
        imagelayout_t layout

    if data is out:
        raise ValueError('cannot encode in-place')

    _image_layout(
        IC_UINT | IC_SZ1 | IC_RGB | IC_ALPHA,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        None,  # bitspersample
        None,  # planar
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if not (
        layout.height > 0
        and layout.height <= INT32_MAX
        and layout.width <= INT32_MAX
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    desc.width = <unsigned int> layout.width
    desc.height = <unsigned int> layout.height
    desc.channels = <unsigned int> layout.samples
    desc.colorspace = (
        qoi.QOI_SRGB if layout.samples == 3 else qoi.QOI_LINEAR
    )

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

        with nogil:
            memcpy(<void*> &dst[0], <const void*> buffer, <size_t> out_len)
    finally:
        free(buffer)

    del dst
    return _return_output(out, dstsize, out_len, outgiven)


def qoi_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded QOI image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t dstbytes
        void* buffer = NULL
        qoi.qoi_desc desc

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > INT32_MAX:
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
        dstbytes = dst.nbytes

        with nogil:
            memcpy(<void*> dst.data, <const void*> buffer, <size_t> dstbytes)
    finally:
        free(buffer)

    return out
