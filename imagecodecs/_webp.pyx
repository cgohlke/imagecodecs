# _webp.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2020, Christoph Gohlke
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

"""WebP codec for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2020.12.22

"""

__version__ = '2020.12.22'

include '_shared.pxi'

from libwebp cimport *


class WEBP:
    """WebP Constants."""


class WebpError(RuntimeError):
    """WebP Exceptions."""

    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            VP8_STATUS_OK: 'VP8_STATUS_OK',
            VP8_STATUS_OUT_OF_MEMORY: 'VP8_STATUS_OUT_OF_MEMORY',
            VP8_STATUS_INVALID_PARAM: 'VP8_STATUS_INVALID_PARAM',
            VP8_STATUS_BITSTREAM_ERROR: 'VP8_STATUS_BITSTREAM_ERROR',
            VP8_STATUS_UNSUPPORTED_FEATURE: 'VP8_STATUS_UNSUPPORTED_FEATURE',
            VP8_STATUS_SUSPENDED: 'VP8_STATUS_SUSPENDED',
            VP8_STATUS_USER_ABORT: 'VP8_STATUS_USER_ABORT',
            VP8_STATUS_NOT_ENOUGH_DATA: 'VP8_STATUS_NOT_ENOUGH_DATA',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def webp_version():
    """Return libwebp library version string."""
    cdef:
        int ver = WebPGetDecoderVersion()

    return 'libwebp {}.{}.{}'.format(ver >> 16, (ver >> 8) & 255, ver & 255)


def webp_check(const uint8_t[::1] data):
    """Return True if data likely contains a WebP image."""
    cdef:
        bytes sig = bytes(data[:12])

    return sig[:4] == b'RIFF' and sig[8:12] == b'WEBP'


def webp_encode(data, level=None, out=None):
    """Return WebP image from numpy array.

    """
    cdef:
        const uint8_t[:, :, :] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* srcptr = <uint8_t*> &src[0, 0, 0]
        uint8_t* output
        ssize_t dstsize
        size_t ret = 0
        int width, height, stride
        float quality_factor = _default_value(level, 75.0, -1.0, 100.0)
        int lossless = quality_factor < 0.0
        int rgba = data.shape[2] == 4

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        data.ndim == 3
        and data.shape[0] < WEBP_MAX_DIMENSION
        and data.shape[1] < WEBP_MAX_DIMENSION
        and data.shape[2] in (3, 4)
        and data.strides[2] == 1
        and data.strides[1] in (3, 4)
        and data.strides[0] >= data.strides[1] * data.strides[2]
        and data.dtype == numpy.uint8
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    height, width = data.shape[:2]
    stride = data.strides[0]

    with nogil:
        if lossless:
            if rgba:
                ret = WebPEncodeLosslessRGBA(
                    <const uint8_t*> srcptr,
                    width,
                    height,
                    stride,
                    &output
                )
            else:
                ret = WebPEncodeLosslessRGB(
                    <const uint8_t*> srcptr,
                    width,
                    height,
                    stride,
                    &output
                )
        elif rgba:
            ret = WebPEncodeRGBA(
                <const uint8_t*> srcptr,
                width,
                height,
                stride,
                quality_factor,
                &output
            )
        else:
            ret = WebPEncodeRGB(
                <const uint8_t*> srcptr,
                width,
                height,
                stride,
                quality_factor,
                &output
            )

    if ret <= 0:
        raise WebpError('WebPEncode', ret)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = ret
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    if <size_t> dstsize < ret:
        raise RuntimeError('output too small')

    with nogil:
        memcpy(<void*> &dst[0], <const void*> output, ret)
        WebPFree(<void*> output)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def webp_decode(data, index=None, out=None):
    """Decode WebP image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        int output_stride
        size_t data_size
        WebPBitstreamFeatures features
        int ret = VP8_STATUS_OK
        uint8_t* pout

    if data is out:
        raise ValueError('cannot decode in-place')

    ret = <int> WebPGetFeatures(&src[0], <size_t> srcsize, &features)
    if ret != VP8_STATUS_OK:
        raise WebpError('WebPGetFeatures', ret)

    # TODO: support features.has_animation

    if features.has_alpha:
        shape = features.height, features.width, 4
    else:
        shape = features.height, features.width, 3

    out = _create_array(out, shape, numpy.uint8, strides=(None, shape[2], 1))
    dst = out
    dstsize = dst.shape[0] * dst.strides[0]
    output_stride = <int> dst.strides[0]

    with nogil:
        if features.has_alpha:
            pout = WebPDecodeRGBAInto(
                &src[0],
                <size_t> srcsize,
                <uint8_t*> dst.data,
                <size_t> dstsize,
                output_stride
            )
        else:
            pout = WebPDecodeRGBInto(
                &src[0],
                <size_t> srcsize,
                <uint8_t*> dst.data,
                <size_t> dstsize,
                output_stride
            )
    if pout == NULL:
        raise WebpError('WebPDecodeRGBAInto', None)

    return out
