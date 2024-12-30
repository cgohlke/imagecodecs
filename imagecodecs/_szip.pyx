# imagecodecs/_szip.pyx
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

"""SZIP codec for the imagecodecs package."""

include '_shared.pxi'

from szlib cimport *


class SZIP:
    """SZIP codec constants."""

    available = True

    class OPTION_MASK(enum.IntEnum):  # IntFlag
        """SZIP codec flags."""

        ALLOW_K13 = SZ_ALLOW_K13_OPTION_MASK
        CHIP = SZ_CHIP_OPTION_MASK
        EC = SZ_EC_OPTION_MASK
        LSB = SZ_LSB_OPTION_MASK
        MSB = SZ_MSB_OPTION_MASK
        NN = SZ_NN_OPTION_MASK
        RAW = SZ_RAW_OPTION_MASK


class SzipError(RuntimeError):
    """SZIP codec exceptions."""

    def __init__(self, func, err):
        msg = {
            SZ_OK: 'SZ_OK',
            SZ_OUTBUFF_FULL: 'SZ_OUTBUFF_FULL',
            SZ_NO_ENCODER_ERROR: 'SZ_NO_ENCODER_ERROR',
            SZ_PARAM_ERROR: 'SZ_PARAM_ERROR',
            SZ_MEM_ERROR: 'SZ_MEM_ERROR',
            # AEC_RSI_OFFSETS_ERROR: 'AEC_RSI_OFFSETS_ERROR',
            -2: 'AEC_STREAM_ERROR',
            -3: 'AEC_DATA_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def szip_version():
    """Return libaec library version string."""
    # f'libaec {AEC_VERSION_MAJOR}.{AEC_VERSION_MINOR}.{AEC_VERSION_PATCH}'
    return 'libaec 1.0.x'


def szip_check(data):
    """Return whether data is SZIP encoded."""


def szip_encode(
    data,
    options_mask,
    pixels_per_block,
    bits_per_pixel,
    pixels_per_scanline,
    *,
    header=False,
    out=None
):
    """Return SZIP encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t dstlen
        SZ_com_t_s param
        int offset = 4 if header else 0
        int ret

    if data is out:
        raise ValueError('cannot encode in-place')

    if srcsize >= 2147483647:
        raise ValueError('src size exceeds 2 GB')

    param.options_mask = options_mask
    param.bits_per_pixel = bits_per_pixel
    param.pixels_per_block = pixels_per_block
    param.pixels_per_scanline = pixels_per_scanline

    if not 0 < param.pixels_per_block <= SZ_MAX_PIXELS_PER_BLOCK:
        raise ValueError(f'invalid pixels_per_block {param.pixels_per_block}')
    if not 0 < param.bits_per_pixel <= 128:
        raise ValueError(f'invalid bits_per_pixel {param.bits_per_pixel}')
    if not 0 < param.pixels_per_scanline <= SZ_MAX_PIXELS_PER_SCANLINE:
        raise ValueError(
            f'invalid pixels_per_scanline {param.pixels_per_scanline}'
        )

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = srcsize + srcsize // 21 + 256 + 1 + offset
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes - offset
    dstlen = <size_t> dstsize

    with nogil:
        ret = SZ_BufftoBuffCompress(
            <void*> &dst[offset],
            &dstlen,
            <const void*> &src[0],
            <size_t> srcsize,
            &param
        )
    if ret != SZ_OK:
        raise SzipError('SZ_BufftoBuffCompress', ret)

    if header:
        pdst = <uint8_t*> &dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize+offset, dstlen+offset, outgiven)


def szip_decode(
    data,
    options_mask,
    pixels_per_block,
    bits_per_pixel,
    pixels_per_scanline,
    *,
    header=False,
    out=None
):
    """Return decoded SZIP data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstsize_header = 0
        size_t dstlen
        SZ_com_t_s param
        int offset = 4 if header else 0
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    param.options_mask = options_mask
    param.bits_per_pixel = bits_per_pixel
    param.pixels_per_block = pixels_per_block
    param.pixels_per_scanline = pixels_per_scanline

    if not 0 < param.pixels_per_block <= SZ_MAX_PIXELS_PER_BLOCK:
        raise ValueError(f'invalid pixels_per_block {param.pixels_per_block}')
    if not 0 < param.bits_per_pixel <= 128:
        raise ValueError(f'invalid bits_per_pixel {param.bits_per_pixel}')
    if not 0 < param.pixels_per_scanline <= SZ_MAX_PIXELS_PER_SCANLINE:
        raise ValueError(
            f'invalid pixels_per_scanline {param.pixels_per_scanline}'
        )

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)
        dstsize_header = dstsize

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 8  # TODO
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <size_t> dstsize

    with nogil:
        ret = SZ_BufftoBuffDecompress(
            <void*> &dst[0],
            &dstlen,
            <const void*> &src[offset],
            <size_t> (srcsize - offset),
            &param
        )
    if ret < 0:
        raise SzipError('SZ_BufftoBuffDecompress', ret)

    if dstsize_header > 0:
        if dstsize < dstsize_header:
            # buffer too small but SZ_BufftoBuffDecompress passed
            raise SzipError('SZ_BufftoBuffDecompress', SZ_OUTBUFF_FULL)
        # dstlen output might be padded up to next scanline
        dstlen = <size_t> dstsize_header

    del dst
    return _return_output(out, dstsize, <ssize_t> dstlen, outgiven)


def szip_params(data, int options_mask=4, int pixels_per_block=32):
    """Return SZIP parameters for numpy array."""
    cdef:
        ssize_t pixels_in_line
        ssize_t pixels_per_scanline
        ssize_t pixels_in_chunk = data.size
        ssize_t bits_per_pixel = data.itemsize * 8
        int ndim = data.ndim

    if (
        not (2 <= pixels_per_block <= SZ_MAX_PIXELS_PER_BLOCK)
        or (pixels_per_block % 2)
    ):
        raise ValueError(f'invalid {pixels_per_block=}')

    if ndim <= 1:
        pixels_in_line = data.size
    elif data.shape[ndim - 1] * data.itemsize in {4, 8}:
        # multiple samples per pixel will be shuffled
        pixels_in_line = data.shape[ndim - 2]
        pixels_in_chunk //= data.shape[ndim - 1]
        bits_per_pixel *= data.shape[ndim - 1]
    else:
        pixels_in_line = data.shape[ndim - 1]

    if pixels_in_line < pixels_per_block:
        if pixels_in_chunk < pixels_per_block:
            # TODO: raise error?
            pixels_per_block = <int> max(2, (pixels_in_chunk // 2) * 2)
        pixels_per_scanline = pixels_per_block
    else:
        pixels_per_scanline = pixels_in_line

    options_mask &= ~(SZ_LSB_OPTION_MASK | SZ_MSB_OPTION_MASK)
    if data.dtype.byteorder != '>':
        options_mask |= SZ_LSB_OPTION_MASK
    else:
        options_mask |= SZ_MSB_OPTION_MASK

    return {
        'options_mask': options_mask,
        'pixels_per_block': pixels_per_block,
        'bits_per_pixel': bits_per_pixel,
        'pixels_per_scanline': min(
            pixels_per_scanline, pixels_per_block * SZ_MAX_BLOCKS_PER_SCANLINE
        )
    }
