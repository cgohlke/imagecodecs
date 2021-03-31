# imagecodecs/_lz4.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2021, Christoph Gohlke
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

"""LZ4 codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from lz4 cimport *


class LZ4:
    """LZ4 Constants."""

    CLEVEL_MIN = LZ4HC_CLEVEL_MIN
    CLEVEL_DEFAULT = LZ4HC_CLEVEL_DEFAULT
    CLEVEL_OPT_MIN = LZ4HC_CLEVEL_OPT_MIN
    CLEVEL_MAX = LZ4HC_CLEVEL_MAX


class Lz4Error(RuntimeError):
    """LZ4 Exceptions."""


def lz4_version():
    """Return LZ4 library version string."""
    return 'lz4 {}.{}.{}'.format(
        LZ4_VERSION_MAJOR, LZ4_VERSION_MINOR, LZ4_VERSION_RELEASE
    )


def lz4_check(data):
    """Return True if data likely contains LZ4 data."""


def lz4_encode(data, level=None, hc=False, header=False, out=None):
    """Compress LZ4.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = src.size
        int dstsize
        int offset = 4 if header else 0
        int ret
        uint8_t* pdst
        int acceleration, compressionlevel

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.size > LZ4_MAX_INPUT_SIZE:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4_compressBound(srcsize) + offset
            if dstsize < 0:
                raise Lz4Error(f'LZ4_compressBound returned {dstsize}')
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int> dst.size - offset

    if dst.size >= 2 ** 31:
        raise ValueError('output too large')

    if hc:
        compressionlevel = _default_value(
            level, LZ4HC_CLEVEL_DEFAULT, LZ4HC_CLEVEL_MIN, LZ4HC_CLEVEL_MAX
        )
        with nogil:
            ret = LZ4_compress_HC(
                <const char*> &src[0],
                <char*> &dst[offset],
                srcsize,
                dstsize,
                compressionlevel
            )
            if ret <= 0:
                raise Lz4Error(f'LZ4_compress_HC returned {ret}')

    else:
        acceleration = _default_value(level, 1, 1, 65537)
        with nogil:
            ret = LZ4_compress_fast(
                <const char*> &src[0],
                <char*> &dst[offset],
                srcsize,
                dstsize,
                acceleration
            )
            if ret <= 0:
                raise Lz4Error(f'LZ4_compress_fast returned {ret}')

    if header:
        pdst = <uint8_t*> &dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def lz4_decode(data, header=False, out=None):
    """Decompress LZ4.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = <int> src.size
        int dstsize
        int offset = 4 if header else 0
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if src.size >= 2 ** 31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming API
            dstsize = max(24, 24 + 255 * (srcsize - offset - 10))  # ugh
            # if dstsize < 0:
            #     raise Lz4Error(f'invalid output size {dstsize}')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int> dst.size

    if dst.size >= 2 ** 31:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_decompress_safe(
            <char*> &src[offset],
            <char*> &dst[0],
            srcsize - offset,
            dstsize
        )
    if ret < 0:
        raise Lz4Error(f'LZ4_decompress_safe returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)
