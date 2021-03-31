# imagecodecs/_brotli.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""Brotli codec for the imagecodecs package."""

__version__ = '2021.2.26'

include '_shared.pxi'

from brotli cimport *


class BROTLI:
    """Brotli Constants."""
    MODE_GENERIC = BROTLI_MODE_GENERIC
    MODE_TEXT = BROTLI_MODE_TEXT
    MODE_FONT = BROTLI_MODE_FONT


class BrotliError(RuntimeError):
    """Brotli Exceptions."""

    def __init__(self, func, err):
        err = {
            True: 'True',
            False: 'False',
            BROTLI_DECODER_RESULT_ERROR: 'BROTLI_DECODER_RESULT_ERROR',
            BROTLI_DECODER_RESULT_SUCCESS: 'BROTLI_DECODER_RESULT_SUCCESS',
            BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT:
                'BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT',
            BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
                'BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {err}'
        super().__init__(msg)


def brotli_version():
    """Return Brotli library version string."""
    cdef:
        uint32_t ver = BrotliDecoderVersion()

    return 'brotli {}.{}.{}'.format(ver >> 24, (ver >> 12) & 4095, ver & 4095)


def brotli_check(data):
    """Return True if data likely contains Brotli data."""


def brotli_encode(data, level=None, mode=None, lgwin=None, out=None):
    """Compress Brotli.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t encoded_size
        BROTLI_BOOL ret = BROTLI_FALSE
        BrotliEncoderMode mode_ = BROTLI_MODE_GENERIC if mode is None else mode
        int quality_ = _default_value(level, 11, 0, 11)
        int lgwin_ = _default_value(lgwin, 22, 10, 24)
        # int lgblock_ = _default_value(lgblock, 0, 16, 24)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming interface with dynamic buffer
            dstsize = <ssize_t> BrotliEncoderMaxCompressedSize(<size_t>srcsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    encoded_size = <size_t> dstsize

    with nogil:
        ret = BrotliEncoderCompress(
            quality_,
            lgwin_,
            mode_,
            <size_t> srcsize,
            <const uint8_t*> &src[0],
            &encoded_size,
            <uint8_t*> &dst[0]
        )
    if ret != BROTLI_TRUE:
        raise BrotliError('BrotliEncoderCompress', bool(ret))

    del dst
    return _return_output(out, dstsize, encoded_size, outgiven)


def brotli_decode(data, out=None):
    """Decompress Brotli.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        size_t decoded_size
        BrotliDecoderResult ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming API with dynamic buffer
            dstsize = srcsize * 4
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    decoded_size = <size_t> dstsize

    with nogil:
        ret = BrotliDecoderDecompress(
            <size_t> srcsize,
            <const uint8_t*> &src[0],
            &decoded_size,
            <uint8_t*> &dst[0]
        )
    if ret != BROTLI_DECODER_RESULT_SUCCESS:
        raise BrotliError('BrotliDecoderDecompress', ret)

    del dst
    return _return_output(out, dstsize, decoded_size, outgiven)
