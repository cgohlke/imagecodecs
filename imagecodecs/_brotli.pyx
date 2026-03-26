# imagecodecs/_brotli.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2019-2026, Christoph Gohlke
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

"""BROTLI codec for the imagecodecs package."""

include '_shared.pxi'

from brotli cimport *


class BROTLI:
    """BROTLI codec constants."""

    available = True

    class MODE(enum.IntEnum):
        """BROTLI codec modes."""

        GENERIC = BROTLI_MODE_GENERIC
        TEXT = BROTLI_MODE_TEXT
        FONT = BROTLI_MODE_FONT


class BrotliError(RuntimeError):
    """BROTLI codec exceptions."""

    def __init__(self, func, err):
        err = {
            None: 'NULL',
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

    return f'brotli {ver >> 24}.{(ver >> 12) & 0xFFF}.{ver & 0xFFF}'


def brotli_check(const uint8_t[::1] data, /):
    """Return whether data is BROTLI encoded or None if unknown."""


def brotli_encode(
    data,
    /,
    level=None,
    *,
    mode=None,
    lgwin=None,
    out=None,
):
    """Return BROTLI encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.nbytes
        ssize_t dstsize
        size_t encoded_size
        BROTLI_BOOL ret = BROTLI_FALSE
        BrotliEncoderMode mode_ = _enum_value(
            mode, BROTLI.MODE, BROTLI_MODE_GENERIC
        )
        int quality_ = _default_value(
            # anything higher than 9 is very slow
            level, 4, BROTLI_MIN_QUALITY, BROTLI_MAX_QUALITY
        )
        int lgwin_ = _default_value(
            lgwin, 22, BROTLI_MIN_WINDOW_BITS, BROTLI_MAX_WINDOW_BITS
        )
        # int lgblock_ = _default_value(
        #     lgblock,
        #     0,
        #     BROTLI_MIN_INPUT_BLOCK_BITS,
        #     BROTLI_MAX_INPUT_BLOCK_BITS
        # )

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming interface with dynamic buffer
            dstsize = <ssize_t> BrotliEncoderMaxCompressedSize(
                <size_t> srcsize
            )
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes
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


def brotli_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded BROTLI data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        size_t decoded_size
        BrotliDecoderResult ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        return _brotli_decode(data, outtype)

    if out is None:
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes
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


cdef _brotli_decode(const uint8_t[::1] src, outtype):
    """Decompress using streaming API."""
    cdef:
        output_t* output = NULL
        uint8_t* next_in = NULL
        uint8_t* next_out = NULL
        size_t srcsize = <size_t> src.nbytes
        size_t available_in, available_out, incsize
        BrotliDecoderState* state = NULL
        BrotliDecoderResult ret = BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT

    try:
        with nogil:
            state = BrotliDecoderCreateInstance(NULL, NULL, NULL)
            if state == NULL:
                raise BrotliError('BrotliDecoderCreateInstance', None)

            output = output_new(NULL, _align_size_t(srcsize * 2))
            if output == NULL:
                raise MemoryError('output_new failed')

            next_in = <uint8_t*> &src[0]
            available_in = srcsize
            next_out = output.data
            available_out = output.size

            while ret == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:

                ret = BrotliDecoderDecompressStream(
                    state,
                    &available_in,
                    <const uint8_t**> &next_in,
                    &available_out,
                    &next_out,
                    NULL
                )

                if (
                    ret == BROTLI_DECODER_RESULT_ERROR
                    or ret == BROTLI_DECODER_RESULT_NEEDS_MORE_INPUT
                ):
                    raise BrotliError('BrotliDecoderDecompressStream', ret)

                output.pos = output.size - available_out

                if ret == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT:
                    incsize = _align_size_t(srcsize)
                    if output_resize(output, output.size + incsize) == 0:
                        raise MemoryError('output_resize failed')
                    next_out = output.data + output.pos
                    available_out += incsize

        if ret != BROTLI_DECODER_RESULT_SUCCESS:
            raise BrotliError('BrotliDecoderDecompressStream', ret)

        out = _create_output(
            outtype, output.pos, <const char*> output.data
        )

    finally:
        output_del(output)
        if state != NULL:
            BrotliDecoderDestroyInstance(state)

    return out


# Output Stream ###############################################################

ctypedef struct output_t:
    uint8_t* data
    size_t size
    size_t pos
    size_t used
    int owner


cdef output_t* output_new(uint8_t* data, size_t size) noexcept nogil:
    """Return new output."""
    cdef:
        output_t* output = <output_t*> calloc(1, sizeof(output_t))

    if output == NULL:
        return NULL
    output.size = size
    output.used = 0
    output.pos = 0
    if data == NULL:
        output.owner = 1
        output.data = <uint8_t*> malloc(size)
    else:
        output.owner = 0
        output.data = data
    if output.data == NULL:
        free(output)
        return NULL
    return output


cdef void output_del(output_t* output) noexcept nogil:
    """Free output."""
    if output != NULL:
        if output.owner != 0:
            free(output.data)
        free(output)


cdef int output_seek(output_t* output, size_t pos) noexcept nogil:
    """Seek output to position."""
    if output == NULL or pos > output.size:
        return 0
    output.pos = pos
    if pos > output.used:
        output.used = pos
    return 1


cdef int output_resize(output_t* output, size_t newsize) noexcept nogil:
    """Resize output."""
    cdef:
        uint8_t* tmp

    if output == NULL or newsize == 0 or output.used > output.size:
        return 0
    if newsize == output.size or output.owner == 0:
        return 1

    tmp = <uint8_t*> realloc(<void*> output.data, newsize)
    if tmp == NULL:
        return 0
    output.data = tmp
    output.size = newsize
    return 1
