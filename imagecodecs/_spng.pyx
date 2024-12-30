# imagecodecs/_spng.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021-2025, Christoph Gohlke
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

"""SPNG codec for the imagecodecs package."""

__version__ = '2023.3.16'

include '_shared.pxi'

from libspng cimport *


class SPNG:
    """SPNG codec constants."""

    available = True

    class FMT(enum.IntEnum):
        """SPNG codec formats."""

        RGBA8 = SPNG_FMT_RGBA8
        RGBA16 = SPNG_FMT_RGBA16
        RGB8 = SPNG_FMT_RGB8
        GA8 = SPNG_FMT_GA8
        GA16 = SPNG_FMT_GA16
        G8 = SPNG_FMT_G8


class SpngError(RuntimeError):
    """SPNG codec exceptions."""

    def __init__(self, func, err):
        cdef:
            char* error_message
            int error_value

        try:
            error_value = int(err)
            error_message = <char*> spng_strerror(error_value)
            msg = error_message.decode().strip()
        except Exception:
            msg = 'NULL' if err is None else f'unknown error {err!r}'
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def spng_version():
    """Return libspng library version string."""
    return 'libspng {}.{}.{}'.format(
        SPNG_VERSION_MAJOR, SPNG_VERSION_MINOR, SPNG_VERSION_PATCH
    )


def spng_check(data):
    """Return whether data is PNG encoded image."""
    cdef:
        bytes sig = bytes(data[:8])

    return sig == b'\x89PNG\r\n\x1a\n'


def spng_encode(data, level=None, out=None):
    """Return PNG encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        size_t srcsize = <size_t> src.nbytes
        int samples = <int> src.shape[2] if src.ndim == 3 else 1
        int clevel = _default_value(level, -1, -1, 9)
        int err = 0
        int flags = 0
        size_t output_size
        void* output = NULL
        spng_ctx* ctx = NULL
        spng_ihdr ihdr

    if not (
        src.dtype in {numpy.uint8, numpy.uint16}
        and src.ndim in {2, 3}
        and src.shape[0] <= 2147483647
        and src.shape[1] <= 2147483647
        and samples <= 4
    ):
        raise ValueError('invalid data shape or dtype')

    ihdr.width = <uint32_t> src.shape[1]
    ihdr.height = <uint32_t> src.shape[0]
    ihdr.bit_depth = <uint8_t> (src.dtype.itemsize * 8)

    # libpng can read these files but spng_decode fails with "invalid format"
    # if ihdr.bit_depth > 8:
    #     if samples == 1:
    #         raise ValueError('SPNG_FMT_G16 not supported')
    #     if samples == 2:
    #         raise ValueError('SPNG_FMT_GA16 not supported')
    #     if samples == 3:
    #         raise ValueError('SPNG_FMT_RGB16 not supported')
    # elif samples == 2:
    #     raise ValueError('SPNG_FMT_GA8 not supported')

    try:
        with nogil:
            ctx = spng_ctx_new(SPNG_CTX_ENCODER)
            if ctx == NULL:
                raise SpngError('spng_ctx_new', None)

            if samples == 1:
                ihdr.color_type = SPNG_COLOR_TYPE_GRAYSCALE
                # ihdr.color_type = SPNG_COLOR_TYPE_INDEXED
            elif samples == 2:
                flags = SPNG_DECODE_TRNS
                ihdr.color_type = SPNG_COLOR_TYPE_GRAYSCALE_ALPHA
            elif samples == 3:
                ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR
            elif samples == 4:
                flags = SPNG_DECODE_TRNS
                ihdr.color_type = SPNG_COLOR_TYPE_TRUECOLOR_ALPHA
            else:
                raise ValueError(f'{samples=} not supported')

            ihdr.compression_method = 0
            ihdr.filter_method = SPNG_FILTER_NONE
            ihdr.interlace_method = SPNG_INTERLACE_NONE

            err = spng_set_ihdr(ctx, &ihdr)
            if err != SPNG_OK:
                raise SpngError('spng_set_ihdr', err)

            err = spng_set_option(ctx, SPNG_IMG_COMPRESSION_LEVEL, clevel)
            if err != SPNG_OK:
                raise SpngError('spng_set_option', err)

            err = spng_set_option(ctx, SPNG_ENCODE_TO_BUFFER, 1)
            if err != SPNG_OK:
                raise SpngError('spng_set_option', err)

            err = spng_encode_image(
                ctx,
                <const void *> src.data,
                srcsize,
                SPNG_FMT_PNG,
                SPNG_ENCODE_FINALIZE
            )
            if err != SPNG_OK:
                raise SpngError('spng_encode_image', err)

            output = spng_get_png_buffer(ctx, &output_size, &err)
            if err != SPNG_OK:
                raise SpngError('spng_get_png_buffer', err)
            if output == NULL:
                raise SpngError('spng_get_png_buffer', None)

        out, dstsize, outgiven, outtype = _parse_output(out)

        if out is None:
            if dstsize < 0:
                dstsize = output_size
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.size
        if <size_t> dstsize < output_size:
            raise RuntimeError('output too small')

        with nogil:
            memcpy(<void*> &dst[0], <const void*> output, output_size)

    finally:
        if ctx != NULL:
            spng_ctx_free(ctx)
        if output != NULL:
            free(output)

    del dst
    return _return_output(out, dstsize, output_size, outgiven)


def spng_decode(data, out=None):
    """Return decoded PNG image.

    Supported formats: G8, RGB8, RGBA8, RGBA16
    Not supported: GA8, GA16, G16, RGB16

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t itemsize
        size_t out_size
        int samples = 0
        int err = 0
        int fmt = 0
        int flags = 0
        spng_ctx* ctx = NULL
        spng_ihdr ihdr

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        with nogil:
            ctx = spng_ctx_new(0)
            if ctx == NULL:
                raise SpngError('spng_ctx_new', None)

            err = spng_set_png_buffer(
                ctx,
                <const void*> &src[0],
                <size_t> srcsize
            )
            if err != SPNG_OK:
                raise SpngError('spng_set_png_buffer', err)

            err = spng_get_ihdr(ctx, &ihdr)
            if err != SPNG_OK:
                raise SpngError('spng_get_ihdr', err)

            itemsize = 1 if ihdr.bit_depth <= 8 else 2
            if ihdr.color_type == SPNG_COLOR_TYPE_GRAYSCALE:
                samples = 1
                if itemsize == 1:
                    fmt = SPNG_FMT_G8
                else:
                    fmt = 128  # TODO: SPNG_FMT_G16 not defined
            elif ihdr.color_type == SPNG_COLOR_TYPE_TRUECOLOR:
                samples = 3
                if itemsize == 1:
                    fmt = SPNG_FMT_RGB8
                else:
                    fmt = 8  # TODO: SPNG_FMT_RGB16 not defined
            elif ihdr.color_type == SPNG_COLOR_TYPE_INDEXED:
                samples = 3
                fmt = SPNG_FMT_RGB8
            elif ihdr.color_type == SPNG_COLOR_TYPE_GRAYSCALE_ALPHA:
                samples = 2
                if itemsize == 1:
                    fmt = SPNG_FMT_GA8  # doesn't work
                else:
                    fmt = SPNG_FMT_GA16  # doesn't work
            elif ihdr.color_type == SPNG_COLOR_TYPE_TRUECOLOR_ALPHA:
                samples = 4
                if itemsize == 1:
                    fmt = SPNG_FMT_RGBA8
                else:
                    fmt = SPNG_FMT_RGBA16
            else:
                raise ValueError(
                    f'invalid ihdr.color_type {int(ihdr.color_type)!r}'
                )

            err = spng_decoded_image_size(ctx, fmt, &out_size)
            if err != SPNG_OK:
                raise SpngError('spng_decoded_image_size', err)

        dtype = numpy.dtype(f'u{itemsize}')
        if samples > 1:
            shape = int(ihdr.height), int(ihdr.width), int(samples)
        else:
            shape = int(ihdr.height), int(ihdr.width)

        out = _create_array(out, shape, dtype)
        dst = out
        if <size_t> dst.nbytes != out_size:
            raise RuntimeError('out.nbytes != spng_decoded_image_size')

        with nogil:
            err = spng_decode_image(
                ctx, <void *> dst.data, out_size, fmt, flags
            )
            if err != SPNG_OK:
                raise SpngError('spng_decode_image', err)

    finally:
        if ctx != NULL:
            spng_ctx_free(ctx)

    return out
