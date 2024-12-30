# imagecodecs/_jpegxs.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2024-2025, Christoph Gohlke
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

"""JPEGXS codec for the imagecodecs package."""

include '_shared.pxi'

from libjxs cimport *


class JPEGXS:
    """JPEGXS codec constants."""

    available = True

    class PROFILE(enum.IntEnum):
        """JPEGXS profiles."""

        AUTO = XS_PROFILE_AUTO
        UNRESTRICTED = XS_PROFILE_UNRESTRICTED
        LIGHT_422_10 = XS_PROFILE_LIGHT_422_10
        LIGHT_444_12 = XS_PROFILE_LIGHT_444_12
        LIGHT_SUBLINE_422_10 = XS_PROFILE_LIGHT_SUBLINE_422_10
        MAIN_420_12 = XS_PROFILE_MAIN_420_12
        MAIN_422_10 = XS_PROFILE_MAIN_422_10
        MAIN_444_12 = XS_PROFILE_MAIN_444_12
        MAIN_4444_12 = XS_PROFILE_MAIN_4444_12
        HIGH_420_12 = XS_PROFILE_HIGH_420_12
        HIGH_444_12 = XS_PROFILE_HIGH_444_12
        HIGH_4444_12 = XS_PROFILE_HIGH_4444_12
        MLS_12 = XS_PROFILE_MLS_12
        LIGHT_BAYER = XS_PROFILE_LIGHT_BAYER
        MAIN_BAYER = XS_PROFILE_MAIN_BAYER
        HIGH_BAYER = XS_PROFILE_HIGH_BAYER

    class LEVEL(enum.IntEnum):
        """JPEGXS levels."""

        AUTO = XS_LEVEL_AUTO
        UNRESTRICTED = XS_LEVEL_UNRESTRICTED
        K1_1 = XS_LEVEL_1K_1
        K2_1 = XS_LEVEL_2K_1
        K4_1 = XS_LEVEL_4K_1
        K4_2 = XS_LEVEL_4K_2
        K4_3 = XS_LEVEL_4K_3
        K8_1 = XS_LEVEL_8K_1
        K8_2 = XS_LEVEL_8K_2
        K8_3 = XS_LEVEL_8K_3
        K10_1 = XS_LEVEL_10K_1

    class SUBLEVEL(enum.IntEnum):
        """JPEGXS sublevels."""

        AUTO = XS_SUBLEVEL_AUTO
        UNRESTRICTED = XS_SUBLEVEL_UNRESTRICTED
        FULL = XS_SUBLEVEL_FULL
        BPP_12 = XS_SUBLEVEL_12_BPP
        BPP_9 = XS_SUBLEVEL_9_BPP
        BPP_6 = XS_SUBLEVEL_6_BPP
        BPP_4 = XS_SUBLEVEL_4_BPP
        BPP_3 = XS_SUBLEVEL_3_BPP
        BPP_2 = XS_SUBLEVEL_2_BPP


class JpegxsError(RuntimeError):
    """JPEGXS codec exceptions."""


def jpegxs_version():
    """Return libjxs library version string."""
    cdef:
        char* version = xs_get_version_str()

    return f'libjxs {version.decode()}'


def jpegxs_check(const uint8_t[::1] data):
    """Return whether data is JPEG XS encoded image."""
    sig = bytes(data[:12])
    return (
        sig == b'\x00\x00\x00\x0C\x4A\x58\x53\x20\x0D\x0A\x87\x0A'  # image
        or sig[:4] == b'\xFF\x10\xFF\x50'  # codestream or video
    )


def jpegxs_encode(
    data, config=None, bitspersample=None, verbose=None, out=None
):
    """Return JPEGXS encoded data."""
    cdef:
        const uint8_t[::1] dst  # must be const to write to bytes
        numpy.ndarray src = numpy.ascontiguousarray(data)
        ssize_t dstsize = 0
        ssize_t s, i, j, itemsize
        ssize_t config_str_max_len = 0
        size_t bitstream_buf_size = 0
        char* config_str = NULL
        uint8_t* src8 = NULL
        uint16_t* src16 = NULL
        xs_data_in_t* comptr = NULL
        xs_enc_context_t* ctx = NULL
        xs_image_t xs_image
        xs_config_t xs_config
        bint ret = False

    memset(&xs_image, 0, sizeof(xs_image))
    memset(&xs_config, 0, sizeof(xs_config))

    xs_config.verbose = int(verbose) if verbose else 0

    # TODO: handle planar config
    if src.ndim == 2:
        xs_image.height = <int> src.shape[0]
        xs_image.width = <int> src.shape[1]
        xs_image.ncomps = 1
    elif src.ndim == 3 and src.shape[2] <= 4:
        xs_image.height = <int> src.shape[0]
        xs_image.width = <int> src.shape[1]
        xs_image.ncomps = <int> src.shape[2]
    else:
        raise ValueError(f'data.ndim={src.ndim} not supported')

    if src.dtype.kind != 'u' or src.dtype.itemsize > 2:
        raise ValueError('{data.dtype=} not supported')

    itemsize = src.dtype.itemsize
    if bitspersample is None:
        xs_image.depth = <int> (itemsize * 8)
    elif 8 <= bitspersample < 16:
        xs_image.depth = bitspersample
    else:
        raise ValueError(f'{bitspersample=} not supported')

    if config is None:
        config = b'p=MLS.12'
        if xs_image.width < 32:
            config += b';nlx=2;nly=2'
        if xs_image.ncomps != 3:
            # TODO: define gains/priorities for other configurations
            raise ValueError(
                f'{xs_image.ncomps=} != 3 not supported without config'
            )
        if xs_image.height < 4 or xs_image.width < 4:
            # TODO: find config for smaller images
            raise ValueError(
                f'{xs_image.height=} or {xs_image.width=} < 4 '
                'not supported without config'
            )
    else:
        config = config.encode()
    config += b'\0'
    config_str = config
    config_str_max_len = len(config)

    try:
        with nogil:

            for s in range(xs_image.ncomps):
                xs_image.sx[s] = 1
                xs_image.sy[s] = 1

            ret = xs_allocate_image(&xs_image, 0)
            if not ret:
                raise JpegxsError('xs_allocate_image failed')

            # copy data
            # TODO: handle planar and strided data
            for s in range(xs_image.ncomps):
                comptr = xs_image.comps_array[s]
                j = 0
                if itemsize == 1:
                    src8 = <uint8_t*> &src.data[s * itemsize]
                    for i in range(xs_image.height * xs_image.width):
                        comptr[i] = <xs_data_in_t> src8[j]
                        j += xs_image.ncomps
                else:
                    src16 = <uint16_t*> &src.data[s * itemsize]
                    for i in range(xs_image.height * xs_image.width):
                        comptr[i] = <xs_data_in_t> src16[j]
                        j += xs_image.ncomps

            ret = xs_config_parse_and_init(
                &xs_config, &xs_image, config_str, config_str_max_len
            )
            if not ret:
                raise JpegxsError('xs_config_parse_and_init failed')

            ret = xs_enc_preprocess_image(&xs_config, &xs_image)
            if not ret:
                raise JpegxsError('xs_enc_preprocess_image failed')

            ctx = xs_enc_init(&xs_config, &xs_image)
            if ctx == NULL:
                raise JpegxsError('xs_enc_init failed')

        out, dstsize, outgiven, outtype = _parse_output(out)

        if out is None:
            if dstsize < 0:
                if xs_config.bitstream_size_in_bytes == <size_t> (-1):
                    dstsize = <ssize_t> (
                        xs_image.width * xs_image.height * xs_image.ncomps
                        * ((xs_image.depth + 7) >> 3) + 1024 * 1024
                    )
                else:
                    dstsize = <ssize_t> (
                        (xs_config.bitstream_size_in_bytes + 7) & (~0x7)
                    )
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.size

        with nogil:
            ret = xs_enc_image(
                ctx,
                &xs_image,
                <uint8_t*> &dst[0],
                <size_t> dstsize,
                &bitstream_buf_size
            )
        if not ret:
            raise JpegxsError('xs_enc_image failed')

    finally:
        # if xs_image.comps_array[0] != NULL:
        xs_free_image(&xs_image)
        if ctx != NULL:
            xs_enc_close(ctx)

    del dst
    return _return_output(out, dstsize, bitstream_buf_size, outgiven)


def jpegxs_decode(data, verbose=None, out=None):
    """Return decoded JPEGXS data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        size_t srcsize = src.size
        ssize_t s, i, j, itemsize
        uint8_t* dst8 = NULL
        uint16_t* dst16 = NULL
        xs_data_in_t* comptr = NULL
        xs_dec_context_t* ctx = NULL
        xs_image_t xs_image
        xs_config_t xs_config
        bint ret = False

    if data is out:
        raise ValueError('cannot decode in-place')

    memset(&xs_image, 0, sizeof(xs_image))
    memset(&xs_config, 0, sizeof(xs_config))

    xs_config.verbose = int(verbose) if verbose else 0

    try:
        with nogil:

            ret = xs_dec_probe(
                <uint8_t *> &src[0], srcsize, &xs_config, &xs_image
            )
            if not ret:
                raise JpegxsError('xs_dec_probe failed')

            ret = xs_allocate_image(&xs_image, 0)
            if not ret:
                raise JpegxsError('xs_allocate_image failed')

            ctx = xs_dec_init(&xs_config, &xs_image)
            if ctx == NULL:
                raise JpegxsError('xs_dec_init failed')

            ret = xs_dec_bitstream(ctx, <void *> &src[0], srcsize, &xs_image)
            if not ret:
                raise JpegxsError('xs_dec_bitstream failed')

            ret = xs_dec_postprocess_image(&xs_config, &xs_image)
            if not ret:
                raise JpegxsError('xs_dec_postprocess_image failed')

        if xs_image.ncomps > 1:
            shape = (xs_image.height, xs_image.width, xs_image.ncomps)
        else:
            shape = (xs_image.height, xs_image.width)
        if 0 < xs_image.depth <= 8:
            dtype = numpy.uint8
        elif 8 < xs_image.depth <= 16:
            dtype = numpy.uint16
        else:
            raise ValueError(f'invalid {xs_image.depth=}')

        out = _create_array(out, shape, dtype)
        dst = out
        itemsize = out.dtype.itemsize

        with nogil:
            # copy data
            # TODO: handle planar output
            for s in range(xs_image.ncomps):
                if xs_image.sx[s] != 1 or xs_image.sy[s] != 1:
                    raise ValueError(
                        f'cannot handle {xs_image.sx[s]=} or {xs_image.sy[s]=}'
                    )
                comptr = xs_image.comps_array[s]
                j = 0
                if itemsize == 1:
                    dst8 = <uint8_t*> &dst.data[s * itemsize]
                    for i in range(xs_image.height * xs_image.width):
                        dst8[j] = <uint8_t> comptr[i]
                        j += xs_image.ncomps
                else:
                    dst16 = <uint16_t*> &dst.data[s * itemsize]
                    for i in range(xs_image.height * xs_image.width):
                        dst16[j] = <uint16_t> comptr[i]
                        j += xs_image.ncomps

    finally:
        # if xs_image.comps_array[0] != NULL:
        xs_free_image(&xs_image)
        if ctx != NULL:
            xs_dec_close(ctx)

    return out
