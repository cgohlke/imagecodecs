# imagecodecs/_ultrahdr.pyx
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

"""Ultra HDR (JPEG_R) codec for the imagecodecs package."""

include '_shared.pxi'

from libultrahdr cimport *


class ULTRAHDR:
    """Ultra HDR codec constants."""

    available = True

    class CG(enum.IntEnum):
        """Ultra HDR color gamut."""

        UNSPECIFIED = UHDR_CG_UNSPECIFIED
        BT_709 = UHDR_CG_BT_709
        DISPLAY_P3 = UHDR_CG_DISPLAY_P3
        BT_2100 = UHDR_CG_BT_2100

    class CT(enum.IntEnum):
        """Ultra HDR color transfer."""

        UNSPECIFIED = UHDR_CT_UNSPECIFIED
        LINEAR = UHDR_CT_LINEAR
        HLG = UHDR_CT_HLG
        PQ = UHDR_CT_PQ
        SRGB = UHDR_CT_SRGB

    class CR(enum.IntEnum):
        """Ultra HDR color range."""

        UNSPECIFIED = UHDR_CR_UNSPECIFIED
        LIMITED_RANGE = UHDR_CR_LIMITED_RANGE
        FULL_RANGE = UHDR_CR_FULL_RANGE

    class CODEC(enum.IntEnum):
        """Ultra HDR codec."""

        JPEG = UHDR_CODEC_JPG
        HEIF = UHDR_CODEC_HEIF
        AVIF = UHDR_CODEC_AVIF

    class USAGE(enum.IntEnum):
        """Ultra HDR codec."""

        REALTIME = UHDR_USAGE_REALTIME
        QUALITY = UHDR_USAGE_BEST_QUALITY


class UltrahdrError(RuntimeError):
    """Ultra HDR codec exceptions."""

    def __init__(self, func, code=None, detail=None):
        if code is None:
            msg = 'NULL'
        elif detail:
            msg = detail.decode()
        else:
            msg = f'error code {code}'
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def ultrahdr_version():
    """Return libultrahdr library version string."""
    return 'libultrahdr ' + UHDR_LIB_VERSION_STR.decode()


def ultrahdr_check(const uint8_t[::1] data):
    """Return whether data is Ultra HDR encoded."""
    return bool(is_uhdr_image(<void*> &data[0], <int> data.size))


def ultrahdr_encode(
    data,
    level=None,
    scale=None,
    gamut=None,
    transfer=None,
    nits=None,
    crange=None,
    usage=None,
    codec=None,
    out=None
):
    """Return Ultra HDR encoded data.

    Only 64bppRGBAHalfFloat and 32bppRGBA1010102 data are supported.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        int quality = _default_value(level, -1, 0, 100)
        int scale_factor = _default_value(scale, 1, 1, 128)
        float cnits = _default_value(nits, 0, 203, 10000)
        uhdr_enc_preset_t preset = UHDR_USAGE_BEST_QUALITY
        uhdr_codec_t output_format = UHDR_CODEC_JPG
        uhdr_raw_image_t raw_image
        uhdr_compressed_image_t* compressed_image
        uhdr_codec_private_t* encoder
        uhdr_error_info_t error

    if src.dtype.char not in 'eLI':
        raise ValueError(f'{src.dtype} not supported')
    if (
        src.ndim not in {2, 3}
        or src.shape[0] > 65535
        or src.shape[1] > 65535
        or src.nbytes > 2147483647
        or (src.ndim == 2 and src.dtype.char not in 'IL')
        or (src.ndim == 3 and (src.dtype.char != 'e' or src.shape[2] != 4))
    ):
        raise ValueError('data shape not supported')

    if usage is not None:
        preset = usage
    if codec is not None:
        output_format = codec

    memset(&raw_image, 0, sizeof(uhdr_raw_image_t))
    raw_image.h = <int> src.shape[0]
    raw_image.w = <int> src.shape[1]
    raw_image.range = UHDR_CR_FULL_RANGE
    raw_image.planes[UHDR_PLANE_PACKED] = <void *> src.data
    raw_image.stride[UHDR_PLANE_PACKED] = <unsigned int> src.shape[1]

    if src.ndim == 2:
        raw_image.fmt = UHDR_IMG_FMT_32bppRGBA1010102
        raw_image.ct = UHDR_CT_HLG
    else:
        raw_image.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
        raw_image.ct = UHDR_CT_LINEAR

    if gamut is not None:
        raw_image.cg = gamut
    if transfer is not None:
        raw_image.ct = transfer
    if crange is not None:
        raw_image.range = crange

    try:
        with nogil:
            encoder = uhdr_create_encoder()
            if encoder == NULL:
                raise UltrahdrError('uhdr_create_encoder')

            error = uhdr_enc_set_preset(encoder, preset)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_enc_set_preset', error.error_code, error.detail
                )

            error = uhdr_enc_set_raw_image(encoder, &raw_image, UHDR_HDR_IMG)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_enc_set_raw_image', error.error_code, error.detail
                )

            if quality >= 0:
                error = uhdr_enc_set_quality(encoder, quality, UHDR_BASE_IMG)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_quality', error.error_code, error.detail
                    )

                error = uhdr_enc_set_quality(
                    encoder, quality, UHDR_GAIN_MAP_IMG
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_quality', error.error_code, error.detail
                    )

            if cnits > 0.0:
                error = uhdr_enc_set_target_display_peak_brightness(
                    encoder, cnits
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_target_display_peak_brightness',
                        error.error_code,
                        error.detail
                    )

            # error = uhdr_enc_set_exif_data(encoder, &exif)
            # if error.error_code != UHDR_CODEC_OK:
            #     raise UltrahdrError(
            #         'uhdr_enc_set_exif_data',
            #         error.error_code,
            #         error.detail
            #     )

            if output_format != UHDR_CODEC_JPG:
                error = uhdr_enc_set_output_format(encoder, output_format)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_output_format',
                        error.error_code,
                        error.detail
                    )

            error = uhdr_enc_set_gainmap_scale_factor(encoder, scale_factor)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_enc_set_gainmap_scale_factor',
                    error.error_code,
                    error.detail
                )

            # error = uhdr_enc_set_using_multi_channel_gainmap(encoder, mcgm)
            # if error.error_code != UHDR_CODEC_OK:
            #     raise UltrahdrError(
            #         'uhdr_enc_set_using_multi_channel_gainmap',
            #         error.error_code,
            #         error.detail
            #     )

            # error = uhdr_enc_set_gainmap_gamma(encoder, gamma)
            # if error.error_code != UHDR_CODEC_OK:
            #     raise UltrahdrError(
            #         'uhdr_enc_set_gainmap_gamma',
            #         error.error_code,
            #         error.detail
            #     )

            error = uhdr_encode(encoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_encode', error.error_code, error.detail
                )

            compressed_image = uhdr_get_encoded_stream(encoder)
            if compressed_image == NULL:
                raise UltrahdrError('uhdr_get_encoded_stream')

        out, dstsize, outgiven, outtype = _parse_output(out)

        if out is None:
            dstsize = <ssize_t> compressed_image.data_sz
            out = _create_output(
                outtype,
                compressed_image.data_sz,
                <const char *> compressed_image.data
            )
        else:
            dst = out
            dstsize = dst.nbytes
            if dstsize < compressed_image.data_sz:
                raise ValueError(
                    f'output too small {dstsize} < {compressed_image.data_sz}'
                )
            memcpy(
                <void *> &dst[0],
                <const void *> compressed_image.data,
                compressed_image.data_sz
            )
            del dst
            out = _return_output(
                out, dstsize, compressed_image.data_sz, outgiven
            )

    finally:
        if encoder != NULL:
            uhdr_release_encoder(encoder)

    return out


def ultrahdr_decode(
    data,
    dtype=None,
    transfer=None,
    boost=None,
    gpu=False,
    out=None
):
    """Return decoded Ultra HDR image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t i, j, instride, outstride, width, height, bpp
        uint32_t temp = 0
        uint16_t color = 0
        bint enable_gpu = bool(gpu)
        float display_boost = 0.0 if boost is None else boost
        char* outptr = NULL
        char* inptr = NULL
        uhdr_codec_private_t* decoder = NULL
        uhdr_compressed_image_t compressed_image
        uhdr_raw_image_t* raw_image
        uhdr_img_fmt_t fmt
        uhdr_color_transfer_t otf
        uhdr_error_info_t error

    if dtype is None:
        dtype = numpy.float16
    dtype = numpy.dtype(dtype)
    if dtype.char == 'e':
        fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
        otf = UHDR_CT_LINEAR
    elif dtype.char == 'B':
        fmt = UHDR_IMG_FMT_32bppRGBA8888
        otf = UHDR_CT_SRGB
    # elif dtype.char == 'H':
    #     # TODO: this is disabled because of crashes in CRT
    #     fmt = UHDR_IMG_FMT_32bppRGBA1010102
    #     otf = UHDR_CT_HLG  # or UHDR_CT_PQ
    elif dtype.kind == 'u' and dtype.itemsize == 4:
        fmt = UHDR_IMG_FMT_32bppRGBA1010102
        otf = UHDR_CT_HLG  # or UHDR_CT_PQ
    else:
        raise ValueError(f'{dtype} not supported')

    if transfer is not None:
        otf = transfer

    memset(<void *> &compressed_image, 0, sizeof(uhdr_compressed_image_t))
    compressed_image.data = <void *> &src[0]
    compressed_image.data_sz = <size_t> src.size
    compressed_image.capacity = <size_t> src.size
    compressed_image.cg = UHDR_CG_UNSPECIFIED
    compressed_image.ct = UHDR_CT_UNSPECIFIED
    compressed_image.range = UHDR_CR_UNSPECIFIED

    try:
        with nogil:
            decoder = uhdr_create_decoder()
            if decoder == NULL:
                raise UltrahdrError('uhdr_create_decoder')

            error = uhdr_dec_set_image(decoder, &compressed_image)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_dec_set_image', error.error_code, error.detail
                )

            error = uhdr_dec_set_out_img_format(decoder, fmt)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_dec_set_out_img_format',
                    error.error_code,
                    error.detail
                )

            error = uhdr_dec_set_out_color_transfer(decoder, otf)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_dec_set_out_color_transfer',
                    error.error_code,
                    error.detail
                )

            if display_boost >= 1.0:
                error = uhdr_dec_set_out_max_display_boost(
                    decoder, display_boost
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_dec_set_out_max_display_boost',
                        error.error_code,
                        error.detail
                    )

            if enable_gpu:
                error = uhdr_enable_gpu_acceleration(decoder, enable_gpu)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enable_gpu_acceleration',
                        error.error_code,
                        error.detail
                    )

            error = uhdr_dec_probe(decoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_dec_probe', error.error_code, error.detail
                )

            width = uhdr_dec_get_image_width(decoder)
            if width < 0:
                raise UltrahdrError('uhdr_dec_get_image_width', width)

            height = uhdr_dec_get_image_height(decoder)
            if height < 0:
                raise UltrahdrError('uhdr_dec_get_image_height', height)

            with gil:
                if dtype.itemsize == 4:
                    out = _create_array(out, (height, width), dtype)
                    bpp = dtype.itemsize
                else:
                    out = _create_array(out, (height, width, 4), dtype)
                    bpp = dtype.itemsize * 4
                dst = out
                outptr = dst.data

            error = uhdr_decode(decoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError(
                    'uhdr_decode', error.error_code, error.detail
                )

            raw_image = uhdr_get_decoded_image(decoder)
            if raw_image == NULL:
                raise UltrahdrError('uhdr_get_decoded_image')

            if raw_image.h != height or raw_image.w != width:
                raise ValueError(
                    f'shape mismatch ({raw_image.h}, {raw_image.w})'
                    f' != ({height}, {width})'
                )

            inptr = <char *> raw_image.planes[UHDR_PLANE_PACKED]
            instride = raw_image.stride[UHDR_PLANE_PACKED] * bpp
            outstride = width * bpp

            if fmt == UHDR_IMG_FMT_32bppRGBA1010102 and bpp == 8:
                # unpack RGBA1010102
                # TODO: this crashes in msvcrt, why?
                instride /= 2
                for i in range(raw_image.h):
                    for j in range(raw_image.w):
                        memcpy(&temp, <const void *> inptr[j * 4], 4)
                        j *= 8
                        # 10 bit red
                        color = <uint16_t> (temp & 4095)
                        memcpy(outptr + j, <const void *> &color, 2)
                        # 10 bit green
                        color = <uint16_t> ((temp >> 10) & 4095)
                        memcpy(outptr + j + 2, <const void *> &color, 2)
                        # 10 bit blue
                        color = <uint16_t> ((temp >> 20) & 4095)
                        memcpy(outptr + j + 4, <const void *> &color, 2)
                        # 2 bit alpha
                        color = <uint16_t> ((temp >> 30) & 3)
                        memcpy(outptr + j + 6, <const void *> &color, 2)
                    inptr += instride
                    outptr += outstride
            else:
                for i in range(raw_image.h):
                    memcpy(
                        outptr + i * outstride,
                        inptr + i * instride,
                        outstride
                    )

    finally:
        if decoder != NULL:
            uhdr_release_decoder(decoder)

    return dst
