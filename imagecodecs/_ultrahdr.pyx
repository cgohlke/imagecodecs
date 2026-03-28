# imagecodecs/_ultrahdr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2024-2026, Christoph Gohlke
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

"""ULTRAHDR (Ultra HDR, JPEG_R) codec for the imagecodecs package."""

include '_shared.pxi'

from libultrahdr cimport *


class ULTRAHDR:
    """ULTRAHDR codec constants."""

    available = True

    class CG(enum.IntEnum):
        """ULTRAHDR color gamut."""

        UNSPECIFIED = UHDR_CG_UNSPECIFIED
        BT_709 = UHDR_CG_BT_709
        DISPLAY_P3 = UHDR_CG_DISPLAY_P3
        BT_2100 = UHDR_CG_BT_2100

    class CT(enum.IntEnum):
        """ULTRAHDR color transfer."""

        UNSPECIFIED = UHDR_CT_UNSPECIFIED
        LINEAR = UHDR_CT_LINEAR
        HLG = UHDR_CT_HLG
        PQ = UHDR_CT_PQ
        SRGB = UHDR_CT_SRGB

    class CR(enum.IntEnum):
        """ULTRAHDR color range."""

        UNSPECIFIED = UHDR_CR_UNSPECIFIED
        LIMITED_RANGE = UHDR_CR_LIMITED_RANGE
        FULL_RANGE = UHDR_CR_FULL_RANGE

    class CODEC(enum.IntEnum):
        """ULTRAHDR codec."""

        JPEG = UHDR_CODEC_JPG
        HEIF = UHDR_CODEC_HEIF
        AVIF = UHDR_CODEC_AVIF

    class USAGE(enum.IntEnum):
        """ULTRAHDR encoder preset."""

        REALTIME = UHDR_USAGE_REALTIME
        QUALITY = UHDR_USAGE_BEST_QUALITY


class UltrahdrError(RuntimeError):
    """ULTRAHDR codec exceptions."""

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


def ultrahdr_check(const uint8_t[::1] data, /):
    """Return whether data is ULTRAHDR encoded or None if unknown."""
    if data.nbytes < 12:
        return False
    return bool(is_uhdr_image(<void*> &data[0], <int> data.nbytes))


def ultrahdr_encode(
    data,  # High Dynamic Range
    /,
    level=None,
    *,
    sdr=None,  # Standard Dynamic Range
    scale=None,
    gamut=None,
    transfer=None,
    nits=None,
    boostmin=None,
    boostmax=None,
    crange=None,
    usage=None,
    codec=None,
    out=None,
):
    """Return ULTRAHDR encoded image.

    Only 64bppRGBAHalfFloat and 32bppRGBA1010102 data are supported.

    SDR is an optional 32bppRGBA8888 sRGB companion image.
    If omitted, the library tone-maps the HDR input automatically.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        numpy.ndarray sdr_src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        int quality = _default_value(level, -1, 0, 100)
        int scale_factor = _default_value(scale, 1, 1, 128)
        float cnits = _default_value(nits, 0.0, None, 10000)
        float cboostmin = 0.0
        float cboostmax = 0.0
        bint has_sdr
        bint has_boost
        uhdr_enc_preset_t preset = _enum_value(
            usage, ULTRAHDR.USAGE, UHDR_USAGE_BEST_QUALITY
        )
        uhdr_codec_t output_format = _enum_value(
            codec, ULTRAHDR.CODEC, UHDR_CODEC_JPG
        )
        uhdr_raw_image_t raw_image
        uhdr_raw_image_t sdr_image
        uhdr_compressed_image_t* compressed_image
        uhdr_codec_private_t* encoder = NULL
        uhdr_error_info_t error

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.dtype.char != 'e' and not (
        src.dtype.kind == 'u' and src.dtype.itemsize == 4
    ):
        raise ValueError(f'{src.dtype} not supported')
    if (
        src.ndim not in {2, 3}
        or src.shape[0] > 65535
        or src.shape[1] > 65535
        or src.nbytes > INT32_MAX
        or (src.ndim == 2 and not (
            src.dtype.kind == 'u' and src.dtype.itemsize == 4
        ))
        or (src.ndim == 3 and (src.dtype.char != 'e' or src.shape[2] != 4))
    ):
        raise ValueError('data shape or dtype not supported')

    memset(<void*> &raw_image, 0, sizeof(uhdr_raw_image_t))
    raw_image.h = <int> src.shape[0]
    raw_image.w = <int> src.shape[1]
    raw_image.range = UHDR_CR_FULL_RANGE
    raw_image.planes[UHDR_PLANE_PACKED] = <void*> src.data
    raw_image.stride[UHDR_PLANE_PACKED] = <unsigned int> src.shape[1]

    if src.ndim == 2:
        raw_image.fmt = UHDR_IMG_FMT_32bppRGBA1010102
        raw_image.ct = UHDR_CT_HLG
        raw_image.cg = UHDR_CG_BT_2100
    else:
        raw_image.fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
        raw_image.ct = UHDR_CT_LINEAR
        raw_image.cg = UHDR_CG_BT_2100

    if gamut is not None:
        raw_image.cg = _enum_value(gamut, ULTRAHDR.CG)
    if transfer is not None:
        raw_image.ct = _enum_value(transfer, ULTRAHDR.CT)
    if crange is not None:
        raw_image.range = _enum_value(crange, ULTRAHDR.CR)

    has_sdr = sdr is not None
    has_boost = boostmin is not None and boostmax is not None

    if has_boost:
        cboostmin = boostmin
        cboostmax = boostmax

    if has_sdr:
        sdr_src = numpy.ascontiguousarray(sdr, dtype=numpy.uint8)
        if (
            sdr_src.ndim != 3
            or sdr_src.shape[0] != src.shape[0]
            or sdr_src.shape[1] != src.shape[1]
            or sdr_src.shape[2] != 4
        ):
            raise ValueError('sdr data shape not supported')
        memset(<void*> &sdr_image, 0, sizeof(uhdr_raw_image_t))
        sdr_image.fmt = UHDR_IMG_FMT_32bppRGBA8888
        sdr_image.ct = UHDR_CT_SRGB
        sdr_image.cg = UHDR_CG_BT_709
        sdr_image.range = UHDR_CR_FULL_RANGE
        sdr_image.w = <unsigned int> sdr_src.shape[1]
        sdr_image.h = <unsigned int> sdr_src.shape[0]
        sdr_image.planes[UHDR_PLANE_PACKED] = <void*> sdr_src.data
        sdr_image.stride[UHDR_PLANE_PACKED] = <unsigned int> sdr_src.shape[1]

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

            if has_sdr:
                error = uhdr_enc_set_raw_image(
                    encoder, &sdr_image, UHDR_SDR_IMG
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_raw_image',
                        error.error_code,
                        error.detail
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

            if has_boost:
                error = uhdr_enc_set_min_max_content_boost(
                    encoder, cboostmin, cboostmax
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_min_max_content_boost',
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
                <const char*> compressed_image.data
            )
        else:
            dst = out
            dstsize = dst.nbytes
            if dstsize < <ssize_t> compressed_image.data_sz:
                raise ValueError(
                    f'output too small {dstsize} < {compressed_image.data_sz}'
                )
            memcpy(
                <void*> &dst[0],
                <const void*> compressed_image.data,
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
    /,
    *,
    dtype=None,
    transfer=None,
    boost=None,
    gpu=False,
    out=None,
):
    """Return decoded ULTRAHDR image.

    dtype controls output format and color transfer:

        - float16 (default): linear RGBA, shape (H, W, 4)
        - uint8: SRGB RGBA, shape (H, W, 4)
        - uint16: unpacked RGBA1010102, Hybrid Log-Gamma (HLG), shape (H, W, 4)
        - uint32: packed RGBA1010102, Hybrid Log-Gamma (HLG), shape (H, W)

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t width, height, bpp
        bint enable_gpu = bool(gpu)
        float display_boost = 0.0 if boost is None else boost
        uhdr_codec_private_t* decoder = NULL
        uhdr_compressed_image_t compressed_image
        uhdr_raw_image_t* raw_image
        uhdr_img_fmt_t fmt
        uhdr_color_transfer_t otf
        uhdr_error_info_t error

    if data is out:
        raise ValueError('cannot decode in-place')

    if dtype is None:
        dtype = numpy.float16
    dtype = numpy.dtype(dtype)
    if dtype.char == 'e':
        fmt = UHDR_IMG_FMT_64bppRGBAHalfFloat
        otf = UHDR_CT_LINEAR
    elif dtype.char == 'B':
        fmt = UHDR_IMG_FMT_32bppRGBA8888
        otf = UHDR_CT_SRGB
    elif dtype.char == 'H':
        fmt = UHDR_IMG_FMT_32bppRGBA1010102
        otf = UHDR_CT_HLG  # or UHDR_CT_PQ
    elif dtype.kind == 'u' and dtype.itemsize == 4:
        fmt = UHDR_IMG_FMT_32bppRGBA1010102
        otf = UHDR_CT_HLG  # or UHDR_CT_PQ
    else:
        raise ValueError(f'{dtype} not supported')

    if transfer is not None:
        otf = transfer

    memset(<void*> &compressed_image, 0, sizeof(uhdr_compressed_image_t))
    compressed_image.data = <void*> &src[0]
    compressed_image.data_sz = <size_t> src.nbytes
    compressed_image.capacity = <size_t> src.nbytes
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

            with gil:
                if dtype.itemsize == 4:
                    out = _create_array(out, (height, width), dtype)
                    bpp = 4
                else:
                    out = _create_array(out, (height, width, 4), dtype)
                    bpp = dtype.itemsize * 4
                dst = out

            if fmt == UHDR_IMG_FMT_32bppRGBA1010102 and bpp == 8:
                _uhdr_unpack_rgba1010102(
                    <uint16_t*> dst.data,
                    <uint32_t*> raw_image.planes[UHDR_PLANE_PACKED],
                    <ssize_t> raw_image.stride[UHDR_PLANE_PACKED],
                    height,
                    width
                )
            else:
                _uhdr_copy_image(
                    dst.data,
                    <char*> raw_image.planes[UHDR_PLANE_PACKED],
                    width * bpp,
                    <ssize_t> raw_image.stride[UHDR_PLANE_PACKED] * bpp,
                    height
                )

    finally:
        if decoder != NULL:
            uhdr_release_decoder(decoder)

    return dst


cdef inline void _uhdr_copy_image(
    char* dst,
    char* src,
    const ssize_t dststride,
    const ssize_t srcstride,
    const ssize_t height,
) noexcept nogil:
    """Copy decoded image."""
    cdef:
        ssize_t i

    if dststride == srcstride:
        memcpy(<void*> dst, <const void*> src, dststride * height)
        return
    for i in range(height):
        memcpy(
            <void*> (dst + i * dststride),
            <const void*> (src + i * srcstride),
            dststride
        )


cdef inline void _uhdr_unpack_rgba1010102(
    uint16_t* dst,
    uint32_t* src,
    const ssize_t stride,
    const ssize_t height,
    const ssize_t width,
) noexcept nogil:
    """Unpack UHDR_IMG_FMT_32bppRGBA1010102 to uint16."""
    cdef:
        ssize_t i, j, k
        uint32_t rgba

    k = 0
    for j in range(height):
        for i in range(width):
            rgba = src[i]
            # 10 bit red
            dst[k] = <uint16_t> (rgba & 0x3ff)
            k += 1
            # 10 bit green
            dst[k] = <uint16_t> ((rgba >> 10) & 0x3ff)
            k += 1
            # 10 bit blue
            dst[k] = <uint16_t> ((rgba >> 20) & 0x3ff)
            k += 1
            # 2 bit alpha
            dst[k] = <uint16_t> ((rgba >> 30) & 0x3)  # << 8
            k += 1
        src += stride
