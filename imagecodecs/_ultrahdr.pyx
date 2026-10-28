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

    def __init__(self, func, err=None):
        if err is None:
            msg = 'NULL'
        elif isinstance(err, dict):
            # converted uhdr_error_info_t
            if err['has_detail'] and err['detail']:
                msg = err['detail'].decode(errors='replace')
            else:
                msg = f'error code {err["error_code"]}'
        else:
            msg = f'error code {err}'
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def ultrahdr_version():
    """Return libultrahdr library version string."""
    return 'libultrahdr ' + UHDR_LIB_VERSION_STR.decode()


def ultrahdr_check(const uint8_t[::1] data not None, /):
    """Return whether data is ULTRAHDR encoded or None if unknown."""
    if data.shape[0] > INT32_MAX:
        return None
    return (
        data.shape[0] >= 12
        and bool(is_uhdr_image(<void*> data._data, <int> data.shape[0]))
    )


def ultrahdr_encode(
    data,  # High Dynamic Range
    /,
    level=None,  # quality 0..100
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
    gainmap_gamma=None,
    gainmap_level=None,  # quality
    multichannel=None,
    out=None,
):
    """Return ULTRAHDR encoded image.

    Only 64bppRGBAHalfFloat and 32bppRGBA1010102 data are supported.

    Linear float16 (64bppRGBAHalfFloat) input must be scaled so that
    1.0 == 203 nits (BT.2100 SDR-reference white). Callers working in a
    different linear-light range (for example, 1.0 == 1 nit scene-linear)
    must pre-scale before encoding.

    SDR is an optional 32bppRGBA8888 sRGB companion image.
    If omitted, the library tone-maps the HDR input automatically.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        numpy.ndarray sdr_src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        int quality = _default_value(level, -1, 0, 100)
        int gainmap_quality = _default_value(gainmap_level, -1, 0, 100)
        int scale_factor = _default_value(scale, 0, 1, 128)
        float cnits = _default_value(nits, 0.0, None, 10000)
        bint has_sdr = sdr is not None
        bint has_boost = boostmin is not None and boostmax is not None
        bint has_multichannel = multichannel is not None
        bint has_gamma = gainmap_gamma is not None
        float cgamma = gainmap_gamma if has_gamma else 0.0
        float cboostmax = boostmax if has_boost else 0.0
        float cboostmin = boostmin if has_boost else 0.0
        bint cmultichannel = bool(multichannel)
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

    if (boostmin is None) != (boostmax is None):
        raise ValueError('boostmin and boostmax must be specified together')

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

    if has_sdr:
        sdr_src = _array_input(sdr, out=None, dtype=numpy.uint8)
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
                raise UltrahdrError('uhdr_enc_set_preset', error)

            error = uhdr_enc_set_raw_image(encoder, &raw_image, UHDR_HDR_IMG)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_enc_set_raw_image', error)

            if has_sdr:
                error = uhdr_enc_set_raw_image(
                    encoder, &sdr_image, UHDR_SDR_IMG
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError('uhdr_enc_set_raw_image', error)

            if quality >= 0:
                error = uhdr_enc_set_quality(encoder, quality, UHDR_BASE_IMG)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError('uhdr_enc_set_quality', error)

            if gainmap_quality >= 0:
                error = uhdr_enc_set_quality(
                    encoder, gainmap_quality, UHDR_GAIN_MAP_IMG
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError('uhdr_enc_set_quality', error)

            if cnits > 0.0:
                error = uhdr_enc_set_target_display_peak_brightness(
                    encoder, cnits
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_target_display_peak_brightness', error
                    )

            if has_boost:
                error = uhdr_enc_set_min_max_content_boost(
                    encoder, cboostmin, cboostmax
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_min_max_content_boost', error
                    )

            # error = uhdr_enc_set_exif_data(encoder, &exif)
            # if error.error_code != UHDR_CODEC_OK:
            #     raise UltrahdrError('uhdr_enc_set_exif_data', error)

            if output_format != UHDR_CODEC_JPG:
                error = uhdr_enc_set_output_format(encoder, output_format)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError('uhdr_enc_set_output_format', error)

            if scale_factor > 0:
                error = uhdr_enc_set_gainmap_scale_factor(
                    encoder, scale_factor
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_gainmap_scale_factor', error
                    )

            if has_multichannel:
                error = uhdr_enc_set_using_multi_channel_gainmap(
                    encoder, cmultichannel
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_enc_set_using_multi_channel_gainmap', error
                    )

            if has_gamma:
                error = uhdr_enc_set_gainmap_gamma(encoder, cgamma)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError('uhdr_enc_set_gainmap_gamma', error)

            error = uhdr_encode(encoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_encode', error)

            compressed_image = uhdr_get_encoded_stream(encoder)
            if compressed_image == NULL:
                raise UltrahdrError('uhdr_get_encoded_stream')

        out, dstsize, outgiven, outtype = _parse_output(out)

        if out is None:
            if 0 <= dstsize < <ssize_t> compressed_image.data_sz:
                raise ValueError(
                    f'output too small {dstsize} < '
                    f'{compressed_image.data_sz}'
                )
            dstsize = <ssize_t> compressed_image.data_sz
            out = _create_output(
                outtype,
                compressed_image.data_sz,
                <const char*> compressed_image.data
            )
        else:
            dst = out
            dstsize = dst.shape[0]
            if dstsize < <ssize_t> compressed_image.data_sz:
                raise ValueError(
                    f'output too small {dstsize} < {compressed_image.data_sz}'
                )
            memcpy(
                <void*> dst._data,
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
    bint _info=False,
):
    """Return decoded ULTRAHDR image.

    dtype controls output format and color transfer:

        - float16 (default): linear RGBA, shape (H, W, 4)
        - uint8: SRGB RGBA, shape (H, W, 4)
        - uint16: unpacked RGBA1010102, Hybrid Log-Gamma (HLG), shape (H, W, 4)
          (alpha is not rescaled and in range 0..3)
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
        raise ValueError('cannot transform in-place')

    if src.shape[0] == 0:
        raise ValueError('src is empty')

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
        otf = _enum_value(transfer, ULTRAHDR.CT)

    bpp = 4 if dtype.itemsize == 4 else dtype.itemsize * 4

    try:
        with nogil:
            memset(
                <void*> &compressed_image, 0, sizeof(uhdr_compressed_image_t)
            )
            compressed_image.data = <void*> src._data
            compressed_image.data_sz = <size_t> src.shape[0]
            compressed_image.capacity = <size_t> src.shape[0]
            compressed_image.cg = UHDR_CG_UNSPECIFIED
            compressed_image.ct = UHDR_CT_UNSPECIFIED
            compressed_image.range = UHDR_CR_UNSPECIFIED

            decoder = uhdr_create_decoder()
            if decoder == NULL:
                raise UltrahdrError('uhdr_create_decoder')

            error = uhdr_dec_set_image(decoder, &compressed_image)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_dec_set_image', error)

            error = uhdr_dec_set_out_img_format(decoder, fmt)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_dec_set_out_img_format', error)

            error = uhdr_dec_set_out_color_transfer(decoder, otf)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_dec_set_out_color_transfer', error)

            if display_boost >= 1.0:
                error = uhdr_dec_set_out_max_display_boost(
                    decoder, display_boost
                )
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError(
                        'uhdr_dec_set_out_max_display_boost', error
                    )

            if enable_gpu:
                error = uhdr_enable_gpu_acceleration(decoder, enable_gpu)
                if error.error_code != UHDR_CODEC_OK:
                    raise UltrahdrError('uhdr_enable_gpu_acceleration', error)

            error = uhdr_dec_probe(decoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_dec_probe', error)

            width = uhdr_dec_get_image_width(decoder)
            if width < 0:
                raise UltrahdrError('uhdr_dec_get_image_width', width)

            height = uhdr_dec_get_image_height(decoder)
            if height < 0:
                raise UltrahdrError('uhdr_dec_get_image_height', height)

        if dtype.itemsize == 4:
            shape = (int(height), int(width))
        else:
            shape = (int(height), int(width), 4)

        if _info:
            return {
                '_codec': 'ultrahdr',
                'shape': shape,
                'dtype': dtype,
                **_ultrahdr_info(decoder, width),
            }

        dst = _create_array(out, shape, dtype)

        with nogil:
            error = uhdr_decode(decoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_decode', error)

            raw_image = uhdr_get_decoded_image(decoder)
            if raw_image == NULL:
                raise UltrahdrError('uhdr_get_decoded_image')

            if raw_image.h != height or raw_image.w != width:
                raise ValueError(
                    f'shape mismatch ({raw_image.h}, {raw_image.w})'
                    f' != ({height}, {width})'
                )

            if raw_image.fmt != fmt:
                raise ValueError(
                    f'format mismatch {raw_image.fmt} != {fmt}'
                )

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


def ultrahdr_info(
    data,
    /,
    *,
    dtype=None,
    transfer=None,
    boost=None,
):
    """Return information about ULTRAHDR image."""
    return ultrahdr_decode(
        data,
        dtype=dtype,
        transfer=transfer,
        boost=boost,
        _info=True
    )


cdef dict _ultrahdr_info(uhdr_codec_private_t* decoder, ssize_t width):
    """Return ULTRAHDR metadata from probed decoder without decoding pixels."""
    cdef:
        uhdr_gainmap_metadata_t* metadata
        uhdr_mem_block_t* icc
        ssize_t gainmap_width, gainmap_height
        bytes iccprofile = None

    with nogil:
        gainmap_width = uhdr_dec_get_gainmap_width(decoder)
        gainmap_height = uhdr_dec_get_gainmap_height(decoder)
        metadata = uhdr_dec_get_gainmap_metadata(decoder)
        icc = uhdr_dec_get_icc(decoder)

    if metadata == NULL:
        raise UltrahdrError('uhdr_dec_get_gainmap_metadata')

    if icc != NULL and icc.data != NULL and icc.data_sz > 0:
        iccprofile = (<const char*> icc.data)[:icc.data_sz]
        if iccprofile.startswith(b'ICC_PROFILE\x00'):
            # strip JPEG APP2 marker, sequence number, and chunk count
            iccprofile = iccprofile[14:]

    return {
        'codec': ULTRAHDR.CODEC.JPEG,
        'scale': int(width // gainmap_width) if gainmap_width > 0 else None,
        'gainmap_shape': (int(gainmap_height), int(gainmap_width)),
        **_uhdr_gainmap_metadata(metadata),
        # libultrahdr sets hdr_capacity_max = peak nits / 203
        'nits': float(metadata.hdr_capacity_max) * 203.0,
        'iccprofile': iccprofile,
    }


cdef dict _uhdr_gainmap_metadata(const uhdr_gainmap_metadata_t* metadata):
    """Return gainmap metadata as ultrahdr_encode_jpeg keyword arguments."""
    return {
        'gainmap_gamma': _scalar3(metadata.gamma),
        'boostmin': _scalar3(metadata.min_content_boost),
        'boostmax': _scalar3(metadata.max_content_boost),
        'offset_sdr': _scalar3(metadata.offset_sdr),
        'offset_hdr': _scalar3(metadata.offset_hdr),
        'hdr_capacity_min': float(metadata.hdr_capacity_min),
        'hdr_capacity_max': float(metadata.hdr_capacity_max),
        'use_base_cg': bool(metadata.use_base_cg),
    }


def ultrahdr_decode_jpeg(data, /):
    """Return base JPEG, gainmap JPEG, and gainmap metadata from UHDR stream.

    Used with ultrahdr_encode_jpeg for lossless round-trips.

    """
    cdef:
        const uint8_t[::1] src = data
        uhdr_codec_private_t* decoder = NULL
        uhdr_compressed_image_t compressed_image
        uhdr_mem_block_t* base_block
        uhdr_mem_block_t* gainmap_block
        uhdr_gainmap_metadata_t* gainmap_metadata
        uhdr_error_info_t error

    if src.shape[0] == 0:
        raise ValueError('src is empty')

    try:
        with nogil:
            memset(
                <void*> &compressed_image, 0, sizeof(uhdr_compressed_image_t)
            )
            compressed_image.data = <void*> src._data
            compressed_image.data_sz = <size_t> src.shape[0]
            compressed_image.capacity = <size_t> src.shape[0]
            compressed_image.cg = UHDR_CG_UNSPECIFIED
            compressed_image.ct = UHDR_CT_UNSPECIFIED
            compressed_image.range = UHDR_CR_UNSPECIFIED

            decoder = uhdr_create_decoder()
            if decoder == NULL:
                raise UltrahdrError('uhdr_create_decoder')

            error = uhdr_dec_set_image(decoder, &compressed_image)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_dec_set_image', error)

            error = uhdr_dec_probe(decoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_dec_probe', error)

            base_block = uhdr_dec_get_base_image(decoder)
            if base_block == NULL or base_block.data == NULL:
                raise UltrahdrError('uhdr_dec_get_base_image')

            gainmap_block = uhdr_dec_get_gainmap_image(decoder)
            if gainmap_block == NULL or gainmap_block.data == NULL:
                raise UltrahdrError('uhdr_dec_get_gainmap_image')

            gainmap_metadata = uhdr_dec_get_gainmap_metadata(decoder)
            if gainmap_metadata == NULL:
                raise UltrahdrError('uhdr_dec_get_gainmap_metadata')

        base_jpeg = bytes(
            (<unsigned char*> base_block.data)[:base_block.data_sz]
        )
        gainmap_jpeg = bytes(
            (<unsigned char*> gainmap_block.data)[:gainmap_block.data_sz]
        )
        metadata = _uhdr_gainmap_metadata(gainmap_metadata)

    finally:
        if decoder != NULL:
            uhdr_release_decoder(decoder)

    return base_jpeg, gainmap_jpeg, metadata


def ultrahdr_encode_jpeg(
    base_jpeg,
    gainmap_jpeg,
    /,
    *,
    gamut=None,
    gainmap_gamma=None,
    boostmin=None,
    boostmax=None,
    offset_sdr=None,
    offset_hdr=None,
    hdr_capacity_min=None,
    hdr_capacity_max=None,
    use_base_cg=None,
):
    """Return UHDR stream from base JPEG, gainmap JPEG, and metadata.

    Used together with ultrahdr_decode_jpeg for lossless UHDR round-trips.

    """
    cdef:
        const uint8_t[::1] base_src = base_jpeg
        const uint8_t[::1] gainmap_src = gainmap_jpeg
        ssize_t i
        uhdr_codec_private_t* encoder = NULL
        uhdr_compressed_image_t base_image
        uhdr_compressed_image_t gainmap_image
        uhdr_gainmap_metadata_t gainmap_metadata
        uhdr_compressed_image_t* out_stream
        uhdr_color_gamut_t base_gamut = _enum_value(
            gamut, ULTRAHDR.CG, UHDR_CG_UNSPECIFIED
        )
        uhdr_error_info_t error

    if base_src.shape[0] < 2:
        raise ValueError('base_jpeg is too small')
    if gainmap_src.shape[0] < 2:
        raise ValueError('gainmap_jpeg is too small')

    gainmap_gamma = _float3(gainmap_gamma, 1.0)
    boostmax = _float3(boostmax, 2.0)
    boostmin = _float3(boostmin, 1.0)
    offset_sdr = _float3(offset_sdr, 0.0)
    offset_hdr = _float3(offset_hdr, 0.0)

    memset(<void*> &gainmap_metadata, 0, sizeof(uhdr_gainmap_metadata_t))
    for i in range(3):
        gainmap_metadata.gamma[i] = gainmap_gamma[i]
        gainmap_metadata.max_content_boost[i] = boostmax[i]
        gainmap_metadata.min_content_boost[i] = boostmin[i]
        gainmap_metadata.offset_sdr[i] = offset_sdr[i]
        gainmap_metadata.offset_hdr[i] = offset_hdr[i]
    gainmap_metadata.hdr_capacity_min = <float> float(
        hdr_capacity_min if hdr_capacity_min is not None else 1.0
    )
    gainmap_metadata.hdr_capacity_max = <float> float(
        hdr_capacity_max
        if hdr_capacity_max is not None
        else max(boostmax)
    )
    gainmap_metadata.use_base_cg = (
        1 if use_base_cg is None or use_base_cg else 0
    )

    try:
        with nogil:
            memset(<void*> &base_image, 0, sizeof(uhdr_compressed_image_t))
            base_image.data = <void*> base_src._data
            base_image.data_sz = <size_t> base_src.shape[0]
            base_image.capacity = <size_t> base_src.shape[0]
            base_image.cg = base_gamut
            base_image.ct = UHDR_CT_SRGB
            base_image.range = UHDR_CR_FULL_RANGE

            memset(<void*> &gainmap_image, 0, sizeof(uhdr_compressed_image_t))
            gainmap_image.data = <void*> gainmap_src._data
            gainmap_image.data_sz = <size_t> gainmap_src.shape[0]
            gainmap_image.capacity = <size_t> gainmap_src.shape[0]
            gainmap_image.cg = UHDR_CG_UNSPECIFIED
            gainmap_image.ct = UHDR_CT_UNSPECIFIED
            gainmap_image.range = UHDR_CR_FULL_RANGE

            encoder = uhdr_create_encoder()
            if encoder == NULL:
                raise UltrahdrError('uhdr_create_encoder')

            error = uhdr_enc_set_compressed_image(
                encoder, &base_image, UHDR_BASE_IMG
            )
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_enc_set_compressed_image', error)

            error = uhdr_enc_set_gainmap_image(
                encoder, &gainmap_image, &gainmap_metadata
            )
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_enc_set_gainmap_image', error)

            error = uhdr_encode(encoder)
            if error.error_code != UHDR_CODEC_OK:
                raise UltrahdrError('uhdr_encode', error)

            out_stream = uhdr_get_encoded_stream(encoder)
            if out_stream == NULL:
                raise UltrahdrError('uhdr_get_encoded_stream')

        out = _create_output(
            bytes, out_stream.data_sz, <const char*> out_stream.data
        )

    finally:
        if encoder != NULL:
            uhdr_release_encoder(encoder)

    return out


def _float3(value, default):
    """Return tuple of three floats from scalar or sequence."""
    if value is None:
        return float(default), float(default), float(default)
    if isinstance(value, (str, bytes)):
        raise ValueError(f'invalid value {value!r}')
    try:
        values = tuple(float(v) for v in value)
    except TypeError:
        value = float(value)
        return value, value, value
    if len(values) != 3:
        raise ValueError(f'expected 3 values, got {len(values)}')
    return values


cdef _scalar3(const float* values):
    """Return float if all three values are equal, else tuple of floats."""
    if values[0] == values[1] == values[2]:
        return float(values[0])
    return float(values[0]), float(values[1]), float(values[2])


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
        ssize_t i, k
        uint32_t rgba

    k = 0
    for _j in range(height):
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
