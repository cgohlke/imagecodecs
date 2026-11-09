# imagecodecs/_jpegls.pyx
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

"""JPEGLS (JPEG Lossless Standard) codec for the imagecodecs package."""

include '_shared.pxi'

from charls cimport *


class JPEGLS:
    """JPEGLS codec constants."""

    available = True

    class COLOR_TRANSFORMATION(enum.IntEnum):
        """CharLS color transformation (HP extension)."""

        NONE = CHARLS_COLOR_TRANSFORMATION_NONE
        HP1 = CHARLS_COLOR_TRANSFORMATION_HP1
        HP2 = CHARLS_COLOR_TRANSFORMATION_HP2
        HP3 = CHARLS_COLOR_TRANSFORMATION_HP3

    class INTERLEAVE(enum.IntEnum):
        """JPEGLS interleave mode."""

        NONE = CHARLS_INTERLEAVE_MODE_NONE
        LINE = CHARLS_INTERLEAVE_MODE_LINE
        SAMPLE = CHARLS_INTERLEAVE_MODE_SAMPLE


class JpeglsError(RuntimeError):
    """JPEGLS codec exceptions."""

    def __init__(self, func, err):
        cdef:
            char* error_message
            charls_jpegls_errc error_value

        try:
            error_value = int(err)
            error_message = <char*> charls_get_error_message(error_value)
            msg = error_message.decode().strip()
        except Exception:
            msg = 'NULL' if err is None else f'unknown error {err!r}'
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def jpegls_version():
    """Return CharLS library version string."""
    return 'charls {}.{}.{}'.format(
        CHARLS_VERSION_MAJOR, CHARLS_VERSION_MINOR, CHARLS_VERSION_PATCH
    )


def jpegls_check(const uint8_t[::1] data not None, /):
    """Return whether data is JPEGLS encoded image or None if unknown."""
    cdef:
        ssize_t size = data.shape[0]
        ssize_t pos = 2
        ssize_t length
        uint8_t marker

    if size < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return False
    if (
        size >= 12
        and data[2] == 0xFF
        and data[3] == 0xE8
        and _startswith(data, b'SPIFF\x00', 6)
    ):
        return True
    # scan marker segments up to the first frame header
    while pos + 1 < size:
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xF7:  # SOF55
            return True
        if 0xC0 <= marker <= 0xCF and marker not in {0xC4, 0xC8, 0xCC}:
            return False  # other JPEG SOF
        if marker == 0xD9 or marker == 0xDA:  # EOI or SOS
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:  # standalone
            pos += 2
            continue
        if pos + 3 >= size:
            return None
        length = (<ssize_t> data[pos + 2] << 8) | data[pos + 3]
        if length < 2:
            return None
        pos += 2 + length
    return None


def jpegls_encode(
    data,
    /,
    level=None,
    *,
    bitspersample=None,
    planar=None,
    transformation=None,
    spiff=None,
    out=None,
):
    """Return JPEGLS encoded image."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        charls_jpegls_errc ret
        charls_jpegls_encoder* encoder = NULL
        charls_frame_info frameinfo
        # charls_jpegls_pc_parameters preset_coding_parameters
        charls_color_transformation color_transformation = \
            <charls_color_transformation> <int> _enum_value(
                transformation, JPEGLS.COLOR_TRANSFORMATION, 0
            )
        charls_interleave_mode interleave_mode
        charls_spiff_color_space spiff_color_space
        bint write_spiff = bool(spiff)
        int32_t near_lossless = _default_value(level, 0, 0, None)
        uint32_t rowstride
        size_t byteswritten
        size_t size_in_bytes
        imagelayout_t layout

    _image_layout(
        IC_UINT
        | IC_SZ1
        | IC_SZ2
        | IC_GRAY
        | IC_RGB
        | IC_ALPHA
        | IC_PLANAR
        | IC_BPS,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        bitspersample,
        planar,
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if not (layout.height > 0 and srcsize <= UINT32_MAX):
        raise ValueError('invalid data shape or dtype')

    rowstride = <uint32_t> src.strides[1 if layout.planar else 0]

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.shape[0]
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        dst = out
        dstsize = dst.shape[0]

    # memset(
    #     <void*> &preset_coding_parameters,
    #     0,
    #     sizeof(charls_jpegls_pc_parameters)
    # )
    # preset_coding_parameters.maximum_sample_value = 0
    # preset_coding_parameters.threshold1 = 0
    # preset_coding_parameters.threshold2 = 0
    # preset_coding_parameters.threshold3 = 0
    # preset_coding_parameters.reset_value = 0

    # memset(<void*> &frameinfo, 0, sizeof(charls_frame_info))
    frameinfo.width = <uint32_t> layout.width
    frameinfo.height = <uint32_t> layout.height
    frameinfo.bits_per_sample = <int32_t> layout.bitspersample

    if layout.samples == 1:
        frameinfo.component_count = 1
        interleave_mode = CHARLS_INTERLEAVE_MODE_NONE
        spiff_color_space = CHARLS_SPIFF_COLOR_SPACE_GRAYSCALE
    elif layout.samples == 3:
        frameinfo.component_count = 3
        interleave_mode = CHARLS_INTERLEAVE_MODE_SAMPLE
        spiff_color_space = CHARLS_SPIFF_COLOR_SPACE_RGB
    elif layout.samples == 4:
        frameinfo.component_count = 4
        # some decoders do not support 4-component sample interleave
        # interleave_mode = CHARLS_INTERLEAVE_MODE_SAMPLE
        interleave_mode = CHARLS_INTERLEAVE_MODE_LINE
        # SPIFF has no RGBA color space
        # spiff_color_space = CHARLS_SPIFF_COLOR_SPACE_RGB
        spiff_color_space = CHARLS_SPIFF_COLOR_SPACE_NONE
    else:
        raise ValueError('invalid shape')

    if layout.planar:
        interleave_mode = CHARLS_INTERLEAVE_MODE_NONE

    try:
        with nogil:
            encoder = charls_jpegls_encoder_create()
            if encoder == NULL:
                raise JpeglsError('charls_jpegls_encoder_create', None)

            ret = charls_jpegls_encoder_set_frame_info(encoder, &frameinfo)
            if ret:
                raise JpeglsError('charls_jpegls_encoder_set_frame_info', ret)

            ret = charls_jpegls_encoder_set_near_lossless(
                encoder,
                near_lossless
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_encoder_set_near_lossless', ret
                )

            ret = charls_jpegls_encoder_set_interleave_mode(
                encoder,
                interleave_mode
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_encoder_set_interleave_mode', ret
                )

            ret = charls_jpegls_encoder_set_color_transformation(
                encoder,
                color_transformation
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_encoder_set_color_transformation', ret
                )

            # ret charls_jpegls_encoder_set_preset_coding_parameters(
            #     encoder,
            #     &preset_coding_parameters
            # )
            # if ret:
            #     raise JpeglsError(
            #         'charls_jpegls_encoder_set_preset_coding_parameters', ret
            #     )

            if dstsize < 0:
                ret = charls_jpegls_encoder_get_estimated_destination_size(
                    encoder,
                    &size_in_bytes
                )
                if ret:
                    raise JpeglsError(
                        'charls_jpegls_encoder_get_estimated_destination_size',
                        ret
                    )
                # pad for incompressible data where estimate may be too small
                if size_in_bytes < <size_t> srcsize + 65536:
                    size_in_bytes = <size_t> srcsize + 65536
                if write_spiff:
                    # SPIFF header and end-of-directory segments
                    size_in_bytes += 64
                dstsize = <ssize_t> size_in_bytes
                with gil:
                    out = _create_output(outtype, dstsize)
                    dst = out
                    dstsize = dst.shape[0]

            ret = charls_jpegls_encoder_set_destination_buffer(
                encoder,
                <void*> dst._data,
                <size_t> dstsize
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_encoder_set_destination_buffer', ret
                )

            if write_spiff:
                ret = charls_jpegls_encoder_write_standard_spiff_header(
                    encoder,
                    spiff_color_space,
                    CHARLS_SPIFF_RESOLUTION_UNITS_DOTS_PER_INCH,
                    300,
                    300
                )
                if ret:
                    raise JpeglsError(
                        'charls_jpegls_encoder_write_standard_spiff_header',
                        ret
                    )

            ret = charls_jpegls_encoder_encode_from_buffer(
                encoder,
                <const void*> src.data,
                <size_t> srcsize,
                <uint32_t> rowstride
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_encoder_encode_from_buffer', ret
                )

            ret = charls_jpegls_encoder_get_bytes_written(
                encoder,
                &byteswritten
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_encoder_get_bytes_written', ret
                )
    finally:
        if encoder != NULL:
            charls_jpegls_encoder_destroy(encoder)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def jpegls_decode(
    data,
    /,
    *,
    planar=None,
    out=None,
    bint _info=False,
):
    """Return decoded JPEGLS image.

    Multi-component images are returned with samples last, regardless of
    the stream's interleave mode, unless `planar` is True.

    """
    cdef:
        numpy.ndarray dst
        numpy.ndarray dstout
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t bandsize = 0
        charls_jpegls_errc ret
        charls_jpegls_decoder* decoder = NULL
        charls_interleave_mode interleave_mode
        charls_color_transformation color_transformation
        charls_frame_info frameinfo
        charls_spiff_header spiff_header
        int32_t header_found = 0
        bint planar_ = bool(planar)
        bint transpose = False

    if data is out:
        raise ValueError('cannot transform in-place')

    try:
        with nogil:
            decoder = charls_jpegls_decoder_create()
            if decoder == NULL:
                raise JpeglsError('charls_jpegls_decoder_create', None)

            ret = charls_jpegls_decoder_set_source_buffer(
                decoder,
                <void*> src._data,
                <size_t> srcsize
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_decoder_set_source_buffer', ret
                )

            if _info:
                # must be called before charls_jpegls_decoder_read_header
                ret = charls_jpegls_decoder_read_spiff_header(
                    decoder,
                    &spiff_header,
                    &header_found
                )
                if ret:
                    raise JpeglsError(
                        'charls_jpegls_decoder_read_spiff_header', ret
                    )

            ret = charls_jpegls_decoder_read_header(decoder)
            if ret:
                raise JpeglsError('charls_jpegls_decoder_read_header', ret)

            # memset(<void*> &frameinfo, 0, sizeof(charls_frame_info))
            ret = charls_jpegls_decoder_get_frame_info(decoder, &frameinfo)
            if ret:
                raise JpeglsError('charls_jpegls_decoder_get_frame_info', ret)

            ret = charls_jpegls_decoder_get_interleave_mode(
                decoder,
                &interleave_mode
            )
            if ret:
                raise JpeglsError(
                    'charls_jpegls_decoder_get_interleave_mode', ret
                )

            with gil:
                # CharLS limits bits_per_sample to 2..16
                if frameinfo.bits_per_sample <= 8:
                    dtype = numpy.uint8
                else:
                    dtype = numpy.uint16

                if frameinfo.component_count == 1:
                    shape = (frameinfo.height, frameinfo.width)
                elif planar_:
                    shape = (
                        frameinfo.component_count,
                        frameinfo.height,
                        frameinfo.width
                    )
                else:
                    shape = (
                        frameinfo.height,
                        frameinfo.width,
                        frameinfo.component_count
                    )

                if _info:
                    ret = charls_jpegls_decoder_get_color_transformation(
                        decoder,
                        &color_transformation
                    )
                    if ret:
                        raise JpeglsError(
                            'charls_jpegls_decoder_get_color_transformation',
                            ret
                        )

                    return {
                        '_codec': 'jpegls',
                        'shape': shape,
                        'dtype': numpy.dtype(dtype),
                        'bitspersample': frameinfo.bits_per_sample,
                        'interleave': _enum_or_int(
                            interleave_mode, JPEGLS.INTERLEAVE
                        ),
                        'transformation': _enum_or_int(
                            color_transformation, JPEGLS.COLOR_TRANSFORMATION
                        ),
                        'spiff': bool(header_found),
                    }

                out = _create_array(out, shape, dtype)
                dstout = out
                bandsize = (
                    <ssize_t> frameinfo.height * <ssize_t> frameinfo.width
                )
                # use temporary buffer if stream and output layouts differ
                transpose = frameinfo.component_count > 1 and (
                    (interleave_mode == CHARLS_INTERLEAVE_MODE_NONE)
                    != planar_
                )
                if not transpose:
                    dst = out
                elif planar_:
                    dst = numpy.empty(
                        (
                            frameinfo.height,
                            frameinfo.width,
                            frameinfo.component_count
                        ),
                        dtype
                    )
                else:
                    dst = numpy.empty(
                        (
                            frameinfo.component_count,
                            frameinfo.height,
                            frameinfo.width
                        ),
                        dtype
                    )
                dstsize = dst.nbytes

            ret = charls_jpegls_decoder_decode_to_buffer(
                decoder,
                <void*> dst.data,
                <size_t> dstsize,
                0
            )

            if ret == 0 and transpose:
                if frameinfo.bits_per_sample <= 8:
                    _jpegls_transpose(
                        <const uint8_t*> dst.data,
                        <uint8_t*> dstout.data,
                        bandsize,
                        frameinfo.component_count,
                        planar_
                    )
                else:
                    _jpegls_transpose(
                        <const uint16_t*> dst.data,
                        <uint16_t*> dstout.data,
                        bandsize,
                        frameinfo.component_count,
                        planar_
                    )

        if ret:
            raise JpeglsError('charls_jpegls_decoder_decode_to_buffer', ret)

    finally:
        if decoder != NULL:
            charls_jpegls_decoder_destroy(decoder)

    return out


def jpegls_info(data, /, *, planar=None):
    """Return information about JPEGLS image.

    The returned shape matches the output of `jpegls_decode` called with
    the same `planar` argument.

    """
    return jpegls_decode(data, planar=planar, _info=True)


ctypedef fused jpegls_sample_t:
    uint8_t
    uint16_t


cdef void _jpegls_transpose(
    const jpegls_sample_t* src,
    jpegls_sample_t* dst,
    const ssize_t bandsize,
    const ssize_t samples,
    const bint to_planar,
) noexcept nogil:
    """Copy samples between contiguous and planar layouts."""
    cdef:
        ssize_t i, j
        const jpegls_sample_t* srcband
        jpegls_sample_t* dstband

    if to_planar:
        for i in range(samples):
            dstband = dst + i * bandsize
            for j in range(bandsize):
                dstband[j] = src[j * samples + i]
    else:
        for i in range(samples):
            srcband = src + i * bandsize
            for j in range(bandsize):
                dst[j * samples + i] = srcband[j]
