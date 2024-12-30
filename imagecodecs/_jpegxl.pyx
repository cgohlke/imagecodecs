# imagecodecs/_jpegxl.pyx
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

"""JPEG XL codec for the imagecodecs package."""

include '_shared.pxi'

from libjxl cimport *


class JPEGXL:
    """JPEGXL codec constants."""

    available = True

    class COLOR_SPACE(enum.IntEnum):
        """JPEGXL codec color spaces."""

        UNKNOWN = JXL_COLOR_SPACE_UNKNOWN
        RGB = JXL_COLOR_SPACE_RGB
        GRAY = JXL_COLOR_SPACE_GRAY
        XYB = JXL_COLOR_SPACE_XYB

    class CHANNEL(enum.IntEnum):
        """JPEGXL codec channel types."""

        UNKNOWN = JXL_CHANNEL_UNKNOWN
        ALPHA = JXL_CHANNEL_ALPHA
        DEPTH = JXL_CHANNEL_DEPTH
        SPOT_COLOR = JXL_CHANNEL_SPOT_COLOR
        SELECTION_MASK = JXL_CHANNEL_SELECTION_MASK
        BLACK = JXL_CHANNEL_BLACK
        CFA = JXL_CHANNEL_CFA
        THERMAL = JXL_CHANNEL_THERMAL
        OPTIONAL = JXL_CHANNEL_OPTIONAL


class JpegxlError(RuntimeError):
    """JPEGXL codec exceptions."""

    def __init__(self, func, err):
        if err is None:
            msg = 'NULL'
        elif 'Decoder' in func:
            msg = {
                JXL_DEC_SUCCESS: 'JXL_DEC_SUCCESS',
                JXL_DEC_ERROR: 'JXL_DEC_ERROR',
                JXL_DEC_NEED_MORE_INPUT: 'JXL_DEC_NEED_MORE_INPUT',
                JXL_DEC_NEED_PREVIEW_OUT_BUFFER:
                    'JXL_DEC_NEED_PREVIEW_OUT_BUFFER',
                JXL_DEC_NEED_IMAGE_OUT_BUFFER: 'JXL_DEC_NEED_IMAGE_OUT_BUFFER',
                JXL_DEC_JPEG_NEED_MORE_OUTPUT: 'JXL_DEC_JPEG_NEED_MORE_OUTPUT',
                JXL_DEC_BASIC_INFO: 'JXL_DEC_BASIC_INFO',
                JXL_DEC_COLOR_ENCODING: 'JXL_DEC_COLOR_ENCODING',
                JXL_DEC_PREVIEW_IMAGE: 'JXL_DEC_PREVIEW_IMAGE',
                JXL_DEC_FRAME: 'JXL_DEC_FRAME',
                JXL_DEC_FULL_IMAGE: 'JXL_DEC_FULL_IMAGE',
                JXL_DEC_JPEG_RECONSTRUCTION: 'JXL_DEC_JPEG_RECONSTRUCTION',
                JXL_DEC_BOX: 'JXL_DEC_BOX',
                JXL_DEC_FRAME_PROGRESSION: 'JXL_DEC_FRAME_PROGRESSION',
            }.get(err, f'unknown error {err!r}')
        elif 'Encoder' in func:
            msg = {
                JXL_ENC_SUCCESS: 'JXL_ENC_SUCCESS',
                JXL_ENC_ERR_GENERIC: 'JXL_ENC_ERR_GENERIC',
                JXL_ENC_ERR_OOM: 'JXL_ENC_ERR_OOM',
                JXL_ENC_ERR_JBRD: 'JXL_ENC_ERR_JBRD',
                JXL_ENC_ERR_BAD_INPUT: 'JXL_ENC_ERR_BAD_INPUT',
                JXL_ENC_ERR_NOT_SUPPORTED: 'JXL_ENC_ERR_NOT_SUPPORTED',
                JXL_ENC_ERR_API_USAGE: 'JXL_ENC_ERR_API_USAGE',
                # JXL_ENC_ERROR: 'JXL_ENC_ERROR',
                # JXL_ENC_NEED_MORE_OUTPUT: 'JXL_ENC_NEED_MORE_OUTPUT',
                # JXL_ENC_NOT_SUPPORTED: 'JXL_ENC_NOT_SUPPORTED',
            }.get(err, f'unknown error {err!r}')
        else:
            msg = f'error {err!r}'
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def jpegxl_version():
    """Return libjxl library version string."""
    cdef:
        uint32_t ver = JxlDecoderVersion()

    return f'libjxl {ver / 1000000}.{(ver / 1000) % 1000}.{ver % 1000}'


def jpegxl_check(const uint8_t[::1] data):
    """Return whether data is JPEGXL encoded image."""
    cdef:
        JxlSignature sig = JxlSignatureCheck(&data[0], min(data.size, 16))

    return sig != JXL_SIG_NOT_ENOUGH_BYTES and sig != JXL_SIG_INVALID


def jpegxl_encode(
    data,
    level=None,  # -inf-100: quality; > 100: lossless
    effort=None,
    distance=None,
    lossless=None,
    decodingspeed=None,
    photometric=None,
    bitspersample=None,
    # extrasamples=None,
    planar=None,
    usecontainer=None,
    numthreads=None,
    out=None
):
    """Return JPEGXL encoded image.

    Float must be in nominal range 0..1.

    Currently, L, LA, RGB, and RGBA images are supported in contig mode.
    Extra channels are only supported for grayscale images in planar mode.

    """
    cdef:
        numpy.ndarray src
        numpy.dtype dtype
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize
        ssize_t frame
        ssize_t ysize, xsize
        ssize_t frames = 1
        ssize_t samples = 1
        int colorspace = -1
        size_t avail_out = 0
        uint8_t* next_out = NULL
        output_t* output = NULL
        size_t framesize = 0
        char* buffer = NULL
        void* runner = NULL
        JxlEncoder* encoder = NULL
        JxlEncoderFrameSettings* frame_settings = NULL
        JxlFrameHeader frame_header
        JxlEncoderStatus status = JXL_ENC_SUCCESS
        JxlBasicInfo basic_info
        JxlPixelFormat pixel_format
        JxlColorEncoding color_encoding
        JxlExtraChannelType extra_channel_type = JXL_CHANNEL_OPTIONAL
        JxlExtraChannelInfo extra_channel_info
        JxlBitDepth bit_depth
        # JxlBlendInfo blend_info
        JXL_BOOL use_container = bool(usecontainer)
        JXL_BOOL option_lossless = lossless is None or bool(lossless)
        int option_tier = _default_value(decodingspeed, 0, 0, 4)
        int option_effort = _default_value(effort, 5, 1, 10)  # 7 is too slow
        float option_distance = _default_value(distance, 1.0, 0.0, 25.0)
        size_t num_threads = <size_t> _default_threads(numthreads)
        size_t channel_index
        uint32_t bits_per_sample
        bint is_planar = bool(planar)

    if data is out:
        raise ValueError('cannot encode in-place')

    if level is not None:
        if level > 100:
            if lossless is None:
                option_lossless = JXL_TRUE
        elif distance is None:
            option_distance = JxlEncoderDistanceFromQuality(level)
            if lossless is None:
                option_lossless = JXL_FALSE
    elif distance is not None and lossless is None:
        option_lossless = JXL_FALSE

    src = numpy.ascontiguousarray(data)
    dtype = src.dtype
    srcsize = src.nbytes
    buffer = src.data

    if not (src.dtype.kind in 'uf' and src.ndim in {2, 3, 4}):
        raise ValueError('invalid data shape or dtype')

    memset(<void*> &bit_depth, 0, sizeof(JxlBitDepth))
    if bitspersample is None or src.dtype.kind == 'f':
        bit_depth.dtype = JXL_BIT_DEPTH_FROM_PIXEL_FORMAT
        bits_per_sample = 0
    else:
        bit_depth.dtype = JXL_BIT_DEPTH_FROM_CODESTREAM
        bits_per_sample = bitspersample

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.nbytes
        output = output_new(<uint8_t*> &dst[0], dstsize)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        dst = out
        dstsize = dst.nbytes
        output = output_new(<uint8_t*> &dst[0], dstsize)
    else:
        output = output_new(
            NULL, max(32768, srcsize // (4 if option_lossless else 16))
        )
    if output == NULL:
        raise MemoryError('output_new failed')

    JxlEncoderInitBasicInfo(&basic_info)
    memset(<void*> &pixel_format, 0, sizeof(JxlPixelFormat))
    memset(<void*> &color_encoding, 0, sizeof(JxlColorEncoding))

    colorspace = _jpegxl_encode_photometric(photometric)

    # TODO: support multi-channel in non-planar mode
    if src.ndim == 2:
        frames = 1
        ysize = src.shape[0]
        xsize = src.shape[1]
        samples = 1
        is_planar = False
    elif is_planar:
        if colorspace != -1 and colorspace != JXL_COLOR_SPACE_GRAY:
            raise NotImplementedError(
                'only grayscale images are supported in planar mode'
            )
        if colorspace == -1:
            colorspace = JXL_COLOR_SPACE_GRAY
        if src.ndim == 3:
            frames = 1
            samples = src.shape[0]
            ysize = src.shape[1]
            xsize = src.shape[2]
        elif src.ndim == 4:
            frames = src.shape[0]
            samples = src.shape[1]
            ysize = src.shape[2]
            xsize = src.shape[3]
        else:
            raise ValueError(f'{src.ndim} dimensions not supported')
        if samples == 1:
            is_planar = False
    elif src.ndim == 3:
        if src.shape[2] > 4 or colorspace == JXL_COLOR_SPACE_GRAY:
            frames = src.shape[0]
            ysize = src.shape[1]
            xsize = src.shape[2]
            samples = 1
        else:
            frames = 1
            ysize = src.shape[0]
            xsize = src.shape[1]
            samples = src.shape[2]
    elif src.ndim == 4:
        frames = src.shape[0]
        ysize = src.shape[1]
        xsize = src.shape[2]
        samples = src.shape[3]
    else:
        raise ValueError(f'{src.ndim} dimensions not supported')

    if colorspace == -1:
        if samples > 2:
            colorspace = JXL_COLOR_SPACE_RGB
        else:
            colorspace = JXL_COLOR_SPACE_GRAY
    elif samples < 3:
        colorspace = JXL_COLOR_SPACE_GRAY

    basic_info.xsize = <uint32_t> xsize
    basic_info.ysize = <uint32_t> ysize
    if colorspace == JXL_COLOR_SPACE_GRAY:
        basic_info.num_color_channels = 1
    else:
        basic_info.num_color_channels = 3
    assert samples - basic_info.num_color_channels >= 0
    basic_info.num_extra_channels = (
        <uint32_t> samples - basic_info.num_color_channels
    )

    if is_planar:
        pixel_format.num_channels = basic_info.num_color_channels
    else:
        pixel_format.num_channels = <uint32_t> samples
    pixel_format.endianness = JXL_NATIVE_ENDIAN
    pixel_format.align = 0  # TODO: allow strides

    if dtype.byteorder == b'<':
        pixel_format.endianness = JXL_LITTLE_ENDIAN
    elif dtype.byteorder == b'>':
        pixel_format.endianness = JXL_BIG_ENDIAN

    if dtype == numpy.uint8:
        pixel_format.data_type = JXL_TYPE_UINT8
        if bits_per_sample < 1 or bits_per_sample > 8:
            bits_per_sample = 8
        basic_info.bits_per_sample = bits_per_sample
    elif dtype == numpy.uint16:
        pixel_format.data_type = JXL_TYPE_UINT16
        if bits_per_sample < 1 or bits_per_sample > 16:
            bits_per_sample = 16
        basic_info.bits_per_sample = bits_per_sample
    elif dtype == numpy.float32:
        pixel_format.data_type = JXL_TYPE_FLOAT
        basic_info.bits_per_sample = 32
        basic_info.exponent_bits_per_sample = 8
    elif dtype == numpy.float16:
        pixel_format.data_type = JXL_TYPE_FLOAT16
        basic_info.bits_per_sample = 16
        basic_info.exponent_bits_per_sample = 5
    else:
        raise ValueError(f'{dtype=!r} not supported')

    try:
        with nogil:

            if option_lossless != 0:
                # avoid lossy colorspace conversion
                basic_info.uses_original_profile = JXL_TRUE

            if is_planar:
                pass
            elif basic_info.num_extra_channels == 1:
                basic_info.alpha_bits = basic_info.bits_per_sample
                basic_info.alpha_exponent_bits = (
                    basic_info.exponent_bits_per_sample
                )
            elif basic_info.num_extra_channels > 1:
                raise NotImplementedError(
                    f'{basic_info.num_extra_channels} extra channels '
                    'not supported in contig mode'
                )
            if frames > 1:
                basic_info.have_animation = JXL_TRUE
                basic_info.animation.tps_numerator = 10
                basic_info.animation.tps_denominator = 1
                basic_info.animation.num_loops = 0
                basic_info.animation.have_timecodes = JXL_FALSE

            framesize = ysize * xsize * dtype.itemsize
            if not is_planar:
                framesize *= samples

            encoder = JxlEncoderCreate(NULL)
            if encoder == NULL:
                raise JpegxlError('JxlEncoderCreate', None)

            if num_threads == 0:
                num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
            if num_threads > 1:
                runner = JxlThreadParallelRunnerCreate(NULL, num_threads)
                if runner == NULL:
                    raise JpegxlError('JxlThreadParallelRunnerCreate', None)

                status = JxlEncoderSetParallelRunner(
                    encoder, JxlThreadParallelRunner, runner
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderSetParallelRunner', status)

            status = JxlEncoderSetBasicInfo(encoder, &basic_info)
            if status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderSetBasicInfo', status)

            # TODO: review this
            if pixel_format.data_type == JXL_TYPE_UINT8:
                JxlColorEncodingSetToSRGB(
                    &color_encoding, colorspace == JXL_COLOR_SPACE_GRAY
                )
            else:
                JxlColorEncodingSetToLinearSRGB(
                    &color_encoding, colorspace == JXL_COLOR_SPACE_GRAY
                )

            status = JxlEncoderSetColorEncoding(encoder, &color_encoding)
            if status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderSetColorEncoding', status)

            status = JxlEncoderUseContainer(encoder, use_container)
            if status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderUseContainer', status)

            frame_settings = JxlEncoderFrameSettingsCreate(encoder, NULL)
            if frame_settings == NULL:
                raise JpegxlError('JxlEncoderFrameSettingsCreate', None)

            if bit_depth.dtype != JXL_BIT_DEPTH_FROM_PIXEL_FORMAT:
                status = JxlEncoderSetFrameBitDepth(frame_settings, &bit_depth)
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderSetFrameBitDepth', status)

            if option_lossless != 0:
                status = JxlEncoderSetFrameLossless(
                    frame_settings, option_lossless
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderSetFrameLossless', status)

            elif option_distance != 1.0:
                status = JxlEncoderSetFrameDistance(
                    frame_settings, option_distance
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderSetFrameDistance', status)

            if option_tier != 0:
                status = JxlEncoderFrameSettingsSetOption(
                    frame_settings,
                    JXL_ENC_FRAME_SETTING_DECODING_SPEED,
                    option_tier
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError(
                        'JxlEncoderFrameSettingsSetOption '
                        'JXL_ENC_FRAME_SETTING_DECODING_SPEED',
                        status
                    )

            if option_effort != 7:
                status = JxlEncoderFrameSettingsSetOption(
                    frame_settings,
                    JXL_ENC_FRAME_SETTING_EFFORT,
                    option_effort
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError(
                        'JxlEncoderFrameSettingsSetOption '
                        'JXL_ENC_FRAME_SETTING_EFFORT',
                        status
                    )

            if is_planar:
                for channel_index in range(basic_info.num_extra_channels):
                    JxlEncoderInitExtraChannelInfo(
                        extra_channel_type, &extra_channel_info
                    )
                    extra_channel_info.bits_per_sample = (
                        basic_info.bits_per_sample
                    )
                    extra_channel_info.exponent_bits_per_sample = (
                        basic_info.exponent_bits_per_sample
                    )
                    status = JxlEncoderSetExtraChannelInfo(
                        encoder, channel_index, &extra_channel_info
                    )
                    if status != JXL_ENC_SUCCESS:
                        raise JpegxlError(
                            'JxlEncoderSetExtraChannelInfo', status
                        )

            JxlEncoderInitFrameHeader(&frame_header)
            frame_header.duration = 1

            for frame in range(frames):

                status = JxlEncoderSetFrameHeader(
                    frame_settings, &frame_header
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderSetFrameHeader', status)

                status = JxlEncoderAddImageFrame(
                    frame_settings,
                    &pixel_format,
                    <const void*> buffer,
                    framesize
                )
                buffer += framesize
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderAddImageFrame', status)

                if is_planar:
                    for channel_index in range(basic_info.num_extra_channels):
                        status = JxlEncoderSetExtraChannelBuffer(
                            frame_settings,
                            &pixel_format,
                            <const void*> buffer,
                            framesize,
                            <uint32_t> channel_index
                        )
                        buffer += framesize
                        if status != JXL_ENC_SUCCESS:
                            raise JpegxlError(
                                'JxlEncoderAddImageFrame', status
                            )

                if frame == frames - 1:
                    JxlEncoderCloseInput(encoder)

                while True:

                    next_out = output.data + output.used
                    avail_out = output.size - output.used

                    status = JxlEncoderProcessOutput(
                        encoder, &next_out, &avail_out
                    )

                    if output_seek(
                        output, output.size - avail_out
                    ) == 0:
                        raise RuntimeError('output_seek returned 0')

                    if (
                        status != JXL_ENC_NEED_MORE_OUTPUT
                        or output.owner == 0
                    ):
                        break

                    if output_resize(
                        output,
                        min(
                            output.size + <size_t> 33554432,  # 32 MB
                            output.size * 2
                        )
                    ) == 0:
                        raise RuntimeError('output_resize returned 0')

                if status == JXL_ENC_ERROR:
                    raise JpegxlError(
                        'JxlEncoderProcessOutput', JxlEncoderGetError(encoder)
                    )
                elif status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderProcessOutput', status)

        if output.owner:
            out = _create_output(
                out, output.used, <const char *> output.data
            )
        else:
            out = _return_output(
                out, output.size, output.used, outgiven
            )

    finally:
        if encoder != NULL:
            JxlEncoderDestroy(encoder)
        if runner != NULL:
            JxlThreadParallelRunnerDestroy(runner)
        if output != NULL:
            output_del(output)

    return out


def jpegxl_decode(
    data,
    index=None,
    keeporientation=None,
    numthreads=None,
    out=None,
):
    """Return decoded JPEGXL image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        size_t srcsize = <size_t> src.size
        size_t dstsize = 0
        size_t samples = 0
        size_t frames = 0
        size_t buffersize = 0
        size_t framesize = 0
        size_t frameindex = 0
        char* buffer = NULL
        void* runner = NULL
        JxlDecoder* decoder = NULL
        JxlDecoderStatus status = JXL_DEC_SUCCESS
        JxlSignature signature
        JxlBasicInfo basic_info
        JxlPixelFormat pixel_format
        JxlBitDepth bit_depth
        size_t num_threads = _default_threads(numthreads)
        size_t channel_index
        bint keep_orientation = bool(keeporientation)
        bint is_planar = False

    signature = JxlSignatureCheck(&src[0], srcsize)
    if signature != JXL_SIG_CODESTREAM and signature != JXL_SIG_CONTAINER:
        raise ValueError('not a JPEG XL codestream')

    if index is None:
        frames = 0
        frameindex = 0
    elif index >= 0:
        frames = 1
        frameindex = index
    else:
        # TODO: implement advanced indexing
        raise NotImplementedError('advanced indexing not supported')

    try:
        with nogil:

            memset(<void*> &basic_info, 0, sizeof(JxlBasicInfo))
            memset(<void*> &pixel_format, 0, sizeof(JxlPixelFormat))
            memset(<void*> &bit_depth, 0, sizeof(JxlBitDepth))

            decoder = JxlDecoderCreate(NULL)
            if decoder == NULL:
                raise JpegxlError('JxlDecoderCreate', None)

            if num_threads == 0:
                num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
            if num_threads > 1:
                runner = JxlThreadParallelRunnerCreate(NULL, num_threads)
                if runner == NULL:
                    raise JpegxlError('JxlThreadParallelRunnerCreate', None)

                status = JxlDecoderSetParallelRunner(
                    decoder, JxlThreadParallelRunner, runner
                )
                if status != JXL_DEC_SUCCESS:
                    raise JpegxlError('JxlDecoderSetParallelRunner', status)

            status = JxlDecoderSetInput(decoder, &src[0], srcsize)
            if status != JXL_DEC_SUCCESS:
                raise JpegxlError('JxlDecoderSetInput', status)
            # JxlDecoderCloseInput(decoder)

            if keep_orientation:
                status = JxlDecoderSetKeepOrientation(decoder, JXL_TRUE)
                if status != JXL_DEC_SUCCESS:
                    raise JpegxlError('JxlDecoderSetKeepOrientation', status)

            if frames == 0:
                frames = jpegxl_framecount(decoder)
                if frames < 1:
                    raise RuntimeError('could not determine frame count')
                status = JxlDecoderSetInput(decoder, &src[0], srcsize)
                if status != JXL_DEC_SUCCESS:
                    raise JpegxlError('JxlDecoderSetInput', status)
                # JxlDecoderCloseInput(decoder)

            if frameindex > 0:
                JxlDecoderSkipFrames(decoder, frameindex)
                frameindex = 0

            status = JxlDecoderSubscribeEvents(
                decoder, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE
                # | JXL_DEC_COLOR_ENCODING
            )
            if status != JXL_DEC_SUCCESS:
                raise JpegxlError('JxlDecoderSubscribeEvents', status)

            while True:
                status = JxlDecoderProcessInput(decoder)
                if (
                    status == JXL_DEC_ERROR
                    or status == JXL_DEC_NEED_MORE_INPUT
                ):
                    raise JpegxlError('JxlDecoderProcessInput', status)

                if status == JXL_DEC_SUCCESS:
                    break

                if status == JXL_DEC_BASIC_INFO:
                    status = JxlDecoderGetBasicInfo(decoder, &basic_info)
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError('JxlDecoderGetBasicInfo', status)

                    samples = (
                        basic_info.num_color_channels
                        + basic_info.num_extra_channels
                    )

                    with gil:

                        if samples == 1:
                            # L
                            shape = (
                                int(basic_info.ysize),
                                int(basic_info.xsize)
                            )
                            is_planar = False
                        elif basic_info.num_extra_channels == 0:
                            # RGB
                            shape = (
                                int(basic_info.ysize),
                                int(basic_info.xsize),
                                int(samples)
                            )
                            is_planar = False
                        elif basic_info.alpha_bits > 0:
                            # LA, RGBA: ignore other extra channels
                            samples = basic_info.num_color_channels + 1
                            shape = (
                                int(basic_info.ysize),
                                int(basic_info.xsize),
                                int(samples)
                            )
                            is_planar = False
                        elif basic_info.num_color_channels != 1:
                            # RGB + C: ignore extra channels
                            samples = basic_info.num_color_channels
                            shape = (
                                int(basic_info.ysize),
                                int(basic_info.xsize),
                                int(samples)
                            )
                            is_planar = False
                        else:
                            # L + C
                            is_planar = True
                            shape = (
                                int(samples),
                                int(basic_info.ysize),
                                int(basic_info.xsize),
                            )

                        if frames > 1:
                            shape = (int(frames), *shape)

                        if is_planar:
                            pixel_format.num_channels = 1
                        else:
                            pixel_format.num_channels = <uint32_t> samples

                        pixel_format.endianness = JXL_NATIVE_ENDIAN
                        pixel_format.align = 0  # TODO: allow strides

                        if basic_info.exponent_bits_per_sample > 0:

                            if basic_info.bits_per_sample == 32:
                                pixel_format.data_type = JXL_TYPE_FLOAT
                                dtype = numpy.float32
                            elif basic_info.bits_per_sample == 16:
                                pixel_format.data_type = JXL_TYPE_FLOAT16
                                dtype = numpy.float16
                            # elif basic_info.bits_per_sample == 64:
                            #     pixel_format.data_type = JXL_TYPE_FLOAT
                            #     dtype = numpy.float64
                            else:
                                raise ValueError(
                                    f'float{basic_info.bits_per_sample}'
                                    ' not supported'
                                )
                        elif basic_info.bits_per_sample <= 8:
                            bit_depth.dtype = JXL_BIT_DEPTH_FROM_CODESTREAM
                            pixel_format.data_type = JXL_TYPE_UINT8
                            dtype = numpy.uint8
                        elif basic_info.bits_per_sample <= 16:
                            bit_depth.dtype = JXL_BIT_DEPTH_FROM_CODESTREAM
                            pixel_format.data_type = JXL_TYPE_UINT16
                            dtype = numpy.uint16
                        else:
                            dtype = numpy.float32
                            pixel_format.data_type = JXL_TYPE_FLOAT

                        out = _create_array(out, shape, dtype)
                        dst = out
                        dstsize = dst.nbytes
                        framesize = dstsize // frames
                        if is_planar:
                            framesize //= samples
                        buffer = dst.data

                # elif status == JXL_DEC_COLOR_ENCODING:
                #     pass

                elif status == JXL_DEC_NEED_IMAGE_OUT_BUFFER:

                    if buffer == NULL:
                        raise RuntimeError('buffer == NULL')

                    if frameindex >= frames:
                        raise RuntimeError(f'{frameindex=} > {frames=}')

                    status = JxlDecoderImageOutBufferSize(
                        decoder, &pixel_format, &buffersize
                    )
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError(
                            'JxlDecoderImageOutBufferSize', status
                        )
                    if buffersize != framesize:
                        raise RuntimeError(f'{buffersize=} != {framesize=}')

                    status = JxlDecoderSetImageOutBuffer(
                        decoder,
                        &pixel_format,
                        <void*> buffer,
                        buffersize
                    )
                    buffer += framesize
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError(
                            'JxlDecoderSetImageOutBuffer', status
                        )

                    if bit_depth.dtype == JXL_BIT_DEPTH_FROM_CODESTREAM:
                        # do not rescale uint images
                        status = JxlDecoderSetImageOutBitDepth(
                            decoder, &bit_depth
                        )
                        if status != JXL_DEC_SUCCESS:
                            raise JpegxlError(
                                'JxlDecoderSetImageOutBitDepth', status
                            )

                    if is_planar:
                        for channel_index in range(
                            basic_info.num_extra_channels
                        ):
                            status = JxlDecoderExtraChannelBufferSize(
                                decoder,
                                &pixel_format,
                                &buffersize,
                                <uint32_t> channel_index
                            )
                            if status != JXL_DEC_SUCCESS:
                                raise JpegxlError(
                                    'JxlDecoderExtraChannelBufferSize', status
                                )
                            if buffersize != framesize:
                                raise RuntimeError(
                                    f'{buffersize=} != {framesize=}'
                                )

                            status = JxlDecoderSetExtraChannelBuffer(
                                decoder,
                                &pixel_format,
                                <void*> buffer,
                                buffersize,
                                <uint32_t> channel_index
                            )
                            buffer += framesize
                            if status != JXL_DEC_SUCCESS:
                                raise JpegxlError(
                                    'JxlDecoderSetExtraChannelBuffer', status
                                )

                elif status == JXL_DEC_FULL_IMAGE:
                    frameindex += 1
                    if frameindex == frames:
                        break

                else:
                    raise RuntimeError(
                        f'JxlDecoderProcessInput unknown {status=}'
                    )

    finally:
        if decoder != NULL:
            JxlDecoderDestroy(decoder)
        if runner != NULL:
            JxlThreadParallelRunnerDestroy(runner)

    return out


def jpegxl_encode_jpeg(
    data,
    usecontainer=None,
    numthreads=None,
    out=None,
):
    """Return JPEGXL encoded image from JPEG stream."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        size_t srcsize = <size_t> src.size
        size_t avail_out = 0
        uint8_t* next_out = NULL
        output_t* output = NULL
        void* runner = NULL
        JxlEncoder* encoder = NULL
        JxlEncoderFrameSettings* frame_settings = NULL
        JxlEncoderStatus status = JXL_ENC_SUCCESS
        JXL_BOOL use_container = bool(usecontainer)
        size_t num_threads = <size_t> _default_threads(numthreads)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.nbytes
        output = output_new(<uint8_t*> &dst[0], dstsize)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        dst = out
        dstsize = dst.nbytes
        output = output_new(<uint8_t*> &dst[0], dstsize)
    else:
        output = output_new(NULL, max(32768, srcsize))
    if output == NULL:
        raise MemoryError('output_new failed')

    encoder = JxlEncoderCreate(NULL)
    if encoder == NULL:
        raise JpegxlError('JxlEncoderCreate', None)

    try:
        with nogil:

            frame_settings = JxlEncoderFrameSettingsCreate(encoder, NULL)
            if frame_settings == NULL:
                raise JpegxlError('JxlEncoderFrameSettingsCreate', None)

            status = JxlEncoderUseContainer(encoder, use_container)
            if status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderUseContainer', status)

            if num_threads == 0:
                num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
            if num_threads > 1:
                runner = JxlThreadParallelRunnerCreate(NULL, num_threads)
                if runner == NULL:
                    raise JpegxlError('JxlThreadParallelRunnerCreate', None)

                status = JxlEncoderSetParallelRunner(
                    encoder, JxlThreadParallelRunner, runner
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderSetParallelRunner', status)

            # JxlEncoderSetBasicInfo
            # JxlEncoderSetColorEncoding
            # JxlEncoderSetICCProfile
            # will implicitly be called with the parameters of the JPEG frame

            status = JxlEncoderStoreJPEGMetadata(encoder, JXL_TRUE)
            if status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderStoreJPEGMetadata', status)

            status = JxlEncoderAddJPEGFrame(
                frame_settings,
                <const uint8_t*> &src[0],
                srcsize
            )
            if status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderAddJPEGFrame', status)

            JxlEncoderCloseInput(encoder)

            while True:
                next_out = output.data + output.used
                avail_out = output.size - output.used

                status = JxlEncoderProcessOutput(
                    encoder, &next_out, &avail_out
                )

                if output_seek(
                    output, output.size - avail_out
                ) == 0:
                    raise RuntimeError('output_seek returned 0')

                if (
                    status != JXL_ENC_NEED_MORE_OUTPUT
                    or output.owner == 0
                ):
                    break

                if output_resize(
                    output,
                    min(
                        output.size + <size_t> 33554432,  # 32 MB
                        output.size * 2
                    )
                ) == 0:
                    raise RuntimeError('output_resize returned 0')

            if status == JXL_ENC_ERROR:
                raise JpegxlError(
                    'JxlEncoderProcessOutput', JxlEncoderGetError(encoder)
                )
            elif status != JXL_ENC_SUCCESS:
                raise JpegxlError('JxlEncoderProcessOutput', status)

        if output.owner:
            out = _create_output(
                out, output.used, <const char *> output.data
            )
        else:
            out = _return_output(
                out, output.size, output.used, outgiven
            )

    finally:
        if encoder != NULL:
            JxlEncoderDestroy(encoder)
        if runner != NULL:
            JxlThreadParallelRunnerDestroy(runner)
        if output != NULL:
            output_del(output)

    return out


def jpegxl_decode_jpeg(
    data,
    numthreads=None,
    out=None,
):
    """Return JPEG encoded image from JPEG XL."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        size_t srcsize = <size_t> src.size
        output_t* output = NULL
        void* runner = NULL
        JxlDecoder* decoder = NULL
        JxlDecoderStatus status = JXL_DEC_SUCCESS
        JxlSignature signature
        size_t num_threads = _default_threads(numthreads)

    signature = JxlSignatureCheck(&src[0], srcsize)
    if signature != JXL_SIG_CODESTREAM and signature != JXL_SIG_CONTAINER:
        raise ValueError('not a JPEG XL codestream')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.nbytes
        output = output_new(<uint8_t*> &dst[0], dstsize)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        dst = out
        dstsize = dst.nbytes
        output = output_new(<uint8_t*> &dst[0], dstsize)
    else:
        output = output_new(NULL, (srcsize * 3) // 2 + 1024)
    if output == NULL:
        raise MemoryError('output_new failed')

    try:
        with nogil:

            decoder = JxlDecoderCreate(NULL)
            if decoder == NULL:
                raise JpegxlError('JxlDecoderCreate', None)

            if num_threads == 0:
                num_threads = JxlThreadParallelRunnerDefaultNumWorkerThreads()
            if num_threads > 1:
                runner = JxlThreadParallelRunnerCreate(NULL, num_threads)
                if runner == NULL:
                    raise JpegxlError('JxlThreadParallelRunnerCreate', None)

                status = JxlDecoderSetParallelRunner(
                    decoder, JxlThreadParallelRunner, runner
                )
                if status != JXL_DEC_SUCCESS:
                    raise JpegxlError('JxlDecoderSetParallelRunner', status)

            status = JxlDecoderSetInput(decoder, &src[0], srcsize)
            if status != JXL_DEC_SUCCESS:
                raise JpegxlError('JxlDecoderSetInput', status)
            # JxlDecoderCloseInput(decoder)

            status = JxlDecoderSubscribeEvents(
                decoder, JXL_DEC_FULL_IMAGE | JXL_DEC_JPEG_RECONSTRUCTION
            )
            if status != JXL_DEC_SUCCESS:
                raise JpegxlError('JxlDecoderSubscribeEvents', status)

            while True:
                status = JxlDecoderProcessInput(decoder)
                if (
                    status == JXL_DEC_ERROR
                    or status == JXL_DEC_NEED_MORE_INPUT
                ):
                    raise JpegxlError('JxlDecoderProcessInput', status)

                if status == JXL_DEC_SUCCESS:
                    break

                elif status == JXL_DEC_JPEG_RECONSTRUCTION:
                    status = JxlDecoderSetJPEGBuffer(
                        decoder, output.data, output.size
                    )
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError('JxlDecoderSetJPEGBuffer', status)

                elif status == JXL_DEC_JPEG_NEED_MORE_OUTPUT:
                    if output.owner == 0:
                        raise ValueError('cannot resize output buffer')

                    output.used = (
                        output.size - JxlDecoderReleaseJPEGBuffer(decoder)
                    )

                    if output_resize(output, (output.size * 3) // 2) == 0:
                        raise RuntimeError('output_resize returned 0')

                    status = JxlDecoderSetJPEGBuffer(
                        decoder,
                        output.data + output.used,
                        output.size - output.used
                    )
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError('JxlDecoderSetJPEGBuffer', status)

                elif status == JXL_DEC_FULL_IMAGE:
                    output.used = (
                        output.size - JxlDecoderReleaseJPEGBuffer(decoder)
                    )
                    break

                elif status == JXL_DEC_NEED_IMAGE_OUT_BUFFER:
                    # TODO: support grayscale or multiple frames
                    raise ValueError(
                        "can't transcode grayscale or animated JPEG XL to JPEG"
                    )

                else:
                    raise RuntimeError(
                        f'JxlDecoderProcessInput unknown {status=}'
                    )

        if output.owner:
            out = _create_output(
                out, output.used, <const char *> output.data
            )
        else:
            out = _return_output(
                out, output.size, output.used, outgiven
            )

    finally:
        if decoder != NULL:
            JxlDecoderDestroy(decoder)
        if runner != NULL:
            JxlThreadParallelRunnerDestroy(runner)
        if output != NULL:
            output_del(output)

    return out


cdef size_t jpegxl_framecount(JxlDecoder* decoder) noexcept nogil:
    """Return number of frames."""
    cdef:
        JxlDecoderStatus status = JXL_DEC_SUCCESS
        size_t framecount = 0

    status = JxlDecoderSubscribeEvents(decoder, JXL_DEC_FRAME)
    if status != JXL_DEC_SUCCESS:
        return -1

    while True:
        status = JxlDecoderProcessInput(decoder)
        if (status == JXL_DEC_ERROR or status == JXL_DEC_NEED_MORE_INPUT):
            break
        if status == JXL_DEC_SUCCESS:
            break
        if status == JXL_DEC_FRAME:
            framecount += 1
        else:
            return -1

    JxlDecoderRewind(decoder)
    return framecount


cdef _jpegxl_encode_photometric(photometric):
    """Return JxlColorSpace value from photometric argument."""
    if photometric is None:
        return -1
    if isinstance(photometric, int):
        if photometric not in {
            -1,
            JXL_COLOR_SPACE_RGB,
            JXL_COLOR_SPACE_GRAY,
            JXL_COLOR_SPACE_XYB,
            JXL_COLOR_SPACE_UNKNOWN
        }:
            raise ValueError(f'{photometric=!r} not supported')
        return photometric
    photometric = photometric.upper()
    if photometric[:3] == 'RGB':
        return JXL_COLOR_SPACE_RGB
    if photometric in {'WHITEISZERO', 'MINISWHITE'}:
        return JXL_COLOR_SPACE_GRAY
    if photometric in {'BLACKISZERO', 'MINISBLACK', 'GRAY'}:
        return JXL_COLOR_SPACE_GRAY
    if photometric == 'XYB':
        return JXL_COLOR_SPACE_XYB
    if photometric == 'UNKNOWN':
        return JXL_COLOR_SPACE_UNKNOWN
    raise ValueError(f'{photometric=!r} not supported')


cdef _jpegxl_encode_extrasamples(extrasample):
    """Return JxlExtraChannelType from extrasample argument."""
    if extrasample is None:
        return JXL_CHANNEL_OPTIONAL
    if isinstance(extrasample, int):
        # if extrasample not in {
        #     -1,
        #     JXL_CHANNEL_ALPHA
        #     JXL_CHANNEL_DEPTH
        #     JXL_CHANNEL_SPOT_COLOR
        #     JXL_CHANNEL_SELECTION_MASK
        #     JXL_CHANNEL_BLACK
        #     JXL_CHANNEL_CFA
        #     JXL_CHANNEL_THERMAL
        #     JXL_CHANNEL_UNKNOWN
        #     JXL_CHANNEL_OPTIONAL
        # }:
        #     raise ValueError('ExtraChannelType not supported')
        return extrasample
    extrasample = extrasample.upper()
    if extrasample == 'ALPHA':
        return JXL_CHANNEL_ALPHA
    if extrasample == 'OPTIONAL':
        return JXL_CHANNEL_OPTIONAL
    if extrasample == 'UNKNOWN':
        return JXL_CHANNEL_UNKNOWN
    if extrasample == 'DEPTH':
        return JXL_CHANNEL_DEPTH
    if extrasample == 'SPOT_COLOR':
        return JXL_CHANNEL_SPOT_COLOR
    if extrasample == 'SELECTION_MASK':
        return JXL_CHANNEL_SELECTION_MASK
    if extrasample == 'BLACK':
        return JXL_CHANNEL_BLACK
    if extrasample == 'CFA':
        return JXL_CHANNEL_CFA
    if extrasample == 'THERMAL':
        return JXL_CHANNEL_THERMAL
    raise ValueError(f'ExtraChannelType {extrasample!r} not supported')


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
