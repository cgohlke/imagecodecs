# imagecodecs/_jpegxl.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021-2022, Christoph Gohlke
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

__version__ = '2022.2.22'

include '_shared.pxi'

from libjxl cimport *


class JPEGXL:
    """JPEG XL Constants."""

    class COLOR_SPACE(enum.IntEnum):
        UNKNOWN = JXL_COLOR_SPACE_UNKNOWN
        RGB = JXL_COLOR_SPACE_RGB
        GRAY = JXL_COLOR_SPACE_GRAY
        XYB = JXL_COLOR_SPACE_XYB

    class CHANNEL(enum.IntEnum):
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
    """JPEG XL Exceptions."""

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
                JXL_DEC_NEED_DC_OUT_BUFFER: 'JXL_DEC_NEED_DC_OUT_BUFFER',
                JXL_DEC_NEED_IMAGE_OUT_BUFFER: 'JXL_DEC_NEED_IMAGE_OUT_BUFFER',
                JXL_DEC_JPEG_NEED_MORE_OUTPUT: 'JXL_DEC_JPEG_NEED_MORE_OUTPUT',
                JXL_DEC_BASIC_INFO: 'JXL_DEC_BASIC_INFO',
                JXL_DEC_EXTENSIONS: 'JXL_DEC_EXTENSIONS',
                JXL_DEC_COLOR_ENCODING: 'JXL_DEC_COLOR_ENCODING',
                JXL_DEC_PREVIEW_IMAGE: 'JXL_DEC_PREVIEW_IMAGE',
                JXL_DEC_FRAME: 'JXL_DEC_FRAME',
                JXL_DEC_DC_IMAGE: 'JXL_DEC_DC_IMAGE',
                JXL_DEC_FULL_IMAGE: 'JXL_DEC_FULL_IMAGE',
                JXL_DEC_JPEG_RECONSTRUCTION: 'JXL_DEC_JPEG_RECONSTRUCTION'
            }.get(err, f'unknown error {err!r}')
        elif 'Encoder' in func:
            msg = {
                JXL_ENC_SUCCESS: 'JXL_ENC_SUCCESS',
                JXL_ENC_ERROR: 'JXL_ENC_ERROR',
                JXL_ENC_NEED_MORE_OUTPUT: 'JXL_ENC_NEED_MORE_OUTPUT',
                JXL_ENC_NOT_SUPPORTED: 'JXL_ENC_NOT_SUPPORTED',
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
    """Return True if data likely contains a JPEG XL image."""
    cdef JxlSignature sig = JxlSignatureCheck(&data[0], min(data.size, 16))

    return sig != JXL_SIG_NOT_ENOUGH_BYTES and sig != JXL_SIG_INVALID


def jpegxl_encode(
    data,
    level=None,  # None or < 0: lossless, 0-4: tier/speed
    effort=None,
    distance=None,
    lossless=None,
    decodingspeed=None,
    photometric=None,
    usecontainer=None,
    numthreads=None,
    out=None
):
    """Return JPEG XL image from numpy array.

    Float must be in nominal range 0..1.

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
        output_t* compressed = NULL
        size_t framesize = 0
        char* buffer = NULL
        void* runner = NULL
        JxlEncoder* encoder = NULL
        JxlEncoderOptions* options = NULL
        JxlEncoderStatus status = JXL_ENC_SUCCESS
        JxlBasicInfo basic_info
        JxlPixelFormat pixel_format
        JxlColorEncoding color_encoding
        JXL_BOOL use_container = bool(usecontainer)
        JXL_BOOL option_lossless = lossless is None or bool(lossless)
        int option_tier = _default_value(decodingspeed, 0, 0, 4)
        int option_effort = _default_value(effort, 3, 3, 9)  # 7 is too slow
        float option_distance = _default_value(distance, 1.0, 0.0, 15.0)
        size_t num_threads = <size_t> _default_threads(numthreads)

    if data is out:
        raise ValueError('cannot encode in-place')

    if isinstance(data, (bytes, bytearray)):
        # input is a JPEG stream
        return jpegxl_from_jpeg(data, use_container, num_threads, out)

    if level is not None:
        if level < 0:
            option_lossless = JXL_TRUE
        elif level > 4:
            option_tier = 4
        else:
            option_tier = level

    src = numpy.ascontiguousarray(data)
    dtype = src.dtype
    srcsize = src.nbytes
    buffer = src.data

    if not (src.dtype.kind in 'uf' and src.ndim in (2, 3, 4)):
        raise ValueError('invalid data shape or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is not None:
        dst = out
        dstsize = dst.nbytes
        compressed = output_new(<uint8_t*> &dst[0], dstsize)
    elif dstsize > 0:
        out = _create_output(outtype, dstsize)
        dst = out
        dstsize = dst.nbytes
        compressed = output_new(<uint8_t*> &dst[0], dstsize)
    else:
        compressed = output_new(
            NULL, max(32768, srcsize // (4 if option_lossless else 16))
        )
    if compressed == NULL:
        raise MemoryError('output_new failed')

    JxlEncoderInitBasicInfo(&basic_info)
    memset(<void*> &pixel_format, 0, sizeof(JxlPixelFormat))
    memset(<void*> &color_encoding, 0, sizeof(JxlColorEncoding))

    if src.ndim == 2:
        frames = 1
        ysize = src.shape[0]
        xsize = src.shape[1]
        samples = 1
    elif src.ndim == 3:
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

    colorspace = jpegxl_encode_photometric(photometric)
    if colorspace == -1:
        if samples > 2:
            colorspace = JXL_COLOR_SPACE_RGB
        else:
            colorspace = JXL_COLOR_SPACE_GRAY
    elif samples < 3:
        colorspace = JXL_COLOR_SPACE_GRAY

    pixel_format.num_channels = <uint32_t> samples
    pixel_format.endianness = JXL_NATIVE_ENDIAN
    pixel_format.align = 0  # TODO: allow strides

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

    if dtype.byteorder == b'<':
        pixel_format.endianness = JXL_LITTLE_ENDIAN
    elif dtype.byteorder == b'>':
        pixel_format.endianness = JXL_BIG_ENDIAN

    if dtype == numpy.uint8:
        pixel_format.data_type = JXL_TYPE_UINT8
        basic_info.bits_per_sample = 8
    elif dtype == numpy.uint16:
        pixel_format.data_type = JXL_TYPE_UINT16
        basic_info.bits_per_sample = 16
    elif dtype == numpy.uint32:
        # TODO: raises JXL_ENC_NOT_SUPPORTED ?
        pixel_format.data_type = JXL_TYPE_UINT32
        basic_info.bits_per_sample = 32
    elif dtype == numpy.float32:
        pixel_format.data_type = JXL_TYPE_FLOAT
        basic_info.bits_per_sample = 32
        basic_info.exponent_bits_per_sample = 8
    elif dtype == numpy.float16:
        pixel_format.data_type = JXL_TYPE_FLOAT16
        basic_info.bits_per_sample = 16
        basic_info.exponent_bits_per_sample = 5
    else:
        raise ValueError(f'dtype {dtype!r} not supported')

    if option_lossless != 0:
        # avoid lossy colorspace conversion
        basic_info.uses_original_profile = JXL_TRUE

    if basic_info.num_extra_channels > 0:
        basic_info.alpha_bits = basic_info.bits_per_sample
        basic_info.alpha_exponent_bits = basic_info.exponent_bits_per_sample

    # TODO: there seems to be no API to add JxlExtraChannelInfo (?)
    if basic_info.num_extra_channels > 1:
        raise ValueError(
            f'{basic_info.num_extra_channels} extra channels not supported'
        )

    if frames > 1:
        # TODO: writing animations not supported by libjxl 0.6.x
        basic_info.have_animation = JXL_TRUE
        basic_info.animation.tps_numerator = 10
        basic_info.animation.tps_denominator = 1
        basic_info.animation.num_loops = 0
        basic_info.animation.have_timecodes = JXL_FALSE

    framesize = ysize * xsize * samples * basic_info.bits_per_sample // 8

    try:
        with nogil:

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

            options = JxlEncoderOptionsCreate(encoder, NULL)
            if options == NULL:
                raise JpegxlError('JxlEncoderOptionsCreate', None)

            if option_lossless != 0:
                status = JxlEncoderOptionsSetLossless(options, option_lossless)
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderOptionsSetLossless', status)

            elif option_distance != 1.0:
                status = JxlEncoderOptionsSetDistance(options, option_distance)
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderOptionsSetDistance', status)

            if option_tier != 0:
                status = JxlEncoderOptionsSetDecodingSpeed(
                    options, option_tier
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError(
                        'JxlEncoderOptionsSetDecodingSpeed', status
                    )

            if option_effort != 7:
                status = JxlEncoderOptionsSetEffort(options, option_effort)
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderOptionsSetEffort', status)

            for frame in range(frames):

                status = JxlEncoderAddImageFrame(
                    options,
                    &pixel_format,
                    <void*> (buffer + frame * framesize),
                    framesize
                )
                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderAddImageFrame', status)

                if frame == frames - 1:
                    JxlEncoderCloseInput(encoder)

                while True:

                    next_out = compressed.data + compressed.used
                    avail_out = compressed.size - compressed.used

                    status = JxlEncoderProcessOutput(
                        encoder, &next_out, &avail_out
                    )

                    if output_seek(
                        compressed, compressed.size - avail_out
                    ) == 0:
                        raise RuntimeError('output_seek returned 0')

                    if (
                        status != JXL_ENC_NEED_MORE_OUTPUT
                        or compressed.owner == 0
                    ):
                        break

                    if output_resize(
                        compressed,
                        min(
                            compressed.size + <size_t> 33554432,  # 32 MB
                            compressed.size * 2
                        )
                    ) == 0:
                        raise RuntimeError('output_resize returned 0')

                if status != JXL_ENC_SUCCESS:
                    raise JpegxlError('JxlEncoderProcessOutput', status)

        if compressed.owner:
            out = _create_output(
                out, compressed.used, <const char *> compressed.data
            )
        else:
            out = _return_output(
                out, compressed.size, compressed.used, outgiven
            )

    finally:
        if encoder != NULL:
            JxlEncoderDestroy(encoder)
        if runner != NULL:
            JxlThreadParallelRunnerDestroy(runner)
        if compressed != NULL:
            output_del(compressed)

    return out


def jpegxl_decode(
    data,
    index=None,
    keeporientation=None,
    tojpeg=None,
    numthreads=None,
    out=None,
):
    """Return numpy array from JPEG XL image.

    """
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
        void* buffer = NULL
        void* runner = NULL
        JxlDecoder* decoder = NULL
        JxlDecoderStatus status = JXL_DEC_SUCCESS
        JxlSignature signature
        JxlBasicInfo basic_info
        JxlPixelFormat pixel_format
        size_t num_threads = _default_threads(numthreads)
        bint keep_orientation = bool(keeporientation)

    signature = JxlSignatureCheck(&src[0], srcsize)
    if signature != JXL_SIG_CODESTREAM and signature != JXL_SIG_CONTAINER:
        raise ValueError('not a JPEG XL codestream')

    if tojpeg:
        # output JPEG stream
        return jpegxl_to_jpeg(src, num_threads, out)

    if index is None:
        frames = 0
        frameindex = 0
    elif index >= 0:
        frames = 1
        frameindex = index
    else:
        # TODO: implement advanced indexing
        raise NotImplementedError('advanced indexing not implemented')

    try:
        with nogil:

            memset(<void*> &basic_info, 0, sizeof(JxlBasicInfo))
            memset(<void*> &pixel_format, 0, sizeof(JxlPixelFormat))

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
                    raise RuntimeError('no frames found')
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

                        if samples > 1:
                            shape = (
                                int(basic_info.ysize),
                                int(basic_info.xsize),
                                int(samples)
                            )
                        else:
                            shape = (
                                int(basic_info.ysize),
                                int(basic_info.xsize)
                            )

                        if frames > 1:
                            shape = (int(frames), *shape)

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
                            pixel_format.data_type = JXL_TYPE_UINT8
                            dtype = numpy.uint8
                        elif basic_info.bits_per_sample <= 16:
                            pixel_format.data_type = JXL_TYPE_UINT16
                            dtype = numpy.uint16
                        elif basic_info.bits_per_sample <= 32:
                            pixel_format.data_type = JXL_TYPE_UINT32
                            dtype = numpy.uint32
                        else:
                            dtype = numpy.float32
                            pixel_format.data_type = JXL_TYPE_FLOAT

                        out = _create_array(out, shape, dtype)
                        dst = out
                        dstsize = dst.nbytes
                        framesize = dstsize // frames
                        buffer = <void*> dst.data

                # elif status == JXL_DEC_COLOR_ENCODING:
                #     pass

                elif status == JXL_DEC_NEED_IMAGE_OUT_BUFFER:

                    if buffer == NULL:
                        raise RuntimeError('buffer == NULL')

                    if frameindex >= frames:
                        raise RuntimeError(
                            'frameindex {frameindex} > frames {frames}'
                        )

                    status = JxlDecoderImageOutBufferSize(
                        decoder, &pixel_format, &buffersize
                    )
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError(
                            'JxlDecoderImageOutBufferSize', status
                        )

                    if buffersize != framesize:
                        raise RuntimeError(
                            f'buffersize {buffersize} != framesize {framesize}'
                        )

                    status = JxlDecoderSetImageOutBuffer(
                        decoder,
                        &pixel_format,
                        <void*> (<uint8_t *> buffer + frameindex * framesize),
                        buffersize
                    )
                    if status != JXL_DEC_SUCCESS:
                        raise JpegxlError(
                            'JxlDecoderSetImageOutBuffer', status
                        )

                elif status == JXL_DEC_FULL_IMAGE:
                    frameindex += 1
                    if frameindex == frames:
                        break

                else:
                    raise RuntimeError(
                        f'JxlDecoderProcessInput unknown status {status}'
                    )

    finally:
        if decoder != NULL:
            JxlDecoderDestroy(decoder)
        if runner != NULL:
            JxlThreadParallelRunnerDestroy(runner)

    return out


cdef object jpegxl_from_jpeg(
    const uint8_t[::1] src,
    JXL_BOOL use_container,
    size_t num_threads,
    out
):
    """Return JPEG XL from JPEG stream."""
    raise NotImplementedError  # TODO: transcoding not yet supported by libjxl


cdef object jpegxl_to_jpeg(
    const uint8_t[::1] src,
    size_t num_threads,
    out
):
    """Return JPEG from JPEG XL stream."""
    raise NotImplementedError  # TODO: transcoding not yet supported by libjxl


cdef size_t jpegxl_framecount(JxlDecoder* decoder) nogil:
    """Return number of frames."""
    cdef:
        JxlDecoderStatus status = JXL_DEC_SUCCESS
        size_t framecount = 0

    status = JxlDecoderSubscribeEvents(decoder, JXL_DEC_FRAME)
    if status != JXL_DEC_SUCCESS:
        raise JpegxlError('JxlDecoderSubscribeEvents', status)

    while True:
        status = JxlDecoderProcessInput(decoder)
        if (status == JXL_DEC_ERROR or status == JXL_DEC_NEED_MORE_INPUT):
            raise JpegxlError('JxlDecoderProcessInput', status)
        if status == JXL_DEC_SUCCESS:
            break
        if status == JXL_DEC_FRAME:
            framecount += 1
        else:
            raise RuntimeError(
                f'JxlDecoderProcessInput unknown status {status}'
            )

    JxlDecoderRewind(decoder)
    return framecount


cdef int jpegxl_encode_photometric(photometric):
    """Return JxlColorSpace value from photometric argument."""
    if photometric is None:
        return -1
    if isinstance(photometric, int):
        if photometric not in (
            -1,
            JXL_COLOR_SPACE_RGB,
            JXL_COLOR_SPACE_GRAY,
            JXL_COLOR_SPACE_XYB,
            JXL_COLOR_SPACE_UNKNOWN
        ):
            raise ValueError('photometric interpretation not supported')
        return photometric
    photometric = photometric.upper()
    if photometric[:3] == 'RGB':
        return JXL_COLOR_SPACE_RGB
    if photometric in ('WHITEISZERO', 'MINISWHITE'):
        return JXL_COLOR_SPACE_GRAY
    if photometric in ('BLACKISZERO', 'MINISBLACK', 'GRAY'):
        return JXL_COLOR_SPACE_GRAY
    if photometric == 'XYB':
        return JXL_COLOR_SPACE_XYB
    if photometric == 'UNKNOWN':
        return JXL_COLOR_SPACE_UNKNOWN
    raise ValueError(
        'photometric interpretation {photometric!r} not supported'
    )


ctypedef struct output_t:
    uint8_t* data
    size_t size
    size_t pos
    size_t used
    int owner


cdef output_t* output_new(uint8_t* data, size_t size) nogil:
    """Return new output."""
    cdef:
        output_t* output = <output_t*> malloc(sizeof(output_t))

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


cdef void output_del(output_t* output) nogil:
    """Free output."""
    if output != NULL:
        if output.owner != 0:
            free(output.data)
        free(output)


cdef int output_seek(output_t* output, size_t pos) nogil:
    """Seek output to position."""
    if output == NULL or pos > output.size:
        return 0
    output.pos = pos
    if pos > output.used:
        output.used = pos
    return 1


cdef int output_resize(output_t* output, size_t newsize) nogil:
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
