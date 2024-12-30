# imagecodecs/_webp.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2025, Christoph Gohlke
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

"""WebP codec for the imagecodecs package."""

include '_shared.pxi'

from libwebp cimport *


class WEBP:
    """WEBP codec constants."""

    available = True


class WebpError(RuntimeError):
    """WEBP codec exceptions."""

    def __init__(self, func, err):
        if func == 'WebPEncode':
            errors = {
                0: 'False',
                VP8_ENC_ERROR_OUT_OF_MEMORY: 'VP8_ENC_ERROR_OUT_OF_MEMORY',
                VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY:
                    'VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY',
                VP8_ENC_ERROR_NULL_PARAMETER: 'VP8_ENC_ERROR_NULL_PARAMETER',
                VP8_ENC_ERROR_INVALID_CONFIGURATION:
                    'VP8_ENC_ERROR_INVALID_CONFIGURATION',
                VP8_ENC_ERROR_BAD_DIMENSION: 'VP8_ENC_ERROR_BAD_DIMENSION',
                VP8_ENC_ERROR_PARTITION0_OVERFLOW:
                    'VP8_ENC_ERROR_PARTITION0_OVERFLOW',
                VP8_ENC_ERROR_PARTITION_OVERFLOW:
                    'VP8_ENC_ERROR_PARTITION_OVERFLOW',
                VP8_ENC_ERROR_BAD_WRITE: 'VP8_ENC_ERROR_BAD_WRITE',
                VP8_ENC_ERROR_FILE_TOO_BIG: 'VP8_ENC_ERROR_FILE_TOO_BIG',
                VP8_ENC_ERROR_USER_ABORT: 'VP8_ENC_ERROR_USER_ABORT',
                VP8_ENC_ERROR_LAST: 'VP8_ENC_ERROR_LAST',
            }
        else:
            errors = {
                0: 'NULL',
                VP8_STATUS_OUT_OF_MEMORY: 'VP8_STATUS_OUT_OF_MEMORY',
                VP8_STATUS_INVALID_PARAM: 'VP8_STATUS_INVALID_PARAM',
                VP8_STATUS_BITSTREAM_ERROR: 'VP8_STATUS_BITSTREAM_ERROR',
                VP8_STATUS_UNSUPPORTED_FEATURE:
                    'VP8_STATUS_UNSUPPORTED_FEATURE',
                VP8_STATUS_SUSPENDED: 'VP8_STATUS_SUSPENDED',
                VP8_STATUS_USER_ABORT: 'VP8_STATUS_USER_ABORT',
                VP8_STATUS_NOT_ENOUGH_DATA: 'VP8_STATUS_NOT_ENOUGH_DATA',
            }
        msg = errors.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def webp_version():
    """Return libwebp library version string."""
    cdef:
        int ver = WebPGetDecoderVersion()

    return 'libwebp {}.{}.{}'.format(ver >> 16, (ver >> 8) & 255, ver & 255)


def webp_check(const uint8_t[::1] data):
    """Return whether data is WebP encoded image."""
    cdef:
        bytes sig = bytes(data[:12])

    return sig[:4] == b'RIFF' and sig[8:12] == b'WEBP'


def webp_encode(
    data, level=None, lossless=None, method=None, numthreads=None, out=None
):
    """Return WebP encoded image.

    Libwebp drops entire alpha channel if all alpha values are 255 (opaque).

    """
    cdef:
        numpy.ndarray src = numpy.asarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* output = NULL
        size_t output_size = 0
        ssize_t dstsize
        int width, height, stride
        float quality_factor = _default_value(level, 75.0, -1.0, 100.0)
        int effort_level = _default_value(method, 4, 0, 6)
        int thread_level = 1 if numthreads else 0
        int lossless_
        int rgba
        WebPConfig config
        WebPPicture picture
        WebPMemoryWriter writer

    if not (
        src.ndim == 3
        and src.shape[0] < WEBP_MAX_DIMENSION
        and src.shape[1] < WEBP_MAX_DIMENSION
        and src.shape[2] in {3, 4}
        and src.strides[2] == 1
        and src.strides[1] in {3, 4}
        and src.strides[0] >= src.strides[1] * src.strides[2]
        and src.dtype == numpy.uint8
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    height = <int> src.shape[0]
    width = <int> src.shape[1]
    stride = <int> src.strides[0]
    rgba = <int> src.shape[2] == 4

    if lossless is None or lossless or quality_factor < 0.0:
        lossless_ = 1
    else:
        lossless_ = 0

    try:
        with nogil:

            if quality_factor < 0.0:
                quality_factor = 75.0

            if WebPPictureInit(&picture) == 0:
                raise WebpError('WebPPictureInit', 0)

            picture.use_argb = lossless_
            picture.width = width
            picture.height = height
            picture.writer = WebPMemoryWrite
            picture.custom_ptr = &writer

            WebPMemoryWriterInit(&writer)

            if WebPConfigPreset(
                &config, WEBP_PRESET_DEFAULT, quality_factor
            ) == 0:
                raise WebpError('WebPConfigPreset', 0)

            config.thread_level = thread_level
            config.method = effort_level  # 0=fast, 6=slower-better
            if lossless_:
                config.lossless = 1
                config.exact = 1  # preserve RGB values under transparent area

            # if WebPValidateConfig(&config) == 0:
            #     raise WebpError('WebPValidateConfig', 0)

            if rgba:
                # TODO: do not remove all-opaque alpha channel
                if WebPPictureImportRGBA(
                    &picture, <const uint8_t*> src.data, stride
                ) == 0:
                    raise WebpError('WebPPictureImportRGBA', 0)
            else:
                if WebPPictureImportRGB(
                    &picture, <const uint8_t*> src.data, stride
                ) == 0:
                    raise WebpError('WebPPictureImportRGB', 0)

            if WebPEncode(&config, &picture) == 0:
                if picture.error_code != VP8_ENC_OK:
                    raise WebpError('WebPEncode', picture.error_code)
                raise WebpError('WebPEncode', 0)

            output = writer.mem
            output_size = <ssize_t>writer.size

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
        WebPMemoryWriterClear(&writer)
        WebPPictureFree(&picture)

    del dst
    return _return_output(out, dstsize, output_size, outgiven)


def webp_decode(data, index=0, hasalpha=None, out=None):
    """Return decoded WebP image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t dstsize, dstoffset
        uint32_t width, height, frames  # , bgcolor, flags
        int frame, output_stride, ret
        WebPData webp_data
        WebPDemuxer* demux = NULL
        WebPIterator iter
        uint8_t* pout = NULL
        bint has_alpha

    if data is out:
        raise ValueError('cannot decode in-place')

    webp_data.bytes = &src[0]
    webp_data.size = <size_t> src.size

    demux = WebPDemux(&webp_data)
    if demux == NULL:
        raise WebpError('WebPDemux', 0)

    try:
        width = WebPDemuxGetI(demux, WEBP_FF_CANVAS_WIDTH)
        height = WebPDemuxGetI(demux, WEBP_FF_CANVAS_HEIGHT)
        frames = WebPDemuxGetI(demux, WEBP_FF_FRAME_COUNT)
        # bgcolor = WebPDemuxGetI(demux, WEBP_FF_BACKGROUND_COLOR)
        # flags = WebPDemuxGetI(demux, WEBP_FF_FORMAT_FLAGS)

        if index is None:
            index = 0
        frame = index + 1 if index >= 0 else frames + index + 1
        if frame < 1 or <uint32_t> frame > frames:
            raise IndexError(f'{index=} out of range {frames}')

        ret = WebPDemuxGetFrame(demux, frame, &iter)
        if ret == 0:
            raise WebpError('WebPDemuxGetFrame', 0)

        if (iter.has_alpha and hasalpha is None) or hasalpha:
            has_alpha = True
            shape = height, width, 4
        else:
            has_alpha = False
            shape = height, width, 3

        out = _create_array(
            out,
            shape,
            numpy.uint8,
            strides=(None, shape[2], 1),
            zero=width != iter.width or height != iter.height
        )
        dst = out
        dstsize = dst.shape[0] * dst.strides[0]
        output_stride = <int> dst.strides[0]

        # TODO: initialize output with bgcolor

        with nogil:
            if has_alpha:
                dstoffset = iter.y_offset * output_stride + iter.x_offset * 4
                pout = WebPDecodeRGBAInto(
                    iter.fragment.bytes,
                    iter.fragment.size,
                    (<uint8_t*> dst.data) + dstoffset,
                    <size_t> dstsize,
                    output_stride
                )
                if pout == NULL:
                    raise WebpError('WebPDecodeRGBAInto', 0)
            else:
                dstoffset = iter.y_offset * output_stride + iter.x_offset * 3
                pout = WebPDecodeRGBInto(
                    iter.fragment.bytes,
                    iter.fragment.size,
                    (<uint8_t*> dst.data) + dstoffset,
                    <size_t> dstsize,
                    output_stride
                )
                if pout == NULL:
                    raise WebpError('WebPDecodeRGBInto', 0)

    finally:
        WebPDemuxReleaseIterator(&iter)
        if demux != NULL:
            WebPDemuxDelete(demux)

    return out
