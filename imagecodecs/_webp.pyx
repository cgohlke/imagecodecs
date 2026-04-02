# imagecodecs/_webp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2018-2026, Christoph Gohlke
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

"""WEBP (Web Portable) codec for the imagecodecs package."""

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

    return f'libwebp {ver >> 16}.{(ver >> 8) & 255}.{ver & 255}'


def webp_check(const uint8_t[::1] data, /):
    """Return whether data is WEBP encoded image or None if unknown."""
    cdef:
        bytes sig = bytes(data[:12])

    return sig[:4] == b'RIFF' and sig[8:12] == b'WEBP'


def webp_encode(
    data,
    /,
    level=None,
    *,
    lossless=None,
    method=None,
    numthreads=None,
    delay=None,
    out=None,
):
    """Return WEBP encoded image.

    Libwebp drops entire alpha channel if all alpha values are 255 (opaque).

    """
    cdef:
        numpy.ndarray src = numpy.asarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* output = NULL
        size_t output_size = 0
        ssize_t dstsize
        int stride
        float quality_factor = _default_value(level, 75.0, -1.0, 100.0)
        int effort_level = _default_value(method, 4, 0, 6)
        int thread_level = 1 if numthreads else 0
        int lossless_
        WebPConfig config
        WebPPicture picture
        WebPMemoryWriter writer
        imagelayout_t layout

    if data is out:
        raise ValueError('cannot encode in-place')

    _image_layout(
        IC_UINT | IC_SZ1 | IC_RGB | IC_ALPHA | IC_FRAMES,
        src.ndim,
        src.shape,
        src.dtype,
        None,  # photometric
        None,  # bitspersample
        None,  # planar
        None,  # frames
        None,  # volumetric
        None,  # extrasample
        &layout,
    )

    if not (
        layout.width > 0
        and layout.height > 0
        and layout.height < WEBP_MAX_DIMENSION
        and layout.width < WEBP_MAX_DIMENSION
    ):
        raise ValueError('invalid data shape or dtype')

    lossless_ = int(lossless is None or lossless or quality_factor < 0.0)

    if layout.frames > 1:
        return _webp_encode_frames(
            src, layout, quality_factor, effort_level,
            thread_level, lossless_, delay, out
        )

    if not (
        src.strides[2] == 1
        and src.strides[1] == layout.samples
        and src.strides[0] >= layout.samples * layout.width
    ):
        raise ValueError('invalid data strides')

    stride = <int> src.strides[0]

    WebPMemoryWriterInit(&writer)
    if WebPPictureInit(&picture) == 0:
        raise WebpError('WebPPictureInit', 0)

    try:
        with nogil:
            if quality_factor < 0.0:
                quality_factor = 75.0

            picture.use_argb = lossless_
            picture.width = <int> layout.width
            picture.height = <int> layout.height
            picture.writer = WebPMemoryWrite
            picture.custom_ptr = &writer

            if WebPConfigPreset(
                &config, WEBP_PRESET_DEFAULT, quality_factor
            ) == 0:
                raise WebpError('WebPConfigPreset', 0)

            config.thread_level = thread_level
            config.method = effort_level  # 0=fast, 6=slower-better
            if lossless_:
                config.lossless = 1
                config.exact = 1  # preserve RGB values under transparent area

            # is this necessary?
            # if WebPValidateConfig(&config) == 0:
            #     raise WebpError('WebPValidateConfig', 0)

            if layout.samples == 4:
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
            output_size = writer.size

        out, dstsize, outgiven, outtype = _parse_output(out)

        if out is None and dstsize < 0:
            out = _create_output(outtype, output_size, <const char*> output)
            return out

        if out is None:
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.nbytes
        if <size_t> dstsize < output_size:
            raise RuntimeError('output too small')

        with nogil:
            memcpy(<void*> &dst[0], <const void*> output, output_size)

    finally:
        WebPMemoryWriterClear(&writer)
        WebPPictureFree(&picture)

    del dst
    return _return_output(out, dstsize, output_size, outgiven)


def webp_decode(
    data,
    /,
    index=None,
    *,
    hasalpha=None,
    out=None,
):
    """Return decoded WEBP image."""
    cdef:
        numpy.ndarray dst
        numpy.ndarray rgba_temp
        const uint8_t[::1] src = data
        ssize_t fi, target_frame, nframes
        ssize_t anim_row_bytes, pi
        uint32_t width, height, frames
        int frame, output_stride, ret, timestamp
        int err = 0
        uint8_t* anim_buf = NULL
        uint8_t* dest = NULL
        uint8_t* pout = NULL
        WebPDemuxer* demux = NULL
        WebPAnimDecoder* anim_dec = NULL
        WebPAnimDecoderOptions anim_options
        WebPAnimInfo anim_info
        WebPIterator iter
        WebPData webp_data
        bint iter_active = 0
        bint has_alpha
        bint all_opaque

    if data is out:
        raise ValueError('cannot decode in-place')

    webp_data.bytes = &src[0]
    webp_data.size = <size_t> src.nbytes

    demux = WebPDemux(&webp_data)
    if demux == NULL:
        raise WebpError('WebPDemux', 0)

    try:
        width = WebPDemuxGetI(demux, WEBP_FF_CANVAS_WIDTH)
        height = WebPDemuxGetI(demux, WEBP_FF_CANVAS_HEIGHT)
        frames = WebPDemuxGetI(demux, WEBP_FF_FRAME_COUNT)
        # flags = WebPDemuxGetI(demux, WEBP_FF_FORMAT_FLAGS)

        if frames > 1:
            # use WebPAnimDecoder for correct frame compositing
            WebPDemuxDelete(demux)
            demux = NULL

            if WebPAnimDecoderOptionsInit(&anim_options) == 0:
                raise WebpError('WebPAnimDecoderOptionsInit', 0)
            anim_options.color_mode = MODE_RGBA
            anim_options.use_threads = 1

            anim_dec = WebPAnimDecoderNew(&webp_data, &anim_options)
            if anim_dec == NULL:
                raise WebpError('WebPAnimDecoderNew', 0)

            try:
                if WebPAnimDecoderGetInfo(anim_dec, &anim_info) == 0:
                    raise WebpError('WebPAnimDecoderGetInfo', 0)

                nframes = <ssize_t> anim_info.frame_count

                if index is not None:
                    target_frame = (
                        index if index >= 0 else nframes + index
                    )
                    if target_frame < 0 or target_frame >= nframes:
                        raise IndexError(
                            f'{index=} out of range [0, {nframes - 1}]'
                        )
                else:
                    target_frame = -1

                anim_row_bytes = (
                    <ssize_t> anim_info.canvas_width * 4
                )
                rgba_temp = numpy.empty(
                    (
                        nframes if target_frame < 0 else 1,
                        <ssize_t> anim_info.canvas_height,
                        <ssize_t> anim_info.canvas_width,
                        4,
                    ),
                    dtype=numpy.uint8,
                )
                all_opaque = 1

                with nogil:
                    for fi in range(nframes):
                        ret = WebPAnimDecoderGetNext(
                            anim_dec, &anim_buf, &timestamp
                        )
                        if ret == 0:
                            err = 1
                            break
                        if target_frame < 0:
                            dest = (
                                <uint8_t*> rgba_temp.data
                                + fi
                                * <ssize_t> anim_info.canvas_height
                                * anim_row_bytes
                            )
                            memcpy(
                                dest,
                                anim_buf,
                                <ssize_t> anim_info.canvas_height
                                * anim_row_bytes,
                            )
                        elif fi == target_frame:
                            memcpy(
                                <uint8_t*> rgba_temp.data,
                                anim_buf,
                                <ssize_t> anim_info.canvas_height
                                * anim_row_bytes,
                            )
                        else:
                            continue
                        if all_opaque:
                            for pi in range(
                                <ssize_t> anim_info.canvas_width
                                * <ssize_t> anim_info.canvas_height
                            ):
                                if anim_buf[pi * 4 + 3] != 255:
                                    all_opaque = 0
                                    break

                if err:
                    raise WebpError('WebPAnimDecoderGetNext', 0)

            finally:
                if anim_dec != NULL:
                    WebPAnimDecoderDelete(anim_dec)

            # determine alpha handling
            if hasalpha is not None:
                has_alpha = <bint> hasalpha
            else:
                has_alpha = not all_opaque

            if target_frame >= 0:
                if has_alpha:
                    result = numpy.ascontiguousarray(
                        rgba_temp[0]
                    )
                else:
                    result = numpy.ascontiguousarray(
                        rgba_temp[0, :, :, :3]
                    )
            else:
                if has_alpha:
                    result = rgba_temp
                else:
                    result = numpy.ascontiguousarray(
                        rgba_temp[:, :, :, :3]
                    )

            if out is not None:
                out = _create_array(
                    out,
                    result.shape,
                    numpy.uint8,
                    strides=None,
                    zero=False,
                )
                numpy.copyto(out, result)
                return out
            return result

        else:
            if index is None:
                index = 0
            frame = index + 1 if index >= 0 else frames + index + 1
            if frame < 1 or <uint32_t> frame > frames:
                raise IndexError(f'{index=} out of range [0, {frames - 1}]')

            ret = WebPDemuxGetFrame(demux, frame, &iter)
            iter_active = 1
            if ret == 0:
                raise WebpError('WebPDemuxGetFrame', 0)

            has_alpha = <bint> (
                (iter.has_alpha and hasalpha is None) or hasalpha
            )
            shape = int(height), int(width), 4 if has_alpha else 3
            out = _create_array(
                out,
                shape,
                numpy.uint8,
                strides=(None, shape[2], 1),
                zero=False
            )
            dst = out
            output_stride = <int> dst.strides[0]
            with nogil:
                if has_alpha:
                    pout = WebPDecodeRGBAInto(
                        iter.fragment.bytes,
                        iter.fragment.size,
                        <uint8_t*> dst.data,
                        <size_t> (dst.shape[0] * output_stride),
                        output_stride,
                    )
                else:
                    pout = WebPDecodeRGBInto(
                        iter.fragment.bytes,
                        iter.fragment.size,
                        <uint8_t*> dst.data,
                        <size_t> (dst.shape[0] * output_stride),
                        output_stride,
                    )
            if pout == NULL:
                raise WebpError(
                    'WebPDecodeRGBAInto'
                    if has_alpha
                    else 'WebPDecodeRGBInto',
                    0
                )

    finally:
        if iter_active:
            WebPDemuxReleaseIterator(&iter)
        if demux != NULL:
            WebPDemuxDelete(demux)

    return out


cdef _webp_encode_frames(
    numpy.ndarray src,
    imagelayout_t layout,
    float quality_factor,
    int effort_level,
    int thread_level,
    int lossless_,
    object delay,
    object out,
):
    """Return animated WEBP encoded from multiple frames."""
    cdef:
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize, output_size
        ssize_t frame_stride = src.strides[0]
        int stride = <int> src.strides[1]
        int delay_ms = 100 if delay is None else <int> delay
        ssize_t i
        uint8_t* frame_ptr
        WebPConfig config
        WebPPicture picture
        WebPAnimEncoder* enc = NULL
        WebPAnimEncoderOptions enc_options
        WebPData webp_data

    if not (
        src.strides[3] == 1
        and src.strides[2] == layout.samples
        and src.strides[1] >= layout.samples * layout.width
    ):
        raise ValueError('invalid data strides')

    if quality_factor < 0.0:
        quality_factor = 75.0

    webp_data.bytes = NULL
    webp_data.size = 0

    try:
        with nogil:
            if WebPAnimEncoderOptionsInit(&enc_options) == 0:
                raise WebpError('WebPAnimEncoderOptionsInit', 0)

            enc = WebPAnimEncoderNew(
                <int> layout.width, <int> layout.height, &enc_options
            )
            if enc == NULL:
                raise WebpError('WebPAnimEncoderNew', 0)

            if WebPConfigPreset(
                &config, WEBP_PRESET_DEFAULT, quality_factor
            ) == 0:
                raise WebpError('WebPConfigPreset', 0)

            config.thread_level = thread_level
            config.method = effort_level
            if lossless_:
                config.lossless = 1
                config.exact = 1

            for i in range(layout.frames):
                if WebPPictureInit(&picture) == 0:
                    raise WebpError('WebPPictureInit', 0)

                picture.use_argb = lossless_
                picture.width = <int> layout.width
                picture.height = <int> layout.height
                frame_ptr = (<uint8_t*> src.data) + i * frame_stride

                if layout.samples == 4:
                    if WebPPictureImportRGBA(
                        &picture, frame_ptr, stride
                    ) == 0:
                        WebPPictureFree(&picture)
                        raise WebpError('WebPPictureImportRGBA', 0)
                else:
                    if WebPPictureImportRGB(
                        &picture, frame_ptr, stride
                    ) == 0:
                        WebPPictureFree(&picture)
                        raise WebpError('WebPPictureImportRGB', 0)

                if WebPAnimEncoderAdd(
                    enc, &picture, <int> i * delay_ms, &config
                ) == 0:
                    WebPPictureFree(&picture)
                    raise WebpError('WebPAnimEncoderAdd', 0)

                WebPPictureFree(&picture)

            # NULL frame marks end of animation
            if WebPAnimEncoderAdd(
                enc, NULL, <int> layout.frames * delay_ms, NULL
            ) == 0:
                raise WebpError('WebPAnimEncoderAdd', 0)

            if WebPAnimEncoderAssemble(enc, &webp_data) == 0:
                raise WebpError('WebPAnimEncoderAssemble', 0)

            output_size = <ssize_t> webp_data.size

        out, dstsize, outgiven, outtype = _parse_output(out)

        if out is None and dstsize < 0:
            out = _create_output(
                outtype, output_size, <const char*> webp_data.bytes
            )
            return out

        if out is None:
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.nbytes
        if <size_t> dstsize < <size_t> output_size:
            raise RuntimeError('output too small')

        with nogil:
            memcpy(
                <void*> &dst[0], <const void*> webp_data.bytes, output_size
            )

    finally:
        WebPDataClear(&webp_data)
        if enc != NULL:
            WebPAnimEncoderDelete(enc)

    del dst
    return _return_output(out, dstsize, output_size, outgiven)
