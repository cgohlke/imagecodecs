# imagecodecs/_apng.pyx
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

"""APNG (Animated PNG) codec for the imagecodecs package."""

include '_shared.pxi'

from zlib cimport *

from libc.setjmp cimport setjmp
from libpng cimport *


class APNG:
    """APNG codec constants."""

    available = True

    class COLOR_TYPE(enum.IntEnum):
        """APNG codec color types."""

        GRAY = PNG_COLOR_TYPE_GRAY
        GRAY_ALPHA = PNG_COLOR_TYPE_GRAY_ALPHA
        RGB = PNG_COLOR_TYPE_RGB
        RGB_ALPHA = PNG_COLOR_TYPE_RGB_ALPHA
        PALETTE = PNG_COLOR_TYPE_PALETTE

    class COMPRESSION(enum.IntEnum):
        """APNG codec compression levels."""

        DEFAULT = Z_DEFAULT_COMPRESSION
        NO = Z_NO_COMPRESSION
        BEST = Z_BEST_COMPRESSION
        SPEED = Z_BEST_SPEED

    class STRATEGY(enum.IntEnum):
        """APNG codec strategies."""

        DEFAULT = Z_DEFAULT_STRATEGY
        FILTERED = Z_FILTERED
        HUFFMAN_ONLY = Z_HUFFMAN_ONLY
        RLE = Z_RLE
        FIXED = Z_FIXED

    class FILTER(enum.IntFlag):
        """APNG codec filters."""

        NO = PNG_NO_FILTERS
        NONE = PNG_FILTER_NONE
        SUB = PNG_FILTER_SUB
        UP = PNG_FILTER_UP
        AVG = PNG_FILTER_AVG
        PAETH = PNG_FILTER_PAETH
        FAST = PNG_FAST_FILTERS
        ALL = PNG_ALL_FILTERS

    class COLOR_PRIMARIES(enum.IntEnum):
        """APNG color primaries (ITU-T H.273)."""

        BT709 = 1
        SRGB = 1
        UNSPECIFIED = 2
        BT470M = 4
        BT470BG = 5
        BT601 = 6
        SMPTE240 = 7
        GENERIC_FILM = 8
        BT2020 = 9
        BT2100 = 9
        XYZ = 10
        SMPTE431 = 11
        SMPTE432 = 12
        DCI_P3 = 12
        EBU3213 = 22

    class TRANSFER_CHARACTERISTICS(enum.IntEnum):
        """APNG transfer characteristics (ITU-T H.273)."""

        BT709 = 1
        UNSPECIFIED = 2
        BT470M = 4
        BT470BG = 5
        BT601 = 6
        SMPTE240 = 7
        LINEAR = 8
        LOG100 = 9
        LOG100_SQRT10 = 10
        IEC61966 = 11
        BT1361 = 12
        SRGB = 13
        BT2020_10BIT = 14
        BT2020_12BIT = 15
        PQ = 16
        SMPTE428 = 17
        HLG = 18

    class MATRIX_COEFFICIENTS(enum.IntEnum):
        """APNG matrix coefficients (ITU-T H.273)."""

        IDENTITY = 0
        BT709 = 1
        UNSPECIFIED = 2
        FCC = 4
        BT470BG = 5
        BT601 = 6
        SMPTE240 = 7
        YCGCO = 8
        BT2020_NCL = 9
        BT2020_CL = 10
        SMPTE2085 = 11
        CHROMA_DERIVED_NCL = 12
        CHROMA_DERIVED_CL = 13
        ICTCP = 14
        YCGCO_RE = 16
        YCGCO_RO = 17


class ApngError(RuntimeError):
    """APNG codec exceptions."""


def apng_version():
    """Return libpng-apng library version string."""
    return 'libpng_apng ' + PNG_LIBPNG_VER_STRING.decode()


def apng_check(const uint8_t[::1] data not None, /):
    """Return whether data is APNG encoded image or None if unknown."""
    return (
        data.shape[0] >= 8
        and png_sig_cmp(<png_const_bytep> data._data, 0, 8) == 0
    )


def apng_encode(
    data,
    /,
    level=None,
    *,
    strategy=None,
    filter=None,
    photometric=None,
    delay=None,
    primaries=None,
    transfer=None,
    matrix=None,
    out=None,
):
    """Return APNG encoded image.

    The delay parameter specifies the frame delay in milliseconds.
    Delays above 65535 ms are stored with reduced precision.

    For fast encoding, matching OpenCV settings, set:

    - level=1 (APNG.COMPRESSION.SPEED)
    - strategy=3 (APNG.STRATEGY.RLE)
    - filter=16 (APNG.FILTER.SUB)

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t row, rowstride
        int color_type = PNG_COLOR_TYPE_GRAY
        int level_ = _default_value(
            level, Z_DEFAULT_COMPRESSION, -1, Z_BEST_COMPRESSION
        )
        int strategy_ = _default_value(
            _enum_value(strategy, APNG.STRATEGY),
            Z_DEFAULT_STRATEGY,
            0,
            Z_FIXED
        )
        int filter_ = _enum_value(filter, APNG.FILTER, -1)
        int delay_ = _default_value(delay, 1000, 0, 3600000)
        int primaries_ = -1
        int transfer_ = -1
        int matrix_ = -1
        png_uint_16 delay_num
        png_uint_16 delay_den
        png_bytep rowptr = NULL
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_bytepp rowpointers = NULL
        mempng_t mempng
        imagelayout_t layout

    hasalpha = None
    if photometric is not None:
        photometric = _enum_value(photometric, APNG.COLOR_TYPE)
        if photometric == APNG.COLOR_TYPE.GRAY:
            photometric, hasalpha = 'GRAY', False
        elif photometric == APNG.COLOR_TYPE.GRAY_ALPHA:
            photometric, hasalpha = 'GRAY', True
        elif photometric == APNG.COLOR_TYPE.RGB:
            photometric, hasalpha = 'RGB', False
        elif photometric == APNG.COLOR_TYPE.RGB_ALPHA:
            photometric, hasalpha = 'RGB', True
        else:
            raise ValueError(f'{photometric=!r} not supported')

    _image_layout(
        IC_UINT | IC_SZ1 | IC_SZ2 | IC_GRAY | IC_RGB | IC_ALPHA | IC_FRAMES,
        src.ndim,
        src.shape,
        src.dtype,
        photometric,
        None,  # bitspersample
        None,  # planar
        None,  # frames
        None,  # volumetric
        'UNASSALPHA' if hasalpha else None,
        &layout,
    )

    if hasalpha is not None and hasalpha != (layout.extracount > 0):
        raise ValueError('photometric color type does not match data shape')

    if not (
        layout.height > 0
        and layout.height <= INT32_MAX
        and layout.width <= INT32_MAX
    ):
        raise ValueError('invalid data shape or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use dynamic mempng
            dstsize = _apng_size_max(srcsize, layout.frames, layout.height)
        out = _create_output(outtype, dstsize)
    mempng.owner = 0

    dst = out
    dstsize = dst.shape[0]

    rowptr = <png_bytep> src.data
    rowstride = layout.width * layout.samples * src.itemsize

    primaries_ = _enum_value(primaries, APNG.COLOR_PRIMARIES, -1)
    transfer_ = _enum_value(transfer, APNG.TRANSFER_CHARACTERISTICS, -1)
    matrix_ = _enum_value(matrix, APNG.MATRIX_COEFFICIENTS, -1)
    if matrix_ > 0:
        # PNG only supports RGB (identity) cICP matrix coefficients
        raise ValueError(f'{matrix=!r} not supported')

    if delay_ <= 65535:
        delay_num = <png_uint_16> delay_
        delay_den = 1000
    elif delay_ <= 655350:
        delay_num = <png_uint_16> ((delay_ + 5) // 10)
        delay_den = 100
    else:
        delay_num = <png_uint_16> ((delay_ + 50) // 100)
        delay_den = 10

    try:
        with nogil:
            mempng.data = <png_bytep> dst._data
            mempng.size = <png_size_t> dstsize
            mempng.offset = 0
            mempng.error = NULL

            if layout.samples == 1:
                color_type = PNG_COLOR_TYPE_GRAY
            elif layout.samples == 2:
                color_type = PNG_COLOR_TYPE_GRAY_ALPHA
            elif layout.samples == 3:
                color_type = PNG_COLOR_TYPE_RGB
            elif layout.samples == 4:
                color_type = PNG_COLOR_TYPE_RGB_ALPHA

            # allocate before setjmp. locals modified after it are clobbered
            rowpointers = <png_bytepp> calloc(layout.height, sizeof(png_bytep))
            if rowpointers == NULL:
                raise MemoryError('failed to allocate row pointers')

            png_ptr = png_create_write_struct(
                PNG_LIBPNG_VER_STRING,
                NULL,
                png_error_callback,
                png_warn_callback
            )
            if png_ptr == NULL:
                raise ApngError('png_create_write_struct returned NULL')

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise ApngError('png_create_info_struct returned NULL')

            if setjmp(png_jmpbuf(png_ptr)):
                if mempng.error != NULL:
                    raise ApngError(mempng.error.decode().strip())
                raise ApngError('unknown error')

            png_set_write_fn(
                png_ptr,
                <png_voidp> &mempng,
                png_write_data_fn,
                png_output_flush_fn
            )

            png_set_IHDR(
                png_ptr,
                info_ptr,
                <png_uint_32> layout.width,
                <png_uint_32> layout.height,
                layout.bitspersample,
                color_type,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT
            )

            if layout.frames > 1:
                png_set_acTL(png_ptr, info_ptr, <png_uint_32> layout.frames, 0)
                # png_set_first_frame_is_hidden(png_ptr, info_ptr, 1)
            if primaries_ >= 0 or transfer_ >= 0 or matrix_ >= 0:
                png_set_cICP(
                    png_ptr,
                    info_ptr,
                    <png_byte> (primaries_ if primaries_ >= 0 else 2),
                    <png_byte> (transfer_ if transfer_ >= 0 else 2),
                    0,
                    1
                )

            png_write_info(png_ptr, info_ptr)
            png_set_compression_level(png_ptr, level_)
            png_set_compression_strategy(png_ptr, strategy_)
            if filter_ >= 0:
                png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, filter_)
            if layout.bitspersample > 8:
                png_set_swap(png_ptr)

            for _frame in range(layout.frames):
                for row in range(layout.height):
                    rowpointers[row] = rowptr
                    rowptr += rowstride
                if layout.frames > 1:
                    png_write_frame_head(
                        png_ptr,
                        info_ptr,
                        rowpointers,
                        <png_uint_32> layout.width,
                        <png_uint_32> layout.height,
                        0,
                        0,
                        delay_num,
                        delay_den,
                        PNG_DISPOSE_OP_NONE,
                        PNG_BLEND_OP_SOURCE
                    )
                png_write_image(png_ptr, rowpointers)
                if layout.frames > 1:
                    png_write_frame_tail(png_ptr, info_ptr)

            png_write_end(png_ptr, info_ptr)

    finally:
        if rowpointers != NULL:
            free(rowpointers)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        elif png_ptr != NULL:
            png_destroy_write_struct(&png_ptr, NULL)

    del dst
    return _return_output(out, dstsize, mempng.offset, outgiven)


def apng_decode(
    data,
    /,
    index=None,
    *,
    out=None,
    bint _info=False,
):
    """Return decoded APNG image.

    By default, all images in the file are returned in one array, including
    hidden frames and those not part of the animation.
    If an image index is specified, ignore the disposal and blending
    modes and return the frame data on black background.
    Negative indices count from the last frame.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.shape[0]
        mempng_t mempng
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_uint_32 ret = 0
        png_uint_32 width = 0
        png_uint_32 height = 0
        png_uint_32 row
        png_uint_32 numframes = 0
        png_uint_32 frame_width = 0
        png_uint_32 frame_height = 0
        png_uint_32 frame_x_offset = 0
        png_uint_32 frame_y_offset = 0
        png_uint_16 frame_delay_num = 0
        png_uint_16 frame_delay_den = 0
        png_byte frame_dispose_op = 0
        png_byte frame_blend_op = 0
        png_bytepp rowpointers = NULL
        png_bytep rowptr
        png_bytep dataptr
        png_bytep framebuffer = NULL
        png_bytep prevbuffer = NULL
        png_bytep bufptr
        int samples = 0
        int bit_depth = 0
        int color_type = -1
        ssize_t rowstride
        ssize_t regionstride
        ssize_t itemsize
        ssize_t frameindex = -1 if index is None else index
        ssize_t frame = 0
        ssize_t framesize = 0
        ssize_t first = 0
        bint allframes = index is None
        bint hidden = False
        bint isapng

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize < 8 or png_sig_cmp(<png_const_bytep> src._data, 0, 8) != 0:
        raise ValueError('not a PNG image')

    try:
        with nogil:
            mempng.data = <png_bytep> src._data
            mempng.size = srcsize
            mempng.offset = 8
            mempng.owner = 0
            mempng.error = NULL

            png_ptr = png_create_read_struct(
                PNG_LIBPNG_VER_STRING, NULL,
                png_error_callback,
                png_warn_callback
            )
            if png_ptr == NULL:
                raise ApngError('png_create_read_struct returned NULL')

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise ApngError('png_create_info_struct returned NULL')

            if setjmp(png_jmpbuf(png_ptr)):
                if mempng.error != NULL:
                    raise ApngError(mempng.error.decode().strip())
                raise ApngError('unknown error')

            png_set_read_fn(png_ptr, <png_voidp> &mempng, png_read_data_fn)
            png_set_sig_bytes(png_ptr, 8)
            png_read_info(png_ptr, info_ptr)
            ret = png_get_IHDR(
                png_ptr,
                info_ptr,
                &width,
                &height,
                &bit_depth,
                &color_type,
                NULL,
                NULL,
                NULL
            )
            if ret != 1:
                raise ApngError(f'png_get_IHDR returned {ret}')

            if bit_depth > 8:
                png_set_swap(png_ptr)
                itemsize = 2
            else:
                itemsize = 1

            if color_type == PNG_COLOR_TYPE_GRAY:
                # samples = 1
                if bit_depth < 8:
                    png_set_expand_gray_1_2_4_to_8(png_ptr)
            elif color_type == PNG_COLOR_TYPE_GRAY_ALPHA:
                # samples = 2
                pass
            elif color_type == PNG_COLOR_TYPE_RGB:
                # samples = 3
                pass
            elif color_type == PNG_COLOR_TYPE_PALETTE:
                # samples = 3 or 4
                png_set_palette_to_rgb(png_ptr)
            elif color_type == PNG_COLOR_TYPE_RGB_ALPHA:
                # samples = 4
                pass
            else:
                raise ValueError(f'PNG {color_type=!r} not supported')

            if png_get_valid(png_ptr, info_ptr, PNG_INFO_acTL):
                numframes = png_get_num_frames(png_ptr, info_ptr)
                if numframes == 0:
                    # TODO: test this
                    # file contains only default image without frame header
                    numframes = 1
                    isapng = False
                else:
                    isapng = True
                    # libpng counts the hidden default image in num_frames
                    hidden = png_get_first_frame_is_hidden(
                        png_ptr, info_ptr
                    ) != 0
                    first = 1 if hidden else 0
            else:
                numframes = 1
                isapng = False

            if not allframes:
                if (
                    frameindex < -<ssize_t> numframes
                    or frameindex >= <ssize_t> numframes
                ):
                    raise IndexError(
                        f'frame index={frameindex} out of range '
                        f'[-{numframes}, {numframes})'
                    )
                if frameindex < 0:
                    frameindex += <ssize_t> numframes

            if png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS):
                png_set_tRNS_to_alpha(png_ptr)

            png_set_interlace_handling(png_ptr)
            png_read_update_info(png_ptr, info_ptr)

            samples = <int> (
                png_get_rowbytes(png_ptr, info_ptr) / (width * itemsize)
            )

        dtype = numpy.dtype(f'u{itemsize}')
        if samples > 1:
            if numframes == 1 or frameindex >= 0:
                shape = int(height), int(width), int(samples)
            else:
                shape = int(numframes), int(height), int(width), int(samples)
        elif numframes == 1 or frameindex >= 0:
            shape = int(height), int(width)
        else:
            shape = int(numframes), int(height), int(width)

        if _info:
            return {
                '_codec': 'apng',
                'shape': shape,
                'dtype': dtype,
                'photometric': 'gray' if samples <= 2 else 'rgb',
                **_apng_info(png_ptr, info_ptr, color_type, bit_depth),
            }

        out = _create_array(out, shape, dtype, strides=None, zero=True)
        dst = out
        dataptr = <png_bytep> dst.data
        rowstride = width * samples * itemsize
        framesize = height * rowstride

        with nogil:
            frame_width = width
            frame_height = height
            frame_x_offset = 0
            frame_y_offset = 0

            # allocate before setjmp; locals modified after it are clobbered
            rowpointers = <png_bytepp> calloc(height, sizeof(png_bytep))
            if rowpointers == NULL:
                raise MemoryError('failed to allocate row pointers')
            if frameindex > 0 or (frameindex < 0 and numframes > 1):
                framebuffer = <png_bytep> malloc(framesize)
                if framebuffer == NULL:
                    raise MemoryError('failed to allocate framebuffer')
            if frameindex < 0 and numframes > 2:
                prevbuffer = <png_bytep> malloc(framesize)
                if prevbuffer == NULL:
                    raise MemoryError('failed to allocate prevbuffer')

            # re-arm error handler for this block
            if setjmp(png_jmpbuf(png_ptr)):
                if mempng.error != NULL:
                    raise ApngError(mempng.error.decode().strip())
                raise ApngError('unknown error')

            for frame in range(<ssize_t> numframes):
                # TODO: add option to skip hidden frames

                if isapng:
                    png_read_frame_head(png_ptr, info_ptr)

                if png_get_valid(png_ptr, info_ptr, PNG_INFO_fcTL):
                    png_get_next_frame_fcTL(
                        png_ptr,
                        info_ptr,
                        &frame_width,
                        &frame_height,
                        &frame_x_offset,
                        &frame_y_offset,
                        &frame_delay_num,
                        &frame_delay_den,
                        &frame_dispose_op,
                        &frame_blend_op
                    )
                elif frame != 0:
                    raise ApngError(f'{frame=} has no fcTL chunk')

                regionstride = frame_width * samples * itemsize

                if frame < frameindex:
                    # skip image
                    # TODO: do not decode image
                    rowptr = framebuffer
                    for row in range(frame_height):
                        rowpointers[row] = rowptr
                        rowptr += rowstride
                    png_read_image(png_ptr, rowpointers)
                    continue

                if frame == frameindex:
                    # return image
                    rowptr = (
                        dataptr
                        + frame_y_offset * rowstride
                        + frame_x_offset * samples * itemsize
                    )
                    for row in range(frame_height):
                        rowpointers[row] = rowptr
                        rowptr += rowstride
                    png_read_image(png_ptr, rowpointers)
                    break

                if (
                    frame_dispose_op == PNG_DISPOSE_OP_PREVIOUS
                    and frame > first
                    and frame < <ssize_t> numframes - 1
                ):
                    # save canvas region before it is drawn over
                    rowptr = (
                        dataptr
                        + frame * framesize
                        + frame_y_offset * rowstride
                        + frame_x_offset * samples * itemsize
                    )
                    bufptr = prevbuffer
                    for _row in range(frame_height):
                        memcpy(
                            <void*> bufptr,
                            <const void*> rowptr,
                            <size_t> regionstride
                        )
                        rowptr += rowstride
                        bufptr += regionstride

                if (
                    frame_blend_op == PNG_BLEND_OP_SOURCE
                    or samples == 1
                    or samples == 3
                    or frame <= first
                ):
                    # overwrite output buffer with frame content
                    rowptr = (
                        dataptr
                        + frame * framesize
                        + frame_y_offset * rowstride
                        + frame_x_offset * samples * itemsize
                    )
                    for row in range(frame_height):
                        rowpointers[row] = rowptr
                        rowptr += rowstride
                    png_read_image(png_ptr, rowpointers)

                elif frame_blend_op == PNG_BLEND_OP_OVER:
                    # composite frame onto output buffer based on its alpha
                    # read frame into buffer
                    rowptr = framebuffer
                    for row in range(frame_height):
                        rowpointers[row] = rowptr
                        rowptr += regionstride
                    png_read_image(png_ptr, rowpointers)
                    # composition
                    rowptr = (
                        dataptr
                        + frame * framesize
                        + frame_y_offset * rowstride
                        + frame_x_offset * samples * itemsize
                    )
                    for row in range(frame_height):
                        rowpointers[row] = rowptr
                        rowptr += rowstride
                    if itemsize == 1:
                        png_composite_uint8(
                            rowpointers,
                            <const png_bytep> framebuffer,
                            <const ssize_t> frame_height,
                            <const ssize_t> frame_width,
                            <const ssize_t> samples
                        )
                    else:
                        png_composite_uint16(
                            <png_uint_16pp> rowpointers,
                            <const png_uint_16p> framebuffer,
                            <const ssize_t> frame_height,
                            <const ssize_t> frame_width,
                            <const ssize_t> samples
                        )
                else:
                    raise ApngError(f'invalid {frame_blend_op=}')

                if frame == numframes - 1:
                    continue

                if frame < first:
                    # hidden default image is not part of the animation
                    continue

                # TODO: do not dispose if next frame is covering this frame
                # and has PNG_BLEND_OP_SOURCE
                if frame_dispose_op == PNG_DISPOSE_OP_NONE:
                    memcpy(
                        <void*> (dataptr + (frame + 1) * framesize),
                        <const void*> (dataptr + frame * framesize),
                        framesize
                    )

                elif (
                    frame_dispose_op == PNG_DISPOSE_OP_BACKGROUND
                    or frame == first
                ):
                    memcpy(
                        <void*> (dataptr + (frame + 1) * framesize),
                        <const void*> (dataptr + frame * framesize),
                        framesize
                    )
                    for row in range(frame_height):
                        memset(
                            <void*> (rowpointers[row] + framesize),
                            0,
                            <size_t> regionstride,
                        )

                elif frame_dispose_op == PNG_DISPOSE_OP_PREVIOUS:
                    memcpy(
                        <void*> (dataptr + (frame + 1) * framesize),
                        <const void*> (dataptr + frame * framesize),
                        framesize
                    )
                    rowptr = (
                        dataptr
                        + (frame + 1) * framesize
                        + frame_y_offset * rowstride
                        + frame_x_offset * samples * itemsize
                    )
                    bufptr = prevbuffer
                    for _row in range(frame_height):
                        memcpy(
                            <void*> rowptr,
                            <const void*> bufptr,
                            <size_t> regionstride
                        )
                        rowptr += rowstride
                        bufptr += regionstride

                else:
                    raise ApngError(f'invalid {frame_dispose_op=}')

            # frames remain unread if decoding stopped at an earlier frame
            if frameindex < 0 or frameindex == <ssize_t> numframes - 1:
                png_read_end(png_ptr, NULL)

    finally:
        if prevbuffer != NULL:
            free(prevbuffer)
        if framebuffer != NULL:
            free(framebuffer)
        if rowpointers != NULL:
            free(rowpointers)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL)
        elif png_ptr != NULL:
            png_destroy_read_struct(&png_ptr, NULL, NULL)

    return out


def apng_info(data, /, index=0):
    """Return information about APNG image."""
    return apng_decode(data, index=index, _info=True)


cdef dict _apng_info(
    png_structp png_ptr,
    png_infop info_ptr,
    int color_type,
    int bit_depth,
):
    """Return metadata from PNG chunks preceding IDAT."""
    cdef:
        png_color_8p sig_bit = NULL
        png_charp iccname = NULL
        png_bytep iccdata = NULL
        png_uint_32 icclen = 0
        png_uint_32 res_x = 0
        png_uint_32 res_y = 0
        png_byte primaries = 0
        png_byte transfer = 0
        png_byte matrix = 0
        png_byte fullrange = 0
        png_uint_32 numframes = 0
        png_uint_16 delay_num = 0
        png_uint_16 delay_den = 0
        int icccomp = 0
        int unit_type = 0
        bint has_cicp
        bint isapng = False

    has_cicp = png_get_cICP(
        png_ptr, info_ptr, &primaries, &transfer, &matrix, &fullrange
    ) != 0

    if png_get_valid(png_ptr, info_ptr, PNG_INFO_acTL):
        numframes = png_get_num_frames(png_ptr, info_ptr)
        isapng = numframes > 0

    delay = None
    # fcTL of default image precedes IDAT
    if png_get_valid(png_ptr, info_ptr, PNG_INFO_fcTL):
        delay_num = png_get_next_frame_delay_num(png_ptr, info_ptr)
        delay_den = png_get_next_frame_delay_den(png_ptr, info_ptr)
        if delay_den == 0:
            delay_den = 100
        delay = (delay_num * 1000 + delay_den // 2) // delay_den

    if png_get_sBIT(png_ptr, info_ptr, &sig_bit) and sig_bit != NULL:
        if color_type in {
            PNG_COLOR_TYPE_RGB,
            PNG_COLOR_TYPE_RGB_ALPHA,
            PNG_COLOR_TYPE_PALETTE,
        }:
            bitspersample = int(
                max(sig_bit.red, sig_bit.green, sig_bit.blue)
            )
        else:
            bitspersample = int(sig_bit.gray)
    elif color_type == PNG_COLOR_TYPE_PALETTE:
        bitspersample = 8
    else:
        bitspersample = bit_depth

    iccprofile = None
    if (
        png_get_iCCP(
            png_ptr, info_ptr, &iccname, &icccomp, &iccdata, &icclen
        )
        and iccdata != NULL
    ):
        iccprofile = iccdata[:icclen]

    resolution = None
    resolutionunit = None
    if png_get_pHYs(png_ptr, info_ptr, &res_x, &res_y, &unit_type):
        resolution = (int(res_x), int(res_y))
        if unit_type == PNG_RESOLUTION_METER:
            resolutionunit = 'meter'

    return {
        'frames': int(numframes) if isapng else 1,
        'bitspersample': bitspersample,
        'bit_depth': bit_depth,
        'color_type': _enum_or_int(color_type, APNG.COLOR_TYPE),
        'interlace': (
            png_get_interlace_type(png_ptr, info_ptr) == PNG_INTERLACE_ADAM7
        ),
        'resolution': resolution,
        'resolutionunit': resolutionunit,
        'primaries': (
            _enum_or_int(primaries, APNG.COLOR_PRIMARIES) if has_cicp else None
        ),
        'transfer': (
            _enum_or_int(transfer, APNG.TRANSFER_CHARACTERISTICS)
            if has_cicp
            else None
        ),
        'matrix': (
            _enum_or_int(matrix, APNG.MATRIX_COEFFICIENTS)
            if has_cicp
            else None
        ),
        'fullrange': bool(fullrange) if has_cicp else None,
        'iccprofile': iccprofile,
        'delay': delay,
        'num_plays': (
            int(png_get_num_plays(png_ptr, info_ptr)) if isapng else None
        ),
        'first_frame_hidden': (
            isapng and png_get_first_frame_is_hidden(png_ptr, info_ptr) != 0
        ),
    }


cdef void png_composite_uint8(
    png_bytepp background_rowpointers,
    const png_bytep foreground,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples,
) noexcept nogil:
    """Composite foreground image against background image."""
    cdef:
        png_bytep background
        uint32_t alpha, u, v, al
        ssize_t row, _col, i, j, _s

    i = 0
    for row in range(height):
        background = background_rowpointers[row]
        j = 0
        for _col in range(width):
            alpha = foreground[i + samples - 1]
            if alpha == 0:
                i += samples
                j += samples
                continue
            if alpha == 255 or background[j + samples - 1] == 0:
                for _s in range(samples):
                    background[j] = foreground[i]
                    i += 1
                    j += 1
                continue
            # non-premultiplied OVER per APNG specification
            u = alpha * 255
            v = (255 - alpha) * background[j + samples - 1]
            al = u + v
            for _s in range(samples - 1):
                background[j] = <png_byte> (
                    (foreground[i] * u + background[j] * v) // al
                )
                i += 1
                j += 1
            background[j] = <png_byte> (al // 255)
            i += 1
            j += 1


cdef void png_composite_uint16(
    png_uint_16pp background_rowpointers,
    const png_uint_16p foreground,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples,
) noexcept nogil:
    """Composite foreground image against background image."""
    cdef:
        png_uint_16p background
        uint64_t alpha, u, v, al
        ssize_t row, _col, i, j, _s

    i = 0
    for row in range(height):
        background = background_rowpointers[row]
        j = 0
        for _col in range(width):
            alpha = foreground[i + samples - 1]
            if alpha == 0:
                i += samples
                j += samples
                continue
            if alpha == 65535 or background[j + samples - 1] == 0:
                for _s in range(samples):
                    background[j] = foreground[i]
                    i += 1
                    j += 1
                continue
            # non-premultiplied OVER per APNG specification
            u = alpha * 65535
            v = (65535 - alpha) * background[j + samples - 1]
            al = u + v
            for _s in range(samples - 1):
                background[j] = <png_uint_16> (
                    (foreground[i] * u + background[j] * v) // al
                )
                i += 1
                j += 1
            background[j] = <png_uint_16> (al // 65535)
            i += 1
            j += 1


cdef void png_error_callback(
    png_structp png_ptr,
    png_const_charp msg,
) noexcept nogil:
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)

    if mempng == NULL:
        return
    mempng.error = msg
    png_longjmp(png_ptr, 1)


cdef void png_warn_callback(
    png_structp png_ptr,
    png_const_charp msg,
) noexcept with gil:
    try:
        _log_warning('PNG warning: %s', msg.decode().strip())
    except Exception:
        pass


ctypedef struct mempng_t:
    png_bytep data
    png_const_charp error
    png_size_t size
    png_size_t offset
    int owner


cdef void png_read_data_fn(
    png_structp png_ptr,
    png_bytep dst,
    png_size_t size,
) noexcept nogil:
    """APNG read callback function."""
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)

    if mempng == NULL:
        png_error(png_ptr, b'png_read_data_fn invalid io pointer')
        return
    if (
        mempng.offset > mempng.size
        or size > mempng.size - mempng.offset
    ):
        png_error(png_ptr, b'png_read_data_fn input stream too small')
        return
    memcpy(
        <void*> dst,
        <const void*> &(mempng.data[mempng.offset]),
        size
    )
    mempng.offset += size


cdef void png_write_data_fn(
    png_structp png_ptr,
    png_bytep src,
    png_size_t size,
) noexcept nogil:
    """APNG write callback function."""
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)
        ssize_t newsize
        png_bytep tmp

    if mempng == NULL:
        return
    if size > mempng.size - mempng.offset:
        if not mempng.owner:
            png_error(png_ptr, b'png_write_data_fn output stream too small')
            return
        # always over-allocate to avoid repeated reallocations
        newsize = _align_ssize_t(
            mempng.offset + size + (mempng.offset + size) // 4
        )
        tmp = <png_bytep> realloc(<void*> mempng.data, newsize)
        if tmp == NULL:
            png_error(png_ptr, b'png_write_data_fn realloc failed')
            return
        mempng.data = tmp
        mempng.size = newsize
    memcpy(
        <void*> &(mempng.data[mempng.offset]),
        <const void*> src,
        size
    )
    mempng.offset += size


cdef void png_output_flush_fn(png_structp png_ptr) noexcept nogil:
    """APNG flush callback function."""


cdef ssize_t _apng_size_max(
    ssize_t size, ssize_t frames, ssize_t height
) noexcept nogil:
    """Return upper bound size of APNG stream from uncompressed image size."""
    # TODO: review this
    size = (size + frames - 1) / frames  # ceiling division per frame
    size += height  # filter type byte per row
    size += ((size + 7) >> 3) + ((size + 63) >> 6) + 11  # ZLIB compression
    size += 12 * (size / PNG_ZBUF_SIZE + 1)  # IDAT
    size += 8 + 25 + 16 + 44 + 12 + 64  # sig IHDR gAMA cHRM IEND
    size *= frames
    return size
