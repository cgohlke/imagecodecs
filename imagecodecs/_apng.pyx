# imagecodecs/_apng.pyx
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

"""APNG codec for the imagecodecs package."""

include '_shared.pxi'

from libc.setjmp cimport setjmp

from zlib cimport *
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

    class FILTER(enum.IntEnum):  # IntFlag
        """APNG codec filters."""

        NO = PNG_NO_FILTERS
        NONE = PNG_FILTER_NONE
        SUB = PNG_FILTER_SUB
        UP = PNG_FILTER_UP
        AVG = PNG_FILTER_AVG
        PAETH = PNG_FILTER_PAETH
        FAST = PNG_FAST_FILTERS
        ALL = PNG_ALL_FILTERS


class ApngError(RuntimeError):
    """APNG codec exceptions."""


def apng_version():
    """Return libpng-apng library version string."""
    return 'libpng_apng ' + PNG_LIBPNG_VER_STRING.decode()


def apng_check(const uint8_t[::1] data):
    """Return whether data is APNG encoded image."""
    cdef:
        bytes sig = bytes(data[:8])

    return sig == b'\x89PNG\r\n\x1a\n'


def apng_encode(
    data,
    level=None,
    strategy=None,
    filter=None,
    photometric=None,
    delay=None,
    out=None
):
    """Return APNG encoded image.

    For fast encoding, matching OpenCV settings, set:

    - level=1 (APNG.LEVEL.SPEED)
    - strategy=3 (APNG.STRATEGY.RLE)
    - filter=16 (APNG.FILTER.SUB)

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t rowstride
        png_bytep rowptr = NULL
        int color_type
        int bit_depth = src.itemsize * 8
        int level_ = _default_value(
            level, Z_DEFAULT_COMPRESSION, -1, Z_BEST_COMPRESSION
        )
        int strategy_ = _default_value(
            strategy, Z_DEFAULT_STRATEGY, 0, Z_FIXED
        )
        int filter_ = -1 if filter is None else <int> filter
        png_uint_16 delay_num = _default_value(delay, 1000, 1, 3600000)
        mempng_t mempng
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_bytepp rowpointers = NULL
        ssize_t frames, height, width, samples
        ssize_t row, frame
        bint isapng

    color_type = _png_colortype(photometric)

    if src.ndim == 2:
        frames = 1
        height = src.shape[0]
        width = src.shape[1]
        samples = 1
    elif src.ndim == 3:
        if src.shape[2] > 4 or color_type == PNG_COLOR_TYPE_GRAY:
            frames = src.shape[0]
            height = src.shape[1]
            width = src.shape[2]
            samples = 1
        else:
            frames = 1
            height = src.shape[0]
            width = src.shape[1]
            samples = src.shape[2]
    elif src.ndim == 4:
        frames = src.shape[0]
        height = src.shape[1]
        width = src.shape[2]
        samples = src.shape[3]
    else:
        raise ValueError(f'{src.ndim} dimensions not supported')

    if not (
        src.dtype in {numpy.uint8, numpy.uint16}
        and height <= 2147483647
        and width <= 2147483647
        and samples <= 4
    ):
        raise ValueError('invalid data shape or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = png_size_max(srcsize, frames)  # TODO: use dynamic mempng
        out = _create_output(outtype, dstsize)
        mempng.owner = 0
    else:
        mempng.owner = 0

    dst = out
    dstsize = dst.nbytes

    rowptr = <png_bytep> &src.data[0]
    rowstride = width * samples * src.itemsize
    bit_depth = src.itemsize * 8

    try:
        with nogil:

            isapng = frames > 1

            mempng.data = <png_bytep> &dst[0]
            mempng.size = <png_size_t> dstsize
            mempng.offset = 0
            mempng.error = NULL

            if samples == 1:
                color_type = PNG_COLOR_TYPE_GRAY
            elif samples == 2:
                color_type = PNG_COLOR_TYPE_GRAY_ALPHA
            elif samples == 3:
                color_type = PNG_COLOR_TYPE_RGB
            elif samples == 4:
                color_type = PNG_COLOR_TYPE_RGB_ALPHA

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
                <png_uint_32> width,
                <png_uint_32> height,
                bit_depth,
                color_type,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT
            )

            if isapng:
                png_set_acTL(png_ptr, info_ptr, <png_uint_32> frames, 0)
                # png_set_first_frame_is_hidden(png_ptr, info_ptr, 1)
            png_write_info(png_ptr, info_ptr)
            png_set_compression_level(png_ptr, level_)
            png_set_compression_strategy(png_ptr, strategy_)
            if filter_ >= 0:
                png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, filter_)
            if bit_depth > 8:
                png_set_swap(png_ptr)

            rowpointers = <png_bytepp> calloc(height, sizeof(png_bytep))
            if rowpointers == NULL:
                raise MemoryError('failed to allocate row pointers')

            for frame in range(frames):
                for row in range(height):
                    rowpointers[row] = rowptr
                    rowptr += rowstride
                if isapng:
                    png_write_frame_head(
                        png_ptr,
                        info_ptr,
                        rowpointers,
                        <png_uint_32> width,
                        <png_uint_32> height,
                        0,
                        0,
                        delay_num,
                        1000,
                        PNG_DISPOSE_OP_NONE,
                        PNG_BLEND_OP_SOURCE
                    )
                png_write_image(png_ptr, rowpointers)
                if isapng:
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


def apng_decode(data, index=None, out=None):
    """Return decoded APNG image.

    By default, all images in the file are returned in one array, including
    hidden frames and those not part of the animation.
    If an image index >= 0 is specified, ignore the disposal and blending
    modes and return the frame data on black background.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        int samples = 0
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
        int bit_depth = 0
        int color_type = -1
        png_bytepp rowpointers = NULL
        png_bytep rowptr
        png_bytep dataptr
        png_bytep framebuffer = NULL
        ssize_t rowstride
        ssize_t itemsize
        ssize_t frameindex = -1 if index is None else index
        ssize_t frame = 0
        ssize_t framesize = 0
        bint isapng

    if data is out:
        raise ValueError('cannot decode in-place')

    if png_sig_cmp(&src[0], 0, 8) != 0:
        raise ValueError('not a PNG image')

    try:
        with nogil:
            mempng.data = <png_bytep> &src[0]
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
            else:
                numframes = 1
                isapng = False

            if frameindex >= 0 and frameindex >= <ssize_t> numframes:
                raise IndexError(f'{frameindex=} out of range {numframes}')

            if png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS):
                png_set_tRNS_to_alpha(png_ptr)

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

            rowpointers = <png_bytepp> calloc(height, sizeof(png_bytep))
            if rowpointers == NULL:
                raise MemoryError('failed to allocate row pointers')

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
                    raise RuntimeError(f'{frame=} has no PNG_INFO_fcTL')
                # elif not apng:
                # TODO: skip default image, which is not part of the animation

                if frame < frameindex:
                    # skip image
                    # TODO: do not decode image
                    if framebuffer == NULL:
                        framebuffer = <png_bytep> malloc(framesize)
                        if framebuffer == NULL:
                            raise MemoryError('failed to allocate framebuffer')
                        rowptr = framebuffer
                        for row in range(height):
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
                    frame_blend_op == PNG_BLEND_OP_SOURCE
                    or samples == 1
                    or samples == 3
                    or frame == 0
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
                    if framebuffer == NULL:
                        framebuffer = <png_bytep> malloc(framesize)
                        if framebuffer == NULL:
                            raise MemoryError('failed to allocate framebuffer')
                    rowptr = framebuffer
                    for row in range(frame_height):
                        rowpointers[row] = rowptr
                        rowptr += frame_width * samples * itemsize
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
                    raise ValueError(f'invalid {frame_blend_op=}')

                if frame == numframes - 1:
                    continue

                # TODO: do not dispose if next frame is covering this frame
                # and has PNG_BLEND_OP_SOURCE
                if frame_dispose_op == PNG_DISPOSE_OP_NONE:
                    memcpy(
                        <void *> (dataptr + (frame + 1) * framesize),
                        <const void *> (dataptr + frame * framesize),
                        framesize
                    )

                elif (
                    frame_dispose_op == PNG_DISPOSE_OP_BACKGROUND
                    or frame == 0
                ):
                    memcpy(
                        <void *> (dataptr + (frame + 1) * framesize),
                        <const void *> (dataptr + frame * framesize),
                        framesize
                    )
                    for row in range(frame_height):
                        memset(
                            <void *> (rowpointers[row] + framesize),
                            0,
                            <size_t> (frame_width * samples * itemsize),
                        )

                elif frame_dispose_op == PNG_DISPOSE_OP_PREVIOUS:
                    memcpy(
                        <void *> (dataptr + (frame + 1) * framesize),
                        <const void *> (dataptr + (frame - 1) * framesize),
                        framesize
                    )

                else:
                    raise ValueError(f'invalid {frame_dispose_op=}')

    finally:
        if framebuffer != NULL:
            free(framebuffer)
        if rowpointers != NULL:
            free(rowpointers)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL)
        elif png_ptr != NULL:
            png_destroy_read_struct(&png_ptr, NULL, NULL)

    return out


cdef void png_composite_uint8(
    png_bytepp background_rowpointers,
    const png_bytep foreground,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples
) noexcept nogil:
    """Composite foreground image against background image."""
    cdef:
        png_bytep background
        png_byte alpha
        ssize_t row, col, i, j

    i = 0
    for row in range(height):
        background = background_rowpointers[row]
        j = 0
        for col in range(width):
            alpha = foreground[i + samples - 1]
            if alpha == 0:
                i += samples
                j += samples
                continue
            if alpha == 255:
                for s in range(samples):
                    background[j] = foreground[i]
                    i += 1
                    j += 1
                continue
            for s in range(samples):
                png_composite(
                    background[j],
                    foreground[i],
                    alpha,
                    background[j]
                )
                i += 1
                j += 1


cdef void png_composite_uint16(
    png_uint_16pp background_rowpointers,
    const png_uint_16p foreground,
    const ssize_t height,
    const ssize_t width,
    const ssize_t samples
) noexcept nogil:
    """Composite foreground image against background image."""
    cdef:
        png_uint_16p background
        png_uint_16 alpha
        ssize_t row, col, i, j

    i = 0
    for row in range(height):
        background = background_rowpointers[row]
        j = 0
        for col in range(width):
            alpha = foreground[i + samples - 1]
            if alpha == 0:
                i += samples
                j += samples
                continue
            if alpha == 255:
                for s in range(samples):
                    background[j] = foreground[i]
                    i += 1
                    j += 1
                continue
            for s in range(samples):
                png_composite(
                    background[j],
                    foreground[i],
                    alpha,
                    background[j]
                )
                i += 1
                j += 1


cdef _png_colortype(photometric):
    """Return color_type value from photometric argument."""
    if photometric is None:
        return -1
    if isinstance(photometric, int):
        if photometric not in {
            -1,
            PNG_COLOR_TYPE_GRAY,
            PNG_COLOR_TYPE_GRAY_ALPHA,
            PNG_COLOR_TYPE_RGB,
            PNG_COLOR_TYPE_RGB_ALPHA,
        }:
            raise ValueError(f'{photometric=!r} not supported')
        return photometric
    photometric = photometric.upper()
    if photometric == 'RGB':
        return PNG_COLOR_TYPE_RGB
    if photometric == 'RGB_ALPHA':
        return PNG_COLOR_TYPE_RGB_ALPHA
    if photometric == 'GRAY_ALPHA':
        return PNG_COLOR_TYPE_GRAY_ALPHA
    if photometric in {
        'GRAY', 'BLACKISZERO', 'MINISBLACK', 'WHITEISZERO', 'MINISWHITE'
    }:
        return PNG_COLOR_TYPE_GRAY
    raise ValueError(f'{photometric=!r} not supported')


cdef void png_error_callback(
    png_structp png_ptr,
    png_const_charp msg
) noexcept nogil:
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)

    if mempng == NULL:
        return
    mempng.error = msg
    png_longjmp(png_ptr, 1)


cdef void png_warn_callback(
    png_structp png_ptr,
    png_const_charp msg
) noexcept with gil:
    _log_warning('PNG warning: %s', msg.decode().strip())


ctypedef struct mempng_t:
    png_bytep data
    png_const_charp error
    png_size_t size
    png_size_t offset
    int owner


cdef void png_read_data_fn(
    png_structp png_ptr,
    png_bytep dst,
    png_size_t size
) noexcept nogil:
    """APNG read callback function."""
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)

    if mempng == NULL:
        return
    if mempng.offset >= mempng.size:
        return
    if size > mempng.size - mempng.offset:
        # size = mempng.size - mempng.offset
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
    png_size_t size
) noexcept nogil:
    """APNG write callback function."""
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)
        ssize_t newsize
        png_bytep tmp

    if mempng == NULL:
        return
    if mempng.offset >= mempng.size:
        return
    if size > mempng.size - mempng.offset:
        if not mempng.owner:
            png_error(png_ptr, b'png_write_data_fn output stream too small')
            return
        newsize = mempng.offset + size
        if newsize <= <ssize_t> (<double> mempng.size * 1.25):
            # moderate upsize: overallocate
            newsize = newsize + newsize // 4
            newsize = (((newsize - 1) // 4096) + 1) * 4096
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


cdef void png_output_flush_fn(
    png_structp png_ptr
) noexcept nogil:
    """APNG flush callback function."""
    pass


cdef ssize_t png_size_max(ssize_t size, ssize_t frames) noexcept nogil:
    """Return upper bound size of APNG stream from uncompressed image size."""
    # TODO: review this
    size /= frames
    size += ((size + 7) >> 3) + ((size + 63) >> 6) + 11  # ZLIB compression
    size += 12 * (size / PNG_ZBUF_SIZE + 1)  # IDAT
    size += 8 + 25 + 16 + 44 + 12 + 64  # sig IHDR gAMA cHRM IEND
    size *= frames
    return size
