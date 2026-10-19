# imagecodecs/_png.pyx
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

"""PNG (Portable Network Graphics) codec for the imagecodecs package."""

include '_shared.pxi'

from zlib cimport *

from libc.setjmp cimport setjmp
from libpng cimport *


class PNG:
    """PNG codec constants."""

    available = True

    class COLOR_TYPE(enum.IntEnum):
        """PNG codec color types."""

        GRAY = PNG_COLOR_TYPE_GRAY
        GRAY_ALPHA = PNG_COLOR_TYPE_GRAY_ALPHA
        RGB = PNG_COLOR_TYPE_RGB
        RGB_ALPHA = PNG_COLOR_TYPE_RGB_ALPHA
        PALETTE = PNG_COLOR_TYPE_PALETTE

    class COMPRESSION(enum.IntEnum):
        """PNG codec compression levels."""

        DEFAULT = Z_DEFAULT_COMPRESSION
        NO = Z_NO_COMPRESSION
        BEST = Z_BEST_COMPRESSION
        SPEED = Z_BEST_SPEED

    class STRATEGY(enum.IntEnum):
        """PNG codec compression strategies."""

        DEFAULT = Z_DEFAULT_STRATEGY
        FILTERED = Z_FILTERED
        HUFFMAN_ONLY = Z_HUFFMAN_ONLY
        RLE = Z_RLE
        FIXED = Z_FIXED

    class FILTER(enum.IntFlag):
        """PNG codec filters."""

        NO = PNG_NO_FILTERS
        NONE = PNG_FILTER_NONE
        SUB = PNG_FILTER_SUB
        UP = PNG_FILTER_UP
        AVG = PNG_FILTER_AVG
        PAETH = PNG_FILTER_PAETH
        FAST = PNG_FAST_FILTERS
        ALL = PNG_ALL_FILTERS

    class COLOR_PRIMARIES(enum.IntEnum):
        """PNG color primaries (ITU-T H.273)."""

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
        """PNG transfer characteristics (ITU-T H.273)."""

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
        """PNG matrix coefficients (ITU-T H.273)."""

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


class PngError(RuntimeError):
    """PNG codec exceptions."""


def png_version():
    """Return libpng library version string."""
    return 'libpng ' + PNG_LIBPNG_VER_STRING.decode()


def png_check(const uint8_t[::1] data not None, /):
    """Return whether data is PNG encoded image or None if unknown."""
    return (
        data.shape[0] >= 8
        and png_sig_cmp(<png_const_bytep> data._data, 0, 8) == 0
    )


def png_encode(
    data,
    /,
    level=None,
    *,
    strategy=None,
    filter=None,
    primaries=None,
    transfer=None,
    matrix=None,
    out=None,
):
    """Return PNG encoded image.

    For fast encoding, matching OpenCV settings, set:

    - level=1 (PNG.COMPRESSION.SPEED)
    - strategy=3 (PNG.STRATEGY.RLE)
    - filter=16 (PNG.FILTER.SUB)

    """
    cdef:
        numpy.ndarray src = _array_input(data, out, dtype=None, contig=False)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t row, rowstride
        int color_type = PNG_COLOR_TYPE_GRAY
        int level_ = _default_value(
            level, Z_DEFAULT_COMPRESSION, -1, Z_BEST_COMPRESSION
        )
        int strategy_ = _default_value(
            _enum_value(strategy, PNG.STRATEGY),
            Z_DEFAULT_STRATEGY,
            0,
            Z_FIXED
        )
        int filter_ = _enum_value(filter, PNG.FILTER, -1)
        int primaries_ = -1
        int transfer_ = -1
        int matrix_ = -1
        png_bytep rowptr = <png_bytep> src.data
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_bytepp rowpointers = NULL
        mempng_t mempng
        imagelayout_t layout

    _image_layout(
        IC_UINT | IC_SZ1 | IC_SZ2 | IC_GRAY | IC_RGB | IC_ALPHA,
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
        layout.height > 0
        and layout.height <= INT32_MAX
        and layout.width <= INT32_MAX
        and src.strides[src.ndim - 1] == src.itemsize
        and (src.ndim == 2 or src.strides[1] == layout.samples * src.itemsize)
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    rowstride = src.strides[0]

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use dynamic mempng
            dstsize = _png_size_max(srcsize, layout.height)
        out = _create_output(outtype, dstsize)
    mempng.owner = 0

    dst = out
    dstsize = dst.shape[0]

    primaries_ = _enum_value(primaries, PNG.COLOR_PRIMARIES, -1)
    transfer_ = _enum_value(transfer, PNG.TRANSFER_CHARACTERISTICS, -1)
    matrix_ = _enum_value(matrix, PNG.MATRIX_COEFFICIENTS, -1)
    if matrix_ > 0:
        # PNG only supports RGB (identity) cICP matrix coefficients
        raise ValueError(f'{matrix=!r} not supported')

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
            for row in range(layout.height):
                rowpointers[row] = rowptr
                rowptr += rowstride

            png_ptr = png_create_write_struct(
                PNG_LIBPNG_VER_STRING,
                NULL,
                png_error_callback,
                png_warn_callback
            )
            if png_ptr == NULL:
                raise PngError('png_create_write_struct returned NULL')

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise PngError('png_create_info_struct returned NULL')

            if setjmp(png_jmpbuf(png_ptr)):
                if mempng.error != NULL:
                    raise PngError(mempng.error.decode().strip())
                raise PngError('unknown error')

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

            png_write_image(png_ptr, rowpointers)
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


def png_decode(
    data,
    /,
    *,
    out=None,
    bint _info=False,
):
    """Return decoded PNG image."""
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
        int samples = 0
        int bit_depth = 0
        int color_type = -1
        png_bytepp rowpointers = NULL
        png_bytep rowptr
        ssize_t rowstride
        ssize_t itemsize

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
                raise PngError('png_create_read_struct returned NULL')

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise PngError('png_create_info_struct returned NULL')

            if setjmp(png_jmpbuf(png_ptr)):
                if mempng.error != NULL:
                    raise PngError(mempng.error.decode().strip())
                raise PngError('unknown error')

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
                raise PngError(f'png_get_IHDR returned {ret}')

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

            if png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS):
                png_set_tRNS_to_alpha(png_ptr)

            png_set_interlace_handling(png_ptr)
            png_read_update_info(png_ptr, info_ptr)

            samples = <int> (
                png_get_rowbytes(png_ptr, info_ptr) / (width * itemsize)
            )

        dtype = numpy.dtype(f'u{itemsize}')
        if samples > 1:
            shape = int(height), int(width), int(samples)
            strides = None, int(samples * itemsize), int(itemsize)
        else:
            shape = int(height), int(width)
            strides = None, int(itemsize)

        if _info:
            return {
                '_codec': 'png',
                'shape': shape,
                'dtype': dtype,
                'photometric': 'gray' if samples <= 2 else 'rgb',
                **_png_info(png_ptr, info_ptr, color_type, bit_depth),
            }

        out = _create_array(out, shape, dtype, strides)
        dst = out
        rowptr = <png_bytep> dst.data
        rowstride = dst.strides[0]

        with nogil:
            rowpointers = <png_bytepp> calloc(height, sizeof(png_bytep))
            if rowpointers == NULL:
                raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                rowpointers[row] = rowptr
                rowptr += rowstride

            # re-arm error handler for this block
            if setjmp(png_jmpbuf(png_ptr)):
                if mempng.error != NULL:
                    raise PngError(mempng.error.decode().strip())
                raise PngError('unknown error')

            png_read_image(png_ptr, rowpointers)
            png_read_end(png_ptr, NULL)

    finally:
        if rowpointers != NULL:
            free(rowpointers)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL)
        elif png_ptr != NULL:
            png_destroy_read_struct(&png_ptr, NULL, NULL)

    return out


def png_info(data, /):
    """Return information about PNG image."""
    return png_decode(data, _info=True)


cdef dict _png_info(
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
        int icccomp = 0
        int unit_type = 0
        bint has_cicp

    has_cicp = png_get_cICP(
        png_ptr, info_ptr, &primaries, &transfer, &matrix, &fullrange
    ) != 0

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
        'bitspersample': bitspersample,
        'bit_depth': bit_depth,
        'color_type': _enum_or_int(color_type, PNG.COLOR_TYPE),
        'interlace': (
            png_get_interlace_type(png_ptr, info_ptr) == PNG_INTERLACE_ADAM7
        ),
        'resolution': resolution,
        'resolutionunit': resolutionunit,
        'primaries': (
            _enum_or_int(primaries, PNG.COLOR_PRIMARIES) if has_cicp else None
        ),
        'transfer': (
            _enum_or_int(transfer, PNG.TRANSFER_CHARACTERISTICS)
            if has_cicp
            else None
        ),
        'matrix': (
            _enum_or_int(matrix, PNG.MATRIX_COEFFICIENTS)
            if has_cicp
            else None
        ),
        'fullrange': bool(fullrange) if has_cicp else None,
        'iccprofile': iccprofile,
    }


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
    """PNG read callback function."""
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
    """PNG write callback function."""
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
    """PNG flush callback function."""


cdef ssize_t _png_size_max(ssize_t size, ssize_t height) noexcept nogil:
    """Return upper bound size of PNG stream from uncompressed image size."""
    size += height  # filter type byte per row
    size += ((size + 7) >> 3) + ((size + 63) >> 6) + 11  # ZLIB compression
    size += 12 * (size / PNG_ZBUF_SIZE + 1)  # IDAT
    size += 8 + 25 + 16 + 44 + 12  # sig IHDR gAMA cHRM IEND
    return size
