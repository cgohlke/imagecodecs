# imagecodecs/_png.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2021, Christoph Gohlke
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

"""PNG codec for the imagecodecs package."""

__version__ = '2021.1.28'

include '_shared.pxi'

from libpng cimport *


class PNG:
    """PNG Constants."""


class PngError(RuntimeError):
    """PNG Exceptions."""


def png_version():
    """Return libpng library version string."""
    return 'libpng ' + PNG_LIBPNG_VER_STRING.decode()


def png_check(const uint8_t[::1] data):
    """Return True if data likely contains a PNG image."""
    cdef:
        bytes sig = bytes(data[:8])

    return sig == b'\x89PNG\r\n\x1a\n'


def png_encode(data, level=None, out=None):
    """Return PNG image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t rowstride = src.strides[0]
        png_bytep rowptr = <png_bytep> &src.data[0]
        int color_type
        int bit_depth = src.itemsize * 8
        int samples = <int> src.shape[2] if src.ndim == 3 else 1
        int compresslevel = _default_value(level, 5, 0, 10)
        mempng_t mempng
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_bytepp image = NULL  # row pointers
        png_uint_32 width = <png_uint_32> src.shape[1]
        png_uint_32 height = <png_uint_32> src.shape[0]
        png_uint_32 row

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (
        src.dtype in (numpy.uint8, numpy.uint16)
        and src.ndim in (2, 3)
        and src.shape[0] < 2 ** 31
        and src.shape[1] < 2 ** 31
        and samples <= 4
        and src.strides[src.ndim - 1] == src.itemsize
        and (src.ndim == 2 or src.strides[1] == samples * src.itemsize)
    ):
        raise ValueError('invalid input shape, strides, or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = png_size_max(srcsize)  # TODO: use dynamic mempng
        out = _create_output(outtype, dstsize)
        mempng.owner = 0
    else:
        mempng.owner = 0

    dst = out
    dstsize = dst.nbytes

    try:
        with nogil:

            mempng.data = <png_bytep> &dst[0]
            mempng.size = <png_size_t> dstsize
            mempng.offset = 0

            if samples == 1:
                color_type = PNG_COLOR_TYPE_GRAY
            elif samples == 2:
                color_type = PNG_COLOR_TYPE_GRAY_ALPHA
            elif samples == 3:
                color_type = PNG_COLOR_TYPE_RGB
            elif samples == 4:
                color_type = PNG_COLOR_TYPE_RGB_ALPHA
            else:
                raise ValueError('PNG color type not supported')

            png_ptr = png_create_write_struct(
                PNG_LIBPNG_VER_STRING,
                NULL,
                png_error_callback,
                png_warn_callback
            )
            if png_ptr == NULL:
                raise PngError('png_create_write_struct returned NULL')

            png_set_write_fn(
                png_ptr,
                <png_voidp> &mempng,
                png_write_data_fn,
                png_output_flush_fn
            )

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                raise PngError('png_create_info_struct returned NULL')

            png_set_IHDR(
                png_ptr,
                info_ptr,
                width,
                height,
                bit_depth,
                color_type,
                PNG_INTERLACE_NONE,
                PNG_COMPRESSION_TYPE_DEFAULT,
                PNG_FILTER_TYPE_DEFAULT
            )

            png_write_info(png_ptr, info_ptr)
            png_set_compression_level(png_ptr, compresslevel)
            if bit_depth > 8:
                png_set_swap(png_ptr)

            image = <png_bytepp> malloc(sizeof(png_bytep) * height)
            if image == NULL:
                raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                image[row] = rowptr
                rowptr += rowstride

            png_write_image(png_ptr, image)
            png_write_end(png_ptr, info_ptr)

    finally:
        if image != NULL:
            free(image)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        elif png_ptr != NULL:
            png_destroy_write_struct(&png_ptr, NULL)

    del dst
    return _return_output(out, dstsize, mempng.offset, outgiven)


def png_decode(data, index=None, out=None):
    """Decode PNG image to numpy array.

    The Animated PNG format (apng) is not supported.

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
        int bit_depth = 0
        int color_type = -1
        png_bytepp image = NULL  # row pointers
        png_bytep rowptr
        ssize_t rowstride
        ssize_t itemsize

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
                raise ValueError(f'PNG color type not supported {color_type}')

            if png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS):
                png_set_tRNS_to_alpha(png_ptr)

            png_read_update_info(png_ptr, info_ptr)

            samples = <int> (
                png_get_rowbytes(png_ptr, info_ptr) / (width * itemsize)
            )

        dtype = numpy.dtype(f'u{itemsize}')
        if samples > 1:
            shape = int(height), int(width), int(samples)
            strides = None, shape[2] * int(itemsize), int(itemsize)
        else:
            shape = int(height), int(width)
            strides = None, int(itemsize)

        out = _create_array(out, shape, dtype, strides)
        dst = out
        rowptr = <png_bytep> &dst.data[0]
        rowstride = dst.strides[0]

        with nogil:
            image = <png_bytepp> malloc(sizeof(png_bytep) * height)
            if image == NULL:
                raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                image[row] = <png_bytep> rowptr
                rowptr += rowstride
            png_read_image(png_ptr, image)

    finally:
        if image != NULL:
            free(image)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL)
        elif png_ptr != NULL:
            png_destroy_read_struct(&png_ptr, NULL, NULL)

    return out


cdef void png_error_callback(
    png_structp png_ptr,
    png_const_charp msg
) with gil:
    raise PngError(msg.decode().strip())


cdef void png_warn_callback(
    png_structp png_ptr,
    png_const_charp msg
) with gil:
    _log_warning('PNG %s', msg.decode().strip())


ctypedef struct mempng_t:
    png_bytep data
    png_size_t size
    png_size_t offset
    int owner


cdef void png_read_data_fn(
    png_structp png_ptr,
    png_bytep dst,
    png_size_t size
) nogil:
    """PNG read callback function."""
    cdef:
        mempng_t* mempng = <mempng_t*> png_get_io_ptr(png_ptr)

    if mempng == NULL:
        return
    if mempng.offset >= mempng.size:
        return
    if size > mempng.size - mempng.offset:
        # size = mempng.size - mempng.offset
        raise PngError(f'PNG input stream too small {mempng.size}')
    memcpy(
        <void*> dst,
        <const void*> &(mempng.data[mempng.offset]),
        size)
    mempng.offset += size


cdef void png_write_data_fn(
    png_structp png_ptr,
    png_bytep src,
    png_size_t size
) nogil:
    """PNG write callback function."""
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
            raise PngError(f'PNG output stream too small {mempng.size}')
        newsize = mempng.offset + size
        if newsize <= <ssize_t> (<double> mempng.size * 1.25):
            # moderate upsize: overallocate
            newsize = newsize + newsize // 4
            newsize = (((newsize - 1) // 4096) + 1) * 4096
        tmp = <png_bytep> realloc(<void*> mempng.data, newsize)
        if tmp == NULL:
            raise MemoryError('png_write_data_fn realloc failed')
        mempng.data = tmp
        mempng.size = newsize
    memcpy(
        <void*> &(mempng.data[mempng.offset]),
        <const void*> src,
        size)
    mempng.offset += size


cdef void png_output_flush_fn(png_structp png_ptr) nogil:
    """PNG flush callback function."""
    pass


cdef ssize_t png_size_max(ssize_t size):
    """Return upper bound size of PNG stream from uncompressed image size."""
    size += ((size + 7) >> 3) + ((size + 63) >> 6) + 11  # ZLIB compression
    size += 12 * (size / PNG_ZBUF_SIZE + 1)  # IDAT
    size += 8 + 25 + 16 + 44 + 12  # sig IHDR gAMA cHRM IEND
    return size
