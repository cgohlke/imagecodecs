# imagecodecs/_bmp.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2023-2026, Christoph Gohlke
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

"""BMP (Bitmap) codec for the imagecodecs package."""

include '_shared.pxi'

from bmp cimport *

from . import jpeg_decode, jpeg_info, png_decode, png_info


class BMP:
    """BMP codec constants."""

    available = True


class BmpError(RuntimeError):
    """BMP codec exceptions."""

    def __init__(self, func, err, bitcount=None, compression=None):
        msg = {
            BMP_OK: 'BMP_OK',
            BMP_ERROR: 'BMP_ERROR',
            BMP_VALUE_ERROR: 'BMP_VALUE_ERROR',
            BMP_OUTPUT_TOO_SMALL: 'BMP_OUTPUT_TOO_SMALL',
            BMP_INVALID_HEADER: 'BMP_INVALID_HEADER',
            BMP_UNSUPPORTED_FORMAT: 'BMP_UNSUPPORTED_FORMAT',
            BMP_DATA_TRUNCATED: 'BMP_DATA_TRUNCATED',
            BMP_IMAGE_TOO_LARGE: 'BMP_IMAGE_TOO_LARGE',
            BMP_INDEX_ERROR: 'BMP_INDEX_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        if bitcount is not None:
            msg += f' ({bitcount=}, {compression=})'
        super().__init__(msg)


def bmp_version():
    """Return bmp codec version string."""
    return 'bmp ' + BMP_VERSION.decode()


def bmp_check(const uint8_t[::1] data not None, /):
    """Return whether data is BMP encoded or None if unknown."""
    return bool(bmp_check_(&data[0], data.shape[0]))


def bmp_encode(
    data,
    /,
    *,
    ppm=None,
    out=None,
):
    """Return BMP encoded image.

    Supported:

    - 8-bit grayscale (uint8, 2D): written as 8-bit paletted with identity
      palette
    - 24-bit RGB (uint8, 3D, 3 samples): written as 24-bit uncompressed BI_RGB
    - 32-bit RGBA (uint8, 3D, 4 samples): written as 32-bit BI_BITFIELDS BGRA
      using BITMAPV4HEADER with explicit BGRA channel masks

    Not supported:

    - 1 and 4-bit paletted
    - 16 and 64-bit direct-color
    - Any compression (BI_RLE8, BI_RLE4)
    - OS/2 file types (BA, CI, CP, IC, PT)

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t* srcptr = <const uint8_t*> src.data
        uint8_t* dstptr = NULL
        ssize_t dstsize, size, ret
        int32_t ppm_ = 3780 if ppm is None else max(1, ppm)  # 96 DPI
        imagelayout_t layout

    _image_layout(
        IC_UINT | IC_SZ1 | IC_GRAY | IC_RGB | IC_ALPHA,
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

    size = bmp_encode_size(layout.width, layout.height, layout.samples)
    if size < 0:
        raise ValueError('invalid data shape or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if 0 <= dstsize < size:
            raise ValueError(f'output too small {dstsize} < {size}')
        out = _create_output(outtype, size)
    dst = out
    dstptr = <uint8_t*> dst._data
    dstsize = dst.shape[0]
    if dstsize < size:
        raise ValueError(f'output too small {dstsize} < {size}')

    with nogil:
        ret = bmp_encode_(
            srcptr,
            layout.width,
            layout.height,
            layout.samples,
            ppm_,
            dstptr,
            dstsize,
        )
    if ret < 0:
        raise BmpError('bmp_encode', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def bmp_decode(
    data,
    /,
    *,
    asrgb=None,
    out=None,
):
    """Return decoded BMP image.

    Supported:

    - 1 and 4-bit paletted (BI_RGB): expanded via palette to grayscale or RGB
    - 8-bit paletted (BI_RGB): grayscale or RGB depending on palette content
    - 4-bit paletted (BI_RLE4): grayscale or RGB depending on palette content
    - 8-bit paletted (BI_RLE8): grayscale or RGB depending on palette content
    - 16-bit direct-color (BI_RGB RGB555, BI_BITFIELDS RGB555/RGB565/RGBA)
    - 24-bit direct-color (BI_RGB BGR)
    - 32-bit direct-color (BI_RGB BGRX treated as RGB, BI_BITFIELDS with
      optional alpha if alpha mask is present)
    - Embedded JPEG (BI_JPEG) and PNG (BI_PNG): delegated to jpeg/png decoders
    - Bottom-up (positive height) and top-down (negative height) layouts
    - OS/2 BITMAPCOREHEADER with three-byte palette entries

    Not supported:

    - 64-bit direct-color (undocumented s2.13 fixed-point BGRA)
    - OS/2 file types (BA, CI, CP, IC, PT)

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t* srcptr = &src[0]
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, ret
        bmp_info_t info

    if data is out:
        raise ValueError('cannot transform in-place')

    shape = _bmp_shape(srcptr, srcsize, asrgb, &info)

    if info.compression == BMP_BI_JPEG:
        return jpeg_decode(src[info.offbits:], out=out)
    if info.compression == BMP_BI_PNG:
        return png_decode(src[info.offbits:], out=out)

    out = _create_array(out, shape, numpy.uint8)
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = bmp_decode_(
            srcptr, srcsize, &info, <uint8_t*> dst.data, dstsize
        )
    if ret == BMP_INDEX_ERROR:
        raise IndexError(f'palette index >= {info.clr_used}')
    if ret < 0:
        raise BmpError('bmp_decode', ret, info.bitcount, info.compression)

    return out


def bmp_info(
    const uint8_t[::1] data not None,
    /,
    asrgb=None,
):
    """Return information about BMP image."""
    cdef:
        numpy.ndarray palette = None
        bmp_info_t info
        bint isjpeg

    shape = _bmp_shape(&data[0], data.shape[0], asrgb, &info)

    if (
        info.compression == BMP_BI_JPEG
        or info.compression == BMP_BI_PNG
    ):
        embedded = data[info.offbits:]
        isjpeg = info.compression == BMP_BI_JPEG
        embedded_info = jpeg_info(embedded) if isjpeg else png_info(embedded)
        shape = embedded_info['shape']
        samples = shape[2] if len(shape) > 2 else 1
        return _bmp_info(
            &info,
            shape,
            embedded_info['dtype'],
            'gray' if samples <= 2 else (
                'cmyk' if isjpeg and samples == 4 else 'rgb'
            ),
            not isjpeg and samples in {2, 4},
            None,
            None,
        )

    if info.bitcount <= 8:
        palette = numpy.empty((int(info.clr_used), 3), dtype=numpy.uint8)
        memcpy(
            <void*> palette.data,
            <const void*> info.palette,
            info.clr_used * 3,
        )
    return _bmp_info(
        &info,
        shape,
        numpy.dtype(numpy.uint8),
        'gray' if info.samples == 1 else 'rgb',
        bool(info.has_alpha),
        (
            (
                int(info.masks[0]),
                int(info.masks[1]),
                int(info.masks[2]),
                int(info.masks[3]),
            )
            if info.compression == BMP_BI_BITFIELDS
            else None
        ),
        palette,
    )


cdef tuple _bmp_shape(
    const uint8_t* src, ssize_t srcsize, asrgb, bmp_info_t* info
):
    """Parse BMP headers into info and return decoded shape."""
    cdef:
        int asrgb_ = -1 if asrgb is None else (1 if asrgb else 0)
        ssize_t ret

    with nogil:
        ret = bmp_info_(src, srcsize, asrgb_, info)
    if ret < 0:
        raise BmpError('bmp_info', ret, info.bitcount, info.compression)
    if info.samples > 1:
        return (int(info.height), int(info.width), int(info.samples))
    return (int(info.height), int(info.width))


cdef dict _bmp_info(
    const bmp_info_t* info,
    shape,
    dtype,
    colorspace,
    alpha,
    masks,
    palette,
):
    """Return BMP info dict."""
    return {
        '_codec': 'bmp',
        'shape': shape,
        'dtype': dtype,
        'bitcount': int(info.bitcount),
        'compression': {
            BMP_BI_RGB: 'BI_RGB',
            BMP_BI_RLE8: 'BI_RLE8',
            BMP_BI_RLE4: 'BI_RLE4',
            BMP_BI_BITFIELDS: 'BI_BITFIELDS',
            BMP_BI_JPEG: 'BI_JPEG',
            BMP_BI_PNG: 'BI_PNG',
        }.get(info.compression, int(info.compression)),
        'top_down': bool(info.topdown),
        'x_ppm': int(info.x_ppm),
        'y_ppm': int(info.y_ppm),
        'colorspace': colorspace,
        'alpha': alpha,
        'masks': masks,
        'palette': palette,
    }
