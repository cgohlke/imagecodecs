# imagecodecs/_wic.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""WIC (Windows Imaging Component) codec for the imagecodecs package.

Provides decode-only access to WIC-supported image formats on Windows
(BMP, GIF, ICO, JPEG, JPEG XR, PNG, TIFF, DDS, HEIF, and WebP).

"""

include '_shared.pxi'

from wic cimport *

import atexit


def _wic_factory_destroy():
    wic_factory_destroy()


cdef int32_t _wic_init_hr = wic_factory_init()
if _wic_init_hr != 0:
    import warnings
    warnings.warn(
        f'WIC factory init failed (HRESULT 0x{<uint32_t> _wic_init_hr:08X})',
        RuntimeWarning,
    )
else:
    atexit.register(_wic_factory_destroy)


class WIC:
    """WIC codec constants."""

    available = True

    class FORMAT(enum.IntEnum):
        """WIC container format."""

        BMP = WIC_FORMAT_BMP
        PNG = WIC_FORMAT_PNG
        JPEG = WIC_FORMAT_JPEG
        TIFF = WIC_FORMAT_TIFF
        GIF = WIC_FORMAT_GIF
        WMP = WIC_FORMAT_WMP
        HEIF = WIC_FORMAT_HEIF
        WEBP = WIC_FORMAT_WEBP


class WicError(RuntimeError):
    """WIC codec exceptions."""


def wic_version():
    """Return WIC codec version string."""
    cdef:
        const char* version = wic_version_string()

    if version == NULL:
        return 'wic n/a'
    return 'wic ' + version.decode('ascii')


def wic_check(const uint8_t[::1] data, /):
    """Return whether data is WIC-decodable, or None if unknown."""
    cdef:
        ssize_t srcsize = data.nbytes
        int32_t ret

    if srcsize < 4:
        return False

    with nogil:
        ret = wic_check_impl(&data[0], <size_t> srcsize)

    return bool(ret > 0)


def wic_encode(data, /, level=None, *, format=None, out=None):
    """Return WIC encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        uint8_t* dst = NULL
        size_t dstsize = 0
        uint32_t width, height, components, bpc
        int32_t fmt = <int32_t> _enum_value(format, WIC.FORMAT, WIC.FORMAT.PNG)
        int32_t quality = -1
        int32_t hr

    if src.dtype == numpy.uint8:
        bpc = 8
    elif src.dtype == numpy.uint16:
        bpc = 16
    else:
        raise ValueError(
            f'invalid dtype={src.dtype}, expected uint8 or uint16'
        )

    if src.ndim == 2:
        height = <uint32_t> src.shape[0]
        width = <uint32_t> src.shape[1]
        components = 1
    elif src.ndim == 3:
        height = <uint32_t> src.shape[0]
        width = <uint32_t> src.shape[1]
        components = <uint32_t> src.shape[2]
    else:
        raise ValueError(f'invalid ndim={src.ndim}, expected 2 or 3')

    if components not in (1, 3, 4):
        raise ValueError(f'invalid {components=}, expected 1, 3, or 4')

    if level is not None:
        quality = <int32_t> int(level)
        if quality < 0 or quality > 100:
            raise ValueError(f'invalid {quality=}')

    try:
        with nogil:
            hr = wic_encode_impl(
                <uint8_t*> src.data,
                width,
                height,
                components,
                bpc,
                fmt,
                quality,
                &dst,
                &dstsize,
            )
        if hr != 0:
            raise WicError(
                f'wic_encode failed (HRESULT 0x{<uint32_t> hr:08X})'
            )
        out = _create_output(out, <ssize_t> dstsize, <const char*> dst)
    finally:
        if dst != NULL:
            wic_encode_free(dst)

    return out


def wic_decode(data, /, index=0, *, out=None):
    """Return decoded image from WIC-supported format."""
    cdef:
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        uint32_t width = 0
        uint32_t height = 0
        uint32_t components = 0
        uint32_t bpc = 0
        uint32_t frame_count = 0
        uint32_t frame_idx = <uint32_t> index
        uint32_t dst_stride
        int32_t hr
        numpy.ndarray dst

    if data is out:
        raise ValueError('cannot decode in-place')

    with nogil:
        hr = wic_get_info(
            &src[0], <size_t> srcsize,
            &width, &height, &components, &bpc, &frame_count,
        )

    if hr != 0:
        raise WicError(f'wic_decode failed (HRESULT 0x{<uint32_t> hr:08X})')

    if frame_idx >= frame_count:
        raise WicError(
            f'wic_decode: {frame_idx=} out of range ({frame_count=})'
        )

    if bpc == 16:
        dtype = numpy.uint16
    else:
        dtype = numpy.uint8

    if components == 1:
        shape = height, width
    else:
        shape = height, width, components

    out = _create_array(out, shape, dtype)
    dst = out
    dst_stride = width * components * (bpc // 8)

    with nogil:
        hr = wic_copy_pixels(
            &src[0], <size_t> srcsize,
            frame_idx,
            <uint8_t*> dst.data, dst_stride, <size_t> dst_stride * height,
        )

    if hr != 0:
        raise WicError(f'wic_decode failed (HRESULT 0x{<uint32_t> hr:08X})')

    return out
