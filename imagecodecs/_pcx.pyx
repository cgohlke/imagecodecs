# imagecodecs/_pcx.pyx
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

"""PCX/DCX codec for the imagecodecs package."""

include '_shared.pxi'

from pcx cimport *


class PCX:
    """PCX codec constants."""

    available = True


class PcxError(RuntimeError):
    """PCX codec exceptions."""

    def __init__(
        self, func, err, index=None, bits_per_pixel=None, nplanes=None
    ):
        msg = {
            PCX_OK: 'PCX_OK',
            PCX_ERROR: 'PCX_ERROR',
            PCX_VALUE_ERROR: 'PCX_VALUE_ERROR',
            PCX_OUTPUT_TOO_SMALL: 'PCX_OUTPUT_TOO_SMALL',
            PCX_INVALID_HEADER: 'PCX_INVALID_HEADER',
            PCX_UNSUPPORTED_FORMAT: 'PCX_UNSUPPORTED_FORMAT',
            PCX_DATA_TRUNCATED: 'PCX_DATA_TRUNCATED',
            PCX_IMAGE_TOO_LARGE: 'PCX_IMAGE_TOO_LARGE',
            PCX_INDEX_ERROR: 'PCX_INDEX_ERROR',
            PCX_SHAPE_MISMATCH: 'PCX_SHAPE_MISMATCH',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        if index is not None:
            msg += f' ({index=}, {bits_per_pixel=}, {nplanes=})'
        super().__init__(msg)


def pcx_version():
    """Return PCX codec version string."""
    return 'pcx ' + PCX_VERSION.decode()


def pcx_check(const uint8_t[::1] data not None, /):
    """Return whether data is PCX or DCX encoded or None if unknown."""
    return bool(pcx_check_(&data[0], data.shape[0]))


def pcx_encode(
    data,
    /,
    *,
    out=None,
):
    """Return PCX or DCX encoded image.

    Supported:

    - 8-bit grayscale (uint8, 2D or 3D with frames): 1 plane, 8 bpp
    - 24-bit RGB (uint8, 3D or 4D with frames): 3 planes, 8 bpp
    - 32-bit RGBA (uint8, 3D or 4D with frames): 4 planes, 8 bpp

    Multi-frame input (4D, or 3D grayscale frames) is encoded as DCX.

    PCX always uses RLE encoding. Scanline data is stored per-plane.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst
        const uint8_t* srcptr = <const uint8_t*> src.data
        uint8_t* dstptr = NULL
        ssize_t dstsize, size, frames
        imagelayout_t layout

    _image_layout(
        IC_UINT | IC_SZ1 | IC_GRAY | IC_RGB | IC_ALPHA | IC_FRAMES,
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

    if layout.samples == 2:
        raise ValueError('grayscale+alpha not supported by PCX format')
    frames = max(1, layout.frames)
    if frames > PCX_DCX_MAXPAGES:
        raise ValueError(f'too many frames {frames} > {PCX_DCX_MAXPAGES}')

    size = pcx_encode_size(layout.width, layout.height, layout.samples, frames)
    if size < 0:
        raise ValueError('invalid data shape or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = size
        out = _create_output(outtype, dstsize)

    dst = out
    dstptr = <uint8_t*> dst._data
    dstsize = dst.shape[0]

    with nogil:
        size = pcx_encode_(
            srcptr,
            layout.width,
            layout.height,
            layout.samples,
            frames,
            dstptr,
            dstsize,
        )
    if size == PCX_OUTPUT_TOO_SMALL:
        raise ValueError(f'output too small {dstsize}')
    if size < 0:
        raise PcxError('pcx_encode', size)

    del dst
    return _return_output(out, dstsize, size, outgiven)


def pcx_decode(
    data,
    /,
    index=None,
    *,
    out=None,
):
    """Return decoded PCX or DCX image.

    Supported:

    - 8 bpp, 1 plane: 8-bit paletted or grayscale
    - 8 bpp, 3 planes: 24-bit RGB
    - 8 bpp, 4 planes: 32-bit RGBA
    - 4 bpp, 1 plane: 16-color paletted (decoded to RGB uint8)
    - 2 bpp, 1 plane: 4-color CGA paletted (decoded to RGB uint8)
    - 1 bpp, 1 plane: 1-bit monochrome (decoded to bool)
    - 1 bpp, 2-4 planes: multi-plane paletted (decoded to RGB uint8)

    DCX files (multi-page PCX container) are supported.
    If ``index`` is None for DCX, return all pages stacked along axis 0.
    If ``index`` is an integer, return a single page.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t* srcptr = &src[0]
        ssize_t srcsize = src.shape[0]
        ssize_t index_ = -1 if index is None else index
        ssize_t dstsize, ret
        pcx_info_t info

    if data is out:
        raise ValueError('cannot transform in-place')

    shape = _pcx_shape(srcptr, srcsize, index, &info)

    out = _create_array(
        out, shape, numpy.bool_ if info.isbool else numpy.uint8
    )
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = pcx_decode_(
            srcptr, srcsize, index_, &info, <uint8_t*> dst.data, dstsize
        )
    if ret < 0:
        raise PcxError(
            'pcx_decode', ret, info.index, info.bits_per_pixel, info.nplanes
        )

    return out


def pcx_info(
    data,
    /,
    index=None,
):
    """Return information about PCX image."""
    cdef:
        const uint8_t[::1] src = data
        pcx_info_t info

    shape = _pcx_shape(&src[0], src.shape[0], index, &info)
    return {
        '_codec': 'pcx',
        'shape': shape,
        'dtype': numpy.bool_ if info.isbool else numpy.uint8,
        'frames': int(info.frames),
        'dcx': bool(info.dcx),
    }


cdef tuple _pcx_shape(
    const uint8_t* src, ssize_t srcsize, index, pcx_info_t* info
):
    """Parse PCX header of page into info and return decoded shape."""
    cdef:
        ssize_t index_ = -1
        ssize_t ret

    if index is not None:
        index_ = index
        if index_ < 0:
            raise IndexError(f'page {index=} out of range')

    with nogil:
        ret = pcx_info_(src, srcsize, index_, info)
    if ret == PCX_INDEX_ERROR:
        raise IndexError(f'page {index=} out of range [0, {info.frames})')
    if ret < 0:
        raise PcxError(
            'pcx_info', ret, info.index, info.bits_per_pixel, info.nplanes
        )

    if info.samples > 1:
        shape = (int(info.height), int(info.width), int(info.samples))
    else:
        shape = (int(info.height), int(info.width))
    if index is None and info.frames > 1:
        shape = (int(info.frames), *shape)
    return shape
