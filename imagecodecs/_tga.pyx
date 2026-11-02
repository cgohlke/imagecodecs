# imagecodecs/_tga.pyx
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

"""TGA (TARGA) codec for the imagecodecs package."""

include '_shared.pxi'

from tga cimport *


class TGA:
    """TGA codec constants."""

    available = True


class TgaError(RuntimeError):
    """TGA codec exceptions."""

    def __init__(self, func, err, image_type=None, pixel_depth=None):
        msg = {
            TGA_OK: 'TGA_OK',
            TGA_ERROR: 'TGA_ERROR',
            TGA_VALUE_ERROR: 'TGA_VALUE_ERROR',
            TGA_OUTPUT_TOO_SMALL: 'TGA_OUTPUT_TOO_SMALL',
            TGA_INVALID_HEADER: 'TGA_INVALID_HEADER',
            TGA_UNSUPPORTED_FORMAT: 'TGA_UNSUPPORTED_FORMAT',
            TGA_DATA_TRUNCATED: 'TGA_DATA_TRUNCATED',
            TGA_IMAGE_TOO_LARGE: 'TGA_IMAGE_TOO_LARGE',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        if image_type is not None:
            msg += f' ({image_type=}, {pixel_depth=})'
        super().__init__(msg)


def tga_version():
    """Return tga codec version string."""
    return 'tga ' + TGA_VERSION.decode()


def tga_check(const uint8_t[::1] data not None, /):
    """Return whether data is TGA encoded or None if unknown."""
    cdef int ret = tga_check_(&data[0], data.shape[0])
    return None if ret < 0 else bool(ret)


def tga_encode(
    data,
    /,
    *,
    rle=False,
    out=None,
):
    """Return TGA encoded image.

    Supported:

    - 8-bit grayscale (uint8, 2D): type 3 or RLE type 11
    - 16-bit grayscale+alpha (uint8, 3D, 2 samples): type 3 or RLE type 11
    - 24-bit RGB (uint8, 3D, 3 samples): type 2 or RLE type 10, stored BGR
    - 32-bit RGBA (uint8, 3D, 4 samples): type 2 or RLE type 10, stored BGRA

    Not supported:

    - Paletted (colormap) images
    - 16-bit direct-color (RGB555)

    A TGA 2.0 footer is appended so that tga_check returns True.
    Images are encoded bottom-up (origin at lower-left), which is standard.

    """
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst
        const uint8_t* srcptr = <const uint8_t*> src.data
        uint8_t* dstptr = NULL
        ssize_t dstsize, minsize, size
        int rle_ = bool(rle)
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

    # 18-byte header + pixel data (worst case) + 26-byte TGA 2.0 footer
    minsize = tga_encode_size(
        layout.width, layout.height, layout.samples, rle_
    )
    if minsize < 0:
        raise ValueError('invalid data shape or dtype')

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = minsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstptr = <uint8_t*> dst._data
    dstsize = dst.shape[0]

    if dstsize < 44:
        raise ValueError(f'output too small {dstsize} < 44')
    if not rle_ and dstsize < minsize:
        raise ValueError(f'output too small {dstsize} < {minsize}')

    with nogil:
        size = tga_encode_(
            srcptr,
            layout.width,
            layout.height,
            layout.samples,
            rle_,
            dstptr,
            dstsize,
        )
    if size == TGA_OUTPUT_TOO_SMALL:
        raise ValueError(f'output too small: buffer exhausted {dstsize=}')
    if size < 0:
        raise TgaError('tga_encode', size)

    del dst
    return _return_output(out, dstsize, size, outgiven)


def tga_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded TGA image.

    Supported:

    - Type 2 true-color: 24-bit (BGR->RGB), 32-bit (BGRA->RGBA),
      15/16-bit (RGB555->RGB, or RGBA if 1 alpha bit)
    - Type 3 grayscale: 8-bit, 16-bit with 8-bit alpha
    - Type 10 RLE true-color: same as type 2 but run-length encoded
    - Type 11 RLE grayscale: same as type 3 but run-length encoded
    - Type 1 color-mapped: 8-bit indices with 15, 16, 24, or 32-bit colormap
    - Type 9 RLE color-mapped: same as type 1 but RLE
    - Bottom-up, top-down, and right-to-left layouts

    Not supported:

    - Types 0 (no image), 32, 33 (compressed)

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t* srcptr = &src[0]
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, ret
        tga_info_t info

    if data is out:
        raise ValueError('cannot transform in-place')

    shape = _tga_shape(srcptr, srcsize, &info)

    out = _create_array(out, shape, numpy.uint8)
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = tga_decode_(
            srcptr, srcsize, &info, <uint8_t*> dst.data, dstsize
        )
    if ret < 0:
        raise TgaError('tga_decode', ret, info.image_type, info.pixel_depth)

    return out


def tga_info(const uint8_t[::1] data not None, /):
    """Return information about TGA image."""
    cdef:
        tga_info_t info

    shape = _tga_shape(&data[0], data.shape[0], &info)
    result = {
        '_codec': 'tga',
        'shape': shape,
        'dtype': numpy.dtype(numpy.uint8),
        'photometric': 'gray' if info.samples <= 2 else 'rgb',
        'image_type': int(info.image_type),
        'rle': bool(info.rle),
        'pixel_depth': int(info.pixel_depth),
        'alpha_bits': int(info.alpha_bits),
        'topdown': bool(info.topdown),
        'righttoleft': bool(info.righttoleft),
        'origin': (int(info.x_origin), int(info.y_origin)),
    }
    if info.colormap_type == 1:
        result['colormap_origin'] = int(info.colormap_origin)
        result['colormap_length'] = int(info.colormap_length)
        result['colormap_depth'] = int(info.colormap_depth)
    if info.id_length > 0:
        result['image_id'] = _tga_str(info.image_id)
    result.update(_tga_extension(&info.extension))
    return result


cdef tuple _tga_shape(
    const uint8_t* src, ssize_t srcsize, tga_info_t* info
):
    """Parse TGA header into info and return decoded shape."""
    cdef ssize_t ret

    with nogil:
        ret = tga_info_(src, srcsize, info)
    if ret < 0:
        raise TgaError('tga_info', ret, info.image_type, info.pixel_depth)
    if info.samples > 1:
        return (int(info.height), int(info.width), int(info.samples))
    return (int(info.height), int(info.width))


cdef str _tga_str(const char* s):
    cdef bytes b = s
    return b.decode('latin-1').strip()


cdef dict _tga_extension(const tga_extension_t* ext):
    """Return metadata from TGA 2.0 footer and extension area."""
    cdef int k

    if ext.version != 2:
        return {'version': 1}

    info = {'version': 2}
    if not ext.valid:
        return info

    info['author'] = _tga_str(ext.author)
    info['comments'] = '\n'.join(
        [_tga_str(ext.comments[k]) for k in range(4)]
    ).rstrip()
    if ext.year != 0:
        import datetime
        try:
            info['datetime'] = datetime.datetime(
                ext.year,
                ext.month,
                ext.day,
                ext.hour,
                ext.minute,
                ext.second,
            )
        except ValueError:
            pass
    info['job_name'] = _tga_str(ext.job_name)
    info['job_time'] = (
        int(ext.job_hours),
        int(ext.job_minutes),
        int(ext.job_seconds),
    )
    info['software'] = _tga_str(ext.software)
    if ext.software_version != 0:
        info['software_version'] = (
            f'{int(ext.software_version) / 100:.2f}'
        ) + (
            chr(ext.software_letter)
            if 32 < ext.software_letter < 127
            else ''
        )
    info['key_color'] = int(ext.key_color)
    if ext.aspect_denominator != 0:
        info['pixel_aspect_ratio'] = (
            int(ext.aspect_numerator),
            int(ext.aspect_denominator),
        )
    if ext.gamma_denominator != 0:
        info['gamma'] = (
            int(ext.gamma_numerator) / int(ext.gamma_denominator)
        )
    info['thumbnail'] = ext.postage_stamp_offset != 0
    # 0: no alpha, 1-2: undefined, 3: alpha, 4: premultiplied alpha
    info['attributes_type'] = int(ext.attributes_type)
    return info
