# imagecodecs/_sperr.pyx
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

"""SPERR (SPECK with Error-bounding) codec for the imagecodecs package."""

include '_shared.pxi'

from sperr cimport *


class SPERR:
    """SPERR codec constants."""

    available = True

    class MODE(enum.IntEnum):
        """SPERR quality mode."""

        BPP = 1  # fixed bit-per-pixel
        PSNR = 2  # fixed peak signal-to-noise ratio
        PWE = 3  # fixed point-wise error


class SperrError(RuntimeError):
    """SPERR codec exceptions."""

    def __init__(self, func, err):
        msg = {
            0: 'success',
            1: '`dst` is not pointing to a NULL pointer',
            2: 'one or more parameters are invalid',
            -1: 'other error',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def sperr_version():
    """Return SPERR library version string."""
    return (
        'sperr '
        f'{SPERR_VERSION_MAJOR}.{SPERR_VERSION_MINOR}.{SPERR_VERSION_PATCH}'
    )


def sperr_check(const uint8_t[::1] data not None, /):
    """Return whether data is SPERR encoded or None if unknown."""
    return None if data.shape[0] >= 10 else False


def sperr_encode(
    data,
    /,
    double level,
    mode,
    *,
    chunks=None,
    header=True,
    numthreads=None,
    out=None,
):
    """Return SPERR encoded data."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        void* dst_ptr = NULL
        size_t dst_len
        size_t dimx, dimy, dimz
        size_t chunk_x, chunk_y, chunk_z
        size_t nthreads = _default_threads(numthreads)
        int out_inc_header = bool(header)
        int is_float, mode_, ret

    if src.dtype == numpy.float64:
        is_float = 0
    elif src.dtype == numpy.float32:
        is_float = 1
    else:
        raise ValueError(f'invalid {src.dtype=}')

    if src.ndim == 2:
        dimz = 1
        dimy = <size_t> src.shape[0]
        dimx = <size_t> src.shape[1]
        chunk_x = 0
        chunk_y = 0
        chunk_z = 0
    elif src.ndim == 3:
        dimz = <size_t> src.shape[0]
        dimy = <size_t> src.shape[1]
        dimx = <size_t> src.shape[2]
        if chunks is None:
            chunk_x = dimx
            chunk_y = dimy
            chunk_z = dimz
        else:
            chunks = tuple(chunks)
            if len(chunks) != 3 or any(i < 0 for i in chunks):
                raise ValueError(f'invalid {chunks=}')
            chunk_z = <size_t> chunks[0]
            chunk_y = <size_t> chunks[1]
            chunk_x = <size_t> chunks[2]
    else:
        raise ValueError(f'invalid {src.ndim=}')

    if (
        dimx * dimy * dimz > <size_t> UINT32_MAX
        or dimx > <size_t> INT32_MAX
        or dimy > <size_t> INT32_MAX
        or dimz > <size_t> INT32_MAX
    ):
        raise ValueError(f'invalid {dimx=}, {dimy=}, or {dimz=}')

    if mode is None:
        raise ValueError('mode not specified')
    mode_ = _enum_value(mode, SPERR.MODE)

    with nogil:
        if src.ndim == 2:
            ret = sperr_comp_2d(
                <const void*> src.data,
                is_float,
                dimx,
                dimy,
                mode_,
                level,
                out_inc_header,
                &dst_ptr,
                &dst_len
            )
            if ret != 0 or dst_ptr == NULL:
                raise SperrError('sperr_comp_2d', ret)
        else:
            ret = sperr_comp_3d(
                <const void*> src.data,
                is_float,
                dimx,
                dimy,
                dimz,
                chunk_x,
                chunk_y,
                chunk_z,
                mode_,
                level,
                nthreads,
                &dst_ptr,
                &dst_len
            )
            if ret != 0 or dst_ptr == NULL:
                raise SperrError('sperr_comp_3d', ret)

    try:
        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            if 0 <= dstsize < <ssize_t> dst_len:
                raise ValueError(
                    f'output buffer too small {dstsize} < {dst_len}'
                )
            dstsize = <ssize_t> dst_len
            out = _create_output(outtype, dstsize, <const char*> dst_ptr)
        else:
            dst = out
            dstsize = dst.shape[0]
            if <size_t> dstsize < dst_len:
                raise ValueError(
                    f'output buffer too small {dstsize} < {dst_len}'
                )
            memcpy(<void*> dst._data, <const void*> dst_ptr, dst_len)
            del dst
    finally:
        free(dst_ptr)

    return _return_output(out, dstsize, dst_len, outgiven)


def sperr_decode(
    data,
    /,
    *,
    shape=None,
    dtype=None,
    header=True,
    numthreads=None,
    out=None,
):
    """Return decoded SPERR data.

    Either a header (always present for 3D), a ndarray output, or
    shape and dtype are required.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        size_t src_len = <size_t> src.shape[0]
        size_t dstlen
        void* dst_ptr = NULL
        size_t dimx = 0
        size_t dimy = 0
        size_t dimz = 1
        size_t nthreads = _default_threads(numthreads)
        int out_inc_header = bool(header)
        int is_3d = 0
        int is_float, output_float, ret

    if data is None:
        raise ValueError('input data is None')

    if out is not None and data is out:
        raise ValueError('cannot transform in-place')

    if src_len == 0:
        raise ValueError('src is empty')

    if isinstance(out, numpy.ndarray):
        if dtype is not None and (
            numpy.dtype(dtype).newbyteorder('=')
            != out.dtype.newbyteorder('=')
        ):
            raise ValueError(f'invalid {out.dtype=}, {dtype=}')
        dtype = out.dtype
        shape = out.shape
    if dtype is not None:
        dtype = numpy.dtype(dtype)
        if dtype.kind != 'f' or dtype.itemsize not in {4, 8}:
            raise ValueError(f'invalid {dtype=}')
    if shape is not None:
        if isinstance(shape, int | numpy.integer):
            shape = (int(shape),)
        else:
            shape = tuple(shape)

    if not out_inc_header:
        if shape is None or dtype is None:
            raise ValueError('shape and dtype required if header=False')
        squeezed = _squeeze_shape(shape, 2)
        if any(i <= 0 for i in squeezed):
            raise ValueError(f'invalid {shape=}')
        if len(squeezed) == 2:
            dimy = <size_t> squeezed[0]
            dimx = <size_t> squeezed[1]
        elif len(squeezed) == 3:
            # header always present
            out_inc_header = 1
        else:
            raise ValueError(f'invalid {shape=}')

    if out_inc_header:
        # 3D header is 14 bytes, 2D header 10 bytes
        if src_len < 10 or (src[1] & 0x40 and src_len < 14):
            raise ValueError(
                f'data too short to contain SPERR header {src_len}'
            )
        is_3d = (src[1] & 0x40) != 0
        sperr_parse_header(
            <const void*> src._data,
            &dimx,
            &dimy,
            &dimz,
            &is_float
        )
        if dimx == 0 or dimy == 0 or dimz == 0:
            raise ValueError("'sperr_parse_header' returned invalid values")
        if is_3d:
            hdrshape = (int(dimz), int(dimy), int(dimx))
        else:
            hdrshape = (int(dimy), int(dimx))
        if shape is None:
            shape = hdrshape
        elif (
            tuple(i for i in shape if i != 1)
            != tuple(i for i in hdrshape if i != 1)
        ):
            raise ValueError(f'{shape=} does not match {hdrshape=}')
        if dtype is None:
            dtype = numpy.dtype('f4' if is_float else 'f8')

    if dimx > INT32_MAX or dimy > INT32_MAX or dimz > INT32_MAX:
        raise ValueError(f'invalid {dimx=}, {dimy=}, or {dimz=}')

    output_float = dtype.itemsize == 4
    out = _create_array(out, shape, dtype)
    dst = out
    dstlen = dst.nbytes

    try:
        with nogil:
            if is_3d:
                ret = sperr_decomp_3d(
                    <const void*> src._data,
                    src_len,
                    output_float,
                    nthreads,
                    &dimx,
                    &dimy,
                    &dimz,
                    &dst_ptr
                )
                if ret != 0 or dst_ptr == NULL:
                    raise SperrError('sperr_decomp_3d', ret)
            else:
                ret = sperr_decomp_2d(
                    <const void*> &src[10 * out_inc_header],
                    src_len - 10 * out_inc_header,
                    output_float,
                    dimx,
                    dimy,
                    &dst_ptr
                )
                if ret != 0 or dst_ptr == NULL:
                    raise SperrError('sperr_decomp_2d', ret)

            if dstlen != dimx * dimy * dimz * (4 if output_float else 8):
                raise ValueError(f'invalid output size {dstlen=}')
            memcpy(<void*> dst.data, <const void*> dst_ptr, dstlen)
    finally:
        free(dst_ptr)

    if not dst.dtype.isnative:
        # library writes native byte order
        dst.byteswap(inplace=True)

    return out


def sperr_info(const uint8_t[::1] data not None, /):
    """Return information about SPERR encoded data with header."""
    cdef:
        size_t src_len = <size_t> data.shape[0]
        size_t dimx = 0
        size_t dimy = 0
        size_t dimz = 1
        uint16_t[3] chunk
        int is_float = 0
        bint is_3d

    if src_len < 10 or (data[1] & 0x40 and src_len < 14):
        raise ValueError(f'data too short to contain SPERR header {src_len}')
    is_3d = (data[1] & 0x40) != 0
    sperr_parse_header(
        <const void*> data._data, &dimx, &dimy, &dimz, &is_float
    )
    if dimx == 0 or dimy == 0 or dimz == 0:
        raise ValueError("'sperr_parse_header' returned invalid values")

    if is_3d:
        shape = (dimz, dimy, dimx)
        chunks = shape
        if data[1] & 0x10:
            # multi-chunk header contains chunk dimensions
            if src_len < 20:
                raise ValueError(
                    f'data too short to contain SPERR header {src_len}'
                )
            memcpy(<void*> chunk, <const void*> &data[14], 6)
            chunks = (chunk[2], chunk[1], chunk[0])
    else:
        shape = (dimy, dimx)
        chunks = None

    return {
        '_codec': 'sperr',
        'shape': shape,
        'dtype': numpy.dtype('f4' if is_float else 'f8'),
        'chunks': chunks,
        'version': data[0],
    }
