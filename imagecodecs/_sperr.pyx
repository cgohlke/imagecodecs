# imagecodecs/_sperr.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2023-2025, Christoph Gohlke
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

"""SPERR codec for the imagecodecs package."""

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


def sperr_check(const uint8_t[::1] data):
    """Return whether data is SPERR encoded."""
    return None


def sperr_encode(
    data,
    double level,
    mode,
    chunks=None,
    header=True,
    numthreads=None,
    out=None
):
    """Return SPERR encoded data."""
    cdef:
        numpy.ndarray src = numpy.asarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        void* dst_ptr = NULL
        size_t dst_len
        size_t dimx, dimy, dimz
        size_t chunk_x, chunk_y, chunk_z
        size_t nthreads = <size_t> _default_threads(numthreads)
        int out_inc_header = bool(header)
        int is_float, mode_, ret

    if src.dtype.char == 'd':
        is_float = 0
    elif src.dtype.char == 'f':
        is_float = 1
    else:
        raise ValueError(f'invalid {src.dtype=}')

    if src.ndim == 2:
        dimz = 1
        dimy = <size_t> src.shape[0]
        dimx = <size_t> src.shape[1]
    elif src.ndim == 3:
        dimz = <size_t> src.shape[0]
        dimy = <size_t> src.shape[1]
        dimx = <size_t> src.shape[2]
        if chunks is None:
            chunk_x = dimx
            chunk_y = dimy
            chunk_z = dimz
        else:
            chunk_z = <size_t> chunks[0]
            chunk_y = <size_t> chunks[1]
            chunk_x = <size_t> chunks[2]
    else:
        raise ValueError(f'invalid {src.ndim=}')
    if (
        dimx * dimy * dimz > <size_t> 4294967295U
        or dimx > 2147483647
        or dimy > 2147483647
        or dimz > 2147483647
    ):
        raise ValueError(f'invalid {dimx=}, {dimy=}, or {dimz=}')

    if mode in {1, 2, 3}:
        mode_ = mode
    elif mode == 'bpp':
        mode_ = 1
    elif mode == 'psnr':
        mode_ = 2
    elif mode == 'pwe':
        mode_ = 3
    else:
        raise ValueError(f'invalid SPERR {mode=!r}')

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
            dstsize = <ssize_t> dst_len
            out = _create_output(outtype, dstsize, <const char *> dst_ptr)
        else:
            dst = out
            dstsize = dst.nbytes
            if <size_t> dstsize < dst_len:
                raise ValueError(
                    f'output buffer too small {dstsize} < {dst.nbytes}'
                )
            memcpy(<void*> &dst[0], dst_ptr, dst_len)
            del dst
    finally:
        free(dst_ptr)

    return _return_output(out, dstsize, dst_len, outgiven)


def sperr_decode(
    data,
    shape=None,
    dtype=None,
    header=True,
    numthreads=None,
    out=None
):
    """Return decoded SPERR data.

    Either a header (always present for 3D), a ndarray output, or shape & dtype
    are required.

    `sperr_decomp_2d` may segfault if header argument is not correct.
    `sperr_decomp_2d` reads beyond data buffer in some cases and may segfault,
    especially if heap protection is enabled.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        size_t src_len = <size_t> src.size
        size_t dstlen
        void* dst_ptr = NULL
        size_t ndim = 0
        size_t dimx, dimy, dimz
        size_t nthreads = <size_t> _default_threads(numthreads)
        int out_inc_header = bool(header)
        int is_float, ret

    if data is out:
        raise ValueError('cannot decode in-place')

    is_ndarray = isinstance(out, numpy.ndarray)
    if is_ndarray or (shape is not None and dtype is not None):
        if is_ndarray:
            dtype = out.dtype
            shape = out.shape
        else:
            dtype = numpy.dtype(dtype)
            shape = tuple(shape)

        if dtype.char == 'd':
            is_float = 0
        elif dtype.char == 'f':
            is_float = 1
        else:
            raise ValueError(f'invalid {dtype=}')

        shape = _squeeze_shape(shape, 2)
        ndim = len(shape)
        if ndim == 2:
            dimz = 1
            dimy = <size_t> shape[0]
            dimx = <size_t> shape[1]
        elif ndim == 3:
            # header always present
            out_inc_header = 1
        else:
            raise ValueError(f'invalid {shape=}')

    if out_inc_header:
        sperr_parse_header(
            <const void*> &src[0],
            &dimx,
            &dimy,
            &dimz,
            &is_float
        )
        if (
            (is_float != 0 and is_float != 1)
            or dimx == 0
            or dimy == 0
            or dimz == 0
        ):
            raise ValueError("'sperr_parse_header' returned invalid values")
        if ndim == 0:
            ndim = 2 if dimz == 1 else 3
        if ndim == 2:
            shape = (int(dimy), int(dimx))
        else:
            shape = (int(dimz), int(dimy), int(dimx))
        dtype = numpy.dtype('f4' if is_float else 'f8')

    elif shape is None or dtype is None:
        raise ValueError('shape and dtype required if header=False')

    if dimx > 2147483647 or dimy > 2147483647 or dimz > 2147483647:
        raise ValueError(f'invalid {dimx=}, {dimy=}, or {dimz=}')

    out = _create_array(out, shape, dtype)
    dst = out
    dstlen = dst.nbytes

    with nogil:
        if ndim == 2:
            ret = sperr_decomp_2d(
                <const void*> &src[10 * out_inc_header],
                src_len - 10 * out_inc_header,
                is_float,
                dimx,
                dimy,
                &dst_ptr
            )
            if ret != 0 or dst_ptr == NULL:
                raise SperrError('sperr_decomp_2d', ret)
        else:
            ret = sperr_decomp_3d(
                <const void*> &src[0],
                src_len,
                is_float,
                nthreads,
                &dimx,
                &dimy,
                &dimz,
                &dst_ptr
            )
            if ret != 0 or dst_ptr == NULL:
                raise SperrError('sperr_decomp_3d', ret)
        try:
            if dstlen != <ssize_t> (
                dimx * dimy * dimz * (4 if is_float else 8)
            ):
                raise ValueError(f'invalid output size {out.nbytes=}')
            memcpy(<void*> &dst.data[0], dst_ptr, dstlen)
        finally:
            free(dst_ptr)

    return out
