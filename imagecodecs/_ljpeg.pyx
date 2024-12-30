# imagecodecs/_ljpeg.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021-2025, Christoph Gohlke
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

"""Lossless JPEG codec for the imagecodecs package."""

include '_shared.pxi'

from lj92 cimport *


class LJPEG:
    """LJPEG codec constants."""

    available = True


class LjpegError(RuntimeError):
    """LJPEG codec exceptions."""

    def __init__(self, func, err):
        msg = {
            LJ92_ERROR_NONE: 'LJ92_ERROR_NONE',
            LJ92_ERROR_CORRUPT: 'LJ92_ERROR_CORRUPT',
            LJ92_ERROR_NO_MEMORY: 'LJ92_ERROR_NO_MEMORY',
            LJ92_ERROR_BAD_HANDLE: 'LJ92_ERROR_BAD_HANDLE',
            LJ92_ERROR_TOO_WIDE: 'LJ92_ERROR_TOO_WIDE',
            LJ92_ERROR_ENCODER: 'LJ92_ERROR_ENCODER',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def ljpeg_version():
    """Return liblj92 library version string."""
    return 'liblj92 ' + LJ92_VERSION.decode()


def ljpeg_check(data):
    """Return whether data is Lossless JPEG encoded image."""


def ljpeg_encode(
    data,
    bitspersample=None,
    delinearize=None,
    out=None
):
    """Return Lossless JPEG encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        numpy.ndarray table
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint16_t* delinearize_table = NULL
        ssize_t dstsize
        uint16_t* srcptr = NULL
        uint8_t* encoded = NULL
        int encoded_length = 0
        int delinearize_length = 0
        int width = 0
        int height = 0
        int bitdepth = 0
        int samples = <int> src.shape[2] if src.ndim == 3 else 1
        int ret = LJ92_ERROR_NONE

    if not (
        src.dtype in {numpy.uint8, numpy.uint16}
        and src.ndim in {2, 3}
        and samples == 1  # {1, 3, 4}  RGB does not work correctly
        and src.shape[0] * src.shape[1] <= 2147483647
        # and numpy.PyArray_ISCONTIGUOUS(src)  # TODO: support strides
    ):
        raise ValueError('invalid data shape or dtype')

    if src.dtype == numpy.uint8:
        src = src.astype(numpy.uint16)

    srcptr = <uint16_t*> src.data
    height = <int> src.shape[0]
    width = <int> src.shape[1]

    if bitspersample is None or src.dtype == numpy.uint8:
        bitdepth = data.itemsize * 8
    elif 8 <= bitspersample <= 16:
        bitdepth = bitspersample
    else:
        raise ValueError('invalid bitspersample')

    if delinearize is not None:
        table = numpy.ascontiguousarray(delinearize)
        if table.ndim != 1 or table.dtype != numpy.uint16:
            raise ValueError('invalid delinearize shape or dtype')
        delinearize_table = <uint16_t*> table.data
        delinearize_length = <int> table.size

    with nogil:
        ret = lj92_encode(
            srcptr,
            width,
            height,
            bitdepth,
            samples,
            height * width * samples,
            0,
            delinearize_table,
            delinearize_length,
            &encoded,
            &encoded_length
        )
    if ret != LJ92_ERROR_NONE:
        if encoded != NULL:
            free(encoded)
        raise LjpegError('lj92_encode', ret)

    try:
        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            if dstsize < 0:
                dstsize = <ssize_t> encoded_length
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.nbytes

        memcpy(
            <void *> &dst[0], <void *> encoded, min(dstsize, encoded_length)
        )
    finally:
        free(encoded)

    del dst
    return _return_output(out, dstsize, <ssize_t> encoded_length, outgiven)


def ljpeg_decode(
    data,
    linearize=None,
    out=None
):
    """Return decoded Lossless JPEG image."""
    cdef:
        numpy.ndarray dst
        numpy.ndarray table
        const uint8_t[::1] src = _writable_input(data)
        uint16_t* target = NULL
        uint16_t* linearize_table = NULL
        ssize_t srcsize = src.size
        int linearize_length = 0
        int width = 0
        int height = 0
        int bitdepth = 0
        int components = 0
        int ret = LJ92_ERROR_NONE
        lj92 lj

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize > 2147483647:
        raise ValueError('data too large')

    if linearize is not None:
        table = numpy.ascontiguousarray(linearize)
        if table.ndim != 1 or table.dtype != numpy.uint16:
            raise ValueError('invalid linearize shape or dtype')
        linearize_table = <uint16_t*> table.data
        linearize_length = <int> table.size

    with nogil:
        ret = lj92_open(
            &lj,
            <uint8_t*> &src[0],
            <int> srcsize,
            &width,
            &height,
            &bitdepth,
            &components
        )
    if ret != LJ92_ERROR_NONE:
        raise LjpegError('lj92_open', ret)

    try:
        if components > 1:
            shape = int(height), int(width), int(components)
        else:
            shape = int(height), int(width)

        if bitdepth > 8:
            dst = _create_array(out, shape, numpy.uint16)
        else:
            dst = numpy.empty(shape, numpy.uint16)

        target = <uint16_t*> dst.data

        with nogil:
            ret = lj92_decode(
                lj,
                target,
                width * height * components,
                0,
                linearize_table,
                linearize_length,
            )
        if ret != LJ92_ERROR_NONE:
            raise LjpegError('lj92_decode', ret)
    finally:
        lj92_close(lj)

    if bitdepth > 8:
        return dst
    if out is None:
        return dst.astype(numpy.uint8)
    out = _create_array(out, shape, numpy.uint8)
    out[:] = dst
    return out
