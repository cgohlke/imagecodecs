# imagecodecs/_rgbe.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2022-2025, Christoph Gohlke
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

"""RGBE codec for the imagecodecs package."""

include '_shared.pxi'

from rgbe cimport *
from imcd cimport imcd_strsearch


class RGBE:
    """RGBE codec constants."""

    available = True


class RgbeError(RuntimeError):
    """RGBE codec exceptions."""

    def __init__(self, func, err=None):
        msg = {
            RGBE_RETURN_SUCCESS: 'SUCCESS',
            RGBE_RETURN_FAILURE: 'FAILURE',
            RGBE_READ_ERROR: 'READ_ERROR',
            RGBE_WRITE_ERROR: 'WRITE_ERROR',
            RGBE_FORMAT_ERROR: 'FORMAT_ERROR',
            RGBE_MEMORY_ERROR: 'MEMORY_ERROR',
        }.get(err, repr(err))
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def rgbe_version():
    """Return RGBE library version string."""
    return 'rgbe ' + RGBE_VERSION.decode()


def rgbe_check(const uint8_t[::1] data):
    """Return whether data is RGBE encoded image."""
    cdef:
        bytes sig = bytes(data[:2])

    return sig == b'#?'


def rgbe_encode(
    data, header=None, rle=None, out=None
):
    """Return RGBE encoded image."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        numpy.ndarray arr
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.nbytes
        ssize_t size = 0
        int dorle = -1 if rle is None else (1 if rle else 0)
        int dohdr = -1 if header is None else (1 if header else 0)
        int width = 1
        int height = 1
        int ret = RGBE_RETURN_SUCCESS
        rgbe_stream_t *stream
        # rgbe_header_info info

    if not (
        srcsize <= 2147483647
        and src.dtype.char == 'f'
        and src.ndim > 0
        and src.shape[src.ndim - 1] == 3
    ):
        raise ValueError('invalid data shape, strides, or dtype')

    if (
        out is not None
        and not header
        and not rle
        and isinstance(out, numpy.ndarray)
        and out.dtype.char == 'B'
        and out.nbytes // 4 == srcsize // 12
    ):
        # write to compatible numpy array
        arr = out
        size = arr.nbytes
        with nogil:
            stream = rgbe_stream_new(size, arr.data)
            if stream == NULL:
                raise MemoryError('rgbe_stream_new failed')
            ret = RGBE_WritePixels(
                stream, <float *> src.data, <int> (size // 4)
            )
            rgbe_stream_del(stream)
        if ret != RGBE_RETURN_SUCCESS:
            raise RgbeError('RGBE_WritePixels', ret)
        del arr
        return out

    if dohdr == -1:
        dohdr = 1
        dorle = 1
    elif dohdr:
        dorle = 1
    elif dorle == -1:
        dorle = 0

    if src.ndim > 1:
        width = <int> src.shape[src.ndim - 2]
        height = <int> (src.size // (3 * width))

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = width * height * 4
            if dohdr:
                dstsize += 256
        elif not dorle and dstsize < width * height * 4:
            raise ValueError('output too small')
        out = _create_output(outtype, dstsize)
    dst = out
    dstsize = dst.nbytes

    stream = rgbe_stream_new(dstsize, <char *> &dst[0])
    if stream == NULL:
        raise MemoryError('rgbe_stream_new failed')

    try:
        with nogil:
            if dohdr:
                ret = RGBE_WriteHeader(stream, width, height, NULL)
                if ret != RGBE_RETURN_SUCCESS:
                    raise RgbeError('RGBE_WriteHeader', ret)
            if dorle:
                ret = RGBE_WritePixels_RLE(
                    stream, <float *> src.data, width, height
                )
                if ret != RGBE_RETURN_SUCCESS:
                    raise RgbeError('RGBE_WritePixels_RLE', ret)
            else:
                ret = RGBE_WritePixels(
                    stream, <float *> src.data, width * height
                )
                if ret != RGBE_RETURN_SUCCESS:
                    raise RgbeError('RGBE_WritePixels', ret)
            size = stream.pos
    finally:
        rgbe_stream_del(stream)

    del dst
    return _return_output(out, dstsize, size, outgiven)


def rgbe_decode(
    data, header=None, rle=None, out=None
):
    """Return decoded RGBE image."""
    cdef:
        numpy.ndarray dst, arr
        const uint8_t[::1] src
        ssize_t srcsize
        ssize_t size = 0
        int dorle = -1 if rle is None else (1 if rle else 0)
        int width, height
        int ret = RGBE_RETURN_SUCCESS
        rgbe_stream_t *stream
        # rgbe_header_info info

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        # bytes
        src = data
        srcsize = src.nbytes
    except ValueError as exc:
        # decode uint8 array of shape (..., 4)
        # no header, no rle
        try:
            arr = numpy.ascontiguousarray(data)
        except Exception:
            raise exc
        if not (
            arr.ndim > 0
            and arr.shape[arr.ndim - 1] == 4
            and arr.dtype.char == 'B'
            and arr.nbytes <= 2147483647
        ):
            raise ValueError('data must be a uint8 RGBE image array')
        out = _create_array(
            out, data.shape[:arr.ndim - 1] + (3, ), numpy.float32
        )
        size = out.size // 3
        dst = out

        stream = rgbe_stream_new(arr.nbytes, <char*> arr.data)
        if stream == NULL:
            raise MemoryError('rgbe_stream_new failed')
        with nogil:
            ret = RGBE_ReadPixels(stream, <float*> dst.data, <int> size)
            rgbe_stream_del(stream)
            if ret != RGBE_RETURN_SUCCESS:
                raise RgbeError('RGBE_ReadPixels', ret)

        del dst
        return out

    if srcsize > 2147483647:
        raise ValueError('input too large')

    stream = rgbe_stream_new(srcsize, <char*> &src[0])
    if stream == NULL:
        raise MemoryError('rgbe_stream_new failed')

    try:
        if header is None:
            if src[0] == 35 and src[1] == 63:
                header = True
            elif imcd_strsearch(
                <const char *> &src[0],
                <ssize_t> min(srcsize, 8192),
                "FORMAT=32-bit_rle_",
                18
            ) >= 0:
                header = True
        if header:
            ret = RGBE_ReadHeader(
                stream,
                &width,
                &height,
                NULL  # &info
            )
            if ret != RGBE_RETURN_SUCCESS:
                raise RgbeError('RGBE_ReadHeader', ret)
            dorle = 1
            out = _create_array(
                out, (int(height), int(width), 3), numpy.float32
            )
        elif isinstance(out, numpy.ndarray):
            shape = _squeeze_shape(out.shape, 3)
            if not (
                len(shape) == 3
                and shape[2] == 3
                and out.nbytes <= 2147483647
                and out.dtype.char == 'f'
                and out.flags['C_CONTIGUOUS']
            ):
                raise ValueError(
                    f'no rgbe header found {out.shape=} {out.dtype=}'
                )
            height = <int> shape[0]
            width = <int> shape[1]
        else:
            raise ValueError('no rgbe header found')

        dst = out

        with nogil:
            size = <ssize_t> width * <ssize_t> height * 4
            if dorle == -1:
                if <ssize_t> (stream.size - stream.pos) == size:
                    dorle = 0
                else:
                    dorle = 1
            elif dorle == 0:
                if <ssize_t> (stream.size - stream.pos) != size:
                    raise ValueError('data size does not match output size')

            if dorle != 0:
                ret = RGBE_ReadPixels_RLE(
                    stream, <float*> dst.data, width, height
                )
                if ret != RGBE_RETURN_SUCCESS:
                    raise RgbeError('RGBE_ReadPixels_RLE', ret)
            else:
                ret = RGBE_ReadPixels(
                    stream, <float*> dst.data, width * height
                )
                if ret != RGBE_RETURN_SUCCESS:
                    raise RgbeError('RGBE_ReadPixels', ret)

            if stream.pos < stream.size:
                raise ValueError('not all input decoded')

    finally:
        rgbe_stream_del(stream)

    del dst
    return out

    # TODO: replace header parsing with Python regex
