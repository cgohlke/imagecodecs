# imagecodecs/_meshopt.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2025-2026, Christoph Gohlke
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

"""MESHOPT (Mesh Optimizer) codec for the imagecodecs package."""

include '_shared.pxi'

from meshoptimizer cimport *


class MESHOPT:
    """MESHOPT codec constants."""

    available = True


class MeshoptError(RuntimeError):
    """MESHOPT codec exceptions."""


def meshopt_version():
    """Return meshoptimizer library version string."""
    return 'meshoptimizer {}.{}.{}'.format(
        MESHOPTIMIZER_VERSION // 1000,
        MESHOPTIMIZER_VERSION % 1000 // 10,
        MESHOPTIMIZER_VERSION % 10
    )


def meshopt_check(const uint8_t[::1] data not None, /):
    """Return whether data is MESHOPT encoded or None if unknown."""
    if data.shape[0] < 2 or (data[0] & 0xF0) != 0xA0:
        return False
    return None


def meshopt_encode(
    data,
    /,
    level=None,
    *,
    items=None,
    out=None,
):
    """Return MESHOPT encoded vertex array.

    The array is encoded as vertices of `items` consecutive array items.
    By default, `items` is derived from the last array dimension.
    The byte size of a vertex must be a multiple of 4 and at most 256.
    Compression `level` ranges from 0 (fastest) to 3 (smallest),
    default 2.
    Data are encoded in vertex codec format version 1, which requires
    meshoptimizer >= 0.23 to decode.

    """
    cdef:
        numpy.ndarray src = _array_input(
            data, out, dtype=None, contig=True, empty=False, native=False
        )
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        size_t vertex_count, vertex_size, ret
        int clevel = _default_value(level, 2, 0, 3)

    vertex_count, vertex_size = _meshopt_vertex_array(
        src, 0 if items is None else items
    )

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = meshopt_encodeVertexBufferBound(
                vertex_count, vertex_size
            )
            # if dstsize == 0:
            #     raise MeshoptError(
            #         f'meshopt_encodeVertexBufferBound returned {dstsize}'
            #     )
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        ret = meshopt_encodeVertexBufferLevel(
            <unsigned char*> dst._data,
            <size_t> dstsize,
            <const void*> src.data,
            vertex_count,
            vertex_size,
            clevel,
            1  # version
        )

    if ret == 0:
        raise MeshoptError('meshopt_encodeVertexBufferLevel returned 0')

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def meshopt_decode(
    data,
    /,
    shape=None,
    dtype=None,
    *,
    items=None,
    out=None,
):
    """Return decoded MESHOPT vertex array.

    The `shape`, `dtype`, and `items` arguments must match the encoded
    array. If `shape` and `dtype` are not specified, they are taken from
    the `out` array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = _readable_input(data, out)
        ssize_t srcsize = src.shape[0]
        size_t vertex_count, vertex_size
        int ret

    if shape is not None and dtype is not None:
        out = _create_array(out, tuple(shape), dtype)
    elif out is None:
        raise TypeError(
            "'meshopt_decode' is missing required arguments: "
            "'shape' and 'dtype'"
        )
    elif isinstance(out, numpy.ndarray):
        out = _create_array(out, out.shape, out.dtype)
    else:
        raise TypeError(
            "'meshopt_decode' requires 'shape' and 'dtype' "
            'if out is not a numpy array'
        )
    dst = out

    vertex_count, vertex_size = _meshopt_vertex_array(
        out, 0 if items is None else items
    )

    with nogil:
        ret = meshopt_decodeVertexBuffer(
            <void*> dst.data,
            vertex_count,
            vertex_size,
            <const unsigned char*> src._data,
            <size_t> srcsize
        )

    if ret != 0:
        raise MeshoptError(f'meshopt_decodeVertexBuffer returned {ret}')

    return out


cdef (size_t, size_t) _meshopt_vertex_array(numpy.ndarray arr, ssize_t items):
    """Return vertex_count and vertex_size from array.

    Items defines how many array items comprise a vertex.
    By default, items is determined from the last array dimension.
    Byte-size of vertices must be a multiple of 4 and not greater than 256.

    """
    cdef:
        size_t vertex_count, vertex_size
        ssize_t lastdim, itemsize, i

    if arr.ndim == 0:
        raise ValueError(f'invalid vertex array dimension {arr.ndim}')

    lastdim = arr.shape[arr.ndim - 1]
    itemsize = arr.itemsize

    if items <= 0:
        items = lastdim
        if arr.ndim == 1 or itemsize * items > 256:
            items = 1
        if (itemsize * items) % 4 != 0:
            # smallest group of items with valid vertex size
            for i in range(2, 256 // itemsize + 1):
                if lastdim % i == 0 and (itemsize * i) % 4 == 0:
                    items = i
                    break
    elif lastdim % items != 0:
        raise ValueError(
            f'last dimension of vertex array {lastdim} '
            f'is not a multiple of {items=}'
        )

    vertex_count = <size_t> (arr.size // items)
    vertex_size = <size_t> (itemsize * items)

    if vertex_size % 4 != 0 or vertex_size > 256:
        raise ValueError(
            f'{vertex_size=} must be a multiple of 4 and not greater than '
            '256; specify a valid number of items per vertex'
        )

    if arr.nbytes != vertex_count * vertex_size:
        raise ValueError(
            f'vertex array size {arr.nbytes} != {vertex_count * vertex_size}'
        )

    return vertex_count, vertex_size
