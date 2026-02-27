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


def meshopt_check(const uint8_t[::1] data, /):
    """Return whether data is MESHOPT encoded or None if unknown."""


def meshopt_encode(
    data,
    /,
    level=None,
    *,
    items=None,
    out=None,
):
    """Return MESHOPT encoded vertex array."""
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        size_t vertex_count, vertex_size, ret
        int clevel = _default_value(level, 3, 0, 3)

    if data is out:
        raise ValueError('cannot encode in-place')

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
    dstsize = dst.nbytes

    with nogil:
        ret = meshopt_encodeVertexBufferLevel(
            <unsigned char*> &dst[0],
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
    """Return decoded MESHOPT vertex array."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        size_t vertex_count, vertex_size
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if out is None:
        if shape is None or dtype is None:
            raise TypeError(
                "'meshopt_decode' is missing required arguments: "
                "'shape' and 'dtype'"
            )
        out = _create_array(out, shape, dtype)
    dst = out

    vertex_count, vertex_size = _meshopt_vertex_array(
        out, 0 if items is None else items
    )

    with nogil:
        ret = meshopt_decodeVertexBuffer(
            <void*> dst.data,
            vertex_count,
            vertex_size,
            <const unsigned char*> &src[0],
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

    if arr.ndim == 0:
        raise ValueError(f'invalid vertex array dimension {arr.ndim}')

    if items <= 0:
        items = arr.shape[arr.ndim - 1]
        if arr.ndim == 1 or arr.itemsize * items > 256:
            items = 1
    elif arr.shape[arr.ndim - 1] % items != 0:
        raise ValueError(
            f'last dimension of vertex array {arr.shape[arr.ndim - 1]} '
            f'is not a multiple of {items=}'
        )

    vertex_count = <size_t> (arr.size // items)
    vertex_size = <size_t> (arr.itemsize * items)

    if vertex_size % 4 != 0 or vertex_size > 256:
        raise ValueError(
            f'{vertex_size=} must be a multiple of 4 and not greater than 256'
        )

    if arr.nbytes != vertex_count * vertex_size:
        raise ValueError(
            f'vertex array size {arr.nbytes} != {vertex_count * vertex_size}'
        )

    return vertex_count, vertex_size
