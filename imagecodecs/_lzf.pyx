# imagecodecs/_lzf.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""Lzf codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from liblzf cimport *


class LZF:
    """LZF Constants."""


class LzfError(RuntimeError):
    """LZF Exceptions."""


def lzf_version():
    """Return LibLZF library version string."""
    return f'liblzf {LZF_VERSION >> 8}.{LZF_VERSION & 255}'


def lzf_check(data):
    """Return True if data likely contains LZF data."""


def lzf_encode(data, level=None, header=False, out=None):
    """Compress LZF.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned int ret
        uint8_t* pdst
        ssize_t offset = 4 if header else 0

    if data is out:
        raise ValueError('cannot encode in-place')

    if srcsize >= 2 ** 31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # dstsize = ((srcsize * 33) >> 5 ) + 1 + offset
            dstsize = srcsize + srcsize // 20 + 32
        else:
            dstsize += 1  # bug in liblzf ?
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size - offset

    if dst.size >= 2 ** 31:
        raise ValueError('output too large')

    with nogil:
        ret = lzf_compress(
            <void*> &src[0],
            <unsigned int> srcsize,
            <void*> &dst[offset],
            <unsigned int> dstsize
        )
    if ret == 0:
        raise LzfError('lzf_compress returned 0')

    if header:
        pdst = <uint8_t*> &dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def lzf_decode(data, header=False, out=None):
    """Decompress LZF.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        unsigned int ret
        ssize_t offset = 4 if header else 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize >= 2 ** 31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int> dst.size

    if dst.size >= 2 ** 31:
        raise ValueError('output too large')

    with nogil:
        ret = lzf_decompress(
            <void*> &src[offset],
            <unsigned int> (srcsize - offset),
            <void*> &dst[0],
            <unsigned int> dstsize
        )
    if ret == 0:
        raise LzfError(f'lzf_decompress returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)
