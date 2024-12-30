# imagecodecs/_lzfse.pyx
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

"""LZFSE codec for the imagecodecs package."""

include '_shared.pxi'

from lzfse cimport *
from imcd cimport imcd_memsearch

from libc.stdint cimport uint32_t


class LZFSE:
    """LZFSE codec constants."""

    available = True


class LzfseError(RuntimeError):
    """LZFSE codec exceptions."""


def lzfse_version():
    """Return LZFSE library version string."""
    return 'lzfse 1.0'


def lzfse_check(const uint8_t[::1] data):
    """Return whether data is LZFSE encoded."""
    cdef:
        bytes sig = bytes(data[:3])

    return sig == b'bvx'


def lzfse_encode(data, out=None):
    """Return LZFSE encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t dst_size

    if data is out:
        raise ValueError('cannot encode in-place')

    if srcsize > 2147483647:
        # arbitrary 2GB limit
        raise ValueError('input too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # https://github.com/lzfse/lzfse/issues/4
            dstsize = srcsize + 12 + srcsize // 4096
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        dst_size = lzfse_encode_buffer(
            <uint8_t *> &dst[0],
            <size_t> dstsize,
            <const uint8_t *> &src[0],
            <size_t> srcsize,
            NULL
        )
        if dst_size == 0:
            raise LzfseError('lzfse_encode_buffer failed')

    del dst
    return _return_output(out, dstsize, dst_size, outgiven)


def lzfse_decode(data, out=None):
    """Return decoded LZFSE data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t dst_size

    if srcsize < 12 or bytes(src[:3]) != b'bvx':
        raise LzfseError('invalid LZFSE stream')

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = lzfse_decoded_size(<char*> &src[0], srcsize)
            if dstsize < 0 or dstsize > 2147483647:
                # arbitrary 2 GB limit
                raise LzfseError(f'lzfse_decoded_size {dstsize} out of bound')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dstsize == 0 or srcsize == 12:
        return _return_output(out, dstsize, 0, outgiven)

    with nogil:
        dst_size = lzfse_decode_buffer(
            <uint8_t *> &dst[0],
            <size_t> dstsize,
            <const uint8_t *> &src[0],
            <size_t> srcsize,
            NULL
        )
        if dst_size == 0:
            raise LzfseError('lzfse_decode_buffer failed')

    del dst
    return _return_output(out, dstsize, dst_size, outgiven)


cdef ssize_t lzfse_decoded_size(char* src, ssize_t srcsize) nogil:
    """Return sum of all block headers n_raw_bytes fields."""
    cdef:
        ssize_t size = 0
        ssize_t index = 0
        char c

    while 1:
        # search for block magic
        index = imcd_memsearch(src, srcsize, b'bvx', 3)
        if index < 0 or srcsize - index < 8:
            break
        src += index + 3
        c = src[0]
        if c == b'$':
            # end of stream
            break
        if c == b'n' or c == b'2' or c == b'1' or c == b'-':
            # next 4 bytes are number of decoded bytes in block
            size += (<uint32_t*> (src + 1))[0]
        src += 5
        srcsize -= index + 8
    return size
