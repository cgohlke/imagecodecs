# imagecodecs/_lzfse.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2022-2026, Christoph Gohlke
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

"""LZFSE (Lempel-Ziv Finite State Entropy) codec for the imagecodecs package.

"""

include '_shared.pxi'

from lzfse cimport *


class LZFSE:
    """LZFSE codec constants."""

    available = True


class LzfseError(RuntimeError):
    """LZFSE codec exceptions."""


def lzfse_version():
    """Return LZFSE library version string."""
    return 'lzfse 1.0'


def lzfse_check(const uint8_t[::1] data not None, /):
    """Return whether data is LZFSE encoded or None if unknown."""
    return _startswith(data, b'bvx')


def lzfse_encode(
    data,
    /,
    *,
    out=None,
):
    """Return LZFSE encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t dst_size

    if srcsize > INT32_MAX:
        # arbitrary 2GB limit
        raise ValueError('input too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # https://github.com/lzfse/lzfse/issues/4
            dstsize = srcsize + 12 + srcsize // 4096
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        dst_size = lzfse_encode_buffer(
            <uint8_t*> dst._data,
            <size_t> dstsize,
            <const uint8_t*> src._data,
            <size_t> srcsize,
            NULL
        )
        if dst_size == 0:
            raise LzfseError('lzfse_encode_buffer failed')

    del dst
    return _return_output(out, dstsize, dst_size, outgiven)


def lzfse_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded LZFSE data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, rawsize
        size_t dst_size

    if srcsize < 4 or not _startswith(src, b'bvx'):
        raise ValueError('invalid LZFSE stream')

    if data is out:
        raise ValueError('cannot transform in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    # exact decoded size, or -1 if block headers cannot be parsed
    rawsize = _lzfse_decoded_size(<const uint8_t*> src._data, <size_t> srcsize)

    if out is None:
        if dstsize < 0:
            if rawsize > INT32_MAX:
                # arbitrary 2 GB limit
                raise ValueError(f'LZFSE decoded size {rawsize} too large')
            if rawsize < 0:
                raise LzfseError('cannot determine decoded size; specify out')
            dstsize = rawsize
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]

    if rawsize == 0 or dstsize == 0:
        # lzfse_decode_buffer returns 0 for empty output and errors
        del dst
        return _return_output(out, dstsize, 0, outgiven)

    with nogil:
        dst_size = lzfse_decode_buffer(
            <uint8_t*> dst._data,
            <size_t> dstsize,
            <const uint8_t*> src._data,
            <size_t> srcsize,
            NULL
        )
        if dst_size == 0:
            raise LzfseError('lzfse_decode_buffer failed')

    del dst
    return _return_output(out, dstsize, dst_size, outgiven)


cdef ssize_t _lzfse_decoded_size(
    const uint8_t* src, size_t srcsize
) noexcept nogil:
    """Return sum of n_raw_bytes of all blocks, or -1 if unknown."""
    cdef:
        size_t pos = 0
        size_t size = 0
        size_t blocksize
        uint64_t v0, v1, v2
        uint8_t kind

    while srcsize - pos >= 4:
        if src[pos] != 0x62 or src[pos + 1] != 0x76 or src[pos + 2] != 0x78:
            return -1  # not 'bvx'
        kind = src[pos + 3]
        if kind == 0x24:  # '$' end of stream
            return <ssize_t> size
        if srcsize - pos < 8:
            return -1
        if kind == 0x2D:  # '-' uncompressed
            blocksize = 8 + <size_t> _le32(src + pos + 4)
        elif kind == 0x6E:  # 'n' LZVN
            if srcsize - pos < 12:
                return -1
            blocksize = 12 + <size_t> _le32(src + pos + 8)
        elif kind == 0x32:  # '2' LZFSE v2
            if srcsize - pos < 32:
                return -1
            v0 = _le64(src + pos + 8)
            v1 = _le64(src + pos + 16)
            v2 = _le64(src + pos + 24)
            # header_size + n_literal_payload_bytes + n_lmd_payload_bytes
            blocksize = <size_t> (
                (v2 & 0xFFFFFFFFU)
                + ((v0 >> 20) & 0xFFFFFU)
                + ((v1 >> 40) & 0xFFFFFU)
            )
        else:
            # LZFSE v1 headers are not parsed
            return -1
        if blocksize > srcsize - pos:
            return -1
        size += _le32(src + pos + 4)
        pos += blocksize
    return -1


cdef inline uint32_t _le32(const uint8_t* p) noexcept nogil:
    return (
        <uint32_t> p[0]
        | (<uint32_t> p[1] << 8)
        | (<uint32_t> p[2] << 16)
        | (<uint32_t> p[3] << 24)
    )


cdef inline uint64_t _le64(const uint8_t* p) noexcept nogil:
    return <uint64_t> _le32(p) | (<uint64_t> _le32(p + 4) << 32)
