# imagecodecs/_lzf.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2019-2026, Christoph Gohlke
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

"""LZF (Lempel-Ziv Fast) codec for the imagecodecs package."""

include '_shared.pxi'

from liblzf cimport *


cdef extern from 'errno.h':
    int errno
    int E2BIG
    int EINVAL


class LZF:
    """LZF codec constants."""

    available = True


class LzfError(RuntimeError):
    """LZF codec exceptions."""


def lzf_version():
    """Return LibLZF library version string."""
    return f'liblzf {LZF_VERSION >> 8}.{LZF_VERSION & 255}'


def lzf_check(const uint8_t[::1] data not None, /):
    """Return whether data is LZF encoded or None if unknown."""
    return None if data.shape[0] > 0 else False


def lzf_encode(
    data,
    /,
    *,
    header=False,
    out=None,
):
    """Return LZF encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        unsigned int ret = 0
        uint8_t* pdst
        ssize_t offset = 4 if header else 0

    if srcsize > INT32_MAX:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # dstsize = ((srcsize * 33) >> 5 ) + 1 + offset
            dstsize = srcsize + srcsize // 20 + 32
        else:
            # liblzf bounds checks need up to 4 spare bytes
            dstsize += 4
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    if dst.shape[0] < offset:
        raise LzfError('output too small')
    if dst.shape[0] > INT32_MAX:
        raise ValueError('output too large')
    dstsize = dst.shape[0] - offset

    if srcsize > 0:
        with nogil:
            ret = lzf_compress(
                <void*> src._data,
                <unsigned int> srcsize,
                <void*> (dst._data + offset),
                <unsigned int> dstsize
            )
        if ret == 0:
            raise LzfError('lzf_compress returned 0 (output too small)')

    if header:
        pdst = <uint8_t*> dst._data
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize + offset, ret + offset, outgiven)


def lzf_decode(
    data,
    /,
    *,
    header=False,
    out=None,
):
    """Return decoded LZF data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, rawsize
        ssize_t hdrsize = -1
        unsigned int ret = 0
        ssize_t offset = 4 if header else 0

    global errno  # keep this

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize > INT32_MAX:
        raise ValueError('data too large')

    if header:
        if srcsize < offset:
            raise ValueError('invalid data size')
        hdrsize = <ssize_t> (
            (<uint32_t> src[0])
            | (<uint32_t> src[1] << 8)
            | (<uint32_t> src[2] << 16)
            | (<uint32_t> src[3] << 24)
        )

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            with nogil:
                rawsize = _lzf_decoded_size(
                    <const uint8_t*> (src._data + offset),
                    <size_t> (srcsize - offset)
                )
            if rawsize < 0:
                raise LzfError('invalid or truncated LZF data')
            if header and rawsize != hdrsize:
                raise LzfError(
                    f'LZF header size {hdrsize} does not match data size '
                    f'{rawsize}'
                )
            dstsize = rawsize
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]

    if dst.shape[0] > INT32_MAX:
        raise ValueError('output too large')

    if srcsize > offset:
        with nogil:
            errno = 0
            ret = lzf_decompress(
                <void*> (src._data + offset),
                <unsigned int> (srcsize - offset),
                <void*> dst._data,
                <unsigned int> dstsize
            )
            if ret == 0:
                if errno == E2BIG:
                    raise LzfError(
                        'lzf_decompress detected the output buffer is '
                        'not large enough to hold the decompressed data'
                    )
                if errno == EINVAL:
                    raise LzfError(
                        'lzf_decompress detected an error in the '
                        'compressed data'
                    )
                raise LzfError('lzf_decompress failed')

    if header and <ssize_t> ret != hdrsize:
        raise LzfError(
            f'LZF header size {hdrsize} does not match decoded size {ret}'
        )

    del dst
    return _return_output(out, dstsize, ret, outgiven)


cdef ssize_t _lzf_decoded_size(
    const uint8_t* src, size_t srcsize
) noexcept nogil:
    """Return decoded size of LZF data, or -1 if invalid or truncated."""
    cdef:
        size_t ip = 0
        size_t op = 0
        size_t ctrl, length, distance

    while ip < srcsize:
        ctrl = src[ip]
        ip += 1
        if ctrl < 32:
            # literal run
            ctrl += 1
            if ctrl > srcsize - ip:
                return -1
            ip += ctrl
            op += ctrl
        else:
            # back reference
            length = ctrl >> 5
            if length == 7:
                if ip >= srcsize:
                    return -1
                length += src[ip]
                ip += 1
            if ip >= srcsize:
                return -1
            distance = ((ctrl & 0x1F) << 8) + src[ip] + 1
            ip += 1
            if distance > op:
                return -1
            op += length + 2
    return <ssize_t> op
