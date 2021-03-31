# imagecodecs/_zstd.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2021, Christoph Gohlke
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

"""Zstd (ZStandard) codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from zstd cimport *


class ZSTD:
    """Zstd Constants."""


class ZstdError(RuntimeError):
    """Zstd Exceptions."""

    def __init__(self, func, msg='', err=0):
        cdef:
            const char* errmsg

        if msg:
            mg = f'{func} returned {msg!r}'
        else:
            errmsg = ZSTD_getErrorName(err)
            msg = f'{func} returned {errmsg.decode()!r}'
        super().__init__(msg)


def zstd_version():
    """Return Zstd library version string."""
    return 'zstd {}.{}.{}'.format(
        ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_RELEASE
    )


def zstd_check(data):
    """Return True if data likely contains Zstd data."""


def zstd_encode(data, level=None, out=None):
    """Compress Zstd.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t srcsize = src.size
        ssize_t dstsize
        size_t ret
        int compresslevel = _default_value(level, 5, 1, 22)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = <ssize_t> ZSTD_compressBound(srcsize)
            if dstsize < 0:
                raise ZstdError('ZSTD_compressBound', f'{dstsize}')
        if dstsize < 64:
            dstsize = 64
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = ZSTD_compress(
            <void*> &dst[0],
            <size_t> dstsize,
            <void*> &src[0],
            srcsize,
            compresslevel
        )
    if ZSTD_isError(ret):
        raise ZstdError('ZSTD_compress', err=ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def zstd_decode(data, out=None):
    """Decompress Zstd.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t srcsize = <size_t> src.size
        ssize_t dstsize
        size_t ret
        uint64_t cntsize

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            cntsize = ZSTD_getFrameContentSize(<void*> &src[0], srcsize)
            if (
                cntsize == ZSTD_CONTENTSIZE_UNKNOWN
                or cntsize == ZSTD_CONTENTSIZE_ERROR
            ):
                # 1 MB; arbitrary
                cntsize = max(<uint64_t> 1048576, <uint64_t> (srcsize * 2))
            # TODO: use stream interface
            # if cntsize == ZSTD_CONTENTSIZE_UNKNOWN:
            #     raise ZstdError(
            #         'ZSTD_getFrameContentSize', 'ZSTD_CONTENTSIZE_UNKNOWN'
            #     )
            # if cntsize == ZSTD_CONTENTSIZE_ERROR:
            #     raise ZstdError(
            #         'ZSTD_getFrameContentSize', 'ZSTD_CONTENTSIZE_ERROR'
            # )
            dstsize = cntsize
            if dstsize < 0:
                raise ZstdError('ZSTD_getFrameContentSize', f'{dstsize}')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <size_t> dst.size

    with nogil:
        ret = ZSTD_decompress(
            <void*> &dst[0],
            <size_t> dstsize,
            <void*> &src[0],
            srcsize
        )
    if ZSTD_isError(ret):
        raise ZstdError('ZSTD_decompress', err=ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)
