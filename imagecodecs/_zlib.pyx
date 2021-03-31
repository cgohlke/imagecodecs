# imagecodecs/_zlib.pyx
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

"""Zlib codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from zlib cimport *


class ZLIB:
    """Zlib Constants."""


class ZlibError(RuntimeError):
    """Zlib Exceptions."""

    def __init__(self, func, err):
        msg = {
            Z_OK: 'Z_OK',
            Z_MEM_ERROR: 'Z_MEM_ERROR',
            Z_BUF_ERROR: 'Z_BUF_ERROR',
            Z_DATA_ERROR: 'Z_DATA_ERROR',
            Z_STREAM_ERROR: 'Z_STREAM_ERROR',
            Z_NEED_DICT: 'Z_NEED_DICT',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def zlib_version():
    """Return zlib library version string."""
    return 'zlib ' + zlibVersion().decode()


def zlib_check(data):
    """Return True if data likely contains Zlib data."""


def zlib_encode(data, level=None, out=None):
    """Compress Zlib.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret
        int compresslevel = _default_value(level, 6, 0, 9)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's zlib module
        import zlib

        return zlib.compress(data, compresslevel)
        # TODO: use zlib streaming API
        # return _zlib_compress(src, compresslevel, outtype)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <unsigned long> dstsize
    srclen = <unsigned long> srcsize

    with nogil:
        ret = compress2(
            <Bytef*> &dst[0],
            &dstlen,
            &src[0],
            srclen,
            compresslevel
        )
    if ret != Z_OK:
        raise ZlibError('compress2', ret)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def zlib_decode(data, out=None):
    """Decompress Zlib.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's zlib module
        import zlib

        return zlib.decompress(data)
        # TODO: use zlib streaming API
        # return _zlib_decompress(src, outtype)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <unsigned long> dstsize
    srclen = <unsigned long> srcsize

    with nogil:
        ret = uncompress2(
            <Bytef*> &dst[0],
            &dstlen,
            &src[0],
            &srclen
        )
    if ret != Z_OK:
        raise ZlibError('uncompress2', ret)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def zlib_crc32(data):
    """Return cyclic redundancy checksum CRC-32 of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        uInt srcsize = <uInt> src.size
        uLong crc = 0

    with nogil:
        crc = crc32(crc, NULL, 0)
        crc = crc32(crc, <Bytef*> &src[0], srcsize)
    return int(crc)
