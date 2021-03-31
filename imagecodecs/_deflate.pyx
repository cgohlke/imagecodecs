# imagecodecs/_deflate.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2020-2021, Christoph Gohlke
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

"""Deflate and GZIP codecs for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from libdeflate cimport *


class DEFLATE:
    """Libdeflate Constants."""


class DeflateError(RuntimeError):
    """Libdeflate Exceptions."""

    def __init__(self, func, err):
        msg = {
            LIBDEFLATE_SUCCESS: 'LIBDEFLATE_SUCCESS',
            LIBDEFLATE_BAD_DATA: 'LIBDEFLATE_BAD_DATA',
            LIBDEFLATE_SHORT_OUTPUT: 'LIBDEFLATE_SHORT_OUTPUT',
            LIBDEFLATE_INSUFFICIENT_SPACE: 'LIBDEFLATE_INSUFFICIENT_SPACE',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def deflate_version():
    """Return libdeflate library version string."""
    return 'libdeflate ' + LIBDEFLATE_VERSION_STRING.decode()


def deflate_check(data):
    """Return True if data likely contains Deflate/Zlib data."""


def deflate_encode(data, level=None, bint raw=False, out=None):
    """Compress Deflate/Zlib.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t srclen, dstlen
        libdeflate_compressor* compressor = NULL
        int compression_level = _default_value(level, 6, 0, 12)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    compressor = libdeflate_alloc_compressor(compression_level)
    if compressor == NULL:
        raise DeflateError('libdeflate_alloc_compressor', 'NULL')

    try:
        if out is None:
            if dstsize < 0:
                if raw:
                    dstlen = libdeflate_deflate_compress_bound(
                        compressor, <size_t> srcsize
                    )
                    if dstlen == 0:
                        raise DeflateError(
                            'libdeflate_deflate_compress_bound', '0'
                        )
                else:
                    dstlen = libdeflate_zlib_compress_bound(
                        compressor, <size_t> srcsize
                    )
                    if dstlen == 0:
                        raise DeflateError(
                            'libdeflate_zlib_compress_bound', '0'
                        )
                dstsize = <ssize_t> dstlen
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.size
        dstlen = <size_t> dstsize
        srclen = <size_t> srcsize

        with nogil:
            if raw:
                dstlen = libdeflate_deflate_compress(
                    compressor,
                    <const void*> &src[0],
                    srclen,
                    <void*> &dst[0],
                    dstlen
                )
                if dstlen == 0:
                    raise DeflateError('libdeflate_deflate_compress', '0')
            else:
                dstlen = libdeflate_zlib_compress(
                    compressor,
                    <const void*> &src[0],
                    srclen,
                    <void*> &dst[0],
                    dstlen
                )
                if dstlen == 0:
                    raise DeflateError('libdeflate_zlib_compress', '0')
    finally:
        if compressor != NULL:
            libdeflate_free_compressor(compressor)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def deflate_decode(data, bint raw=False, out=None):
    """Decompress Deflate/Zlib.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t srclen, dstlen
        size_t actual_out_nbytes_ret
        libdeflate_result ret
        libdeflate_decompressor* decompressor = NULL

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            if raw:
                raise NotImplementedError

            # use Python's zlib module if output size is unknwon
            import zlib

            return zlib.decompress(data)

        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <size_t> dstsize
    srclen = <size_t> srcsize

    try:
        with nogil:
            decompressor = libdeflate_alloc_decompressor()
            if decompressor == NULL:
                raise DeflateError('libdeflate_alloc_decompressor', 'NULL')

            if raw:
                ret = libdeflate_deflate_decompress(
                    decompressor,
                    <const void*> &src[0],
                    srclen,
                    <void*> &dst[0],
                    dstlen,
                    &actual_out_nbytes_ret
                )
                if ret != LIBDEFLATE_SUCCESS:
                    raise DeflateError('libdeflate_deflate_decompress', ret)
            else:
                ret = libdeflate_zlib_decompress(
                    decompressor,
                    <const void*> &src[0],
                    srclen,
                    <void*> &dst[0],
                    dstlen,
                    &actual_out_nbytes_ret
                )
                # if ret == LIBDEFLATE_INSUFFICIENT_SPACE:
                    # allow partial decompression for cases found in TIFF
                    # _log_warning(
                    #    'libdeflate_zlib_decompress '
                    #    'LIBDEFLATE_INSUFFICIENT_SPACE'
                    # )
                    # pass
                if ret != LIBDEFLATE_SUCCESS:
                    raise DeflateError('libdeflate_zlib_decompress', ret)

    finally:
        if decompressor != NULL:
            libdeflate_free_decompressor(decompressor)

    del dst
    return _return_output(out, dstsize, actual_out_nbytes_ret, outgiven)


# GZIP ########################################################################

GZIP = DEFLATE
GzipError = DeflateError
gzip_version = deflate_version


def gzip_check(data):
    """Return True if data likely contains GZIP data."""
    cdef:
        bytes sig = bytes(data[:2])

    return sig == b'\x1f\x8b'


def gzip_encode(data, level=None, out=None):
    """Compress GZIP.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t srclen, dstlen
        libdeflate_compressor* compressor = NULL
        int compression_level = _default_value(level, 6, 0, 12)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    compressor = libdeflate_alloc_compressor(compression_level)
    if compressor == NULL:
        raise GzipError('libdeflate_alloc_compressor', 'NULL')

    try:
        if out is None:
            if dstsize < 0:
                dstlen = libdeflate_gzip_compress_bound(
                    compressor, <size_t> srcsize
                )
                if dstlen == 0:
                    raise GzipError('libdeflate_gzip_compress_bound', '0')
                dstsize = <ssize_t> dstlen
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.size
        dstlen = <size_t> dstsize
        srclen = <size_t> srcsize

        with nogil:
            dstlen = libdeflate_gzip_compress(
                compressor,
                <const void*> &src[0],
                srclen,
                <void*> &dst[0],
                dstlen
            )
            if dstlen == 0:
                raise GzipError('libdeflate_gzip_compress', '0')

    finally:
        if compressor != NULL:
            libdeflate_free_compressor(compressor)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def gzip_decode(data, out=None):
    """Decompress GZIP.

    Supports only single-member streams < 2^32.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t srclen, dstlen
        size_t actual_out_nbytes_ret
        libdeflate_result ret
        libdeflate_decompressor* decompressor = NULL

    if data is out:
        raise ValueError('cannot decode in-place')

    if src[0] != 0x1F or src[1] != 0x8B:
        raise ValueError('invalid GZIP header')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # the last 4 bytes contain the size of the original, uncompressed
            # input data modulo 2^32 in little endian order
            dstsize = (
                (src[srcsize - 4] << 0) |
                (src[srcsize - 3] << 8) |
                (src[srcsize - 2] << 16) |
                (src[srcsize - 1] << 24)
            )
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <size_t> dstsize
    srclen = <size_t> srcsize

    try:
        with nogil:
            decompressor = libdeflate_alloc_decompressor()
            if decompressor == NULL:
                raise GzipError('libdeflate_alloc_decompressor', 'NULL')

            ret = libdeflate_gzip_decompress(
                decompressor,
                <const void*> &src[0],
                srclen,
                <void*> &dst[0],
                dstlen,
                &actual_out_nbytes_ret
            )
            if ret != LIBDEFLATE_SUCCESS:
                raise GzipError('libdeflate_gzip_decompress', ret)

    finally:
        if decompressor != NULL:
            libdeflate_free_decompressor(decompressor)

    del dst
    return _return_output(out, dstsize, actual_out_nbytes_ret, outgiven)


# CRC #########################################################################

def deflate_crc32(data):
    """Return cyclic redundancy checksum CRC-32 of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        uint32_t crc = 0

    with nogil:
        crc = libdeflate_crc32(crc, <const void*> &src[0], srcsize)
    return int(crc)


def deflate_adler32(data):
    """Return Adler-32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        uint32_t adler = 1

    with nogil:
        adler = libdeflate_adler32(adler, <const void*> &src[0], srcsize)
    return int(adler)
