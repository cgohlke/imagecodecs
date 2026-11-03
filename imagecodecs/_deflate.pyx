# imagecodecs/_deflate.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2020-2026, Christoph Gohlke
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

include '_shared.pxi'

from libdeflate cimport *

try:
    from ._zlib import ZlibError, _zlib_decode
except ImportError:
    _zlib_decode = None
    ZlibError = None


class DEFLATE:
    """DEFLATE codec constants."""

    available = True


class DeflateError(RuntimeError):
    """DEFLATE codec exceptions."""

    def __init__(self, func, err):
        if err is None:
            msg = 'NULL'
        elif isinstance(err, str):
            msg = repr(err)
        else:
            msg = {
                LIBDEFLATE_SUCCESS: 'LIBDEFLATE_SUCCESS',
                LIBDEFLATE_BAD_DATA: 'LIBDEFLATE_BAD_DATA',
                LIBDEFLATE_SHORT_OUTPUT: 'LIBDEFLATE_SHORT_OUTPUT',
                LIBDEFLATE_INSUFFICIENT_SPACE: 'LIBDEFLATE_INSUFFICIENT_SPACE',
            }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        if err == LIBDEFLATE_INSUFFICIENT_SPACE:
            msg += ' (output too small)'
        super().__init__(msg)


def deflate_version():
    """Return libdeflate library version string."""
    return 'libdeflate ' + LIBDEFLATE_VERSION_STRING.decode()


def deflate_check(const uint8_t[::1] data not None, /):
    """Return whether data is DEFLATE encoded or None if unknown."""
    cdef:
        unsigned int cmf, flg

    if data.shape[0] < 2:
        return False
    cmf = data[0]
    flg = data[1]
    # ZLIB header: CM=8, CINFO<=7, no preset dictionary, valid FCHECK
    if (
        (cmf & 0x0F) == 8
        and (cmf >> 4) <= 7
        and (flg & 0x20) == 0
        and ((cmf << 8) | flg) % 31 == 0
    ):
        return True
    return None


def deflate_encode(
    data,
    /,
    level=None,
    *,
    bint raw=False,
    out=None,
):
    """Return DEFLATE encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t srclen, dstlen
        libdeflate_compressor* compressor = NULL
        int compression_level = _default_value(level, 6, 0, 12)

    out, dstsize, outgiven, outtype = _parse_output(out)

    compressor = libdeflate_alloc_compressor(compression_level)
    if compressor == NULL:
        raise DeflateError('libdeflate_alloc_compressor', None)

    try:
        if out is None:
            if dstsize < 0:
                if raw:
                    dstlen = libdeflate_deflate_compress_bound(
                        compressor, <size_t> srcsize
                    )
                    if dstlen == 0:
                        raise DeflateError(
                            'libdeflate_deflate_compress_bound',
                            'input too large',
                        )
                else:
                    dstlen = libdeflate_zlib_compress_bound(
                        compressor, <size_t> srcsize
                    )
                    if dstlen == 0:
                        raise DeflateError(
                            'libdeflate_zlib_compress_bound', 'input too large'
                        )
                dstsize = <ssize_t> dstlen
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.shape[0]
        dstlen = <size_t> dstsize
        srclen = <size_t> srcsize

        with nogil:
            if raw:
                dstlen = libdeflate_deflate_compress(
                    compressor,
                    <const void*> src._data,
                    srclen,
                    <void*> dst._data,
                    dstlen
                )
                if dstlen == 0:
                    raise DeflateError(
                        'libdeflate_deflate_compress', 'output too small'
                    )
            else:
                dstlen = libdeflate_zlib_compress(
                    compressor,
                    <const void*> src._data,
                    srclen,
                    <void*> dst._data,
                    dstlen
                )
                if dstlen == 0:
                    raise DeflateError(
                        'libdeflate_zlib_compress', 'output too small'
                    )
    finally:
        if compressor != NULL:
            libdeflate_free_compressor(compressor)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def deflate_decode(
    data,
    /,
    *,
    bint raw=False,
    out=None,
):
    """Return decoded DEFLATE data.

    This implementation uses the zlib codec if the output size is unknown.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t srclen, dstlen
        size_t actual_out_nbytes_ret
        libdeflate_result ret
        libdeflate_decompressor* decompressor = NULL

    if data is out:
        raise ValueError('cannot transform in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            if _zlib_decode is None:
                raise NotImplementedError(
                    'DEFLATE decoding requires output size'
                )
            try:
                return _zlib_decode(src, outtype, -15 if raw else 15)
            except ZlibError as exc:
                raise DeflateError('zlib', str(exc)) from exc

        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]
    dstlen = <size_t> dstsize
    srclen = <size_t> srcsize

    try:
        with nogil:
            decompressor = libdeflate_alloc_decompressor()
            if decompressor == NULL:
                raise DeflateError('libdeflate_alloc_decompressor', None)

            if raw:
                ret = libdeflate_deflate_decompress(
                    decompressor,
                    <const void*> src._data,
                    srclen,
                    <void*> dst._data,
                    dstlen,
                    &actual_out_nbytes_ret
                )
                if ret != LIBDEFLATE_SUCCESS:
                    raise DeflateError('libdeflate_deflate_decompress', ret)
            else:
                ret = libdeflate_zlib_decompress(
                    decompressor,
                    <const void*> src._data,
                    srclen,
                    <void*> dst._data,
                    dstlen,
                    &actual_out_nbytes_ret
                )
                # if ret == LIBDEFLATE_INSUFFICIENT_SPACE:
                #    allow partial decompression for cases found in TIFF
                #    _log_warning(
                #       'libdeflate_zlib_decompress '
                #       'LIBDEFLATE_INSUFFICIENT_SPACE'
                #    )
                #    pass
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


def gzip_check(const uint8_t[::1] data not None, /):
    """Return whether data is GZIP encoded or None if unknown."""
    return _startswith(data, b'\x1f\x8b')


def gzip_encode(
    data,
    /,
    level=None,
    *,
    out=None,
):
    """Return GZIP encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t srclen, dstlen
        libdeflate_compressor* compressor = NULL
        int compression_level = _default_value(level, 6, 0, 12)

    out, dstsize, outgiven, outtype = _parse_output(out)

    compressor = libdeflate_alloc_compressor(compression_level)
    if compressor == NULL:
        raise GzipError('libdeflate_alloc_compressor', None)

    try:
        if out is None:
            if dstsize < 0:
                dstlen = libdeflate_gzip_compress_bound(
                    compressor, <size_t> srcsize
                )
                if dstlen == 0:
                    raise GzipError(
                        'libdeflate_gzip_compress_bound', 'input too large'
                    )
                dstsize = <ssize_t> dstlen
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.shape[0]
        dstlen = <size_t> dstsize
        srclen = <size_t> srcsize

        with nogil:
            dstlen = libdeflate_gzip_compress(
                compressor,
                <const void*> src._data,
                srclen,
                <void*> dst._data,
                dstlen
            )
            if dstlen == 0:
                raise GzipError(
                    'libdeflate_gzip_compress', 'output too small'
                )

    finally:
        if compressor != NULL:
            libdeflate_free_compressor(compressor)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def gzip_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded GZIP data.

    If the output size is unknown and the GZIP size field does not match,
    for example, for multi-member streams, the zlib codec is used.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t srclen, dstlen
        size_t srcpos = 0
        size_t dstpos = 0
        size_t actual_in_nbytes_ret
        size_t actual_out_nbytes_ret
        bint sizegiven
        bint fallback = False
        libdeflate_result ret
        libdeflate_decompressor* decompressor = NULL

    if data is out:
        raise ValueError('cannot transform in-place')

    # 10 byte header, empty DEFLATE block, 8 byte trailer
    if srcsize < 20 or src[0] != 0x1F or src[1] != 0x8B:
        raise ValueError('invalid GZIP header')

    out, dstsize, outgiven, outtype = _parse_output(out)
    sizegiven = out is not None or dstsize >= 0

    if out is None:
        if dstsize < 0:
            # the last 4 bytes contain the size of the original, uncompressed
            # input data modulo 2^32 in little endian order
            dstsize = <ssize_t> (
                (<uint32_t> src[srcsize - 4])
                | (<uint32_t> src[srcsize - 3] << 8)
                | (<uint32_t> src[srcsize - 2] << 16)
                | (<uint32_t> src[srcsize - 1] << 24)
            )
            # DEFLATE cannot expand data by more than ~1032:1
            if dstsize < 0 or dstsize // 1032 > srcsize:
                fallback = True
        if not fallback:
            out = _create_output(outtype, dstsize)

    if not fallback:
        dst = _inplace_input(out) if outgiven else out
        dstsize = dst.shape[0]
        dstlen = <size_t> dstsize
        srclen = <size_t> srcsize

        try:
            with nogil:
                decompressor = libdeflate_alloc_decompressor()
                if decompressor == NULL:
                    raise GzipError('libdeflate_alloc_decompressor', None)

                while True:
                    ret = libdeflate_gzip_decompress_ex(
                        decompressor,
                        <const void*> (src._data + srcpos),
                        srclen - srcpos,
                        <void*> (dst._data + dstpos),
                        dstlen - dstpos,
                        &actual_in_nbytes_ret,
                        &actual_out_nbytes_ret
                    )
                    if ret != LIBDEFLATE_SUCCESS:
                        if (
                            ret == LIBDEFLATE_INSUFFICIENT_SPACE
                            and not sizegiven
                        ):
                            fallback = True
                            break
                        raise GzipError('libdeflate_gzip_decompress_ex', ret)
                    srcpos += actual_in_nbytes_ret
                    dstpos += actual_out_nbytes_ret
                    # ignore zero padding after members
                    while srcpos < srclen and src[srcpos] == 0:
                        srcpos += 1
                    if srcpos >= srclen:
                        break
                    if not sizegiven:
                        fallback = True
                        break

        finally:
            if decompressor != NULL:
                libdeflate_free_decompressor(decompressor)

        del dst

    if fallback:
        # size field is not reliable for multi-member or padded streams
        if _zlib_decode is None:
            raise ValueError(
                'GZIP size field does not match data; output size required'
            )
        try:
            return _zlib_decode(src, outtype, 31)
        except ZlibError as exc:
            raise GzipError('zlib', str(exc)) from exc
    return _return_output(out, dstsize, dstpos, outgiven)


# CRC #########################################################################

def deflate_crc32(
    data,
    /,
    value=None,
):
    """Return CRC32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        uint32_t crc = 0 if value is None else value

    with nogil:
        crc = libdeflate_crc32(crc, <const void*> src._data, srcsize)
    return int(crc)


def deflate_adler32(
    data,
    /,
    value=None,
):
    """Return Adler-32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.shape[0]
        uint32_t adler = 1 if value is None else value

    with nogil:
        adler = libdeflate_adler32(adler, <const void*> src._data, srcsize)
    return int(adler)
