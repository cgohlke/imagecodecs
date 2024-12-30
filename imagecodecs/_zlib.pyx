# imagecodecs/_zlib.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2025, Christoph Gohlke
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

include '_shared.pxi'

from zlib cimport *


class ZLIB:
    """ZLIB codec constants."""

    available = True

    class COMPRESSION(enum.IntEnum):
        """ZLIB codec compression levels."""

        DEFAULT = Z_DEFAULT_COMPRESSION
        NO = Z_NO_COMPRESSION
        BEST = Z_BEST_COMPRESSION
        SPEED = Z_BEST_SPEED

    class STRATEGY(enum.IntEnum):
        """ZLIB codec compression strategies."""

        DEFAULT = Z_DEFAULT_STRATEGY
        FILTERED = Z_FILTERED
        HUFFMAN_ONLY = Z_HUFFMAN_ONLY
        RLE = Z_RLE
        FIXED = Z_FIXED


class ZlibError(RuntimeError):
    """ZLIB codec exceptions."""

    def __init__(self, func, err):
        msg = {
            Z_OK: 'Z_OK',
            Z_STREAM_END: 'Z_STREAM_END',
            Z_NEED_DICT: 'Z_NEED_DICT',
            Z_ERRNO: 'Z_ERRNO',
            Z_STREAM_ERROR: 'Z_STREAM_ERROR',
            Z_DATA_ERROR: 'Z_DATA_ERROR',
            Z_MEM_ERROR: 'Z_MEM_ERROR',
            Z_BUF_ERROR: 'Z_BUF_ERROR',
            Z_VERSION_ERROR: 'Z_VERSION_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def zlib_version():
    """Return zlib library version string."""
    return 'zlib ' + zlibVersion().decode()


def zlib_check(data):
    """Return whether data is DEFLATE encoded."""
    cdef:
        bytes sig = bytes(data[:2])

    # most common ZLIB headers
    if (
        sig == b'\x78\x9C'
        or sig == b'\x78\x5E'
        or sig == b'\x78\x01'
        or sig == b'\x78\xDA'
    ):
        return True
    return None  # maybe


def zlib_encode(data, level=None, out=None):
    """Return DEFLATE encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret
        int compresslevel = _default_value(
            level, Z_DEFAULT_COMPRESSION, -1, Z_BEST_COMPRESSION
        )

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming APIs
            dstsize = <ssize_t> compressBound(<unsigned long> srcsize)
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
    """Return decoded DEFLATE data."""
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        return _zlib_decode(data, outtype)
        # use Python's zlib module
        # import zlib
        # return zlib.decompress(data)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError  # TODO
        out = _create_output(outtype, dstsize)

    src = data
    dst = out
    dstsize = dst.size
    dstlen = <unsigned long> dstsize
    srclen = <unsigned long> src.size

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


cdef _zlib_decode(const uint8_t[::1] src, outtype):
    """Decompress using streaming API."""
    cdef:
        output_t* output = NULL
        z_stream stream
        ssize_t srcsize = <size_t> src.size
        size_t size, left
        int ret

    try:
        with nogil:

            stream.next_in = <Bytef*> &src[0]  # <z_const Bytef*>
            stream.avail_in = 0
            stream.zalloc = NULL
            stream.zfree = NULL
            stream.opaque = NULL

            ret = inflateInit(&stream)
            if ret != Z_OK:
                raise ZlibError('inflateInit', ret)

            output = output_new(
                NULL,
                max(4096, (srcsize * 2) + (4096 - srcsize * 2) % 4096)
            )
            if output == NULL:
                raise MemoryError('output_new failed')

            stream.next_out = <Bytef*> output.data
            stream.avail_out = 0
            left = <size_t> output.size
            size = <size_t> srcsize

            while ret == Z_OK or ret == Z_BUF_ERROR:

                if stream.avail_out == 0:
                    if left == 0:
                        left = output.size * 2
                        if output_resize(output, output.size + left) == 0:
                            raise MemoryError('output_resize failed')
                        stream.next_out = (
                            <Bytef*> output.data + (output.size - left)
                        )
                    if left > <size_t> 4294967295U:
                        stream.avail_out = <uInt> 4294967295U
                    else:
                        stream.avail_out = <uInt> left
                    left -= stream.avail_out

                if stream.avail_in == 0:
                    if ret == Z_BUF_ERROR:
                        # ret = Z_STREAM_END
                        break
                    if size > <size_t> 4294967295U:
                        stream.avail_in = <uInt> 4294967295U
                    else:
                        stream.avail_in = <uInt> size
                    size -= stream.avail_in

                ret = inflate(&stream, Z_NO_FLUSH)

            if ret != Z_STREAM_END:
                raise ZlibError('inflate', ret)

        out = _create_output(
            outtype, stream.total_out, <const char *> output.data
        )

    finally:
        output_del(output)
        ret = inflateEnd(&stream)
        if ret != Z_OK:
            raise ZlibError('inflateEnd', ret)

    return out


# CRC #########################################################################

def zlib_crc32(data, value=None):
    """Return CRC32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        uInt srcsize = <uInt> src.size
        uLong crc = 0 if value is None else value

    with nogil:
        # crc = crc32(crc, NULL, 0)
        crc = crc32(crc, <const Bytef*> &src[0], srcsize)
    return int(crc)


def zlib_adler32(data, value=None):
    """Return Adler-32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        uInt srcsize = <uInt> src.size
        uLong adler = 1 if value is None else value

    with nogil:
        # adler = adler32(adler, NULL, 0)
        adler = adler32(adler, <const Bytef*> &src[0], srcsize)
    return int(adler)


# Output Stream ###############################################################

ctypedef struct output_t:
    uint8_t* data
    size_t size
    size_t pos
    size_t used
    int owner


cdef output_t* output_new(uint8_t* data, size_t size) noexcept nogil:
    """Return new output."""
    cdef:
        output_t* output = <output_t*> calloc(1, sizeof(output_t))

    if output == NULL:
        return NULL
    output.size = size
    output.used = 0
    output.pos = 0
    if data == NULL:
        output.owner = 1
        output.data = <uint8_t*> malloc(size)
    else:
        output.owner = 0
        output.data = data
    if output.data == NULL:
        free(output)
        return NULL
    return output


cdef void output_del(output_t* output) noexcept nogil:
    """Free output."""
    if output != NULL:
        if output.owner != 0:
            free(output.data)
        free(output)


cdef int output_seek(output_t* output, size_t pos) noexcept nogil:
    """Seek output to position."""
    if output == NULL or pos > output.size:
        return 0
    output.pos = pos
    if pos > output.used:
        output.used = pos
    return 1


cdef int output_resize(output_t* output, size_t newsize) noexcept nogil:
    """Resize output."""
    cdef:
        uint8_t* tmp

    if output == NULL or newsize == 0 or output.used > output.size:
        return 0
    if newsize == output.size or output.owner == 0:
        return 1

    tmp = <uint8_t*> realloc(<void*> output.data, newsize)
    if tmp == NULL:
        return 0
    output.data = tmp
    output.size = newsize
    return 1
