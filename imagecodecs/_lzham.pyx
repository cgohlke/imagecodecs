# imagecodecs/_lzham.pyx
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

"""LZHAM codec for the imagecodecs package."""

include '_shared.pxi'

from lzham cimport *


class LZHAM:
    """LZHAM codec constants."""

    available = True

    class COMPRESSION(enum.IntEnum):
        """LZHAM codec compression levels."""

        DEFAULT = LZHAM_Z_DEFAULT_COMPRESSION
        NO = LZHAM_Z_NO_COMPRESSION
        BEST = LZHAM_Z_BEST_COMPRESSION
        SPEED = LZHAM_Z_BEST_SPEED
        UBER = LZHAM_Z_UBER_COMPRESSION

    class STRATEGY(enum.IntEnum):
        """LZHAM codec compression strategies."""

        DEFAULT = LZHAM_Z_DEFAULT_STRATEGY
        FILTERED = LZHAM_Z_FILTERED
        HUFFMAN_ONLY = LZHAM_Z_HUFFMAN_ONLY
        RLE = LZHAM_Z_RLE
        FIXED = LZHAM_Z_FIXED


class LzhamError(RuntimeError):
    """LZHAM codec exceptions."""

    def __init__(self, func, err):
        cdef:
            const char* error = lzham_z_error(err)

        msg = f'unknown error {err!r}' if error == NULL else error.decode()
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def lzham_version():
    """Return LZHAM library version string."""
    return 'lzham ' + lzham_z_version().decode()


def lzham_check(data):
    """Return whether data is LZHAM encoded."""


def lzham_encode(data, level=None, out=None):
    """Return LZHAM encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        lzham_z_ulong srclen, dstlen
        int ret
        int compresslevel = _default_value(
            level, LZHAM_Z_DEFAULT_COMPRESSION, -1, LZHAM_Z_UBER_COMPRESSION
        )

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming APIs
            dstsize = <ssize_t> lzham_z_compressBound(<lzham_z_ulong> srcsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <lzham_z_ulong> dstsize
    srclen = <lzham_z_ulong> srcsize

    with nogil:
        ret = lzham_z_compress2(
            <unsigned char*> &dst[0],
            &dstlen,
            &src[0],
            srclen,
            compresslevel
        )
    if ret != LZHAM_Z_OK:
        raise LzhamError('lzham_z_compress2', ret)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def lzham_decode(data, out=None):
    """Return decoded LZHAM data."""
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        lzham_z_ulong srclen, dstlen
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        return _lzham_decode(data, outtype)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError  # TODO
            # lzham_z_ulong lzham_z_deflateBound(
            #     lzham_z_streamp pStream, lzham_z_ulong source_len);
        out = _create_output(outtype, dstsize)

    src = data
    dst = out
    dstsize = dst.size
    dstlen = <lzham_z_ulong> dstsize
    srclen = <lzham_z_ulong> src.size

    with nogil:
        ret = lzham_z_uncompress(
            <unsigned char*> &dst[0],
            &dstlen,
            <const unsigned char *> &src[0],
            srclen
        )
    if ret != LZHAM_Z_OK:
        raise LzhamError('lzham_z_uncompress', ret)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


cdef _lzham_decode(const uint8_t[::1] src, outtype):
    """Decompress using streaming API."""
    cdef:
        output_t* output = NULL
        lzham_z_stream stream
        ssize_t srcsize = <size_t> src.size
        size_t size, left
        int ret

    try:
        with nogil:

            stream.next_in = <unsigned char*> &src[0]
            stream.avail_in = 0
            stream.zalloc = NULL
            stream.zfree = NULL
            stream.opaque = NULL

            ret = lzham_z_inflateInit(&stream)
            if ret != LZHAM_Z_OK:
                raise LzhamError('lzham_z_inflateInit', ret)

            output = output_new(
                NULL,
                max(4096, (srcsize * 2) + (4096 - srcsize * 2) % 4096)
            )
            if output == NULL:
                raise MemoryError('output_new failed')

            stream.next_out = <unsigned char*> output.data
            stream.avail_out = 0
            left = <size_t> output.size
            size = <size_t> srcsize

            while ret == LZHAM_Z_OK or ret == LZHAM_Z_BUF_ERROR:

                if stream.avail_out == 0:
                    if left == 0:
                        left = output.size * 2
                        if output_resize(output, output.size + left) == 0:
                            raise MemoryError('output_resize failed')
                        stream.next_out = (
                            <unsigned char*> output.data + (output.size - left)
                        )
                    if left > <size_t> 4294967295U:
                        stream.avail_out = <unsigned int> 4294967295U
                    else:
                        stream.avail_out = <unsigned int> left
                    left -= stream.avail_out

                if stream.avail_in == 0:
                    if ret == LZHAM_Z_BUF_ERROR:
                        break
                    if size > <size_t> 4294967295U:
                        stream.avail_in = <unsigned int> 4294967295U
                    else:
                        stream.avail_in = <unsigned int> size
                    size -= stream.avail_in

                ret = lzham_z_inflate(&stream, LZHAM_Z_NO_FLUSH)

            if ret != LZHAM_Z_STREAM_END:
                raise LzhamError('lzham_z_inflate', ret)

        out = _create_output(
            outtype, stream.total_out, <const char *> output.data
        )

    finally:
        output_del(output)
        ret = lzham_z_inflateEnd(&stream)
        if ret != LZHAM_Z_OK:
            raise LzhamError('lzham_z_inflateEnd', ret)

    return out


# CRC #########################################################################

def lzham_crc32(data):
    """Return cyclic redundancy checksum CRC-32 of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        lzham_z_ulong crc = 0

    with nogil:
        crc = lzham_z_crc32(crc, NULL, 0)
        crc = lzham_z_crc32(crc, <const unsigned char*> &src[0], srcsize)
    return int(crc)


def lzham_adler32(data):
    """Return Adler-32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        size_t srcsize = <size_t> src.size
        lzham_z_ulong adler

    with nogil:
        adler = lzham_z_adler32(0, NULL, 0)
        adler = lzham_z_adler32(adler, <const unsigned char*> &src[0], srcsize)
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
