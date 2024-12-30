# imagecodecs/_zstd.pyx
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

"""Zstd (ZStandard) codec for the imagecodecs package."""

include '_shared.pxi'

from zstd cimport *


class ZSTD:
    """ZSTD codec constants."""

    available = True


class ZstdError(RuntimeError):
    """ZSTD codec exceptions."""

    def __init__(self, func, msg='', err=0):
        cdef:
            const char* errmsg

        if msg:
            msg = f'{func} returned {msg!r}'
        else:
            errmsg = ZSTD_getErrorName(err)
            msg = f'{func} returned {errmsg.decode()!r}'
        super().__init__(msg)


def zstd_version():
    """Return Zstandard library version string."""
    return 'zstd {}.{}.{}'.format(
        ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_RELEASE
    )


def zstd_check(data):
    """Return whether data is ZSTD encoded."""
    cdef:
        bytes sig = bytes(data[:4])

    return sig == b'\x28\xB5\x2F\xFD'


def zstd_encode(data, level=None, out=None):
    """Return ZSTD encoded data."""
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
            if dstsize <= 0:
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
    """Return decoded ZSTD data."""
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
            if cntsize == ZSTD_CONTENTSIZE_UNKNOWN or cntsize > 2147483647:
                # use streaming API for unknown or suspiciously large sizes
                return _zstd_decode(data, outtype)
            if cntsize == ZSTD_CONTENTSIZE_ERROR:
                raise ZstdError('ZSTD_getFrameContentSize', f'{cntsize}')
            dstsize = cntsize
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


cdef _zstd_decode(const uint8_t[::1] src, outtype):
    """Decompress using streaming API."""
    cdef:
        output_t* output = NULL
        ZSTD_DCtx* dctx = NULL
        ZSTD_inBuffer zinput
        ZSTD_outBuffer zoutput
        size_t ret
        size_t srcsize = <size_t> src.size
        size_t outsize = ZSTD_DStreamOutSize()
        # increase output size by ~1/2 input size, min 128 KB
        size_t incsize = max((srcsize // outsize) * outsize // 2, outsize)

    try:
        with nogil:

            dctx = ZSTD_createDCtx()
            if dctx == NULL:
                raise ZstdError('ZSTD_createDCtx', 'NULL')

            # allocate ~3/2 input size, min 384 KB, for output
            output = output_new(NULL, incsize * 3)
            if output == NULL:
                raise MemoryError('output_new failed')

            zoutput.dst = <void*> output.data
            zoutput.size = <size_t> output.size
            zoutput.pos = output.pos

            zinput.src = <void*> &src[0]
            zinput.size = srcsize
            zinput.pos = 0

            while zinput.pos < zinput.size:
                if output.size - output.used < outsize:
                    ret = output_resize(output, output.used + incsize)
                    if ret == 0:
                        raise MemoryError('output_resize failed')
                    zoutput.dst = <void*> output.data
                    zoutput.size = <size_t> output.size
                    zoutput.pos = output.pos

                ret = ZSTD_decompressStream(dctx, &zoutput , &zinput)
                if ZSTD_isError(ret):
                    raise ZstdError('ZSTD_decompressStream', err=ret)

                output.pos = zoutput.pos
                output.used = zoutput.pos

        out = _create_output(outtype, output.pos, <const char*> output.data)

    finally:
        output_del(output)
        ZSTD_freeDCtx(dctx)

    return out


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
