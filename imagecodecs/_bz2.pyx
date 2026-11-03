# imagecodecs/_bz2.pyx
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

"""BZ2 (BZip2) codec for the imagecodecs package."""

include '_shared.pxi'

from libbzip2 cimport *


class BZ2:
    """BZ2 codec constants."""

    available = True


class Bz2Error(RuntimeError):
    """BZ2 codec exceptions."""

    def __init__(self, func, err):
        msg = repr(err) if isinstance(err, str) else {
            BZ_OK: 'BZ_OK',
            BZ_RUN_OK: 'BZ_RUN_OK',
            BZ_FLUSH_OK: 'BZ_FLUSH_OK',
            BZ_FINISH_OK: 'BZ_FINISH_OK',
            BZ_STREAM_END: 'BZ_STREAM_END',
            BZ_SEQUENCE_ERROR: 'BZ_SEQUENCE_ERROR',
            BZ_PARAM_ERROR: 'BZ_PARAM_ERROR',
            BZ_MEM_ERROR: 'BZ_MEM_ERROR',
            BZ_DATA_ERROR: 'BZ_DATA_ERROR',
            BZ_DATA_ERROR_MAGIC: 'BZ_DATA_ERROR_MAGIC',
            BZ_IO_ERROR: 'BZ_IO_ERROR',
            BZ_UNEXPECTED_EOF: 'BZ_UNEXPECTED_EOF',
            BZ_OUTBUFF_FULL: 'BZ_OUTBUFF_FULL',
            BZ_CONFIG_ERROR: 'BZ_CONFIG_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def bz2_version():
    """Return libbzip2 library version string."""
    return 'libbzip2 ' + BZ2_bzlibVersion().decode().split(',')[0]


def bz2_check(const uint8_t[::1] data not None, /):
    """Return whether data is BZ2 encoded or None if unknown."""
    return _startswith(data, b'BZh')


def bz2_encode(
    data,
    /,
    level=None,
    *,
    out=None,
):
    """Return BZ2 encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t dstlen = 0
        size_t inleft, outleft, chunk
        int action = BZ_RUN
        int compresslevel = _default_value(level, 9, 1, 9)
        int ret
        bz_stream strm

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # worst case documented in bzlib.h
            dstsize = srcsize + srcsize // 100 + 600
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    memset(<void*> &strm, 0, sizeof(bz_stream))

    try:
        with nogil:
            ret = BZ2_bzCompressInit(&strm, compresslevel, 0, 0)
            if ret != BZ_OK:
                raise Bz2Error('BZ2_bzCompressInit', ret)

            # avail_in/avail_out are unsigned int; feed in chunks
            inleft = <size_t> srcsize
            outleft = <size_t> dstsize
            strm.next_in = <char*> src._data
            strm.next_out = <char*> dst._data
            while True:
                if strm.avail_in == 0 and inleft > 0:
                    chunk = min(inleft, <size_t> UINT32_MAX)
                    strm.avail_in = <unsigned int> chunk
                    inleft -= chunk
                if strm.avail_out == 0 and outleft > 0:
                    chunk = min(outleft, <size_t> UINT32_MAX)
                    strm.avail_out = <unsigned int> chunk
                    outleft -= chunk
                if inleft == 0:
                    action = BZ_FINISH
                ret = BZ2_bzCompress(&strm, action)
                if ret == BZ_STREAM_END:
                    break
                if ret != BZ_RUN_OK and ret != BZ_FINISH_OK:
                    break
                if strm.avail_out == 0 and outleft == 0:
                    break
            dstlen = dstsize - <ssize_t> (outleft + strm.avail_out)
        if ret == BZ_RUN_OK or ret == BZ_FINISH_OK:
            raise Bz2Error('BZ2_bzCompress', 'output too small')
        if ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzCompress', ret)
    finally:
        ret = BZ2_bzCompressEnd(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def bz2_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded BZ2 data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* buffer = NULL
        uint8_t* tmp
        size_t srcsize = <size_t> src.shape[0]
        size_t incsize = 0
        size_t bufsize, inleft, outleft, chunk
        ssize_t dstsize
        ssize_t dstlen = 0
        bint grow
        bint initialized = False
        bint truncated = False
        int ret
        bz_stream strm

    if data is out:
        raise ValueError('cannot transform in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    grow = out is None and dstsize < 0
    if grow:
        incsize = _align_size_t(srcsize // 2)
        if incsize > 268435456:  # 256 MB
            incsize = 268435456
        bufsize = 3 * incsize  # 3/2 srcsize
    else:
        if out is None:
            out = _create_output(outtype, dstsize)
        dst = _inplace_input(out) if outgiven else out
        dstsize = dst.shape[0]
        bufsize = <size_t> dstsize

    memset(<void*> &strm, 0, sizeof(bz_stream))

    try:
        with nogil:
            ret = BZ2_bzDecompressInit(&strm, 0, 0)
            if ret != BZ_OK:
                raise Bz2Error('BZ2_bzDecompressInit', ret)
            initialized = True

            if grow:
                buffer = <uint8_t*> malloc(bufsize)
                if buffer == NULL:
                    raise MemoryError('failed to allocate output buffer')
                strm.next_out = <char*> buffer
            else:
                strm.next_out = <char*> dst._data

            # avail_in/avail_out are unsigned int; feed in chunks
            inleft = srcsize
            outleft = bufsize
            strm.next_in = <char*> src._data
            while True:
                if strm.avail_in == 0 and inleft > 0:
                    chunk = min(inleft, <size_t> UINT32_MAX)
                    strm.avail_in = <unsigned int> chunk
                    inleft -= chunk
                if strm.avail_out == 0:
                    if outleft == 0 and grow:
                        outleft = max(incsize, bufsize // 2)
                        if bufsize > SIZE_MAX - outleft:
                            raise MemoryError('output buffer size overflow')
                        tmp = <uint8_t*> realloc(
                            <void*> buffer, bufsize + outleft
                        )
                        if tmp == NULL:
                            raise MemoryError('failed to resize output buffer')
                        buffer = tmp
                        strm.next_out = <char*> (buffer + bufsize)
                        bufsize += outleft
                    chunk = min(outleft, <size_t> UINT32_MAX)
                    strm.avail_out = <unsigned int> chunk
                    outleft -= chunk
                ret = BZ2_bzDecompress(&strm)
                if ret == BZ_OK:
                    if strm.avail_out == 0:
                        if outleft == 0 and not grow:
                            break  # output buffer too small
                    elif strm.avail_in == 0 and inleft == 0:
                        truncated = True
                        break
                elif ret == BZ_STREAM_END:
                    # decode concatenated streams like Python's bz2 module
                    if strm.avail_in == 0 and inleft == 0:
                        break
                    if strm.avail_out == 0 and outleft == 0 and not grow:
                        break
                    if strm.avail_in < 3 or (
                        strm.next_in[0] != b'B'
                        or strm.next_in[1] != b'Z'
                        or strm.next_in[2] != b'h'
                    ):
                        break  # ignore trailing data
                    initialized = False
                    BZ2_bzDecompressEnd(&strm)
                    ret = BZ2_bzDecompressInit(&strm, 0, 0)
                    if ret != BZ_OK:
                        break
                    initialized = True
                else:
                    break
            dstlen = <ssize_t> (bufsize - (outleft + strm.avail_out))
        if truncated:
            raise Bz2Error('BZ2_bzDecompress', 'truncated data')
        if ret != BZ_OK and ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzDecompress', ret)
        if grow:
            return _create_output(outtype, dstlen, <const char*> buffer)
    finally:
        free(buffer)
        if initialized:
            BZ2_bzDecompressEnd(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)
