# imagecodecs/_bz2.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""BZ2 codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from libbzip2 cimport *


class BZ2:
    """Bz2 Constants."""


class Bz2Error(RuntimeError):
    """BZ2 Exceptions."""

    def __init__(self, func, err):
        msg = {
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


def bz2_check(data):
    """Return True if data likely contains BZ2 data."""


def bz2_encode(data, level=None, out=None):
    """Compress BZ2.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen = 0
        int ret
        bz_stream strm
        int compresslevel = _default_value(level, 9, 1, 9)

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's bz2 module
        import bz2
        return bz2.compress(data, compresslevel)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(bz_stream))
    ret = BZ2_bzCompressInit(&strm, compresslevel, 0, 0)
    if ret != BZ_OK:
        raise Bz2Error('BZ2_bzCompressInit', ret)

    try:
        with nogil:
            strm.next_in = <char*> &src[0]
            strm.avail_in = <unsigned int> srcsize
            strm.next_out = <char*> &dst[0]
            strm.avail_out = <unsigned int> dstsize
            # while True
            ret = BZ2_bzCompress(&strm, BZ_FINISH)
            #    if ret == BZ_STREAM_END:
            #        break
            #    elif ret != BZ_OK:
            #        break
            dstlen = dstsize - <ssize_t> strm.avail_out
        if ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzCompress', ret)
    finally:
        ret = BZ2_bzCompressEnd(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def bz2_decode(data, out=None):
    """Decompress BZ2.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen = 0
        int ret
        bz_stream strm

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's bz2 module
        import bz2

        return bz2.decompress(data)

    if out is None:
        if dstsize < 0:
            raise NotImplementedError
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(bz_stream))
    ret = BZ2_bzDecompressInit(&strm, 0, 0)
    if ret != BZ_OK:
        raise Bz2Error('BZ2_bzDecompressInit', ret)

    try:
        with nogil:
            strm.next_in = <char*> &src[0]
            strm.avail_in = <unsigned int> srcsize
            strm.next_out = <char*> &dst[0]
            strm.avail_out = <unsigned int> dstsize
            ret = BZ2_bzDecompress(&strm)
            dstlen = dstsize - <ssize_t> strm.avail_out
        if ret == BZ_OK:
            pass  # output buffer too small
        elif ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzDecompress', ret)
    finally:
        ret = BZ2_bzDecompressEnd(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)
