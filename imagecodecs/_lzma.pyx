# imagecodecs/_lzma.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2018-2026, Christoph Gohlke
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

"""LZMA (Lempel-Ziv-Markov Chain Algorithm) codec for the imagecodecs package.

"""

include '_shared.pxi'

from liblzma cimport *


class LZMA:
    """LZMA codec constants."""

    available = True

    class CHECK(enum.IntEnum):
        """LZMA codec checksums."""

        NONE = LZMA_CHECK_NONE
        CRC32 = LZMA_CHECK_CRC32
        CRC64 = LZMA_CHECK_CRC64
        SHA256 = LZMA_CHECK_SHA256


class LzmaError(RuntimeError):
    """LZMA codec exceptions."""

    def __init__(self, func, err):
        msg = repr(err) if isinstance(err, str) else {
            LZMA_OK: 'LZMA_OK',
            LZMA_STREAM_END: 'LZMA_STREAM_END',
            LZMA_NO_CHECK: 'LZMA_NO_CHECK',
            LZMA_UNSUPPORTED_CHECK: 'LZMA_UNSUPPORTED_CHECK',
            LZMA_GET_CHECK: 'LZMA_GET_CHECK',
            LZMA_MEM_ERROR: 'LZMA_MEM_ERROR',
            LZMA_MEMLIMIT_ERROR: 'LZMA_MEMLIMIT_ERROR',
            LZMA_FORMAT_ERROR: 'LZMA_FORMAT_ERROR',
            LZMA_OPTIONS_ERROR: 'LZMA_OPTIONS_ERROR',
            LZMA_DATA_ERROR: 'LZMA_DATA_ERROR',
            LZMA_BUF_ERROR: 'LZMA_BUF_ERROR',
            LZMA_PROG_ERROR: 'LZMA_PROG_ERROR',
            # LZMA_SEEK_NEEDED: 'LZMA_SEEK_NEEDED',  # version 3.4.0
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def lzma_version():
    """Return liblzma library version string."""
    return 'liblzma {}.{}.{}'.format(
        LZMA_VERSION_MAJOR, LZMA_VERSION_MINOR, LZMA_VERSION_PATCH
    )


def lzma_check(const uint8_t[::1] data not None, /):
    """Return whether data is LZMA encoded or None if unknown."""
    return _startswith(data, b'\xfd7zXZ\x00')


def lzma_encode(
    data,
    /,
    level=None,
    *,
    check=None,
    out=None,
):
    """Return LZMA encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t dstlen
        uint32_t preset = _default_value(level, 6, 0, 9)
        lzma_check_t check_ = _enum_value(check, LZMA.CHECK, LZMA_CHECK_CRC64)
        lzma_stream strm
        lzma_ret ret

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = lzma_stream_buffer_bound(srcsize)
            if dstsize == 0:
                raise LzmaError('lzma_stream_buffer_bound', 'input too large')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    try:
        with nogil:
            memset(<void*> &strm, 0, sizeof(lzma_stream))
            ret = lzma_easy_encoder(&strm, preset, check_)
            if ret != LZMA_OK:
                raise LzmaError('lzma_easy_encoder', ret)
            strm.next_in = <uint8_t*> src._data
            strm.avail_in = <size_t> srcsize
            strm.next_out = <uint8_t*> dst._data
            strm.avail_out = <size_t> dstsize
            ret = LZMA_OK
            while ret == LZMA_OK:
                ret = lzma_code(&strm, LZMA_FINISH)
            dstlen = dstsize - <ssize_t> strm.avail_out
        if ret == LZMA_BUF_ERROR and strm.avail_out == 0:
            raise LzmaError('lzma_code', 'output too small')
        if ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def lzma_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded LZMA data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t dstlen
        uint64_t memlimit = lzma_physmem()
        lzma_ret ret
        lzma_stream strm

    if data is out:
        raise ValueError('cannot transform in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = _lzma_uncompressed_size(src)
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]

    try:
        with nogil:
            memset(<void*> &strm, 0, sizeof(lzma_stream))
            if memlimit == 0:
                memlimit = UINT64_MAX
            ret = lzma_stream_decoder(&strm, memlimit, LZMA_CONCATENATED)
            if ret != LZMA_OK:
                raise LzmaError('lzma_stream_decoder', ret)
            strm.next_in = <uint8_t*> src._data
            strm.avail_in = <size_t> srcsize
            strm.next_out = <uint8_t*> dst._data
            strm.avail_out = <size_t> dstsize
            ret = LZMA_OK
            while ret == LZMA_OK:
                ret = lzma_code(&strm, LZMA_FINISH)
            dstlen = dstsize - <ssize_t> strm.avail_out
        if ret == LZMA_BUF_ERROR:
            # truncated output is allowed
            if strm.avail_out > 0:
                raise LzmaError('lzma_code', 'truncated data')
        elif ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def lzma_info(const uint8_t[::1] data not None, /):
    """Return information about LZMA encoded data."""
    return {'_codec': 'lzma', 'size': _lzma_uncompressed_size(data)}


cdef ssize_t _lzma_uncompressed_size(const uint8_t[::1] data) except -1:
    """Return size of decompressed LZMA data."""
    cdef:
        ssize_t size = data.shape[0]
        lzma_ret ret
        lzma_index* index = NULL
        lzma_stream strm
        lzma_vli usize = 0

    if size < 2 * LZMA_STREAM_HEADER_SIZE:
        raise ValueError('invalid LZMA data')

    try:
        with nogil:
            memset(<void*> &strm, 0, sizeof(lzma_stream))
            # parse indices of all concatenated streams and padding
            ret = lzma_file_info_decoder(
                &strm, &index, UINT64_MAX, <uint64_t> size
            )
            if ret != LZMA_OK:
                raise LzmaError('lzma_file_info_decoder', ret)
            strm.next_in = <uint8_t*> &data[0]
            strm.avail_in = <size_t> size
            while True:
                ret = lzma_code(&strm, LZMA_RUN)
                if ret == LZMA_SEEK_NEEDED:
                    if strm.seek_pos > <uint64_t> size:
                        break
                    strm.next_in = <uint8_t*> &data[0] + strm.seek_pos
                    strm.avail_in = <size_t> (size - <ssize_t> strm.seek_pos)
                elif ret != LZMA_OK:
                    break
            if ret != LZMA_STREAM_END:
                raise LzmaError('lzma_file_info_decoder', ret)
            usize = lzma_index_uncompressed_size(index)
    finally:
        lzma_end(&strm)
        lzma_index_end(index, NULL)

    if usize > <lzma_vli> (SIZE_MAX >> 1):
        raise ValueError(f'invalid uncompressed size {usize}')
    return <ssize_t> usize
