# imagecodecs/_lzma.pyx
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

"""LZMA codec for the imagecodecs package."""

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
        msg = {
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


def lzma_check(const uint8_t[::1] data):
    """Return whether data is LZMA encoded."""


def lzma_encode(data, level=None, check=None, out=None):
    """Return LZMA encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen
        uint32_t preset = _default_value(level, 6, 0, 9)
        lzma_check_t check_ = LZMA_CHECK_CRC64 if check is None else check
        lzma_stream strm
        lzma_ret ret

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = lzma_stream_buffer_bound(srcsize)
            if dstsize == 0:
                raise LzmaError('lzma_stream_buffer_bound', '0')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    try:
        with nogil:
            memset(&strm, 0, sizeof(lzma_stream))
            ret = lzma_easy_encoder(&strm, preset, check_)
            if ret != LZMA_OK:
                raise LzmaError('lzma_easy_encoder', ret)
            strm.next_in = <uint8_t *> &src[0]
            strm.avail_in = <size_t> srcsize
            strm.next_out = <uint8_t*> &dst[0]
            strm.avail_out = <size_t> dstsize
            ret = lzma_code(&strm, LZMA_RUN)
            if ret == LZMA_OK or ret == LZMA_STREAM_END:
                ret = lzma_code(&strm, LZMA_FINISH)
            dstlen = dstsize - <ssize_t> strm.avail_out
        if ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


def lzma_decode(data, out=None):
    """Return decoded LZMA data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen
        lzma_ret ret
        lzma_stream strm

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = _lzma_uncompressed_size(src, srcsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    try:
        with nogil:
            memset(&strm, 0, sizeof(lzma_stream))
            ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED)
            if ret != LZMA_OK:
                raise LzmaError('lzma_stream_decoder', ret)
            strm.next_in = <uint8_t *> &src[0]
            strm.avail_in = <size_t> srcsize
            strm.next_out = <uint8_t*> &dst[0]
            strm.avail_out = <size_t> dstsize
            ret = lzma_code(&strm, LZMA_RUN)
            dstlen = dstsize - <ssize_t> strm.avail_out
        if ret != LZMA_OK and ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    del dst
    return _return_output(out, dstsize, dstlen, outgiven)


cdef _lzma_uncompressed_size(const uint8_t[::1] data, ssize_t size):
    """Return size of decompressed LZMA data."""
    cdef:
        lzma_ret ret
        lzma_index* index
        lzma_stream_flags options
        lzma_vli usize = 0
        uint64_t memlimit = UINT64_MAX
        ssize_t offset
        size_t pos = 0

    if size < LZMA_STREAM_HEADER_SIZE:
        raise ValueError('invalid LZMA data')
    try:
        index = lzma_index_init(NULL)
        offset = size - LZMA_STREAM_HEADER_SIZE
        ret = lzma_stream_footer_decode(&options, &data[offset])
        if ret != LZMA_OK:
            raise LzmaError('lzma_stream_footer_decode', ret)
        offset -= <ssize_t> options.backward_size
        ret = lzma_index_buffer_decode(
            &index,
            &memlimit,
            NULL,
            &data[offset],
            &pos,
            <ssize_t> options.backward_size
        )
        if ret != LZMA_OK or pos != options.backward_size:
            raise LzmaError('lzma_index_buffer_decode', ret)
        usize = lzma_index_uncompressed_size(index)
    finally:
        lzma_index_end(index, NULL)
    return <ssize_t> usize
