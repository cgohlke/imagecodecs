# imagecodecs/_isal.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""ISA-L deflate and GZIP codec for the imagecodecs package."""

include '_shared.pxi'

from isal cimport *


class ISAL:
    """ISAL codec constants."""

    available = True

    class COMPRESSION(enum.IntEnum):
        """ISAL codec compression levels."""

        DEFAULT = 1
        NO = 0  # store, ISAL_DEF_MIN_LEVEL
        BEST = 3  # ISAL_DEF_MAX_LEVEL
        SPEED = 1  # fast


class IsalError(RuntimeError):
    """ISAL codec exceptions."""

    def __init__(self, func, err):
        msg = {
            ISAL_DECOMP_OK: 'ISAL_DECOMP_OK',
            ISAL_END_INPUT: 'ISAL_END_INPUT',
            ISAL_OUT_OVERFLOW: 'ISAL_OUT_OVERFLOW',
            ISAL_NAME_OVERFLOW: 'ISAL_NAME_OVERFLOW',
            ISAL_COMMENT_OVERFLOW: 'ISAL_COMMENT_OVERFLOW',
            ISAL_EXTRA_OVERFLOW: 'ISAL_EXTRA_OVERFLOW',
            ISAL_NEED_DICT: 'ISAL_NEED_DICT',
            ISAL_INVALID_BLOCK: 'ISAL_INVALID_BLOCK',
            ISAL_INVALID_SYMBOL: 'ISAL_INVALID_SYMBOL',
            ISAL_INVALID_LOOKBACK: 'ISAL_INVALID_LOOKBACK',
            ISAL_INVALID_WRAPPER: 'ISAL_INVALID_WRAPPER',
            ISAL_UNSUPPORTED_METHOD: 'ISAL_UNSUPPORTED_METHOD',
            ISAL_INCORRECT_CHECKSUM: 'ISAL_INCORRECT_CHECKSUM',
            COMP_OK: 'COMP_OK',
            STATELESS_OVERFLOW: 'STATELESS_OVERFLOW',
            ISAL_INVALID_STATE: 'ISAL_INVALID_STATE',
            ISAL_INVALID_LEVEL: 'ISAL_INVALID_LEVEL',
            ISAL_INVALID_LEVEL_BUF: 'ISAL_INVALID_LEVEL_BUF',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def isal_version():
    """Return isa-l library version string."""
    return (
        f'isa-l'
        f' {ISAL_MAJOR_VERSION}.{ISAL_MINOR_VERSION}.{ISAL_PATCH_VERSION}'
    )


def isal_check(const uint8_t[::1] data, /):
    """Return whether data is ZLIB or GZIP encoded or None if unknown."""
    cdef:
        bytes sig = bytes(data[:2])

    if (
        sig == b'\x1f\x8b'  # GZIP magic bytes
        # most common ZLIB headers
        or sig == b'\x78\x9C'
        or sig == b'\x78\x5E'
        or sig == b'\x78\x01'
        or sig == b'\x78\xDA'
    ):
        return True
    return None


def isal_encode(
    data,
    /,
    level=None,
    *,
    bint raw=False,
    bint gzip=False,
    out=None,
):
    """Return DEFLATE, ZLIB, or GZIP encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        uint8_t* level_buf = NULL
        uint32_t level_buf_size = 0
        uint16_t gzip_flag = IGZIP_ZLIB
        isal_zstream stream
        int compresslevel = _default_value(level, 1, 0, 3)
        int ret

    if raw and gzip:
        raise ValueError('raw and gzip are mutually exclusive')
    if raw:
        gzip_flag = IGZIP_DEFLATE
    elif gzip:
        gzip_flag = IGZIP_GZIP

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = _isal_compress_bound(srcsize)
            if gzip:
                dstsize += 32
            elif not raw:
                dstsize += 6
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    try:
        with nogil:
            if compresslevel == 1:
                level_buf_size = <uint32_t> ISAL_DEF_LVL1_DEFAULT
            elif compresslevel == 2:
                level_buf_size = <uint32_t> ISAL_DEF_LVL2_DEFAULT
            elif compresslevel == 3:
                level_buf_size = <uint32_t> ISAL_DEF_LVL3_DEFAULT

            if level_buf_size > 0:
                level_buf = <uint8_t*> malloc(level_buf_size)
                if level_buf == NULL:
                    raise MemoryError('failed to allocate ISA-L level buffer')

            isal_deflate_stateless_init(&stream)
            stream.next_in = <uint8_t*> src._data
            stream.avail_in = <uint32_t> srcsize
            stream.next_out = <uint8_t*> dst._data
            stream.avail_out = <uint32_t> dstsize
            stream.end_of_stream = 1
            stream.flush = NO_FLUSH
            stream.gzip_flag = gzip_flag
            stream.level = <uint32_t> compresslevel
            if level_buf_size > 0:
                stream.level_buf = level_buf
                stream.level_buf_size = level_buf_size
            ret = isal_deflate_stateless(&stream)

        if ret != COMP_OK:
            raise IsalError('isal_deflate_stateless', ret)
    finally:
        if level_buf != NULL:
            free(level_buf)

    del dst
    return _return_output(out, dstsize, <ssize_t> stream.total_out, outgiven)


def isal_decode(
    data,
    /,
    *,
    bint raw=False,
    out=None,
):
    """Return decoded DEFLATE, ZLIB, or GZIP data.

    The format is auto-detected unless raw=True.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        inflate_state state
        int crc_flag = ISAL_ZLIB
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if raw:
        # raw deflate: no wrapper, caller must know the format
        crc_flag = ISAL_DEFLATE
    elif srcsize >= 2 and src[0] == 0x1F and src[1] == 0x8B:
        crc_flag = ISAL_GZIP

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            if crc_flag == ISAL_GZIP and srcsize >= 18:
                # use trailing 4-byte size field (little-endian, mod 2^32)
                dstsize = (
                    (src[srcsize - 4] << 0) |
                    (src[srcsize - 3] << 8) |
                    (src[srcsize - 2] << 16) |
                    (src[srcsize - 1] << 24)
                )
                if dstsize <= 0:
                    return _isal_decode(src, outtype, crc_flag)
            else:
                return _isal_decode(src, outtype, crc_flag)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        isal_inflate_init(&state)
        state.crc_flag = crc_flag
        state.next_in = <uint8_t*> src._data
        state.avail_in = <uint32_t> srcsize
        state.next_out = <uint8_t*> dst._data
        state.avail_out = <uint32_t> dstsize
        ret = isal_inflate_stateless(&state)

    if state.block_state != ISAL_BLOCK_FINISH:
        raise IsalError('isal_inflate_stateless', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> state.total_out, outgiven)


# CRC #########################################################################

def isal_crc32(
    data,
    /,
    value=None,
):
    """Return CRC32 (gzip/zlib reflected) checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        uint64_t srcsize = <uint64_t> src.shape[0]
        uint32_t crc = 0 if value is None else value

    with nogil:
        crc = crc32_gzip_refl(crc, <const unsigned char*> src._data, srcsize)
    return int(crc)


def isal_crc32c(
    data,
    /,
    value=None,
):
    """Return CRC32C (Castagnoli/iSCSI) checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        int srcsize = <int> src.shape[0]
        unsigned int crc = 0 if value is None else value

    with nogil:
        crc = crc32_iscsi(
            <unsigned char*> src._data, srcsize, crc
        )
    return int(crc)


def isal_adler32(
    data,
    /,
    value=None,
):
    """Return Adler-32 checksum of data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        uint64_t srcsize = <uint64_t> src.shape[0]
        uint32_t adler = 1 if value is None else value

    with nogil:
        adler = isal_adler32_c(
            adler, <const unsigned char*> src._data, srcsize
        )
    return int(adler)


# Streaming inflate ###########################################################

def _isal_decode(const uint8_t[::1] src, outtype, int crc_flag):
    """Decompress using ISA-L streaming inflate API."""
    cdef:
        output_t* output = NULL
        inflate_state state
        size_t srcsize = <size_t> src.shape[0]
        size_t incsize = _align_size_t(srcsize // 2)
        size_t size, left
        int ret = ISAL_DECOMP_OK

    try:
        with nogil:
            isal_inflate_init(&state)
            state.crc_flag = crc_flag
            state.next_in = <uint8_t*> src._data
            state.avail_in = 0

            if incsize > 268435456:  # 256 MB
                incsize = 268435456
            output = output_new(NULL, 3 * incsize)  # 3/2 srcsize
            if output == NULL:
                raise MemoryError('output_new failed')

            state.next_out = <uint8_t*> output.data
            state.avail_out = 0
            left = <size_t> output.size
            size = srcsize

            while ret == ISAL_DECOMP_OK or ret == ISAL_OUT_OVERFLOW:

                if state.avail_in == 0 and size > 0:
                    # feed next input chunk
                    if size > <size_t> UINT32_MAX:
                        state.avail_in = <uint32_t> UINT32_MAX
                    else:
                        state.avail_in = <uint32_t> size
                    size -= state.avail_in

                if state.avail_out == 0:
                    if left == 0:
                        # grow output buffer
                        left = incsize
                        if output.size > SIZE_MAX - left:
                            raise MemoryError('output buffer size overflow')
                        if output_resize(output, output.size + left) == 0:
                            raise MemoryError('output_resize failed')
                        state.next_out = (
                            <uint8_t*> output.data + (output.size - left)
                        )
                    if left > <size_t> UINT32_MAX:
                        state.avail_out = <uint32_t> UINT32_MAX
                    else:
                        state.avail_out = <uint32_t> left
                    left -= state.avail_out

                ret = isal_inflate(&state)

                if (
                    ret == ISAL_DECOMP_OK
                    and state.avail_in == 0
                    and size == 0
                    and state.block_state != ISAL_BLOCK_FINISH
                ):
                    raise IsalError('isal_inflate', 'truncated input')

                if state.block_state == ISAL_BLOCK_FINISH:
                    break

            if state.block_state != ISAL_BLOCK_FINISH:
                raise IsalError('isal_inflate', ret)

        out = _create_output(
            outtype,
            state.next_out - <uint8_t*> output.data,
            <const char*> output.data
        )

    finally:
        output_del(output)

    return out


cdef ssize_t _isal_compress_bound(ssize_t srcsize) noexcept nogil:
    """Return upper bound on compressed size (deflate payload only)."""
    return srcsize + (srcsize >> 12) + (srcsize >> 14) + (srcsize >> 25) + 13


# Output stream ###############################################################

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
