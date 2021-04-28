# imagecodecs/_pglz.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021, Christoph Gohlke
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

"""PGLZ (PostgreSQL LZ) codec for the imagecodecs package."""

__version__ = '2021.4.28'

include '_shared.pxi'

from pg_lzcompress cimport *


class PGLZ:
    """PGLZ Constants."""


class PglzError(RuntimeError):
    """PGLZ Exceptions."""


def pglz_version():
    """Return PostgreSQL library version string."""
    return f'pg_lzcompress {PG_LZCOMPRESS_VERSION.decode()}'


def pglz_check(data):
    """Return True if data likely contains LZF data."""


def pglz_encode(data, level=None, header=False, strategy=None, out=None):
    """Compress PGLZ.

    Raise PglzError if pglz_compress is unable to significantly compress
    the data and no header is used.

    Raise ValueError if output buffer is smaller than len(data) + 4.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        int32 ret
        uint8_t* pdst
        ssize_t offset = 4 if header else 0
        PGLZ_Strategy* pglz_strategy = <PGLZ_Strategy*> PGLZ_strategy_default
        PGLZ_Strategy custom_strategy

    if data is out:
        raise ValueError('cannot encode in-place')

    if strategy is None:
        pass
    elif strategy in ('always', 'ALWAYS'):
        pglz_strategy = <PGLZ_Strategy*> PGLZ_strategy_always
    elif isinstance(strategy, (list, tuple)):
        pglz_strategy = &custom_strategy
        custom_strategy.min_input_size = strategy[0]
        custom_strategy.max_input_size = strategy[1]
        custom_strategy.min_comp_rate = strategy[2]
        custom_strategy.first_success_by = strategy[3]
        custom_strategy.match_size_good = strategy[4]
        custom_strategy.match_size_drop = strategy[5]

    if srcsize >= 2 ** 31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = PGLZ_MAX_OUTPUT(srcsize) + offset
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size - offset

    if dst.size >= 2 ** 31:
        raise ValueError('output too large')

    if dstsize < PGLZ_MAX_OUTPUT(srcsize):
        raise ValueError('output too small')

    with nogil:
        ret = pglz_compress(
            <const char*> &src[0],
            <int32> srcsize,
            <char*> &dst[offset],
            pglz_strategy
        )
    if header:
        pdst = <uint8_t*> &dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255
        if ret < 0:
            # copy uncompressed
            if srcsize > dstsize:
                raise ValueError('output too small')
            memcpy(<void*> &dst[offset], &src[0], srcsize)
            ret = srcsize
    elif ret < 0:
        raise PglzError(f'pglz_compress returned {ret}')

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def pglz_decode(data, header=False, checkcomplete=None, out=None):
    """Decompress PGLZ.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        ssize_t rawsize = 0
        int32 ret
        bint check_complete = bool(checkcomplete)  # allow partial results
        ssize_t offset = 4 if header else 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize >= 2 ** 31:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header:
        if srcsize < offset:
            raise ValueError('invalid data size')
        rawsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)
        if dstsize < 0:
            dstsize = rawsize
            if checkcomplete is None:
                check_complete = True

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 4  # TODO
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dst.size >= 2 ** 31:
        raise ValueError('output too large')

    if header and srcsize == offset + rawsize:
        # copy uncompressed
        if rawsize > dstsize:
            raise ValueError('output too small')
        memcpy(<void*> &dst[0], &src[offset], rawsize)
        ret = rawsize

    else:
        # decompress
        with nogil:
            ret = pglz_decompress(
                <const char*> &src[offset],
                <int32> (srcsize - offset),
                <char*> &dst[0],
                <int32> dstsize,
                check_complete
            )
        if ret < 0:
            raise PglzError(f'pglz_decompress returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)
