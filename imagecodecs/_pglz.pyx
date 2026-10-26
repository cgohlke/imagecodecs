# imagecodecs/_pglz.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2021-2026, Christoph Gohlke
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

include '_shared.pxi'

cimport cython
from pg_lzcompress cimport *


cdef cython.pymutex global_lock


class PGLZ:
    """PGLZ codec constants."""

    available = True


class PglzError(RuntimeError):
    """PGLZ codec exceptions."""


def pglz_version():
    """Return PostgreSQL library version string."""
    return f'pg_lzcompress {PG_LZCOMPRESS_VERSION.decode()}'


def pglz_check(const uint8_t[::1] data not None, /):
    """Return whether data is PGLZ encoded or None if unknown."""
    return None if data.shape[0] > 0 else False


def pglz_encode(
    data,
    /,
    *,
    header=False,
    strategy=None,
    out=None,
):
    """Return PGLZ encoded data.

    Raise PglzError if pglz_compress is unable to significantly compress
    the data and no header is used. With header, incompressible data are
    stored uncompressed.

    Raise ValueError if the output buffer is smaller than len(data) + 4,
    plus 4 bytes for the header.

    strategy:
        'default', 'always', or a sequence of six integers
        (min_input_size, max_input_size, min_comp_rate, first_success_by,
        match_size_good, match_size_drop).

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        int32 ret
        uint8_t* pdst
        ssize_t offset = 4 if header else 0
        PGLZ_Strategy* pglz_strategy = <PGLZ_Strategy*> PGLZ_strategy_default
        PGLZ_Strategy custom_strategy

    if strategy is None:
        pass
    elif isinstance(strategy, str):
        if strategy.lower() == 'always':
            pglz_strategy = <PGLZ_Strategy*> PGLZ_strategy_always
        elif strategy.lower() != 'default':
            raise ValueError(f'invalid {strategy=!r}')
    else:
        try:
            values = tuple(int(v) for v in strategy)
        except TypeError as exc:
            raise ValueError(f'invalid {strategy=!r}') from exc
        if len(values) != 6 or min(values) < 0:
            raise ValueError(f'invalid {strategy=!r}')
        pglz_strategy = &custom_strategy
        custom_strategy.min_input_size = values[0]
        custom_strategy.max_input_size = values[1]
        custom_strategy.min_comp_rate = values[2]
        custom_strategy.first_success_by = values[3]
        custom_strategy.match_size_good = values[4]
        custom_strategy.match_size_drop = values[5]

    if srcsize > INT32_MAX:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = PGLZ_MAX_OUTPUT(srcsize) + offset
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0] - offset

    if dst.shape[0] > INT32_MAX:
        raise ValueError('output too large')

    if dstsize < PGLZ_MAX_OUTPUT(srcsize):
        raise ValueError('output too small')

    with nogil:
        pdst = <uint8_t*> dst._data
        with global_lock:
            ret = pglz_compress(
                <const char*> src._data,
                <int32> srcsize,
                <char*> (pdst + offset),
                pglz_strategy
            )

        if offset > 0:
            pdst[0] = srcsize & 255
            pdst[1] = (srcsize >> 8) & 255
            pdst[2] = (srcsize >> 16) & 255
            pdst[3] = (srcsize >> 24) & 255
            if ret < 0:
                # store uncompressed
                memcpy(
                    <void*> (pdst + offset), <const void*> src._data, srcsize
                )
                ret = <int32> srcsize

    if ret < 0:
        raise PglzError(
            'pglz_compress failed to compress data of size '
            f'{srcsize} with the requested strategy'
        )

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def pglz_decode(
    data,
    /,
    *,
    header=False,
    checkcomplete=None,
    out=None,
):
    """Return decoded PGLZ data.

    checkcomplete:
        Require that the data decode exactly to the output size.
        By default, True unless `out` is specified.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t* psrc = &src[0] if src.shape[0] > 0 else NULL
        ssize_t dstsize
        ssize_t srcsize = src.shape[0]
        ssize_t rawsize = -1
        int64 decodedsize
        int32 ret
        bint check_complete
        ssize_t offset = 4 if header else 0

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize > INT32_MAX:
        raise ValueError('data too large')

    check_complete = out is None if checkcomplete is None else checkcomplete

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header:
        if srcsize < offset:
            raise ValueError('invalid data size')
        rawsize = <ssize_t> (
            <uint32_t> src[0]
            | (<uint32_t> src[1] << 8)
            | (<uint32_t> src[2] << 16)
            | (<uint32_t> src[3] << 24)
        )

    stored = header and srcsize == offset + rawsize

    if out is None and dstsize < 0:
        if stored:
            dstsize = rawsize
        else:
            with nogil:
                decodedsize = pglz_decompressed_size(
                    <const char*> (psrc + offset), <int32> (srcsize - offset)
                )
            if decodedsize < 0:
                raise PglzError('corrupted PGLZ data')
            if header and decodedsize != rawsize:
                raise PglzError(
                    f'decoded size {decodedsize} != header size {rawsize}'
                )
            dstsize = <ssize_t> decodedsize
        if dstsize > INT32_MAX:
            raise ValueError('output too large')

    if out is None:
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]

    if dstsize > INT32_MAX:
        raise ValueError('output too large')

    if stored:
        if rawsize > dstsize:
            raise ValueError('output too small')
        if rawsize > 0:
            with nogil:
                memcpy(
                    <void*> dst._data, <const void*> (psrc + offset), rawsize
                )
        ret = <int32> rawsize

    else:
        with nogil:
            ret = pglz_decompress(
                <const char*> (psrc + offset),
                <int32> (srcsize - offset),
                <char*> dst._data,
                <int32> dstsize,
                check_complete
            )
        if ret < 0:
            raise PglzError(
                'pglz_decompress failed: corrupted data'
                + (' or output size mismatch' if check_complete else '')
            )

    del dst
    return _return_output(out, dstsize, ret, outgiven)
