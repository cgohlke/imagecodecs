# imagecodecs/_blosc.pyx
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

"""BLOSC codec for the imagecodecs package."""

include '_shared.pxi'

from blosc cimport *


class BLOSC:
    """BLOSC codec constants."""

    available = True

    class SHUFFLE(enum.IntEnum):
        """BLOSC codec shuffle types."""

        NOSHUFFLE = BLOSC_NOSHUFFLE
        SHUFFLE = BLOSC_SHUFFLE
        BITSHUFFLE = BLOSC_BITSHUFFLE

    class COMPRESSOR(enum.IntEnum):
        """BLOSC codec compressors."""

        BLOSCLZ = BLOSC_BLOSCLZ
        LZ4 = BLOSC_LZ4
        LZ4HC = BLOSC_LZ4HC
        SNAPPY = BLOSC_SNAPPY
        ZLIB = BLOSC_ZLIB
        ZSTD = BLOSC_ZSTD


class BloscError(RuntimeError):
    """BLOSC codec exceptions."""


def blosc_version():
    """Return C-Blosc library version string."""
    return 'c-blosc ' + BLOSC_VERSION_STRING.decode()


def blosc_check(const uint8_t[::1] data not None, /):
    """Return whether data is BLOSC encoded or None if unknown."""
    return None if data.shape[0] >= BLOSC_MIN_HEADER_LENGTH else False


def blosc_encode(
    data,
    /,
    level=None,
    *,
    compressor=None,
    shuffle=None,
    typesize=None,
    blocksize=None,
    numthreads=None,
    out=None,
):
    """Return BLOSC encoded data."""
    cdef:
        ssize_t itemsize
        const uint8_t[::1] src = _readable_input(data, out, &itemsize)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t cblocksize = 0 if blocksize is None else blocksize
        size_t ctypesize = itemsize if typesize is None else typesize
        const char* compname = NULL
        int clevel = _default_value(level, 9, 0, 9)
        int numinternalthreads = _default_threads(numthreads)
        int compcode = _enum_value(
            compressor, BLOSC.COMPRESSOR, BLOSC.COMPRESSOR.BLOSCLZ
        )
        int doshuffle = _enum_value(
            shuffle, BLOSC.SHUFFLE, BLOSC.SHUFFLE.SHUFFLE
        )
        int ret

    if ctypesize == 0:
        raise ValueError('invalid typesize 0')

    if blosc_compcode_to_compname(compcode, &compname) < 0:
        raise ValueError(f'compressor {compressor!r} not supported')

    if srcsize > INT32_MAX - BLOSC_MAX_OVERHEAD:
        raise ValueError('data size larger than 2 GB')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + BLOSC_MAX_OVERHEAD
        if dstsize < 17:
            dstsize = 17
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]
    if dstsize > INT32_MAX:
        dstsize = INT32_MAX  # c-blosc stores destsize as int32

    with nogil:
        if numinternalthreads == 0:
            numinternalthreads = blosc_get_nthreads()

        ret = blosc_compress_ctx(
            clevel,
            doshuffle,
            ctypesize,
            <size_t> srcsize,
            <const void*> src._data,
            <void*> dst._data,
            <size_t> dstsize,
            <const char*> compname,
            cblocksize,
            numinternalthreads
        )
    if ret <= 0:
        raise BloscError(f'blosc_compress_ctx returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc_decode(
    data,
    /,
    *,
    numthreads=None,
    out=None,
):
    """Return decoded BLOSC data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t nbytes, cbytes, blocksize
        int numinternalthreads = _default_threads(numthreads)
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize > INT32_MAX:
        raise ValueError('data size larger than 2 GB')

    if srcsize < BLOSC_MIN_HEADER_LENGTH:
        raise BloscError('invalid blosc data: header too short')

    # blosc_decompress_ctx trusts the compressed size in the header
    blosc_cbuffer_sizes(
        <const void*> src._data,
        &nbytes,
        &cbytes,
        &blocksize
    )
    if (
        (nbytes == 0 and blocksize == 0)
        or nbytes > <size_t> INT32_MAX  # negative int32 read as size_t
        or cbytes < <size_t> BLOSC_MIN_HEADER_LENGTH
        or cbytes > <size_t> srcsize
    ):
        raise BloscError('blosc_cbuffer_sizes returned invalid blosc data')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = <ssize_t> nbytes
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]
    if dstsize > INT32_MAX:
        raise ValueError('output size larger than 2 GB')

    with nogil:
        if numinternalthreads == 0:
            numinternalthreads = blosc_get_nthreads()

        ret = blosc_decompress_ctx(
            <const void*> src._data,
            <void*> dst._data,
            dstsize,
            numinternalthreads
        )
    if ret < 0:
        raise BloscError(f'blosc_decompress_ctx returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc_info(const uint8_t[::1] data not None, /):
    """Return information about BLOSC data."""
    cdef:
        const void* src = <const void*> data._data
        ssize_t srcsize = data.shape[0]
        size_t nbytes, cbytes, blocksize, typesize
        int flags, version, compversion

    if srcsize < BLOSC_MIN_HEADER_LENGTH:
        raise BloscError('invalid blosc data: header too short')

    blosc_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize)
    blosc_cbuffer_metainfo(src, &typesize, &flags)
    blosc_cbuffer_versions(src, &version, &compversion)

    if flags & 0x4:  # BLOSC_DOBITSHUFFLE
        shuffle = BLOSC.SHUFFLE.BITSHUFFLE
    elif flags & 0x1:  # BLOSC_DOSHUFFLE
        shuffle = BLOSC.SHUFFLE.SHUFFLE
    else:
        shuffle = BLOSC.SHUFFLE.NOSHUFFLE

    # compressor format in upper flag bits; LZ4HC shares the LZ4 format
    compressor = {
        0: BLOSC.COMPRESSOR.BLOSCLZ,
        1: BLOSC.COMPRESSOR.LZ4,
        2: BLOSC.COMPRESSOR.SNAPPY,
        3: BLOSC.COMPRESSOR.ZLIB,
        4: BLOSC.COMPRESSOR.ZSTD,
    }.get(data[2] >> 5, data[2] >> 5)

    return {
        '_codec': 'blosc',
        'nbytes': nbytes,
        'cbytes': cbytes,
        'blocksize': blocksize,
        'typesize': typesize,
        'compressor': compressor,
        'shuffle': shuffle,
        'memcpyed': bool(flags & 0x2),  # BLOSC_MEMCPYED
        'version': version,
        'compversion': compversion,
    }
