# imagecodecs/_lz4.pyx
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

"""LZ4 codec for the imagecodecs package."""

include '_shared.pxi'

from lz4 cimport *


class LZ4:
    """LZ4 codec constants."""

    available = True

    class CLEVEL(enum.IntEnum):
        """LZ4 codec compression levels."""

        DEFAULT = LZ4HC_CLEVEL_DEFAULT
        MIN = LZ4HC_CLEVEL_MIN
        MAX = LZ4HC_CLEVEL_MAX
        OPT_MIN = LZ4HC_CLEVEL_OPT_MIN


class Lz4Error(RuntimeError):
    """LZ4 codec exceptions."""


def lz4_version():
    """Return LZ4 library version string."""
    return 'lz4 {}.{}.{}'.format(
        LZ4_VERSION_MAJOR, LZ4_VERSION_MINOR, LZ4_VERSION_RELEASE
    )


def lz4_check(data):
    """Return whether data is LZ4 encoded."""


def lz4_encode(
    data, level=None, hc=False, header=False, out=None
):
    """Return LZ4 encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = <int> src.size
        int dstsize
        int offset = 4 if header else 0
        int ret
        uint8_t* pdst
        int acceleration, compressionlevel

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.size > LZ4_MAX_INPUT_SIZE:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4_compressBound(srcsize) + offset
            if dstsize < 0:
                raise Lz4Error(f'LZ4_compressBound returned {dstsize}')
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int> dst.size - offset

    if dst.size > 2147483647:
        raise ValueError('output too large')

    if hc:
        compressionlevel = _default_value(
            level, LZ4HC_CLEVEL_DEFAULT, LZ4HC_CLEVEL_MIN, LZ4HC_CLEVEL_MAX
        )
        with nogil:
            ret = LZ4_compress_HC(
                <const char*> &src[0],
                <char*> &dst[offset],
                srcsize,
                dstsize,
                compressionlevel
            )
            if ret <= 0:
                raise Lz4Error(f'LZ4_compress_HC returned {ret}')

    else:
        acceleration = _default_value(level, 1, 1, 65537)
        with nogil:
            ret = LZ4_compress_fast(
                <const char*> &src[0],
                <char*> &dst[offset],
                srcsize,
                dstsize,
                acceleration
            )
            if ret <= 0:
                raise Lz4Error(f'LZ4_compress_fast returned {ret}')

    if header:
        pdst = <uint8_t*> &dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize+offset, ret+offset, outgiven)


def lz4_decode(data, header=False, out=None):
    """Return decoded LZ4 data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = <int> src.size
        int dstsize
        int offset = 4 if header else 0
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if src.size > 2147483647:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None:
        if dstsize < 0:
            # TODO: use streaming API
            dstsize = max(24, 24 + 255 * (srcsize - offset - 10))  # ugh
            # if dstsize < 0:
            #     raise Lz4Error(f'invalid output size {dstsize}')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int> dst.size

    if dst.size > 2147483647:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_decompress_safe(
            <char*> &src[offset],
            <char*> &dst[0],
            srcsize - offset,
            dstsize
        )
    if ret < 0:
        raise Lz4Error(f'LZ4_decompress_safe returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)

###############################################################################

# LZ4H5 implements H5Z_FILTER_LZ4
# https://support.hdfgroup.org/services/filters/HDF5_LZ4.pdf
#
# file header: orisize: >i8,  blksize: >i4
# block: lz4size: >i4, data: bytes(lz4size)


class LZ4H5:
    """LZ4H5 codec constants."""

    available = True

    CLEVEL = LZ4.CLEVEL


class Lz4h5Error(RuntimeError):
    """LZ4H5 codec exceptions."""


lz4h5_version = lz4_version


def lz4h5_check(data):
    """Return whether data is likely LZ4H5 encoded."""
    if len(data) < 12:
        return False
    return None


def lz4h5_encode(
    data, level=None, blocksize=None, out=None
):
    """Return LZ4H5 encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstpos = 12
        ssize_t srcpos = 0
        ssize_t nblocks
        ssize_t blksize
        int lz4size
        int acceleration = _default_value(level, 1, 1, 65537)

    if data is out:
        raise ValueError('cannot encode in-place')

    if blocksize is None:
        blksize = min(<ssize_t> 1073741823, max(srcsize, 1))
    elif 0 < blocksize <= LZ4_MAX_INPUT_SIZE:
        blksize = blocksize
    else:
        raise ValueError('invalid block size {blocksize}')

    nblocks = max((srcsize - 1) // blksize + 1, 1)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4_compressBound(<int> blksize)
            if dstsize < 0:
                raise Lz4h5Error(f'LZ4_compressBound returned {dstsize}')
            dstsize = nblocks * dstsize + nblocks * 4 + 12
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    if dstsize < nblocks * 4 + 12:
        raise Lz4h5Error(f'output too small {dstsize} < {nblocks}*4+12')

    with nogil:
        write_i8be(<uint8_t*> &dst[0], <uint64_t> srcsize)
        write_i4be(<uint8_t*> &dst[8], <uint32_t> blksize)

        if srcsize == 0:
            write_i4be(<uint8_t*> &dst[dstpos], <uint32_t> 0)
            dstpos += 4

        while srcpos < srcsize and dstpos < dstsize:
            dstpos += 4
            blksize = min(blksize, srcsize - srcpos)
            lz4size = LZ4_compress_fast(
                <const char*> &src[srcpos],
                <char*> &dst[dstpos],
                <int> blksize,
                <int> (dstsize - dstpos),
                acceleration
            )
            if (
                # compression succeeded, but no space savings
                lz4size >= blksize
                or (
                    # compression failed, not enough output space
                    lz4size <= 0
                    and blksize == srcsize - srcpos
                    and blksize <= dstsize - dstpos
                )
            ):
                memcpy(&dst[dstpos], <const char*> &src[srcpos], blksize)
                lz4size = blksize
            if lz4size <= 0:
                raise Lz4h5Error(f'LZ4_compress_fast returned {lz4size}')

            write_i4be(<uint8_t*> &dst[dstpos - 4], <uint32_t> lz4size)
            srcpos += blksize
            dstpos += lz4size

    del dst
    return _return_output(out, dstsize, dstpos, outgiven)


def lz4h5_decode(data, out=None):
    """Return decoded LZ4H5 data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t orisize
        ssize_t srcpos = 12
        ssize_t dstpos = 0
        int lz4size
        int blksize
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if src.size < 16:
        raise Lz4h5Error(f'LZ4H5 data too short {src.size} < 16')

    orisize = <ssize_t> read_i8be(&src[0])
    blksize = <int> min(read_i4be(&src[8]), orisize)

    if orisize < 0 or blksize < 0 or blksize > LZ4_MAX_INPUT_SIZE:
        raise Lz4h5Error('invalid values in LZ4H5 header')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = orisize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dstsize < orisize:
        raise Lz4h5Error(
            f'output size does not match header {dstsize} != {orisize}'
        )

    with nogil:
        while srcpos < srcsize and dstpos < dstsize:
            lz4size = <int> read_i4be(&src[srcpos])
            srcpos += 4

            if lz4size == 0:
                break
            if lz4size < 0 or srcpos + lz4size > srcsize:
                raise Lz4h5Error(
                    f'invalid block size {lz4size} @{srcpos} of {srcsize}'
                )

            blksize = <int> min(
                <int64_t> blksize, <int64_t> (dstsize - dstpos)
            )

            if blksize == lz4size:
                memcpy(
                    <char*> &dst[dstpos],
                    <char*> &src[srcpos],
                    <size_t> blksize
                )
                ret = blksize

            else:
                ret = LZ4_decompress_safe(
                    <char*> &src[srcpos],
                    <char*> &dst[dstpos],
                    lz4size,
                    blksize
                )
                if ret < 0:
                    raise Lz4h5Error(f'LZ4_decompress_safe returned {ret}')

            srcpos += lz4size
            dstpos += ret

    del dst
    return _return_output(out, dstsize, dstpos, outgiven)


cdef inline uint32_t read_i4be(const uint8_t* psrc) noexcept nogil:
    # read >i4 to value
    return (
        ((<uint32_t> psrc[0]) << 24) |
        ((<uint32_t> psrc[1]) << 16) |
        ((<uint32_t> psrc[2]) << 8) |
        (<uint32_t> psrc[3])
    )


cdef inline uint64_t read_i8be(const uint8_t* psrc) noexcept nogil:
    # read >i8 to value
    return (
        ((<uint64_t> psrc[0]) << 56) |
        ((<uint64_t> psrc[1]) << 48) |
        ((<uint64_t> psrc[2]) << 40) |
        ((<uint64_t> psrc[3]) << 32) |
        ((<uint64_t> psrc[4]) << 24) |
        ((<uint64_t> psrc[5]) << 16) |
        ((<uint64_t> psrc[6]) << 8) |
        (<uint64_t> psrc[7])
    )


cdef inline void write_i4be(uint8_t* pdst, uint32_t value) noexcept nogil:
    # write >i4 to pdst
    pdst[0] = (value >> 24) & 255
    pdst[1] = (value >> 16) & 255
    pdst[2] = (value >> 8) & 255
    pdst[3] = value & 255


cdef inline void write_i8be(uint8_t* pdst, uint64_t value) noexcept nogil:
    # write >i8 to pdst
    pdst[0] = (value >> 56) & 255
    pdst[1] = (value >> 48) & 255
    pdst[2] = (value >> 40) & 255
    pdst[3] = (value >> 32) & 255
    pdst[4] = (value >> 24) & 255
    pdst[5] = (value >> 16) & 255
    pdst[6] = (value >> 8) & 255
    pdst[7] = value & 255
