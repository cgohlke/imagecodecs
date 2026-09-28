# imagecodecs/_lz4.pyx
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

"""LZ4 (Lempel-Ziv version 4) codecs for the imagecodecs package.

- LZ4: raw block compression
- LZ4F: framed format with header, checksums, content-size, and block settings
- LZ4H5: implements H5Z_FILTER_LZ4

"""

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


def lz4_check(const uint8_t[::1] data, /):
    """Return whether data is LZ4 encoded or None if unknown."""
    return None if data.shape[0] > 0 else False


def lz4_encode(
    data,
    /,
    level=None,
    *,
    hc=False,
    header=False,
    out=None,
):
    """Return LZ4 encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = <int> src.shape[0]
        int dstsize
        int offset = 4 if header else 0
        int ret
        uint8_t* pdst
        int acceleration, compressionlevel

    if src.shape[0] > LZ4_MAX_INPUT_SIZE:
        raise ValueError('data too large')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4_compressBound(srcsize)
            if dstsize < 0:
                raise Lz4Error(f'LZ4_compressBound returned {dstsize}')
            dstsize += offset
        if dstsize < offset:
            dstsize = offset
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <int> dst.shape[0] - offset

    if dst.shape[0] > INT32_MAX:
        raise ValueError('output too large')

    if hc:
        compressionlevel = _default_value(
            level, LZ4HC_CLEVEL_DEFAULT, LZ4HC_CLEVEL_MIN, LZ4HC_CLEVEL_MAX
        )
        with nogil:
            ret = LZ4_compress_HC(
                <const char*> src._data,
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
                <const char*> src._data,
                <char*> &dst[offset],
                srcsize,
                dstsize,
                acceleration
            )
            if ret <= 0:
                raise Lz4Error(f'LZ4_compress_fast returned {ret}')

    if header:
        pdst = <uint8_t*> dst._data
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    del dst
    return _return_output(out, dstsize + offset, ret + offset, outgiven)


def lz4_decode(
    data,
    /,
    *,
    header=False,
    out=None,
):
    """Return decoded LZ4 data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        int srcsize = <int> src.shape[0]
        int dstsize
        int offset = 4 if header else 0
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if src.shape[0] > INT32_MAX:
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
    dstsize = <int> dst.shape[0]

    if dst.shape[0] > INT32_MAX:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_decompress_safe(
            <char*> &src[offset],
            <char*> dst._data,
            srcsize - offset,
            dstsize
        )
    if ret < 0:
        raise Lz4Error(f'LZ4_decompress_safe returned {ret}')

    del dst
    return _return_output(out, dstsize, ret, outgiven)


# LZ4F ########################################################################


class LZ4F:
    """LZ4F codec constants."""

    available = True

    VERSION = LZ4F_VERSION


class Lz4fError(RuntimeError):
    """LZ4F codec exceptions."""

    def __init__(self, func, err):
        cdef:
            char* errormessage
            LZ4F_errorCode_t errorcode

        try:
            errorcode = <LZ4F_errorCode_t> err
            errormessage = <char*> LZ4F_getErrorName(errorcode)
            if errormessage == NULL:
                raise RuntimeError('LZ4F_getErrorName returned NULL')
            msg = errormessage.decode().strip()
        except Exception:
            msg = 'NULL' if err is None else f'unknown error {err!r}'
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def lz4f_version():
    """Return LZ4 library version string."""
    return 'lz4 {}.{}.{}'.format(
        LZ4_VERSION_MAJOR, LZ4_VERSION_MINOR, LZ4_VERSION_RELEASE
    )


def lz4f_check(const uint8_t[::1] data, /):
    """Return whether data is LZ4F encoded or None if unknown."""
    cdef:
        bytes sig = bytes(data[:4])

    return sig == b'\x04\x22\x4d\x18'  # LZ4_MAGIC_NUMBER 0x184d2204


def lz4f_encode(
    data,
    /,
    level=None,
    *,
    blocksizeid=None,
    blockchecksum=None,
    contentchecksum=None,
    contentsize=None,
    out=None,
):
    """Return LZ4F encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t srcsize = <size_t> src.shape[0]
        ssize_t dstsize
        size_t ret
        LZ4F_preferences_t prefs

    memset(<void*> &prefs, 0, sizeof(LZ4F_preferences_t))
    if contentsize is None or contentsize:
        prefs.frameInfo.contentSize = srcsize
    if level is not None:
        prefs.compressionLevel = _default_value(level, 0, -1, LZ4HC_CLEVEL_MAX)
    if blocksizeid is not None:
        prefs.frameInfo.blockSizeID = <LZ4F_blockSizeID_t> blocksizeid
    if blockchecksum:
        prefs.frameInfo.blockChecksumFlag = LZ4F_blockChecksumEnabled
    if contentchecksum:
        prefs.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4F_compressFrameBound(srcsize, &prefs)
            if dstsize < 0:
                raise Lz4fError(f'LZ4F_compressFrameBound returned {dstsize}')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        ret = LZ4F_compressFrame(
            <void*> dst._data,
            <size_t> dstsize,
            <void*> src._data,
            srcsize,
            &prefs
        )
        if LZ4F_isError(<LZ4F_errorCode_t> ret):
            raise Lz4fError('LZ4F_compressFrame', <LZ4F_errorCode_t> ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def lz4f_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded LZ4F data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* outbuf = NULL
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        size_t byteswritten, bytesread, ret, blksize
        size_t srcindex = 0
        LZ4F_dctx* dctx = NULL
        LZ4F_decompressOptions_t* options = NULL
        LZ4F_frameInfo_t frameinfo

    if data is out:
        raise ValueError('cannot transform in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            try:
                # read content size from frame header
                ret = LZ4F_headerSize(
                    <const void*> src._data, <size_t> srcsize
                )
                if LZ4F_isError(<LZ4F_errorCode_t> ret):
                    raise Lz4fError('LZ4F_headerSize', ret)
                if ret > <size_t> srcsize:
                    raise Lz4fError('LZ4F_headerSize', 'invalid input')
                srcindex = ret
                ret = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION)
                if LZ4F_isError(<LZ4F_errorCode_t> ret):
                    raise Lz4fError('LZ4F_createDecompressionContext', ret)
                ret = LZ4F_getFrameInfo(
                    dctx,
                    &frameinfo,
                    <const void*> src._data,
                    &srcindex
                )
                if LZ4F_isError(<LZ4F_errorCode_t> ret):
                    raise Lz4fError('LZ4F_getFrameInfo', ret)
                if frameinfo.contentSize > 0:
                    dstsize = <ssize_t> frameinfo.contentSize
                else:
                    # content size not in header
                    with nogil:
                        blksize = _lz4f_blocksize(frameinfo.blockSizeID)
                        ret = _lz4f_decode_chunked(
                            dctx,
                            <const uint8_t*> src._data,
                            <size_t> srcsize,
                            srcindex,
                            &outbuf,
                            blksize
                        )
                        LZ4F_freeDecompressionContext(dctx)
                        dctx = NULL

                    if LZ4F_isError(<LZ4F_errorCode_t> ret):
                        if ret == SIZE_MAX:
                            raise MemoryError('Failed to allocatedecode chunk')
                        raise Lz4fError('LZ4F_decompress', ret)
                    out = _create_output(
                        outtype, <ssize_t> ret, <const char*> outbuf
                    )
                    free(outbuf)
                    return out
            except Exception:
                if outbuf != NULL:
                    free(outbuf)
                LZ4F_freeDecompressionContext(dctx)
                raise

        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]
    byteswritten = <size_t> dstsize
    bytesread = <size_t> srcsize - srcindex

    try:
        with nogil:
            if dctx == NULL:
                ret = LZ4F_createDecompressionContext(&dctx, LZ4F_VERSION)
                if LZ4F_isError(<LZ4F_errorCode_t> ret):
                    raise Lz4fError('LZ4F_createDecompressionContext', ret)

            ret = LZ4F_decompress(
                dctx,
                <void*> dst._data,
                &byteswritten,
                <void*> &src[srcindex],
                &bytesread,
                options
            )
            if LZ4F_isError(<LZ4F_errorCode_t> ret):
                raise Lz4fError('LZ4F_decompress', <LZ4F_errorCode_t> ret)
    finally:
        LZ4F_freeDecompressionContext(dctx)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


cdef inline size_t _lz4f_blocksize(
    LZ4F_blockSizeID_t blocksizeid
) noexcept nogil:
    # Return block size in bytes for given block size ID
    # replaces static LZ4F_getBlockSize function
    if blocksizeid == LZ4F_default or blocksizeid == LZ4F_max64KB:
        return <size_t> 65536
    if blocksizeid == LZ4F_max256KB:
        return <size_t> 262144
    if blocksizeid == LZ4F_max1MB:
        return <size_t> 1048576
    if blocksizeid == LZ4F_max4MB:
        return <size_t> 4194304
    return <size_t> 65536


cdef size_t _lz4f_decode_chunked(
    LZ4F_dctx* dctx,
    const uint8_t* src,
    size_t srcsize,
    size_t srcindex,
    uint8_t** pdst,
    const size_t blksize
) noexcept nogil:
    """Decode LZ4F compressed data in chunks.

    Return bytes written to ``pdst`` or LZ4F error code on decode error,
    SIZE_MAX on malloc failure.

    """
    cdef:
        size_t cap = blksize * 2
        uint8_t* buf = <uint8_t*> malloc(cap)
        uint8_t* newbuf
        size_t used = 0
        size_t src0, dst0, ret

    pdst[0] = NULL
    if buf == NULL:
        return SIZE_MAX

    while True:
        if cap - used < blksize:
            cap *= 2
            newbuf = <uint8_t*> realloc(buf, cap)
            if newbuf == NULL:
                free(buf)
                return SIZE_MAX
            buf = newbuf
        dst0 = cap - used
        src0 = srcsize - srcindex
        ret = LZ4F_decompress(
            dctx, buf + used, &dst0, src + srcindex, &src0, NULL
        )
        if LZ4F_isError(<LZ4F_errorCode_t> ret):
            free(buf)
            return ret
        used += dst0
        srcindex += src0
        if ret == 0:
            break
        if src0 == 0 and dst0 == 0:
            free(buf)
            return ret

    pdst[0] = buf
    return used


# LZ4H5 #######################################################################

# LZ4H5 implements H5Z_FILTER_LZ4
# https://github.com/HDFGroup/hdf5_plugins/blob/master/LZ4/LZ4_HDF5_format.md
#
# file header: orisize: >i8,  blksize: >i4
# block: lz4size: >i4, data: bytes(lz4size)


class LZ4H5:
    """LZ4H5 codec constants."""

    available = True

    class CLEVEL(enum.IntEnum):
        """LZ4 codec compression levels."""

        DEFAULT = LZ4HC_CLEVEL_DEFAULT
        MIN = LZ4HC_CLEVEL_MIN
        MAX = LZ4HC_CLEVEL_MAX
        OPT_MIN = LZ4HC_CLEVEL_OPT_MIN


class Lz4h5Error(RuntimeError):
    """LZ4H5 codec exceptions."""


lz4h5_version = lz4_version


def lz4h5_check(const uint8_t[::1] data, /):
    """Return whether data is LZ4H5 encoded or None if unknown."""
    return None if data.shape[0] >= 12 else False


def lz4h5_encode(
    data,
    /,
    level=None,
    *,
    blocksize=None,
    out=None,
):
    """Return LZ4H5 encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t dstpos = 12
        ssize_t srcpos = 0
        ssize_t nblocks
        int blksize
        int lz4size, lz4dstsize
        int acceleration = _default_value(level, 1, 1, 65537)

    if blocksize is None:
        blksize = <int> min(1073741823, max(srcsize, 1))
    elif 0 < blocksize <= LZ4_MAX_INPUT_SIZE:
        blksize = blocksize
    else:
        raise ValueError(f'invalid block size {blocksize}')

    nblocks = (srcsize - 1) // blksize + 1

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4_compressBound(blksize)
            if dstsize < 0:
                raise Lz4h5Error(f'LZ4_compressBound returned {dstsize}')
            dstsize = nblocks * dstsize + nblocks * 4 + 12
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]
    if dstsize < nblocks * 4 + 12:
        raise Lz4h5Error(f'output too small {dstsize} < {nblocks}*4+12')

    with nogil:
        write_i8be(<uint8_t*> dst._data, <uint64_t> srcsize)
        write_i4be(<uint8_t*> &dst[8], <uint32_t> blksize)

        while srcpos < srcsize:
            if dstsize - dstpos < 5:
                raise Lz4h5Error('output too small')

            dstpos += 4
            lz4dstsize = <int> min(dstsize - dstpos, INT32_MAX)
            blksize = <int> min(blksize, srcsize - srcpos)
            lz4size = LZ4_compress_fast(
                <const char*> &src[srcpos],
                <char*> &dst[dstpos],
                blksize,
                lz4dstsize,
                acceleration
            )
            if (
                # compression succeeded without space savings
                # or compression failed, likely due to not enough output space
                lz4size >= blksize or (lz4size <= 0 and blksize <= lz4dstsize)
            ):
                memcpy(
                    <void*> &dst[dstpos], <const void*> &src[srcpos], blksize
                )
                lz4size = blksize
            elif lz4size <= 0:
                raise Lz4h5Error(f'LZ4_compress_fast returned {lz4size}')

            write_i4be(<uint8_t*> &dst[dstpos - 4], <uint32_t> lz4size)
            srcpos += blksize
            dstpos += lz4size

    del dst
    return _return_output(out, dstsize, dstpos, outgiven)


def lz4h5_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded LZ4H5 data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t orisize
        ssize_t srcpos = 12
        ssize_t dstpos = 0
        int lz4size
        int blksize
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize < 12:
        raise Lz4h5Error(f'LZ4H5 data too short {srcsize} < 12')

    orisize = <ssize_t> read_i8be(&src[0])
    blksize = <int> read_i4be(&src[8])

    if orisize < 0 or blksize <= 0 or blksize > LZ4_MAX_INPUT_SIZE:
        raise Lz4h5Error(
            f'invalid values in LZ4H5 header: {orisize=}, {blksize=}'
        )

    blksize = <int> min(blksize, orisize)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = orisize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    if dstsize < orisize:
        raise Lz4h5Error(
            f'output size does not match header {dstsize} != {orisize}'
        )

    with nogil:
        while srcpos < srcsize and dstpos < dstsize:
            if srcsize - srcpos < 4:
                raise Lz4h5Error('LZ4H5 data too short')

            lz4size = <int> read_i4be(&src[srcpos])
            srcpos += 4

            if lz4size == 0:
                break
            if lz4size < 0 or srcsize - srcpos < lz4size:
                raise Lz4h5Error(
                    f'invalid block size {lz4size} @{srcpos} of {srcsize}'
                )

            blksize = <int> min(
                <int64_t> blksize, <int64_t> (dstsize - dstpos)
            )

            if blksize == lz4size:
                memcpy(
                    <void*> &dst[dstpos],
                    <const void*> &src[srcpos],
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
