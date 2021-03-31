# imagecodecs/_lz4f.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2020-2021, Christoph Gohlke
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

"""LZ4 Frame codec for the imagecodecs package."""

__version__ = '2020.3.31'

include '_shared.pxi'

from lz4 cimport *


class LZ4F:
    """LZ4F Constants."""

    VERSION = LZ4F_VERSION


class Lz4fError(RuntimeError):
    """LZ4F Exceptions."""

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


def lz4f_check(data):
    """Return True if data likely contains LZ4 Frame data."""
    cdef:
        bytes sig = bytes(data[:4])

    return sig == b'\x04\x22\x4d\x18'  # LZ4_MAGIC_NUMBER 0x184d2204


def lz4f_encode(
    data,
    level=None,
    blocksizeid=None,
    contentchecksum=None,
    blockchecksum=None,
    out=None
):
    """Compress LZ4 Frame.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t srcsize = <size_t> src.size
        ssize_t dstsize
        size_t ret
        LZ4F_preferences_t prefs

    if data is out:
        raise ValueError('cannot encode in-place')

    memset(&prefs, 0, sizeof(LZ4F_preferences_t))
    prefs.frameInfo.contentSize = srcsize
    if level:
        prefs.compressionLevel = _default_value(level, 0, -1, LZ4HC_CLEVEL_MAX)
    if blocksizeid:
        prefs.frameInfo.blockSizeID = <LZ4F_blockSizeID_t> blocksizeid
    if contentchecksum:
        prefs.frameInfo.contentChecksumFlag = LZ4F_contentChecksumEnabled
    if blockchecksum:
        prefs.frameInfo.blockChecksumFlag = LZ4F_blockChecksumEnabled

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = LZ4F_compressFrameBound(srcsize, &prefs)
            if dstsize < 0:
                raise Lz4fError(f'LZ4F_compressFrameBound returned {dstsize}')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = <size_t> dst.size

    with nogil:
        ret = LZ4F_compressFrame(
            <void*> &dst[0],
            <size_t> dstsize,
            <void*> &src[0],
            srcsize,
            &prefs
        )
        if LZ4F_isError(<LZ4F_errorCode_t> ret):
            raise Lz4fError('LZ4F_compressFrame', <LZ4F_errorCode_t> ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def lz4f_decode(data, out=None):
    """Decompress LZ4 Frame.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = <size_t> src.size
        ssize_t dstsize
        size_t byteswritten, bytesread, ret
        size_t srcindex = 0
        LZ4F_dctx* dctx = NULL
        LZ4F_decompressOptions_t* options = NULL
        LZ4F_frameInfo_t frameinfo

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            try:
                # read content size from frame header
                ret = LZ4F_headerSize(<const void*> &src[0], <size_t> srcsize)
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
                    <const void*> &src[0],
                    &srcindex
                )
                if LZ4F_isError(<LZ4F_errorCode_t> ret):
                    raise Lz4fError('LZ4F_getFrameInfo', ret)
                if frameinfo.contentSize > 0:
                    dstsize = <ssize_t> frameinfo.contentSize
                else:
                    # TODO: use streaming API
                    dstsize = (
                        LZ4F_HEADER_SIZE_MAX
                        + LZ4F_BLOCK_HEADER_SIZE
                        + LZ4F_BLOCK_CHECKSUM_SIZE
                        + LZ4F_CONTENT_CHECKSUM_SIZE
                    )
                    # ugh
                    dstsize = max(dstsize, dstsize + 24 + 255 * (srcsize - 10))
            except Exception:
                LZ4F_freeDecompressionContext(dctx)
                raise

        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
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
                <void*> &dst[0],
                &byteswritten,
                <void*> &src[srcindex],
                &bytesread,
                options
            )
            if LZ4F_isError(<LZ4F_errorCode_t> ret):
                raise Lz4fError('LZ4F_decompress', <LZ4F_errorCode_t> ret)
    finally:
        ret = LZ4F_freeDecompressionContext(dctx)
        if LZ4F_isError(<LZ4F_errorCode_t> ret):
            raise Lz4fError(
                'LZ4F_freeDecompressionContext', <LZ4F_errorCode_t> ret
            )

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)
