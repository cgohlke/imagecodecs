# imagecodecs/_blosc2.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2021-2022, Christoph Gohlke
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

"""Blosc2 codec for the imagecodecs package."""

__version__ = '2022.9.26'

include '_shared.pxi'

from blosc2 cimport *


class BLOSC2:
    """Blosc2 Constants."""

    NOFILTER = BLOSC_NOFILTER
    NOSHUFFLE = BLOSC_NOSHUFFLE
    SHUFFLE = BLOSC_SHUFFLE
    BITSHUFFLE = BLOSC_BITSHUFFLE
    DELTA = BLOSC_DELTA
    TRUNC_PREC = BLOSC_TRUNC_PREC

    BLOSCLZ = BLOSC_BLOSCLZ
    LZ4 = BLOSC_LZ4
    LZ4HC = BLOSC_LZ4HC
    ZLIB = BLOSC_ZLIB
    ZSTD = BLOSC_ZSTD


class Blosc2Error(RuntimeError):
    """Blosc2 Exceptions."""

    def __init__(self, func, err=None, ret=None):
        if err is not None:
            msg = {
                BLOSC2_ERROR_SUCCESS: 'Success',
                BLOSC2_ERROR_FAILURE: 'Generic failure',
                BLOSC2_ERROR_STREAM: 'Bad stream',
                BLOSC2_ERROR_DATA: 'Invalid data',
                BLOSC2_ERROR_MEMORY_ALLOC: 'Memory alloc/realloc failure',
                BLOSC2_ERROR_READ_BUFFER: 'Not enough space to read',
                BLOSC2_ERROR_WRITE_BUFFER: 'Not enough space to write',
                BLOSC2_ERROR_CODEC_SUPPORT: 'Codec not supported',
                BLOSC2_ERROR_CODEC_PARAM:
                    'Invalid parameter supplied to codec',
                BLOSC2_ERROR_CODEC_DICT: 'Codec dictionary error',
                BLOSC2_ERROR_VERSION_SUPPORT: 'Version not supported',
                BLOSC2_ERROR_INVALID_HEADER: 'Invalid value in header',
                BLOSC2_ERROR_INVALID_PARAM:
                    'Invalid parameter supplied to function',
                BLOSC2_ERROR_FILE_READ: 'File read failure',
                BLOSC2_ERROR_FILE_WRITE: 'File write failure',
                BLOSC2_ERROR_FILE_OPEN: 'File open failure',
                BLOSC2_ERROR_NOT_FOUND: 'Not found',
                BLOSC2_ERROR_RUN_LENGTH: 'Bad run length encoding',
                BLOSC2_ERROR_FILTER_PIPELINE: 'Filter pipeline error',
                BLOSC2_ERROR_CHUNK_INSERT: 'Chunk insert failure',
                BLOSC2_ERROR_CHUNK_APPEND: 'Chunk append failure',
                BLOSC2_ERROR_CHUNK_UPDATE: 'Chunk update failure',
                BLOSC2_ERROR_2GB_LIMIT: 'Sizes larger than 2gb not supported',
                BLOSC2_ERROR_SCHUNK_COPY: 'Super-chunk copy failure',
                BLOSC2_ERROR_FRAME_TYPE: 'Wrong type for frame',
                BLOSC2_ERROR_FILE_TRUNCATE: 'File truncate failure',
                BLOSC2_ERROR_THREAD_CREATE:
                    'Thread or thread context creation failure',
                BLOSC2_ERROR_POSTFILTER: 'Postfilter failure',
                BLOSC2_ERROR_FRAME_SPECIAL: 'Special frame failure',
                BLOSC2_ERROR_SCHUNK_SPECIAL: 'Special super-chunk failure',
                BLOSC2_ERROR_PLUGIN_IO: 'IO plugin error',
                BLOSC2_ERROR_FILE_REMOVE: 'Remove file failure',
            }.get(err, f'unknown error {err!r}')
            msg = f'{func} returned {msg}'
        elif ret is not None:
            msg = f'{func} returned {ret!r}'
        else:
            msg = f'{func}'
        super().__init__(msg)


def blosc2_version():
    """Return Blosc library version string."""
    return 'c-blosc2 ' + BLOSC2_VERSION_STRING.decode()


def blosc2_check(data):
    """Return True if data likely contains Blosc2 data."""


def blosc2_encode(
    data,
    level=None,
    compressor=None,
    typesize=None,
    blocksize=None,
    shuffle=None,  # TODO: enable filters
    numthreads=None,
    out=None
):
    """Encode Blosc2.

    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        int32_t cblocksize
        int32_t ctypesize
        uint8_t compcode
        uint8_t cfilter
        int16_t nthreads = <int16_t> _default_threads(numthreads)
        int clevel = _default_value(level, 5, 0, 9)
        blosc2_context* context = NULL
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS
        int ret

    if data is out:
        raise ValueError('cannot encode in-place')

    try:
        src = data  # common case: contiguous bytes
        ctypesize = 1
    except Exception:
        view = memoryview(data)
        if view.contiguous:
            src = view.cast('B')  # view as bytes
        else:
            src = view.tobytes()  # copy non-contiguous
        ctypesize = view.itemsize

    srcsize = src.size

    if srcsize > 2147483647 - BLOSC2_MAX_OVERHEAD:
        raise ValueError('data size larger than 2 GB')

    if blocksize is None:
        cblocksize = 0
    else:
        cblocksize = blocksize

    if compressor is None:
        compcode = BLOSC_BLOSCLZ
    elif isinstance(compressor, str):
        compressor = compressor.lower().encode()
        if compressor == BLOSC_BLOSCLZ_COMPNAME:
            compcode = BLOSC_BLOSCLZ
        elif compressor == BLOSC_LZ4_COMPNAME:
            compcode = BLOSC_LZ4
        elif compressor == BLOSC_LZ4HC_COMPNAME:
            compcode = BLOSC_LZ4HC
        elif compressor == BLOSC_ZLIB_COMPNAME:
            compcode = BLOSC_ZLIB
        elif compressor == BLOSC_ZSTD_COMPNAME:
            compcode = BLOSC_ZSTD
        else:
            raise ValueError(f'unknown Blosc2 compressor {compressor!r}')
    else:
        compcode = compressor

    if shuffle is None:
        cfilter = BLOSC_SHUFFLE
    elif not shuffle:
        cfilter = BLOSC_NOFILTER
    elif isinstance(shuffle, str):
        shuffle = shuffle.lower()
        if shuffle[:2] == 'no':
            cfilter = BLOSC_NOSHUFFLE
        elif shuffle == 'shuffle':
            cfilter = BLOSC_SHUFFLE
        elif shuffle == 'bitshuffle':
            cfilter = BLOSC_BITSHUFFLE
        elif shuffle == 'delta':
            cfilter = BLOSC_DELTA
        elif shuffle == 'trunc_prec':
            cfilter = BLOSC_TRUNC_PREC
        else:
            raise ValueError(f'unknown Blosc2 filter {shuffle!r}')
    else:
        cfilter = shuffle

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + BLOSC2_MAX_OVERHEAD
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:

        if nthreads == 0:
            nthreads = blosc2_get_nthreads()

        cparams.typesize = ctypesize
        cparams.blocksize = cblocksize
        cparams.compcode = compcode
        cparams.clevel = clevel
        cparams.nthreads = nthreads
        cparams.filters[BLOSC2_MAX_FILTERS - 1] = cfilter

        context = blosc2_create_cctx(cparams)
        if context == NULL:
            raise Blosc2Error(f'blosc2_create_cctx', ret='NULL')

        ret = blosc2_compress_ctx(
            context,
            <const void*> &src[0],
            <int32_t> srcsize,
            <void*> &dst[0],
            <int32_t> dstsize
        )

        blosc2_free_ctx(context)

    if ret <= 0:
        raise Blosc2Error(f'blosc2_compress_ctx', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc2_decode(data, numthreads=None, out=None):
    """Decode Blosc2.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        int32_t nbytes, cbytes, blocksize
        int16_t nthreads = <int16_t> _default_threads(numthreads)
        blosc2_context* context = NULL
        blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if src.size > 2147483647:
        raise ValueError('data size larger than 2 GB')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            blosc2_cbuffer_sizes(
                <const void*> &src[0],
                &nbytes,
                &cbytes,
                &blocksize
            )
            if nbytes == 0 and blocksize == 0:
                raise Blosc2Error(
                    'blosc2_cbuffer_sizes returned invalid blosc data'
                )
            dstsize = <ssize_t> nbytes
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    if dstsize > 2147483647:
        raise ValueError('output size larger than 2 GB')

    with nogil:
        if nthreads == 0:
            nthreads = blosc2_get_nthreads()

        dparams.nthreads = nthreads

        context = blosc2_create_dctx(dparams)
        if context == NULL:
            raise Blosc2Error(f'blosc2_create_dctx', ret='NULL')

        ret = blosc2_decompress_ctx(
            context,
            <const void*> &src[0],
            <int32_t> srcsize,
            <void*> &dst[0],
            <int32_t> dstsize
        )

        blosc2_free_ctx(context)

    if ret < 0:
        raise Blosc2Error(f'blosc2_decompress_ctx', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)
