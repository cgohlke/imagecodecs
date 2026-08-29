# imagecodecs/_blosc2.pyx
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

"""BLOSC2 codec for the imagecodecs package."""

include '_shared.pxi'

from blosc2 cimport *

blosc2_init()


class BLOSC2:
    """BLOSC2 codec constants."""

    available = True

    class FILTER(enum.IntEnum):
        """BLOSC2 codec filters."""

        NOFILTER = BLOSC_NOFILTER
        NOSHUFFLE = BLOSC_NOSHUFFLE
        SHUFFLE = BLOSC_SHUFFLE  # default
        BITSHUFFLE = BLOSC_BITSHUFFLE
        DELTA = BLOSC_DELTA
        TRUNC_PREC = BLOSC_TRUNC_PREC

    class COMPRESSOR(enum.IntEnum):
        """BLOSC2 codec compressors."""

        BLOSCLZ = BLOSC_BLOSCLZ
        LZ4 = BLOSC_LZ4
        LZ4HC = BLOSC_LZ4HC
        ZLIB = BLOSC_ZLIB
        ZSTD = BLOSC_ZSTD  # default

    class SPLIT(enum.IntEnum):
        """BLOSC2 split modes."""

        ALWAYS = BLOSC_ALWAYS_SPLIT  # default
        NEVER = BLOSC_NEVER_SPLIT
        AUTO = BLOSC_AUTO_SPLIT
        FORWARD_COMPAT = BLOSC_FORWARD_COMPAT_SPLIT
        FORWARD = BLOSC_FORWARD_COMPAT_SPLIT  # alias


class Blosc2Error(RuntimeError):
    """BLOSC2 codec exceptions."""

    def __init__(self, func, err=None, ret=None):
        if err is not None:
            msg = f'{func} returned {blosc2_error_string(err).decode()!r}'
        elif ret is not None:
            msg = f'{func} returned {ret!r}'
        else:
            msg = f'{func}'
        super().__init__(msg)


def blosc2_version():
    """Return C-Blosc2 library version string."""
    return 'c-blosc2 ' + BLOSC2_VERSION_STRING.decode()


def blosc2_check(const uint8_t[::1] data, /):
    """Return whether data is BLOSC2 encoded or None if unknown."""
    return None if data.shape[0] > 0 else False


def blosc2_encode(
    data,
    /,
    level=None,
    *,
    compressor=None,
    shuffle=None,  # TODO: enable filters
    splitmode=None,
    typesize=None,
    blocksize=None,
    numthreads=None,
    out=None,
):
    """Return BLOSC2 encoded data."""
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        int32_t cblocksize
        int32_t ctypesize
        uint8_t compcode
        uint8_t cfilter
        int32_t csplitmode
        int16_t nthreads = _default_threads(numthreads)
        int clevel = _default_value(level, 1, 0, 9)
        blosc2_context* context = NULL
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    try:
        src = data  # common case: contiguous bytes
        ctypesize = 8
    except Exception:
        view = memoryview(data)
        if view.contiguous:
            src = view.cast('B')  # view as bytes
        else:
            src = view.tobytes()  # copy non-contiguous
        ctypesize = <int32_t> view.itemsize

    srcsize = src.shape[0]

    if srcsize > INT32_MAX - BLOSC2_MAX_OVERHEAD:
        raise ValueError('data size larger than 2 GB')

    cblocksize = 0 if blocksize is None else blocksize

    compcode = _enum_value(
        compressor, BLOSC2.COMPRESSOR, BLOSC2.COMPRESSOR.ZSTD
    )

    cfilter = _enum_value(shuffle, BLOSC2.FILTER, BLOSC2.FILTER.SHUFFLE)

    if splitmode is not None and not splitmode:
        csplitmode = BLOSC2.SPLIT.NEVER
    else:
        csplitmode = _enum_value(splitmode, BLOSC2.SPLIT, BLOSC2.SPLIT.ALWAYS)

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + BLOSC2_MAX_OVERHEAD
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        if nthreads == 0:
            nthreads = blosc2_get_nthreads()

        cparams.typesize = ctypesize
        cparams.blocksize = cblocksize
        cparams.compcode = compcode
        cparams.clevel = clevel
        cparams.splitmode = csplitmode
        cparams.nthreads = nthreads
        cparams.filters[BLOSC2_MAX_FILTERS - 1] = cfilter

        context = blosc2_create_cctx(cparams)
        if context == NULL:
            raise Blosc2Error('blosc2_create_cctx', ret='NULL')

        ret = blosc2_compress_ctx(
            context,
            <const void*> src._data,
            <int32_t> srcsize,
            <void*> dst._data,
            <int32_t> dstsize
        )

        blosc2_free_ctx(context)

    if ret <= 0:
        raise Blosc2Error('blosc2_compress_ctx', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def blosc2_decode(
    data,
    /,
    *,
    numthreads=None,
    out=None,
):
    """Return decoded BLOSC2 data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.shape[0]
        int32_t nbytes, cbytes, blocksize
        int16_t nthreads = _default_threads(numthreads)
        blosc2_context* context = NULL
        blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS
        int ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize > INT32_MAX:
        raise ValueError('data size larger than 2 GB')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            blosc2_cbuffer_sizes(
                <const void*> src._data,
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
    dstsize = dst.shape[0]
    if dstsize > INT32_MAX:
        raise ValueError('output size larger than 2 GB')

    with nogil:
        if nthreads == 0:
            nthreads = blosc2_get_nthreads()

        dparams.nthreads = nthreads

        context = blosc2_create_dctx(dparams)
        if context == NULL:
            raise Blosc2Error('blosc2_create_dctx', ret='NULL')

        ret = blosc2_decompress_ctx(
            context,
            <const void*> src._data,
            <int32_t> srcsize,
            <void*> dst._data,
            <int32_t> dstsize
        )

        blosc2_free_ctx(context)

    if ret < 0:
        raise Blosc2Error('blosc2_decompress_ctx', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


##############################################################################

class B2ND:
    """B2ND codec constants."""

    available = True

    class FILTER(enum.IntEnum):
        """B2ND codec filters."""

        NOFILTER = BLOSC_NOFILTER
        NOSHUFFLE = BLOSC_NOSHUFFLE
        SHUFFLE = BLOSC_SHUFFLE  # default
        BITSHUFFLE = BLOSC_BITSHUFFLE
        DELTA = BLOSC_DELTA
        TRUNC_PREC = BLOSC_TRUNC_PREC

    class COMPRESSOR(enum.IntEnum):
        """B2ND codec compressors."""

        BLOSCLZ = BLOSC_BLOSCLZ
        LZ4 = BLOSC_LZ4
        LZ4HC = BLOSC_LZ4HC
        ZLIB = BLOSC_ZLIB
        ZSTD = BLOSC_ZSTD  # default


class B2ndError(Blosc2Error):
    """B2ND codec exceptions."""


def b2nd_version():
    """Return c-blosc2 library version string."""
    return blosc2_version()


def b2nd_check(const uint8_t[::1] data, /):
    """Return whether data is B2ND encoded or None if unknown."""
    cdef:
        bytes sig = bytes(data[:10])

    return sig == b'\x9E\xA8\x62\x32\x66\x72\x61\x6D\x65\x00'


def b2nd_encode(
    data,
    /,
    level=None,
    *,
    chunkshape=None,
    blockshape=None,
    compressor=None,
    shuffle=None,
    numthreads=None,
    out=None,
):
    """Return B2ND encoded data."""
    cdef:
        numpy.ndarray src = _array_input(data, out)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        int64_t srcsize = src.nbytes
        int8_t ndim = <int8_t> src.ndim
        int32_t typesize = <int32_t> src.itemsize
        int64_t cframe_len = 0
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS
        blosc2_dparams dparams = BLOSC2_DPARAMS_DEFAULTS
        blosc2_storage storage = BLOSC2_STORAGE_DEFAULTS
        b2nd_context_t *ctx = NULL
        b2nd_array_t *array = NULL
        const char* dtype_bytes = NULL
        uint8_t *cframe = NULL
        int64_t[16] cshape  # B2ND_MAX_DIM
        int32_t[16] cchunkshape
        int32_t[16] cblockshape
        uint8_t compcode, cfilter
        int16_t nthreads = _default_threads(numthreads)
        int clevel = _default_value(level, 1, 0, 9)
        bool needs_free = 0
        ssize_t i
        int64_t bytes_per_row, rows_per_block
        bint default_blockshape = 1
        int ret

    dtype = src.dtype.str.encode()
    dtype_bytes = dtype

    compcode = _enum_value(compressor, B2ND.COMPRESSOR, B2ND.COMPRESSOR.ZSTD)
    cfilter = _enum_value(shuffle, B2ND.FILTER, B2ND.FILTER.SHUFFLE)

    for i in range(ndim):
        cshape[i] = src.shape[i]
        cchunkshape[i] = (
            chunkshape[i] if chunkshape is not None else <int32_t> src.shape[i]
        )

    if blockshape is not None:
        default_blockshape = 0
        for i in range(ndim):
            cblockshape[i] = blockshape[i]

    try:
        with nogil:
            if nthreads == 0:
                nthreads = blosc2_get_nthreads()

            if default_blockshape:
                # target ~256 KB blocks; split the first dimension only
                bytes_per_row = typesize
                for i in range(1, ndim):
                    bytes_per_row *= cchunkshape[i]
                rows_per_block = (
                    (256 * 1024) // bytes_per_row
                    if bytes_per_row > 0
                    else cchunkshape[0]
                )
                if rows_per_block < 1:
                    rows_per_block = 1
                if rows_per_block > cchunkshape[0]:
                    rows_per_block = cchunkshape[0]
                cblockshape[0] = <int32_t> rows_per_block
                for i in range(1, ndim):
                    cblockshape[i] = cchunkshape[i]

            cparams.compcode = compcode
            cparams.clevel = clevel
            cparams.typesize = typesize
            cparams.nthreads = nthreads
            cparams.filters[BLOSC2_MAX_FILTERS - 1] = cfilter
            dparams.nthreads = nthreads
            storage.cparams = &cparams
            storage.dparams = &dparams

            ctx = b2nd_create_ctx(
                &storage,
                ndim,
                cshape,
                cchunkshape,
                cblockshape,
                dtype_bytes,
                DTYPE_NUMPY_FORMAT,
                NULL,
                0,
            )
            if ctx == NULL:
                raise B2ndError('b2nd_create_ctx', ret='NULL')

            ret = b2nd_from_cbuffer(
                ctx, &array, <const void *> src.data, srcsize
            )
            if ret < 0:
                raise B2ndError('b2nd_from_cbuffer', ret)

            ret = b2nd_to_cframe(array, &cframe, &cframe_len, &needs_free)
            if ret < 0:
                if needs_free and cframe != NULL:
                    free(<void *> cframe)
                    cframe = NULL
                raise B2ndError('b2nd_to_cframe', ret)

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            dstsize = <ssize_t> cframe_len
            out = _create_output(outtype, dstsize, <const char*> cframe)
        else:
            dst = out
            dstsize = dst.shape[0]
            if <int64_t> dstsize < cframe_len:
                raise ValueError(
                    f'output buffer too small {dstsize} < {cframe_len}'
                )
            memcpy(
                <void*> dst._data, <const void*> cframe, <size_t> cframe_len
            )
            del dst

    finally:
        if ctx != NULL:
            b2nd_free_ctx(ctx)
        if array != NULL:
            b2nd_free(array)
        if needs_free and cframe != NULL:
            free(<void *> cframe)

    return _return_output(out, dstsize, <ssize_t> cframe_len, outgiven)


def b2nd_decode(
    data,
    /,
    *,
    numthreads=None,
    out=None,
):
    """Return decoded B2ND data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        int64_t srcsize = src.shape[0]
        int64_t dstsize
        b2nd_array_t* array = NULL
        int16_t nthreads = _default_threads(numthreads)
        int16_t old_nthreads
        char* dtype_str
        int ret

    try:
        if nthreads == 0:
            nthreads = blosc2_get_nthreads()
        old_nthreads = blosc2_set_nthreads(nthreads)

        with nogil:
            ret = b2nd_from_cframe(<uint8_t *> src._data, srcsize, 0, &array)
        if ret < 0:
            raise B2ndError('b2nd_from_cframe', ret)

        shape = tuple(array.shape[i] for i in range(array.ndim))
        dtype_str = array.dtype
        if dtype_str == NULL:
            dtype = numpy.dtype('uint8')
        else:
            dtype = numpy.dtype(dtype_str.decode())

        out = _create_array(out, shape, dtype)
        dst = out
        dstsize = <int64_t> dst.nbytes

        with nogil:
            ret = b2nd_to_cbuffer(array, <void *> dst.data, dstsize)

    finally:
        blosc2_set_nthreads(old_nthreads)
        if array != NULL:
            b2nd_free(array)

    if ret < 0:
        raise B2ndError('b2nd_to_cbuffer', ret)

    return out
