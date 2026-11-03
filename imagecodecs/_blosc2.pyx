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
        BYTEDELTA = BLOSC_FILTER_BYTEDELTA
        INT_TRUNC = BLOSC_FILTER_INT_TRUNC
        # require b2nd metalayer, only supported by B2ND:
        # NDCELL = BLOSC_FILTER_NDCELL
        # NDMEAN = BLOSC_FILTER_NDMEAN

    class CODEC(enum.IntEnum):
        """BLOSC2 codec compressors."""

        BLOSCLZ = BLOSC_BLOSCLZ
        LZ4 = BLOSC_LZ4
        LZ4HC = BLOSC_LZ4HC
        ZLIB = BLOSC_ZLIB
        ZSTD = BLOSC_ZSTD  # default
        # require b2nd metalayer, only supported by B2ND:
        # NDLZ = BLOSC_CODEC_NDLZ
        # ZFP_ACC = BLOSC_CODEC_ZFP_FIXED_ACCURACY
        # ZFP_PREC = BLOSC_CODEC_ZFP_FIXED_PRECISION
        # ZFP_RATE = BLOSC_CODEC_ZFP_FIXED_RATE

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


def blosc2_check(const uint8_t[::1] data not None, /):
    """Return whether data is BLOSC2 encoded or None if unknown."""
    return None if data.shape[0] >= BLOSC_MIN_HEADER_LENGTH else False


def blosc2_encode(
    data,
    /,
    level=None,
    *,
    codec=None,
    filters=None,
    filters_meta=None,
    splitmode=None,
    typesize=None,
    blocksize=None,
    numthreads=None,
    out=None,
):
    """Return BLOSC2 encoded data."""
    cdef:
        ssize_t itemsize
        const uint8_t[::1] src = _readable_input(data, out, &itemsize)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        int32_t cblocksize = 0 if blocksize is None else blocksize
        int32_t ctypesize = (
            <int32_t> itemsize if typesize is None else typesize
        )
        uint8_t compcode = _enum_value(codec, BLOSC2.CODEC, BLOSC2.CODEC.ZSTD)
        int32_t csplitmode
        int16_t nthreads = _default_threads(numthreads)
        int clevel = _default_value(level, 1, 0, 9)
        blosc2_context* context = NULL
        blosc2_cparams cparams = BLOSC2_CPARAMS_DEFAULTS
        int ret

    if typesize is None and ctypesize == 1:
        ctypesize = 8  # default for byte input

    if ctypesize < 1:
        raise ValueError(f'invalid typesize {ctypesize}')

    if cblocksize < 0:
        raise ValueError(f'invalid blocksize {cblocksize}')

    if srcsize > INT32_MAX - BLOSC2_MAX_OVERHEAD:
        raise ValueError('data size larger than 2 GB')

    _set_filters(&cparams, filters, filters_meta, BLOSC2.FILTER, ctypesize)

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
    if dstsize > INT32_MAX:
        dstsize = INT32_MAX

    with nogil:
        if nthreads == 0:
            nthreads = blosc2_get_nthreads()

        cparams.typesize = ctypesize
        cparams.blocksize = cblocksize
        cparams.compcode = compcode
        cparams.clevel = clevel
        cparams.splitmode = csplitmode
        cparams.nthreads = nthreads

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

    if srcsize < BLOSC_MIN_HEADER_LENGTH:
        raise Blosc2Error('invalid blosc2 data: header too short')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            ret = blosc2_cbuffer_sizes(
                <const void*> src._data,
                &nbytes,
                &cbytes,
                &blocksize
            )
            if ret < 0:
                raise Blosc2Error('blosc2_cbuffer_sizes', ret)
            dstsize = <ssize_t> nbytes
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
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


def blosc2_info(const uint8_t[::1] data not None, /):
    """Return information about BLOSC2 encoded data."""
    cdef:
        ssize_t srcsize = data.shape[0]
        int32_t nbytes, cbytes, blocksize
        uint8_t flags, compformat
        ssize_t i
        int ret, compcode

    if srcsize < BLOSC_MIN_HEADER_LENGTH:
        raise ValueError('data too short for BLOSC2 header')

    ret = blosc2_cbuffer_sizes(
        <const void*> &data[0], &nbytes, &cbytes, &blocksize
    )
    if ret < 0:
        raise Blosc2Error('blosc2_cbuffer_sizes', ret)

    flags = data[BLOSC2_CHUNK_FLAGS]
    if (flags & BLOSC_DOSHUFFLE) and (flags & BLOSC_DOBITSHUFFLE):
        # extended BLOSC2 header
        if srcsize < BLOSC_EXTENDED_HEADER_LENGTH:
            raise ValueError('data too short for BLOSC2 extended header')
        filters = [
            data[BLOSC2_CHUNK_FILTER_CODES + i]
            for i in range(BLOSC2_MAX_FILTERS)
        ]
        filters_meta = [
            data[BLOSC2_CHUNK_FILTER_META + i]
            for i in range(BLOSC2_MAX_FILTERS)
        ]
        udcompcode = data[BLOSC2_CHUNK_FILTER_CODES + BLOSC2_MAX_FILTERS]
        codec_meta = data[BLOSC2_CHUNK_FILTER_CODES + BLOSC2_MAX_FILTERS + 1]
    else:
        # BLOSC1 header
        filters = [BLOSC_NOFILTER] * BLOSC2_MAX_FILTERS
        filters_meta = [0] * BLOSC2_MAX_FILTERS
        if flags & BLOSC_DOSHUFFLE:
            filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_SHUFFLE
        if flags & BLOSC_DOBITSHUFFLE:
            filters[BLOSC2_MAX_FILTERS - 1] = BLOSC_BITSHUFFLE
        if flags & BLOSC_DODELTA:
            filters[BLOSC2_MAX_FILTERS - 2] = BLOSC_DELTA
        udcompcode = 0
        codec_meta = 0

    compformat = flags >> 5
    if compformat == BLOSC_BLOSCLZ_FORMAT:
        compcode = BLOSC_BLOSCLZ
    elif compformat == BLOSC_LZ4_FORMAT:
        compcode = BLOSC_LZ4  # LZ4HC shares the LZ4 format
    elif compformat == BLOSC_ZLIB_FORMAT:
        compcode = BLOSC_ZLIB
    elif compformat == BLOSC_ZSTD_FORMAT:
        compcode = BLOSC_ZSTD
    elif compformat == BLOSC_UDCODEC_FORMAT:
        compcode = udcompcode
    else:
        compcode = compformat

    return {
        '_codec': 'blosc2',
        'nbytes': int(nbytes),
        'cbytes': int(cbytes),
        'blocksize': int(blocksize),
        'typesize': int(data[BLOSC2_CHUNK_TYPESIZE]),
        'codec': _enum_or_int(compcode, BLOSC2.CODEC),
        'codec_meta': int(codec_meta),
        'filters': [_enum_or_int(f, BLOSC2.FILTER) for f in filters],
        'filters_meta': [int(m) for m in filters_meta],
        'splitmode': (
            BLOSC2.SPLIT.NEVER if flags & 0x10 else BLOSC2.SPLIT.ALWAYS
        ),
        'version': int(data[BLOSC2_CHUNK_VERSION]),
    }


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
        NDCELL = BLOSC_FILTER_NDCELL
        NDMEAN = BLOSC_FILTER_NDMEAN
        BYTEDELTA = BLOSC_FILTER_BYTEDELTA
        INT_TRUNC = BLOSC_FILTER_INT_TRUNC

    class CODEC(enum.IntEnum):
        """B2ND codec compressors."""

        BLOSCLZ = BLOSC_BLOSCLZ
        LZ4 = BLOSC_LZ4
        LZ4HC = BLOSC_LZ4HC
        ZLIB = BLOSC_ZLIB
        ZSTD = BLOSC_ZSTD  # default
        NDLZ = BLOSC_CODEC_NDLZ
        ZFP_ACC = BLOSC_CODEC_ZFP_FIXED_ACCURACY
        ZFP_PREC = BLOSC_CODEC_ZFP_FIXED_PRECISION
        ZFP_RATE = BLOSC_CODEC_ZFP_FIXED_RATE


class B2ndError(Blosc2Error):
    """B2ND codec exceptions."""


def b2nd_version():
    """Return c-blosc2 library version string."""
    return blosc2_version()


def b2nd_check(const uint8_t[::1] data not None, /):
    """Return whether data is B2ND encoded or None if unknown."""
    return _startswith(data, b'\x9E\xA8\x62\x32\x66\x72\x61\x6D\x65\x00')


def b2nd_encode(
    data,
    /,
    level=None,
    *,
    chunkshape=None,
    blockshape=None,
    codec=None,
    codec_meta=None,
    filters=None,
    filters_meta=None,
    numthreads=None,
    out=None,
):
    """Return B2ND encoded data."""
    cdef:
        numpy.ndarray src = _array_input(
            data, out, dtype=None, contig=True, empty=False, native=False
        )
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
        uint8_t compcode
        uint8_t compmeta = _filter_meta(codec_meta)
        int16_t nthreads = _default_threads(numthreads)
        int clevel = _default_value(level, 1, 0, 9)
        bool needs_free = 0
        ssize_t i
        int64_t bytes_per_row, rows_per_block
        bint default_blockshape = 1
        int ret

    if src.ndim > B2ND_MAX_DIM:
        raise ValueError(f'{src.ndim=} exceeds {B2ND_MAX_DIM=}')
    if chunkshape is not None and len(chunkshape) != ndim:
        raise ValueError(f'{len(chunkshape)=} != {src.ndim=}')
    if blockshape is not None and len(blockshape) != ndim:
        raise ValueError(f'{len(blockshape)=} != {src.ndim=}')

    dtype = src.dtype.str.encode()
    dtype_bytes = dtype

    compcode = _enum_value(codec, B2ND.CODEC, B2ND.CODEC.ZSTD)
    _set_filters(&cparams, filters, filters_meta, B2ND.FILTER, typesize)

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
            cparams.compcode_meta = compmeta
            cparams.clevel = clevel
            cparams.typesize = typesize
            cparams.nthreads = nthreads
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
            if 0 <= dstsize < cframe_len:
                raise ValueError(
                    f'output buffer too small {dstsize} < {cframe_len}'
                )
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
        # int16_t old_nthreads
        char* dtype_str
        int ret

    try:
        if nthreads == 0:
            nthreads = blosc2_get_nthreads()
        # TODO: blosc2_set_nthreads is not thread safe
        # old_nthreads = blosc2_set_nthreads(nthreads)

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
        # blosc2_set_nthreads(old_nthreads)
        if array != NULL:
            b2nd_free(array)

    if ret < 0:
        raise B2ndError('b2nd_to_cbuffer', ret)

    return out


def b2nd_info(const uint8_t[::1] data not None, /):
    """Return information about B2ND encoded data."""
    cdef:
        const uint8_t[::1] src = data
        int64_t srcsize = src.shape[0]
        b2nd_array_t* array = NULL
        blosc2_schunk* sc
        char* dtype_str
        ssize_t i
        int ret

    with nogil:
        ret = b2nd_from_cframe(<uint8_t *> src._data, srcsize, 0, &array)
    if ret < 0:
        raise B2ndError('b2nd_from_cframe', ret)

    try:
        sc = array.sc
        dtype_str = array.dtype
        if dtype_str == NULL:
            dtype = numpy.dtype('uint8')
        else:
            dtype = numpy.dtype(dtype_str.decode())
        info = {
            '_codec': 'b2nd',
            'shape': tuple(array.shape[i] for i in range(array.ndim)),
            'dtype': dtype,
            'chunkshape': tuple(
                array.chunkshape[i] for i in range(array.ndim)
            ),
            'blockshape': tuple(
                array.blockshape[i] for i in range(array.ndim)
            ),
            'level': int(sc.clevel),
            'codec': _enum_or_int(sc.compcode, B2ND.CODEC),
            'codec_meta': int(sc.compcode_meta),
            'filters': [
                _enum_or_int(sc.filters[i], B2ND.FILTER)
                for i in range(BLOSC2_MAX_FILTERS)
            ],
            'filters_meta': [
                int(sc.filters_meta[i]) for i in range(BLOSC2_MAX_FILTERS)
            ],
            'typesize': int(sc.typesize),
            'nchunks': int(sc.nchunks),
            'nbytes': int(sc.nbytes),
            'cbytes': int(sc.cbytes),
        }
    finally:
        b2nd_free(array)

    return info


cdef uint8_t _filter_meta(value) except? 0:
    """Return filter or codec meta value as uint8."""
    if value is None:
        return 0
    value = int(value)
    if value < -128 or value > 255:
        raise ValueError(f'meta value {value} out of range [-128, 255]')
    # negative values are interpreted as int8 by some plugins
    return <uint8_t> (value & 0xFF)


cdef _set_filters(
    blosc2_cparams* cparams,
    filters,
    filters_meta,
    enum_class,
    int32_t typesize
):
    """Set filter pipeline and meta values in cparams like python-blosc2."""
    cdef:
        ssize_t i, nfilters

    if filters is None:
        codes = [enum_class.NOFILTER] * (BLOSC2_MAX_FILTERS - 1)
        codes.append(enum_class.SHUFFLE)
    elif isinstance(filters, int | str):
        raise TypeError('filters must be a sequence')
    else:
        codes = [_enum_value(f, enum_class) for f in filters]
    nfilters = len(codes)
    if nfilters > BLOSC2_MAX_FILTERS:
        raise ValueError(
            f'number of filters {nfilters} > {BLOSC2_MAX_FILTERS}'
        )

    if filters_meta is None:
        metas = [0] * nfilters
    else:
        metas = list(filters_meta)
        if len(metas) != nfilters:
            raise ValueError(
                f'number of filters_meta {len(metas)} != '
                f'number of filters {nfilters}'
            )

    for i in range(BLOSC2_MAX_FILTERS):
        cparams.filters[i] = BLOSC_NOFILTER
        cparams.filters_meta[i] = 0

    for i in range(nfilters):
        if codes[i] == BLOSC_FILTER_BYTEDELTA:
            if typesize == 1:
                # BYTEDELTA does not work with typesize 1
                continue
            if metas[i] == 0:
                # BYTEDELTA requires typesize in meta outside of super-chunks
                metas[i] = typesize
        cparams.filters[i] = <uint8_t> codes[i]
        cparams.filters_meta[i] = _filter_meta(metas[i])
