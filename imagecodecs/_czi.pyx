# imagecodecs/_czi.pyx
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

"""Carl Zeiss Image ZSTD1 and CHUNKED codecs for the imagecodecs package.

ZSTD1 is Zeiss CZI compression type 6. It wraps a standard Zstd frame
with a small proprietary header and an optional hi/lo byte-shuffle step.

CHUNKED is Zeiss CZI compression mode 7.
See https://zeiss.github.io/libczi/pages/chunked_compression.html
The format divides the pixel byte stream into independently compressed
chunks and stores the compressed chunk sizes in a compact header.
The per-chunk codec is zstd (default) or lz4. An optional hi/lo byte-packing
preprocessing step may be applied before compression.

"""

include '_shared.pxi'

from lz4 cimport *
from zstd cimport *


class ZSTD1:
    """ZSTD1 codec constants."""

    available = True


class Zstd1Error(RuntimeError):
    """ZSTD1 codec exceptions."""

    def __init__(self, func, msg='', err=0):
        cdef:
            const char* errmsg

        if msg:
            msg = f'{func} returned {msg!r}'
        else:
            errmsg = ZSTD_getErrorName(err)
            msg = f'{func} returned {errmsg.decode()!r}'
        super().__init__(msg)


def zstd1_version():
    """Return Zstandard library version string."""
    return 'zstd {}.{}.{}'.format(
        ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR, ZSTD_VERSION_RELEASE
    )


def zstd1_check(const uint8_t[::1] data, /):
    """Return whether data is ZSTD1 encoded or None if unknown."""
    cdef:
        ssize_t srcsize = data.shape[0]
        uint8_t header_size

    if srcsize < 2:
        return False
    header_size = data[0]
    if header_size < 1 or <ssize_t> header_size >= srcsize:
        return False
    if srcsize - header_size < 4:
        return False
    return bytes(data[header_size:header_size + 4]) == b'\x28\xB5\x2F\xFD'


def zstd1_encode(
    data,
    /,
    level=None,
    *,
    ssize_t itemsize=1,
    hilo=False,
    numthreads=None,
    out=None,
):
    """Return ZSTD1 encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst
        uint8_t* dstptr
        uint8_t* tmpptr
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, hdr_len
        int clevel = 0 if level is None else level
        int workers = _default_threads(numthreads)
        bint do_hilo = hilo and itemsize > 1
        ZSTD_CCtx* cctx = NULL
        size_t ret

    if itemsize < 1:
        raise ValueError(f'{itemsize=} < 1')

    if do_hilo and srcsize % itemsize != 0:
        raise ValueError('data size not a multiple of itemsize')

    # header: 1 byte (no chunks) or 3 bytes (type-1 chunk with hilo flag)
    hdr_len = 3 if do_hilo else 1

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = hdr_len + <ssize_t> ZSTD_compressBound(<size_t> srcsize)
            if dstsize <= hdr_len:
                raise Zstd1Error('ZSTD_compressBound', 'input too large')
        if dstsize < hdr_len + 64:
            dstsize = hdr_len + 64
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    if workers > 1:
        try:
            with nogil:
                cctx = ZSTD_createCCtx()
                if cctx == NULL:
                    raise Zstd1Error('ZSTD_createCCtx', 'NULL')
                ret = ZSTD_CCtx_setParameter(
                    cctx, ZSTD_c_compressionLevel, clevel
                )
                if ZSTD_isError(ret):
                    raise Zstd1Error('ZSTD_CCtx_setParameter', err=ret)
                ret = ZSTD_CCtx_setParameter(cctx, ZSTD_c_nbWorkers, workers)
                if ZSTD_isError(ret):
                    raise Zstd1Error('ZSTD_CCtx_setParameter', err=ret)
                dstptr = <uint8_t*> dst._data
                dstptr[0] = <uint8_t> hdr_len  # header_size includes itself
                if do_hilo:
                    dstptr[1] = 1  # chunk type 1
                    dstptr[2] = 1  # payload: hilo flag set
                    tmpptr = <uint8_t*> malloc(<size_t> srcsize)
                    if tmpptr == NULL:
                        raise MemoryError('zstd1_encode malloc failed')
                    _hilo_pack(
                        <const uint8_t*> src._data,
                        tmpptr,
                        <size_t> srcsize,
                        <size_t> itemsize,
                    )
                    ret = ZSTD_compress2(
                        cctx,
                        <void*> (dstptr + hdr_len),
                        <size_t> (dstsize - hdr_len),
                        <void*> tmpptr,
                        <size_t> srcsize,
                    )
                    free(tmpptr)
                else:
                    ret = ZSTD_compress2(
                        cctx,
                        <void*> (dstptr + hdr_len),
                        <size_t> (dstsize - hdr_len),
                        <void*> src._data,
                        <size_t> srcsize,
                    )
                if ZSTD_isError(ret):
                    raise Zstd1Error('ZSTD_compress2', err=ret)
                ret += <size_t> hdr_len
        finally:
            if cctx != NULL:
                ZSTD_freeCCtx(cctx)
    else:
        with nogil:
            dstptr = <uint8_t*> dst._data
            dstptr[0] = <uint8_t> hdr_len  # header_size includes itself
            if do_hilo:
                dstptr[1] = 1  # chunk type 1
                dstptr[2] = 1  # payload: hilo flag set
                tmpptr = <uint8_t*> malloc(<size_t> srcsize)
                if tmpptr == NULL:
                    raise MemoryError('zstd1_encode malloc failed')
                _hilo_pack(
                    <const uint8_t *> src._data,
                    tmpptr,
                    <size_t> srcsize,
                    <size_t> itemsize
                )
                ret = ZSTD_compress(
                    <void*> (dstptr + hdr_len),
                    <size_t> (dstsize - hdr_len),
                    <void*> tmpptr,
                    <size_t> srcsize,
                    clevel,
                )
                free(tmpptr)
            else:
                ret = ZSTD_compress(
                    <void*> (dstptr + hdr_len),
                    <size_t> (dstsize - hdr_len),
                    <void*> src._data,
                    <size_t> srcsize,
                    clevel,
                )
            if ZSTD_isError(ret):
                raise Zstd1Error('ZSTD_compress', err=ret)
            ret += <size_t> hdr_len

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def zstd1_decode(
    data,
    /,
    *,
    ssize_t itemsize=1,
    ssize_t samples=1,
    out=None,
):
    """Return decoded ZSTD1 data.

    Also accepts plain Zstd streams (magic bytes 0x28B52FFD).
    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, hdr_pos, items
        uint64_t cntsize
        uint8_t header_size, chunk_type
        uint8_t* tmp = NULL
        bint hilo = False
        size_t ret

    if data is out:
        raise ValueError('cannot transform in-place')

    if srcsize < 2:
        raise ValueError(f'{srcsize=} < 2')

    if itemsize < 1:
        raise ValueError(f'{itemsize=} < 1')

    if samples not in {1, 3, 4}:
        raise ValueError(f'{samples=} not in {{1, 3, 4}}')

    if (
        srcsize >= 4
        and src[0] == 0x28
        and src[1] == 0xB5
        and src[2] == 0x2F
        and src[3] == 0xFD
    ):
        # plain Zstd stream: no ZSTD1 header, no hilo shuffle
        header_size = 0
    else:
        header_size = src[0]
        if header_size < 1:
            raise ValueError(f'{header_size=} < 1')
        if <ssize_t> header_size >= srcsize:
            raise ValueError(f'{header_size=} >= {srcsize=}')

        # parse typed chunks in bytes [1 .. header_size-1]
        hdr_pos = 1
        while hdr_pos < header_size:
            chunk_type = src[hdr_pos]
            hdr_pos += 1
            if chunk_type == 1:
                if hdr_pos >= header_size:
                    raise ValueError('truncated chunk type 1')
                hilo = (src[hdr_pos] & 1) != 0
                hdr_pos += 1
            else:
                break  # unknown chunk type; forward-compatible stop

    out, dstsize, outgiven, outtype = _parse_output(out)

    # compressed frame starts at src[header_size]
    if out is None:
        if dstsize < 0:
            cntsize = _zstd1_content_size(
                <uint8_t*> &src[header_size],
                <size_t> (srcsize - header_size),
            )
            if cntsize == ZSTD_CONTENTSIZE_ERROR:
                raise Zstd1Error(
                    'ZSTD_getFrameContentSize', 'ZSTD_CONTENTSIZE_ERROR'
                )
            if cntsize == ZSTD_CONTENTSIZE_UNKNOWN or cntsize > SIZE_MAX >> 1:
                # use streaming API for unknown or suspiciously large sizes
                return _zstd1_decode_stream(
                    src, header_size, itemsize, samples, hilo, outtype
                )
            dstsize = <ssize_t> cntsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    with nogil:
        if hilo and itemsize > 1:
            # decompress into tmp, unshuffle directly into dst to avoid memcpy
            tmp = <uint8_t*> malloc(<size_t> dstsize)
            if tmp == NULL:
                raise MemoryError('failed to allocate hilo_unpack buffer')
            ret = ZSTD_decompress(
                tmp,
                <size_t> dstsize,
                <void*> &src[header_size],
                <size_t> (srcsize - header_size),
            )
            if ZSTD_isError(ret):
                free(tmp)
                raise Zstd1Error('ZSTD_decompress', err=ret)
            items = <ssize_t> ret // itemsize
            if items * itemsize == <ssize_t> ret:
                _hilo_unpack(tmp, <uint8_t*> dst._data, ret, itemsize)
            free(tmp)

            if items * itemsize != <ssize_t> ret:
                raise ValueError(
                    'decompressed size not a multiple of itemsize'
                )
        else:
            ret = ZSTD_decompress(
                <void*> dst._data,
                <size_t> dstsize,
                <void*> &src[header_size],
                <size_t> (srcsize - header_size),
            )
            if ZSTD_isError(ret):
                raise Zstd1Error('ZSTD_decompress', err=ret)

        if samples == 3 or samples == 4:
            items = <ssize_t> ret // (samples * itemsize)
            if items * samples * itemsize != <ssize_t> ret:
                raise ValueError(
                    'decompressed size not a multiple of samples*itemsize'
                )
            _bgr_swap(<uint8_t*> dst._data, items, samples, itemsize)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


cdef _zstd1_decode_stream(
    const uint8_t[::1] src,
    ssize_t header_offset,
    ssize_t itemsize,
    ssize_t samples,
    bint hilo,
    outtype,
):
    """Decompress ZSTD1 using streaming API, then optionally unshuffle."""
    cdef:
        output_t* output = NULL
        ZSTD_DCtx* dctx = NULL
        ZSTD_inBuffer zinput
        ZSTD_outBuffer zoutput
        ssize_t size, items
        size_t srcsize = <size_t> (src.shape[0] - header_offset)
        size_t outsize = ZSTD_DStreamOutSize()
        size_t incsize = max((srcsize // outsize) * outsize // 2, outsize)
        uint8_t* tmp = NULL
        size_t ret

    try:
        with nogil:
            dctx = ZSTD_createDCtx()
            if dctx == NULL:
                raise Zstd1Error('ZSTD_createDCtx', 'NULL')

            output = output_new(NULL, incsize * 3)
            if output == NULL:
                raise MemoryError('output_new failed')

            zoutput.dst = <void*> output.data
            zoutput.size = <size_t> output.size
            zoutput.pos = output.pos

            zinput.src = <void*> &src[header_offset]
            zinput.size = srcsize
            zinput.pos = 0

            while zinput.pos < zinput.size:
                if output.size - output.used < outsize:
                    ret = output_resize(output, output.used + incsize)
                    if ret == 0:
                        raise MemoryError('output_resize failed')
                    zoutput.dst = <void*> output.data
                    zoutput.size = <size_t> output.size
                    zoutput.pos = output.pos

                ret = ZSTD_decompressStream(dctx, &zoutput, &zinput)
                if ZSTD_isError(ret):
                    raise Zstd1Error('ZSTD_decompressStream', err=ret)

                # output.pos = zoutput.pos
                output.used = zoutput.pos

            if hilo and itemsize > 1:
                size = <ssize_t> output.pos
                items = size // itemsize
                if items * itemsize != size:
                    raise ValueError(
                        'decompressed size not a multiple of itemsize'
                    )
                tmp = <uint8_t*> malloc(<size_t> size)
                if tmp == NULL:
                    raise MemoryError('failed to allocate hilo_unpack buffer')
                # unshuffle output.data to tmp
                # _create_output will copy from tmp
                _hilo_unpack(
                    output.data, tmp, <size_t> size, <size_t> itemsize
                )

                if samples == 3 or samples == 4:
                    items = size // (samples * itemsize)
                    if items * samples * itemsize != size:
                        raise ValueError(
                            'decompressed size not a multiple of '
                            'samples*itemsize'
                        )
                    _bgr_swap(tmp, items, samples, itemsize)
            elif samples == 3 or samples == 4:
                size = <ssize_t> output.pos
                items = size // (samples * itemsize)
                if items * samples * itemsize != size:
                    raise ValueError(
                        'decompressed size not a multiple of samples*itemsize'
                    )
                _bgr_swap(output.data, items, samples, itemsize)

        if hilo and itemsize > 1:
            out = _create_output(outtype, size, <const char*> tmp)
        else:
            out = _create_output(
                outtype, output.pos, <const char*> output.data
            )

    finally:
        free(tmp)
        output_del(output)
        ZSTD_freeDCtx(dctx)

    return out


cdef uint64_t _zstd1_content_size(
    const uint8_t* src,
    const size_t srcsize,
) noexcept nogil:
    """Return uncompressed size of all frames in buffer."""
    cdef:
        uint64_t frame_content_size
        size_t frame_compressed_size
        size_t offset = 0
        uint64_t dstsize = 0

    while offset < srcsize:
        frame_compressed_size = ZSTD_findFrameCompressedSize(
            <const void*> (src + offset), srcsize - offset
        )
        if ZSTD_isError(frame_compressed_size):
            return ZSTD_CONTENTSIZE_ERROR  # frame_compressed_size

        frame_content_size = ZSTD_getFrameContentSize(
            <const void*> (src + offset), frame_compressed_size
        )
        if (
            frame_content_size == ZSTD_CONTENTSIZE_ERROR
            or frame_content_size == ZSTD_CONTENTSIZE_UNKNOWN
        ):
            return frame_content_size

        dstsize += frame_content_size
        offset += frame_compressed_size

    return dstsize


# CHUNKED #####################################################################

cdef:
    const int CHUNK_EOH = 0
    const int CHUNK_SIZES = 1
    const int CHUNK_CODEC = 2
    const int CHUNK_DECOMP = 3
    const int CHUNK_PREPROC = 4

    const int CODEC_ZSTD = 0
    const int CODEC_LZ4 = 1

    const int ERROR_TRUNCATED = -1
    const int ERROR_VARINT2 = -2
    const int ERROR_VARINT3 = -3
    const int ERROR_VARINT4 = -4
    const int ERROR_OVERRUN = -5
    const int ERROR_NOMEM = -6
    const int ERROR_INVALID = -7


class CHUNKED:
    """CHUNKED codec constants."""

    available = True

    class CODEC(enum.IntEnum):
        """CHUNKED per-chunk codec."""

        ZSTD = CODEC_ZSTD
        LZ4 = CODEC_LZ4


class ChunkedError(RuntimeError):
    """CHUNKED codec exceptions."""

    def __init__(self, func, msg=''):
        msg = f'{func} returned {msg!r}' if msg else f'{func} failed'
        super().__init__(msg)


def chunked_version():
    """Return CHUNKED codec version string."""
    return 'chunked 2026.10.10'


def chunked_check(const uint8_t[::1] data, /):
    """Return whether data is CHUNKED encoded or None if unknown."""
    cdef:
        uint32_t* comp_sizes = NULL
        uint32_t* decomp_sizes = NULL
        ssize_t nchunks = 0
        ssize_t srcsize = data.shape[0]
        ssize_t header_size = 0
        ssize_t comp_size = 0
        ssize_t i
        int codec_id
        bint hilo_tmp

    if srcsize < 1:
        return False
    try:
        header_size = _chunked_header_parse(
            data,
            &codec_id,
            &hilo_tmp,
            &comp_sizes,
            &nchunks,
            &decomp_sizes
        )
        if header_size < 0:
            return False
        for i in range(nchunks):
            comp_size += <ssize_t> comp_sizes[i]
        if header_size + comp_size == srcsize:
            return True
        if header_size + comp_size < srcsize:
            return None  # valid header but trailing bytes. could be embedded
        return False  # compressed sizes would exceed buffer
    finally:
        free(comp_sizes)
        free(decomp_sizes)


def chunked_encode(
    data,
    /,
    level=None,
    *,
    codec=None,
    chunksize=None,
    ssize_t itemsize=1,
    bint hilo=False,
    out=None,
):
    """Return CHUNKED encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data, out)
        const uint8_t[::1] dst
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize, nchunks, chunk_start, chunk_size_i, i
        ssize_t max_compressed_size, comp_compressed
        ssize_t actual_header_size, built
        uint32_t* comp_sizes = NULL
        uint8_t* compressed_buffer = NULL
        uint8_t* hilo_buffer = NULL
        const uint8_t* chunk_src_ptr = NULL
        ssize_t chunk_size = 65536 if chunksize is None else chunksize
        int codec_id = CODEC_ZSTD
        int clevel, ret_l4
        bint do_hilo = hilo and itemsize > 1
        size_t ret_zstd, zbound

    if chunk_size < 1:
        raise ValueError(f'{chunksize=} < 1')
    if itemsize < 1:
        raise ValueError(f'{itemsize=} < 1')
    if srcsize < 1:
        raise ValueError('chunked_encode: empty input not supported')

    if codec is not None:
        codec_id = _enum_value(codec, CHUNKED.CODEC)

    if codec_id == CODEC_ZSTD:
        clevel = _default_value(
            level, ZSTD_CLEVEL_DEFAULT, 0, 22  # ZSTD_CLEVEL_DEFAULT = 3
        )
    else:
        clevel = 0  # not used

    nchunks = (srcsize + chunk_size - 1) // chunk_size
    comp_sizes = <uint32_t*> malloc(<size_t> nchunks * sizeof(uint32_t))
    if comp_sizes == NULL:
        raise MemoryError('failed to allocate comp_sizes')

    try:
        # compute maximum compressed size (upper bound for temp buffer)
        max_compressed_size = 0
        for i in range(nchunks):
            chunk_start = i * chunk_size
            chunk_size_i = srcsize - chunk_start
            if chunk_size_i > chunk_size:
                chunk_size_i = chunk_size
            if codec_id == CODEC_ZSTD:
                zbound = ZSTD_compressBound(<size_t> chunk_size_i)
                if zbound == 0:
                    raise ChunkedError('ZSTD_compressBound', 'returned 0')
                max_compressed_size += <ssize_t> zbound
            else:
                if chunk_size_i > LZ4_MAX_INPUT_SIZE:
                    raise ValueError(
                        f'chunk size {chunk_size_i} exceeds LZ4_MAX_INPUT_SIZE'
                    )
                max_compressed_size += LZ4_compressBound(<int> chunk_size_i)

        compressed_buffer = <uint8_t*> malloc(<size_t> max_compressed_size)
        if compressed_buffer == NULL:
            raise MemoryError('failed to allocate compressed_buffer')

        if do_hilo:
            hilo_buffer = <uint8_t*> malloc(<size_t> chunk_size)
            if hilo_buffer == NULL:
                raise MemoryError('failed to allocate hilo_buffer')

        with nogil:
            comp_compressed = 0

            # TODO: parallelize chunk compression?
            for i in range(nchunks):
                chunk_start = i * chunk_size
                chunk_size_i = srcsize - chunk_start
                if chunk_size_i > chunk_size:
                    chunk_size_i = chunk_size

                if do_hilo:
                    _hilo_pack(
                        <const uint8_t*> (src._data + chunk_start),
                        hilo_buffer,
                        <size_t> chunk_size_i,
                        2,
                    )
                    chunk_src_ptr = hilo_buffer
                else:
                    chunk_src_ptr = <const uint8_t*> (src._data + chunk_start)

                if codec_id == CODEC_ZSTD:
                    ret_zstd = ZSTD_compress(
                        <void*> (compressed_buffer + comp_compressed),
                        <size_t> (max_compressed_size - comp_compressed),
                        <const void*> chunk_src_ptr,
                        <size_t> chunk_size_i,
                        clevel,
                    )
                    if ZSTD_isError(ret_zstd):
                        raise ChunkedError(
                            'ZSTD_compress',
                            ZSTD_getErrorName(ret_zstd).decode()
                        )
                    comp_sizes[i] = <uint32_t> ret_zstd
                    comp_compressed += <ssize_t> ret_zstd
                else:
                    ret_l4 = LZ4_compress_default(
                        <const char*> chunk_src_ptr,
                        <char*> (compressed_buffer + comp_compressed),
                        <int> chunk_size_i,
                        <int> (max_compressed_size - comp_compressed),
                    )
                    if ret_l4 <= 0:
                        raise ChunkedError(
                            'LZ4_compress_default', 'returned 0'
                        )
                    comp_sizes[i] = <uint32_t> ret_l4
                    comp_compressed += <ssize_t> ret_l4

            # compute actual header size now that all comp_sizes are known
            actual_header_size = _chunked_header_actual_size(
                comp_sizes, nchunks, chunk_size, srcsize, codec_id, do_hilo
            )

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            dstsize = actual_header_size + comp_compressed
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.shape[0]

        if dstsize < actual_header_size + comp_compressed:
            raise ChunkedError(
                'chunked_encode', 'output buffer too small'
            )

        with nogil:
            built = _chunked_header_build(
                <uint8_t*> dst._data,
                <size_t> dstsize,
                comp_sizes,
                nchunks,
                chunk_size,
                srcsize,
                codec_id,
                do_hilo,
            )
            if built != actual_header_size:
                raise ChunkedError(
                    'chunked_encode', 'failed to build header'
                )
            memcpy(
                <uint8_t*> dst._data + actual_header_size,
                compressed_buffer,
                <size_t> comp_compressed,
            )

    finally:
        free(comp_sizes)
        free(compressed_buffer)
        free(hilo_buffer)

    del dst
    return _return_output(
        out, dstsize, actual_header_size + comp_compressed, outgiven
    )


def chunked_decode(
    data,
    /,
    *,
    ssize_t itemsize=1,
    ssize_t samples=1,
    out=None,
):
    """Return decoded CHUNKED data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst
        uint32_t* comp_sizes = NULL
        uint32_t* decomp_sizes = NULL
        uint8_t* dstptr = NULL
        uint8_t* hilo_buffer = NULL
        uint8_t* chunk_dst = NULL
        ssize_t srcsize = src.shape[0]
        ssize_t nchunks, header_size, decomp_size, comp_size, decomp_size_max
        ssize_t src_offset, dst_offset, dstsize, items, i
        ssize_t comp_size_i, decomp_size_i
        size_t ret_zstd
        int ret_l4, codec_id
        bint do_hilo

    if data is out:
        raise ValueError('cannot transform in-place')

    if itemsize < 1:
        raise ValueError(f'{itemsize=} < 1')

    if samples not in {1, 3, 4}:
        raise ValueError(f'{samples=} not in {{1, 3, 4}}')

    try:
        header_size = _chunked_header_parse(
            src,
            &codec_id,
            &do_hilo,
            &comp_sizes,
            &nchunks,
            &decomp_sizes,
        )
        if header_size < 0:
            raise ChunkedError(
                '_chunked_header_parse', f'returned {header_size}'
            )

        comp_size = 0
        for i in range(nchunks):
            comp_size += <ssize_t> comp_sizes[i]
        decomp_size = 0
        decomp_size_max = 0
        for i in range(nchunks):
            decomp_size_i = decomp_sizes[i]
            decomp_size += decomp_size_i
            if decomp_size_max < decomp_size_i:
                decomp_size_max = decomp_size_i
        if header_size + comp_size > srcsize:
            raise ChunkedError(
                'chunked_decode', 'compressed chunks exceed buffer'
            )

        out, dstsize, outgiven, outtype = _parse_output(out)
        if out is None:
            dstsize = decomp_size
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.shape[0]

        if dstsize < decomp_size:
            raise ChunkedError(
                'chunked_decode', 'output buffer too small'
            )

        if nchunks == 0:
            del dst
            return _return_output(out, dstsize, 0, outgiven)

        with nogil:
            dstptr = <uint8_t*> dst._data
            src_offset = header_size
            dst_offset = 0

            if do_hilo and decomp_size_max > 1:
                hilo_buffer = <uint8_t*> malloc(decomp_size_max)
                if hilo_buffer == NULL:
                    raise MemoryError('failed to allocate hilo_buffer')

            # TODO: parallelize this loop?
            for i in range(nchunks):
                comp_size_i = comp_sizes[i]
                decomp_size_i = decomp_sizes[i]

                # when hilo: decompress into hilo_buffer, unshuffle into dst
                chunk_dst = (
                    hilo_buffer if hilo_buffer != NULL
                    else dstptr + dst_offset
                )

                if codec_id == CODEC_ZSTD:
                    ret_zstd = ZSTD_decompress(
                        <void*> chunk_dst,
                        <size_t> decomp_size_i,
                        <const void*> (src._data + src_offset),
                        <size_t> comp_size_i,
                    )
                    if ZSTD_isError(ret_zstd):
                        raise ChunkedError(
                            'ZSTD_decompress',
                            ZSTD_getErrorName(ret_zstd).decode()
                        )
                    if ret_zstd != <size_t> decomp_size_i:
                        raise ChunkedError(
                            'chunked_decode', 'unexpected decompressed size'
                        )
                else:
                    if comp_size_i > INT32_MAX or decomp_size_i > INT32_MAX:
                        raise ChunkedError(
                            'chunked_decode',
                            'LZ4 chunk size exceeds INT32_MAX'
                        )
                    ret_l4 = LZ4_decompress_safe(
                        <const char*> (src._data + src_offset),
                        <char*> chunk_dst,
                        <int> comp_size_i,
                        <int> decomp_size_i,
                    )
                    if ret_l4 < 0:
                        raise ChunkedError('LZ4_decompress_safe', 'failed')
                    if ret_l4 != <int> decomp_size_i:
                        raise ChunkedError(
                            'chunked_decode', 'unexpected decompressed size'
                        )

                if hilo_buffer != NULL:
                    _hilo_unpack(
                        hilo_buffer,
                        dstptr + dst_offset,
                        <size_t> decomp_size_i,
                        2,
                    )

                src_offset += comp_size_i
                dst_offset += decomp_size_i

            # BGR swap is applied to full output
            # because chunk boundaries need not align with pixel boundaries
            if samples == 3 or samples == 4:
                items = decomp_size // (samples * itemsize)
                if items * samples * itemsize != decomp_size:
                    raise ChunkedError(
                        'chunked_decode',
                        'decompressed size not a multiple of samples*itemsize'
                    )
                _bgr_swap(dstptr, items, samples, itemsize)

    finally:
        free(comp_sizes)
        free(decomp_sizes)
        free(hilo_buffer)

    del dst
    return _return_output(out, dstsize, decomp_size, outgiven)


# Helpers #####################################################################


cdef ssize_t _chunked_header_parse(
    const uint8_t[::1] src,
    int* out_codec_id,
    bint* out_hilo,
    uint32_t** out_comp_sizes,
    ssize_t* out_nchunks,
    uint32_t** out_decomp_sizes,
) noexcept nogil:
    """Parse CHUNKED header and return header byte size or error code."""
    # caller must free *out_comp_sizes and *out_decomp_sizes, even on failure
    cdef:
        ssize_t srcsize = src.shape[0]
        ssize_t pos = 0
        ssize_t ret
        uint32_t* decomp_buf = NULL
        uint32_t* decomp_expanded = NULL
        ssize_t comp_count = 0
        ssize_t decomp_count = 0
        uint32_t payload_len = 0
        uint32_t chunk_id = 0
        uint8_t pp
        int codec_id = CODEC_ZSTD
        bint hilo = False

    out_nchunks[0] = 0
    out_comp_sizes[0] = NULL
    out_decomp_sizes[0] = NULL

    while True:
        ret = _varint2_read(src, pos, srcsize, &chunk_id)
        if ret < 0:
            free(decomp_buf)
            return ret
        pos += ret

        if chunk_id == CHUNK_EOH:
            break

        ret = _varint3_read(src, pos, srcsize, &payload_len)
        if ret < 0:
            free(decomp_buf)
            return ret
        pos += ret

        if pos + <ssize_t> payload_len > srcsize:
            free(decomp_buf)
            return ERROR_TRUNCATED

        if chunk_id == CHUNK_SIZES:
            # id=1: ChunkSizes (free previous if duplicate)
            free(out_comp_sizes[0])
            ret = _varint4_array_read(
                src,
                pos,
                <ssize_t> payload_len,
                srcsize,
                out_comp_sizes,
                &comp_count,
            )
            if ret < 0:
                free(decomp_buf)
                return ret

        elif chunk_id == CHUNK_DECOMP:
            # id=3: DecompressedSizes (free previous if duplicate)
            free(decomp_buf)
            ret = _varint4_array_read(
                src,
                pos,
                <ssize_t> payload_len,
                srcsize,
                &decomp_buf,
                &decomp_count,
            )
            if ret < 0:
                return ret

        elif chunk_id == CHUNK_CODEC:
            # id=2: CompressionMethod
            if payload_len != 1:
                free(decomp_buf)
                return ERROR_INVALID
            codec_id = src[pos]
            if codec_id != CODEC_ZSTD and codec_id != CODEC_LZ4:
                free(decomp_buf)
                return ERROR_INVALID

        elif chunk_id == CHUNK_PREPROC:
            # id=4: Preprocessing
            if payload_len != 1:
                free(decomp_buf)
                return ERROR_INVALID
            pp = src[pos]
            if pp > 1:
                free(decomp_buf)
                return ERROR_INVALID
            hilo = pp == 1

        else:
            free(decomp_buf)
            return ERROR_INVALID

        pos += <ssize_t> payload_len

    if out_comp_sizes[0] == NULL or decomp_buf == NULL:
        free(decomp_buf)
        return ERROR_INVALID

    if comp_count > 0:
        decomp_expanded = <uint32_t*> malloc(
            <size_t> comp_count * sizeof(uint32_t)
        )
        if decomp_expanded == NULL:
            free(decomp_buf)
            return ERROR_NOMEM

    # set out_decomp_sizes before expand so caller frees it even on failure
    out_decomp_sizes[0] = decomp_expanded
    ret = _chunked_decomp_expand(
        decomp_expanded,
        comp_count,
        decomp_buf,
        decomp_count
    )
    free(decomp_buf)
    if ret < 0:
        return ERROR_INVALID

    out_codec_id[0] = codec_id
    out_hilo[0] = hilo
    out_nchunks[0] = comp_count
    return pos


cdef ssize_t _varint2_read(
    const uint8_t[::1] src,
    ssize_t pos,
    ssize_t srcsize,
    uint32_t* value,
) noexcept nogil:
    """Read 1-2 byte header chunkid varint into value and return bytes used."""
    cdef:
        uint8_t b0, b1

    if pos >= srcsize:
        return ERROR_TRUNCATED
    b0 = src[pos]
    if not (b0 & 0x80):
        value[0] = b0 & 0x7F
        return 1
    if pos + 1 >= srcsize:
        return ERROR_VARINT2
    b1 = src[pos + 1]
    value[0] = (b0 & 0x7F) | (<uint32_t> b1 << 7)
    return 2


cdef ssize_t _varint3_read(
    const uint8_t[::1] src,
    ssize_t pos,
    ssize_t srcsize,
    uint32_t* value,
) noexcept nogil:
    """Read 1-3 byte payload length varint into value and return bytes used."""
    cdef:
        uint8_t b0, b1, b2

    if pos >= srcsize:
        return ERROR_TRUNCATED
    b0 = src[pos]
    if not (b0 & 0x80):
        value[0] = b0 & 0x7F
        return 1
    if pos + 1 >= srcsize:
        return ERROR_VARINT2
    b1 = src[pos + 1]
    if not (b1 & 0x80):
        value[0] = (b0 & 0x7F) | (<uint32_t> b1 << 7)
        return 2
    if pos + 2 >= srcsize:
        return ERROR_VARINT3
    b2 = src[pos + 2]
    value[0] = (
        (b0 & 0x7F) | ((<uint32_t> (b1 & 0x7F)) << 7) | (<uint32_t> b2 << 14)
    )
    return 3


cdef int _varint4_array_read(
    const uint8_t[::1] src,
    ssize_t pos,
    ssize_t payload_len,
    ssize_t srcsize,
    uint32_t** out_buf,
    ssize_t* out_count,
) noexcept nogil:
    """Fill *out_buf with uint32 array decoded from varint4 payload region."""
    # Return 0 on success or a negative error code on failure
    # On error: *out_buf is NULL and *out_count is 0
    # On success: *out_buf is a malloc'd buffer that the caller must free
    cdef:
        uint32_t* buf = NULL
        ssize_t end = pos + payload_len
        ssize_t size = 0
        ssize_t i, j
        uint8_t b0, b1, b2, b3

    out_buf[0] = NULL
    out_count[0] = 0

    # first pass: count varints and validate bounds within the payload region
    i = pos
    while i < end:
        size += 1
        if not (src[i] & 0x80):
            i += 1
            continue
        if i + 1 >= end:
            return ERROR_VARINT2
        if not (src[i + 1] & 0x80):
            i += 2
            continue
        if i + 2 >= end:
            return ERROR_VARINT3
        if not (src[i + 2] & 0x80):
            i += 3
            continue
        if i + 3 >= end:
            return ERROR_VARINT4
        i += 4
    if i != end:
        return ERROR_OVERRUN

    buf = <uint32_t*> malloc(<size_t> size * sizeof(uint32_t))
    if buf == NULL:
        return ERROR_NOMEM

    # second pass: decode varints, cannot fail
    i = pos
    for j in range(size):
        b0 = src[i]
        if not (b0 & 0x80):
            buf[j] = b0 & 0x7F
            i += 1
            continue
        b1 = src[i + 1]
        if not (b1 & 0x80):
            buf[j] = (b0 & 0x7F) | (<uint32_t> b1 << 7)
            i += 2
            continue
        b2 = src[i + 2]
        if not (b2 & 0x80):
            buf[j] = (
                (b0 & 0x7F)
                | ((<uint32_t> (b1 & 0x7F)) << 7)
                | (<uint32_t> b2 << 14)
            )
            i += 3
            continue
        b3 = src[i + 3]
        buf[j] = (
            (b0 & 0x7F)
            | ((<uint32_t> (b1 & 0x7F)) << 7)
            | ((<uint32_t> (b2 & 0x7F)) << 14)
            | (<uint32_t> b3 << 21)
        )
        i += 4

    out_buf[0] = buf
    out_count[0] = size
    return 0


cdef int _chunked_decomp_expand(
    uint32_t* dst,
    ssize_t nchunks,
    const uint32_t* decomp_vals,
    ssize_t n,
) noexcept nogil:
    """Fill dst with decompressed size per chunk and return error code."""
    # decomp_vals is in compact suffix form:
    # - 1 value: all chunks share that same size.
    # - 2 values: all-but-last use first value, last uses second value.
    # - n values: second-to-last value repeats for all prefix chunks not
    #   explicitly listed. Then explicit values apply in order.
    cdef:
        ssize_t prefix, i
        uint32_t repeating

    if n == 0:
        return ERROR_INVALID
    if nchunks == 0:
        return 0
    if n > nchunks:
        return ERROR_INVALID
    if n == 1:
        for i in range(nchunks):
            dst[i] = decomp_vals[0]
        return 0

    # general suffix form:
    # leading (nchunks - n) chunks repeat second-to-last explicit value
    prefix = nchunks - n
    repeating = decomp_vals[n - 2]
    for i in range(nchunks):
        if i < prefix:
            dst[i] = repeating
        else:
            dst[i] = decomp_vals[i - prefix]
    return 0


cdef inline uint32_t _varint3_encoded_size(uint32_t v) noexcept nogil:
    """Return bytes needed to encode v as varint3 (payload lengths)."""
    if v < 128:
        return 1
    if v < 16384:
        return 2
    if v < 4194304:
        return 3
    return 0  # error


cdef ssize_t _chunked_header_actual_size(
    const uint32_t* comp_sizes,
    ssize_t nchunks,
    ssize_t chunksize,
    ssize_t srcsize,
    int codec_id,
    bint do_hilo,
) noexcept nogil:
    """Return header byte count for given encoding parameters."""
    cdef:
        ssize_t total, payload_size, i
        uint32_t decomp_common, decomp_last
        bint two_decomp

    # ChunkSizes: id(1) + length_varint + payload
    payload_size = 0
    for i in range(nchunks):
        payload_size += _varint4_encoded_size(comp_sizes[i])
    total = 1
    total += <ssize_t> _varint3_encoded_size(<uint32_t> payload_size)
    total += payload_size

    # DecompressedSizes: id(1) + length_varint + payload
    if nchunks == 0:
        payload_size = _varint4_encoded_size(0)
    elif nchunks == 1:
        payload_size = _varint4_encoded_size(<uint32_t> srcsize)
    else:
        decomp_common = <uint32_t> chunksize
        decomp_last = <uint32_t> (srcsize - (nchunks - 1) * chunksize)
        two_decomp = decomp_last != decomp_common
        payload_size = _varint4_encoded_size(decomp_common)
        if two_decomp:
            payload_size += _varint4_encoded_size(decomp_last)
    total += 1
    total += <ssize_t> _varint3_encoded_size(<uint32_t> payload_size)
    total += payload_size

    # CompressionMethod (optional, 3 bytes if not zstd)
    if codec_id != CODEC_ZSTD:
        total += 3

    # Preprocessing (optional, 3 bytes if hilo)
    if do_hilo:
        total += 3

    # EndOfHeader
    total += 1

    return total


cdef inline size_t _varint4_write(
    uint8_t* p,
    size_t avail,
    uint32_t v
) noexcept nogil:
    """Write v as 1-4 byte varint and return number of bytes written."""
    if v < 128:
        if avail < 1:
            return 0
        p[0] = <uint8_t> v
        return 1
    if v < 16384:
        if avail < 2:
            return 0
        p[0] = <uint8_t> ((v & 0x7F) | 0x80)
        p[1] = <uint8_t> (v >> 7)
        return 2
    if v < 4194304:
        if avail < 3:
            return 0
        p[0] = <uint8_t> ((v & 0x7F) | 0x80)
        p[1] = <uint8_t> (((v >> 7) & 0x7F) | 0x80)
        p[2] = <uint8_t> (v >> 14)
        return 3
    if v < 536870912:
        if avail < 4:
            return 0
        p[0] = <uint8_t> ((v & 0x7F) | 0x80)
        p[1] = <uint8_t> (((v >> 7) & 0x7F) | 0x80)
        p[2] = <uint8_t> (((v >> 14) & 0x7F) | 0x80)
        p[3] = <uint8_t> (v >> 21)
        return 4
    return 0  # value exceeds 29-bit range


cdef inline size_t _varint3_write(
    uint8_t* p,
    size_t avail,
    uint32_t v
) noexcept nogil:
    """Write v as 1-3 byte varint and return number of bytes written."""
    if v < 128:
        if avail < 1:
            return 0
        p[0] = <uint8_t> v
        return 1
    if v < 16384:
        if avail < 2:
            return 0
        p[0] = <uint8_t> ((v & 0x7F) | 0x80)
        p[1] = <uint8_t> (v >> 7)
        return 2
    if v < 4194304:
        if avail < 3:
            return 0
        p[0] = <uint8_t> ((v & 0x7F) | 0x80)
        p[1] = <uint8_t> (((v >> 7) & 0x7F) | 0x80)
        p[2] = <uint8_t> (v >> 14)
        return 3
    return 0  # value exceeds 22-bit range


cdef inline uint32_t _varint4_encoded_size(
    uint32_t v
) noexcept nogil:
    """Return number of bytes needed to encode v as varint4."""
    if v < 128:
        return 1
    if v < 16384:
        return 2
    if v < 4194304:
        return 3
    if v < 536870912:
        return 4
    return 0  # error: value out of range


cdef ssize_t _chunked_header_build(
    uint8_t* dst,
    size_t max_size,
    const uint32_t* comp_sizes,
    ssize_t nchunks,
    ssize_t chunksize,
    ssize_t srcsize,
    int codec_id,
    bint do_hilo,
) noexcept nogil:
    """Serialize CHUNKED header into dst and return number bytes written."""
    cdef:
        size_t n, offset = 0
        ssize_t i, payload_size
        uint32_t decomp_common = 0
        uint32_t decomp_last = 0
        bint two_decomp = False

    # id=1: ChunkSizes (required)
    # compute byte length of payload (sum of per-value encoded sizes)
    payload_size = 0
    for i in range(nchunks):
        payload_size += _varint4_encoded_size(comp_sizes[i])

    # id byte (value 1 fits in one byte)
    if offset + 1 > max_size:
        return -1
    dst[offset] = 1
    offset += 1

    # payload length (varint3)
    n = _varint3_write(
        dst + offset, max_size - offset, <uint32_t> payload_size
    )
    if n == 0:
        return -1
    offset += n

    # compressed-size values
    for i in range(nchunks):
        n = _varint4_write(dst + offset, max_size - offset, comp_sizes[i])
        if n == 0:
            return -1
        offset += n

    # id=3: DecompressedSizes (required)
    # determine compact representation of per-chunk uncompressed sizes.
    # each full chunk is exactly chunksize bytes. last chunk may be smaller.
    if nchunks == 0:
        decomp_common = 0
        two_decomp = False
    elif nchunks == 1:
        decomp_common = <uint32_t> srcsize
        two_decomp = False
    else:
        decomp_common = <uint32_t> chunksize
        decomp_last = <uint32_t> (srcsize - (nchunks - 1) * chunksize)
        two_decomp = decomp_last != decomp_common

    payload_size = _varint4_encoded_size(decomp_common)
    if two_decomp:
        payload_size += _varint4_encoded_size(decomp_last)

    # id byte
    if offset + 1 > max_size:
        return -1
    dst[offset] = 3
    offset += 1

    # payload length (varint3)
    n = _varint3_write(
        dst + offset, max_size - offset, <uint32_t> payload_size
    )
    if n == 0:
        return -1
    offset += n

    # common (repeating) uncompressed size
    n = _varint4_write(dst + offset, max_size - offset, decomp_common)
    if n == 0:
        return -1
    offset += n

    # last-chunk uncompressed size (only when different from common)
    if two_decomp:
        n = _varint4_write(dst + offset, max_size - offset, decomp_last)
        if n == 0:
            return -1
        offset += n

    # id=2: CompressionMethod (omitted when codec is zstd)
    if codec_id != CODEC_ZSTD:
        # id(1) + length(1 byte, value=1) + payload(1 byte) = 3 bytes total
        if offset + 3 > max_size:
            return -1
        dst[offset] = 2  # id=2
        offset += 1
        dst[offset] = 1  # payload length = 1
        offset += 1
        dst[offset] = <uint8_t> codec_id  # codec value
        offset += 1

    # id=4: Preprocessing (omitted when no hi/lo packing)
    if do_hilo:
        # id(1) + length(1 byte, value=1) + payload(1 byte) = 3 bytes total
        if offset + 3 > max_size:
            return -1
        dst[offset] = 4  # id=4
        offset += 1
        dst[offset] = 1  # payload length = 1
        offset += 1
        dst[offset] = 1  # preprocessing applied
        offset += 1

    # id=0: EndOfHeader
    if offset + 1 > max_size:
        return -1
    dst[offset] = 0
    offset += 1

    return <ssize_t> offset  # number of bytes written (actual header size)


cdef void _hilo_pack(
    const uint8_t* src,
    uint8_t* dst,
    size_t size,
    size_t itemsize,
) noexcept nogil:
    """Group bytes by byte-position within itemsize-stride."""
    cdef:
        size_t ngroups = size // itemsize
        size_t k, g

    for k in range(itemsize):
        for g in range(ngroups):
            dst[k * ngroups + g] = src[g * itemsize + k]
    for g in range(ngroups * itemsize, size):
        dst[g] = src[g]


cdef void _hilo_unpack(
    const uint8_t* src,
    uint8_t* dst,
    size_t size,
    size_t itemsize,
) noexcept nogil:
    """Ungroup bytes by byte-position within itemsize-stride."""
    cdef:
        size_t ngroups = size // itemsize
        size_t k, g

    for k in range(itemsize):
        for g in range(ngroups):
            dst[g * itemsize + k] = src[k * ngroups + g]
    for g in range(ngroups * itemsize, size):
        dst[g] = src[g]


cdef void _bgr_swap(
    uint8_t* data,
    ssize_t items,
    ssize_t samples,
    ssize_t itemsize,
) noexcept nogil:
    """Swap channel 0, 2 in-place (BGR->RGB) and force alpha=255 for BGRA32."""
    cdef:
        ssize_t stride = samples * itemsize
        ssize_t i, j
        uint8_t  tmp_u8
        uint16_t tmp_u16
        uint32_t tmp_u32
        uint64_t tmp_u64
        uint16_t* p16
        uint32_t* p32
        uint64_t* p64

    if itemsize == 1:
        for i in range(items):
            tmp_u8 = data[i * stride]
            data[i * stride] = data[i * stride + 2]
            data[i * stride + 2] = tmp_u8
        if samples == 4:
            for i in range(items):
                data[i * stride + 3] = 0xFF
    elif itemsize == 2:
        for i in range(items):
            p16 = <uint16_t*> (data + i * stride)
            tmp_u16 = p16[0]
            p16[0] = p16[2]
            p16[2] = tmp_u16
    elif itemsize == 4:
        for i in range(items):
            p32 = <uint32_t*> (data + i * stride)
            tmp_u32 = p32[0]
            p32[0] = p32[2]
            p32[2] = tmp_u32
    elif itemsize == 8:
        for i in range(items):
            p64 = <uint64_t*> (data + i * stride)
            tmp_u64 = p64[0]
            p64[0] = p64[2]
            p64[2] = tmp_u64
    else:
        for i in range(items):
            for j in range(itemsize):
                tmp_u8 = data[i * stride + j]
                data[i * stride + j] = data[i * stride + 2 * itemsize + j]
                data[i * stride + 2 * itemsize + j] = tmp_u8


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
