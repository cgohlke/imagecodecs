# imagecodecs/_wavpack.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2026, Christoph Gohlke
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

"""WavPack codec for the imagecodecs package."""

include '_shared.pxi'

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
)
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memcpy, memset
from wavpack cimport *


class WAVPACK:
    """WavPack codec constants."""

    available = True

    class LEVEL(enum.IntEnum):
        """Compression level."""

        FAST = 1
        DEFAULT = 2
        HIGH = 3
        VERY_HIGH = 4


class WavpackError(RuntimeError):
    """WavPack codec exceptions."""

    def __init__(self, func, msg=''):
        msg = f'{func} returned {msg!r}'
        super().__init__(msg)


def wavpack_version():
    """Return WavPack library version string."""
    return 'libwavpack ' + WavpackGetLibraryVersionString().decode()


def wavpack_check(const uint8_t[::1] data, /):
    """Return whether data is WavPack encoded, or None if unknown."""
    if data.nbytes < 4:
        return False
    return bytes(data[:4]) == b'wvpk'


def wavpack_encode(
    data,
    /,
    level=None,
    *,
    bitrate=None,
    numthreads=None,
    out=None,
):
    """Return WavPack encoded data.

    Supports 1-D ``(nsamples,)`` or 2-D ``(nsamples, nchannels)`` arrays
    with dtypes int8, uint8, int16, uint16, int32, or float32.
    WavPack supports up to 4096 channels.

    """
    cdef:
        numpy.ndarray src = numpy.ascontiguousarray(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        const void* srcptr = src.data
        ssize_t srcsize = <ssize_t> src.size
        ssize_t dstsize
        int itemsize = <int> src.itemsize
        char dtype_char = <char> ord(src.dtype.char)
        WavpackContext* wpc = NULL
        WavpackWriteContext wctx
        WavpackConfig config
        int32_t* ibuf = NULL
        int64_t nsamples
        int32_t block_samples, config_flags_val
        float config_bitrate = <float> (0.0 if bitrate is None else bitrate)
        ssize_t initcap, i
        int nchannels, ret, config_float_norm_exp, config_qmode
        int clevel = _default_value(level, 2, 1, 4)
        uint8_t nthreads = _default_threads(numthreads)

    if data is out:
        raise ValueError('cannot encode in-place')

    if dtype_char not in b'bBhHilf' or itemsize > 4:
        raise ValueError(f'invalid data {src.dtype=!r}')

    if src.ndim == 1:
        nsamples = src.shape[0]
        nchannels = 1
    elif src.ndim == 2:
        nsamples = src.shape[0]
        nchannels = <int> src.shape[1]
    else:
        raise ValueError(f'invalid data {src.ndim=}D')

    if nsamples == 0:
        raise ValueError('data contains no samples')

    out, dstsize, outgiven, outtype = _parse_output(out)

    try:
        with nogil:
            wctx.data = NULL
            initcap = srcsize * itemsize + 65536
            wctx.data = <uint8_t*> malloc(initcap)
            if wctx.data == NULL:
                raise MemoryError('wavpack_encode')
            wctx.size = 0
            wctx.capacity = initcap
            wctx.overflow = False

            ibuf = <int32_t*> malloc(srcsize * sizeof(int32_t))
            if ibuf == NULL:
                raise MemoryError('wavpack_encode')

            # widen source samples to int32 (required by WavPack API)
            if dtype_char == b'b':
                for i in range(srcsize):
                    ibuf[i] = <int32_t> (<const int8_t*> srcptr)[i]
            elif dtype_char == b'B':
                for i in range(srcsize):
                    ibuf[i] = <int32_t> (<const uint8_t*> srcptr)[i]
            elif dtype_char == b'h':
                for i in range(srcsize):
                    ibuf[i] = <int32_t> (<const int16_t*> srcptr)[i]
            elif dtype_char == b'H':
                for i in range(srcsize):
                    ibuf[i] = <int32_t> (<const uint16_t*> srcptr)[i]
            elif dtype_char == b'f':
                memcpy(ibuf, srcptr, srcsize * sizeof(float))
            else:
                memcpy(ibuf, srcptr, srcsize * sizeof(int32_t))

            config_float_norm_exp = 0
            config_qmode = 0
            if dtype_char == b'b':
                config_qmode = QMODE_SIGNED_BYTES
            elif dtype_char == b'H':
                config_qmode = QMODE_UNSIGNED_WORDS
            elif dtype_char == b'f':
                config_float_norm_exp = 127

            if nsamples > 131072:
                block_samples = 131072
            else:
                block_samples = <int32_t> nsamples

            config_flags_val = CONFIG_PAIR_UNDEF_CHANS
            if clevel == 1:
                config_flags_val |= CONFIG_FAST_FLAG
            elif clevel == 3:
                config_flags_val |= CONFIG_HIGH_FLAG
            elif clevel == 4:
                config_flags_val |= CONFIG_HIGH_FLAG | CONFIG_VERY_HIGH_FLAG
            if config_bitrate != 0.0:
                config_flags_val |= CONFIG_HYBRID_FLAG

            memset(&config, 0, sizeof(WavpackConfig))
            config.num_channels = nchannels
            config.sample_rate = 44100
            config.bytes_per_sample = itemsize
            config.bits_per_sample = itemsize * 8
            config.float_norm_exp = config_float_norm_exp
            config.qmode = config_qmode
            config.block_samples = block_samples
            config.flags = config_flags_val
            config.bitrate = config_bitrate
            config.worker_threads = <int32_t> nthreads - 1
            if config.worker_threads < 0:
                config.worker_threads = 0

            wpc = WavpackOpenFileOutput(wavpack_block_output, &wctx, NULL)
            if wpc == NULL:
                raise WavpackError('WavpackOpenFileOutput', 'returned NULL')

            ret = WavpackSetConfiguration64(wpc, &config, nsamples, NULL)
            if ret == 0:
                raise WavpackError(
                    'WavpackSetConfiguration64',
                    WavpackGetErrorMessage(wpc).decode()
                )

            ret = WavpackPackInit(wpc)
            if ret == 0:
                raise WavpackError(
                    'WavpackPackInit',
                    WavpackGetErrorMessage(wpc).decode()
                )

            ret = WavpackPackSamples(wpc, ibuf, <uint32_t> nsamples)
            if ret == 0:
                raise WavpackError(
                    'WavpackPackSamples',
                    WavpackGetErrorMessage(wpc).decode()
                )

            ret = WavpackFlushSamples(wpc)
            if ret == 0:
                raise WavpackError(
                    'WavpackFlushSamples',
                    WavpackGetErrorMessage(wpc).decode()
                )

        if wctx.overflow:
            raise WavpackError('wavpack_encode', 'output buffer overflow')

        if out is None:
            out = _create_output(outtype, wctx.size)
        dst = out
        dstsize = dst.nbytes
        if dstsize < <ssize_t> wctx.size:
            raise WavpackError(
                'wavpack_encode',
                f'output buffer too small: {dstsize} < {wctx.size}'
            )
        memcpy(<void*> &dst[0], wctx.data, wctx.size)
        del dst

    finally:
        if wpc != NULL:
            WavpackCloseFile(wpc)
            wpc = NULL
        free(ibuf)
        ibuf = NULL
        if wctx.data != NULL:
            free(wctx.data)
            wctx.data = NULL

    return _return_output(out, dstsize, wctx.size, outgiven)


def wavpack_decode(
    data,
    /,
    *,
    correction=None,
    numthreads=None,
    out=None,
):
    """Return decoded WavPack data as numpy array.

    Pass correction data (content of a .wvc file) to enable lossless
    decoding of hybrid-mode streams.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] src_c
        numpy.ndarray dst
        void* dstptr = NULL
        void* wvc_id = NULL
        WavpackContext* wpc = NULL
        WavpackReadContext rctx, rctx_c
        WavpackStreamReader64 reader
        char[256] errbuf
        ssize_t dstsize, i
        int open_flags, nchannels, bps, float_norm, qmode
        char dtype_char
        int64_t nsamples
        uint32_t unpacked
        int32_t* ibuf = NULL
        uint8_t nthreads = _default_threads(numthreads)

    if data is out:
        raise ValueError('cannot decode in-place')

    if correction is not None:
        src_c = _readable_input(correction)
        rctx_c.src = &src_c[0]
        rctx_c.pos = 0
        rctx_c.size = <int64_t> src_c.shape[0]
        rctx_c.unget = -1
        wvc_id = &rctx_c

    try:
        with nogil:
            rctx.src = &src[0]
            rctx.pos = 0
            rctx.size = <int64_t> src.shape[0]
            rctx.unget = -1

            reader.read_bytes = wavpack_read_bytes
            reader.write_bytes = NULL
            reader.get_pos = wavpack_get_pos
            reader.set_pos_abs = wavpack_set_pos_abs
            reader.set_pos_rel = wavpack_set_pos_rel
            reader.push_back_byte = wavpack_push_back_byte
            reader.get_length = wavpack_get_length
            reader.can_seek = wavpack_can_seek
            reader.truncate_here = wavpack_truncate_here
            reader.close = wavpack_close_reader

            open_flags = OPEN_ALT_TYPES | (<int> nthreads << OPEN_THREADS_SHFT)
            memset(errbuf, 0, 256)

            wpc = WavpackOpenFileInputEx64(
                &reader, &rctx, wvc_id, errbuf, open_flags, 0
            )
            if wpc == NULL:
                raise WavpackError(
                    'WavpackOpenFileInputEx64', errbuf.decode()
                )

            nsamples = WavpackGetNumSamples64(wpc)
            if nsamples <= 0:
                raise WavpackError(
                    'WavpackGetNumSamples64', 'cannot determine sample count'
                )

            nchannels = WavpackGetNumChannels(wpc)
            bps = WavpackGetBytesPerSample(wpc)
            float_norm = WavpackGetFloatNormExp(wpc)
            qmode = WavpackGetQualifyMode(wpc)

            dstsize = <ssize_t> nsamples * nchannels
            if (
                dstsize <= 0
                or dstsize // nchannels != <ssize_t> nsamples
                or dstsize > <ssize_t> ((<size_t> -1) // sizeof(int32_t))
            ):
                raise WavpackError(
                    'wavpack_decode', 'sample count too large'
                )
            ibuf = <int32_t*> malloc(dstsize * sizeof(int32_t))
            if ibuf == NULL:
                raise MemoryError('failed to allocate sample buffer')

            unpacked = WavpackUnpackSamples(wpc, ibuf, <uint32_t> nsamples)
            if unpacked != <uint32_t> nsamples:
                raise WavpackError(
                    'WavpackUnpackSamples',
                    f'expected {nsamples} samples, got {unpacked}'
                )

        # determine numpy dtype from bitstream metadata
        if float_norm == 127 and bps == 4:
            dtype = numpy.dtype(numpy.float32)
        elif bps == 1:
            if qmode & QMODE_SIGNED_BYTES:
                dtype = numpy.dtype(numpy.int8)
            else:
                dtype = numpy.dtype(numpy.uint8)
        elif bps == 2:
            if qmode & QMODE_UNSIGNED_WORDS:
                dtype = numpy.dtype(numpy.uint16)
            else:
                dtype = numpy.dtype(numpy.int16)
        else:
            dtype = numpy.dtype(numpy.int32)
        dtype_char = <char> ord(dtype.char)

        if nchannels == 1:
            shape = (int(nsamples),)
        else:
            shape = (int(nsamples), int(nchannels))

        out = _create_array(out, shape, dtype)
        dst = out
        dstptr = dst.data

        with nogil:
            # narrow int32 -> target dtype
            if dtype_char == b'b':
                for i in range(dstsize):
                    (<int8_t*> dstptr)[i] = <int8_t> ibuf[i]
            elif dtype_char == b'B':
                for i in range(dstsize):
                    (<uint8_t*> dstptr)[i] = <uint8_t> ibuf[i]
            elif dtype_char == b'h':
                for i in range(dstsize):
                    (<int16_t*> dstptr)[i] = <int16_t> ibuf[i]
            elif dtype_char == b'H':
                for i in range(dstsize):
                    (<uint16_t*> dstptr)[i] = <uint16_t> ibuf[i]
            else:
                memcpy(dstptr, ibuf, dstsize * sizeof(int32_t))

    finally:
        if wpc != NULL:
            WavpackCloseFile(wpc)
            wpc = NULL
        free(ibuf)
        ibuf = NULL

    return out


def wavpack_info(data, /):
    """Return metadata from a WavPack stream."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        WavpackContext* wpc = NULL
        WavpackReadContext rctx
        WavpackStreamReader64 reader
        char[256] errbuf
        int version, nchannels, bps, bits, sample_rate, mode, float_norm, qmode

    try:
        with nogil:
            rctx.src = &src[0]
            rctx.pos = 0
            rctx.size = <int64_t> src.shape[0]
            rctx.unget = -1

            reader.read_bytes = wavpack_read_bytes
            reader.write_bytes = NULL
            reader.get_pos = wavpack_get_pos
            reader.set_pos_abs = wavpack_set_pos_abs
            reader.set_pos_rel = wavpack_set_pos_rel
            reader.push_back_byte = wavpack_push_back_byte
            reader.get_length = wavpack_get_length
            reader.can_seek = wavpack_can_seek
            reader.truncate_here = wavpack_truncate_here
            reader.close = wavpack_close_reader

            memset(errbuf, 0, 256)
            wpc = WavpackOpenFileInputEx64(
                &reader, &rctx, NULL, errbuf, OPEN_ALT_TYPES, 0
            )
            if wpc == NULL:
                raise WavpackError('WavpackOpenFileInputEx64', errbuf.decode())

            version = WavpackGetVersion(wpc)
            nchannels = WavpackGetNumChannels(wpc)
            bps = WavpackGetBytesPerSample(wpc)
            bits = WavpackGetBitsPerSample(wpc)
            sample_rate = <int> WavpackGetSampleRate(wpc)
            mode = WavpackGetMode(wpc)
            float_norm = WavpackGetFloatNormExp(wpc)
            qmode = WavpackGetQualifyMode(wpc)

        nsamples = int(WavpackGetNumSamples64(wpc))
        channel_identities = numpy.empty(nchannels + 1, dtype=numpy.uint8)
        WavpackGetChannelIdentities(
            wpc, <unsigned char*> (<numpy.ndarray> channel_identities).data
        )
    finally:
        if wpc != NULL:
            WavpackCloseFile(wpc)

    if float_norm == 127 and bps == 4:
        dtype = numpy.dtype(numpy.float32)
    elif bps == 1:
        if qmode & QMODE_SIGNED_BYTES:
            dtype = numpy.dtype(numpy.int8)
        else:
            dtype = numpy.dtype(numpy.uint8)
    elif bps == 2:
        if qmode & QMODE_UNSIGNED_WORDS:
            dtype = numpy.dtype(numpy.uint16)
        else:
            dtype = numpy.dtype(numpy.int16)
    else:
        dtype = numpy.dtype(numpy.int32)

    shape = (nsamples,) if nchannels == 1 else (nsamples, nchannels)

    return {
        'shape': shape,
        'dtype': dtype,
        'version': version,
        'samplerate': sample_rate,
        'bitspersample': bits,
        'channel_identities': channel_identities[:nchannels].copy(),
        'lossless': bool(mode & MODE_LOSSLESS),
    }


cdef struct WavpackWriteContext:
    # encode callback: accumulates output WavPack blocks in growable buffer
    uint8_t* data
    ssize_t size  # bytes written so far
    ssize_t capacity  # allocated bytes
    bint overflow  # set on realloc failure


cdef struct WavpackReadContext:
    # decode callbacks: read-only view of compressed input bytes
    const uint8_t* src
    int64_t pos  # current read position
    int64_t size  # total bytes in src
    int unget  # single-byte push-back; -1 = empty


cdef int wavpack_block_output(
    void* id, void* data, int32_t bcount
) noexcept nogil:
    cdef:
        WavpackWriteContext* ctx = <WavpackWriteContext *> id
        ssize_t newcap
        uint8_t* newdata

    if ctx.overflow:
        return 0

    if ctx.size + bcount > ctx.capacity:
        newcap = <ssize_t> _align_ssize_t(ctx.capacity * 2 + <ssize_t> bcount)
        newdata = <uint8_t *> realloc(ctx.data, newcap)
        if newdata == NULL:
            ctx.overflow = True
            return 0
        ctx.data = newdata
        ctx.capacity = newcap

    memcpy(ctx.data + ctx.size, data, bcount)
    ctx.size += bcount
    return 1


cdef int32_t wavpack_read_bytes(
    void* id, void* data, int32_t bcount
) noexcept nogil:
    cdef:
        WavpackReadContext* ctx = <WavpackReadContext *> id
        int32_t avail
        int32_t nread = 0
        int64_t avail64

    if ctx.unget >= 0 and bcount > 0:
        (<uint8_t *> data)[0] = <uint8_t> ctx.unget
        ctx.unget = -1
        ctx.pos += 1
        data = (<uint8_t *> data) + 1
        bcount -= 1
        nread = 1

    avail64 = ctx.size - ctx.pos
    if avail64 < 0:
        avail64 = 0
    if avail64 > <int64_t> bcount:
        avail = bcount
    else:
        avail = <int32_t> avail64
    if bcount > avail:
        bcount = avail
    if bcount > 0:
        memcpy(data, ctx.src + ctx.pos, bcount)
        ctx.pos += bcount
        nread += bcount
    return nread


cdef int64_t wavpack_get_pos(void* id) noexcept nogil:
    return (<WavpackReadContext *> id).pos


cdef int wavpack_set_pos_abs(void* id, int64_t pos) noexcept nogil:
    cdef:
        WavpackReadContext* ctx = <WavpackReadContext *> id

    ctx.pos = pos
    ctx.unget = -1
    return 0


cdef int wavpack_set_pos_rel(
    void* id, int64_t delta, int mode
) noexcept nogil:
    cdef:
        WavpackReadContext* ctx = <WavpackReadContext *> id

    # mode: 0=SEEK_SET, 1=SEEK_CUR, 2=SEEK_END
    if mode == 0:
        ctx.pos = delta
    elif mode == 1:
        ctx.pos += delta
    else:
        ctx.pos = ctx.size + delta
    ctx.unget = -1
    return 0


cdef int wavpack_push_back_byte(void* id, int c) noexcept nogil:
    cdef:
        WavpackReadContext* ctx = <WavpackReadContext *> id

    ctx.unget = c
    if c >= 0:
        ctx.pos -= 1
    return c


cdef int64_t wavpack_get_length(void* id) noexcept nogil:
    return (<WavpackReadContext *> id).size


cdef int wavpack_can_seek(void* id) noexcept nogil:
    return 1


cdef int wavpack_truncate_here(void* id) noexcept nogil:
    return 0


cdef int wavpack_close_reader(void* id) noexcept nogil:
    return 0
