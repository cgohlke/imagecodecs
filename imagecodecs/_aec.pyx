# imagecodecs/_aec.pyx
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

"""AEC (Adaptive Entropy Coding) codec for the imagecodecs package."""

include '_shared.pxi'

from libaec cimport *


class AEC:
    """AEC codec constants."""

    available = True

    class FLAG(enum.IntEnum):  # IntFlag
        """AEC codec flags."""

        DATA_SIGNED = AEC_DATA_SIGNED
        DATA_3BYTE = AEC_DATA_3BYTE
        DATA_PREPROCESS = AEC_DATA_PREPROCESS
        RESTRICTED = AEC_RESTRICTED
        PAD_RSI = AEC_PAD_RSI
        NOT_ENFORCE = AEC_NOT_ENFORCE


class AecError(RuntimeError):
    """AEC codec exceptions."""

    def __init__(self, func, err):
        msg = {
            AEC_OK: 'AEC_OK',
            AEC_CONF_ERROR: 'AEC_CONF_ERROR',
            AEC_STREAM_ERROR: 'AEC_STREAM_ERROR',
            AEC_DATA_ERROR: 'AEC_DATA_ERROR',
            AEC_MEM_ERROR: 'AEC_MEM_ERROR',
            AEC_RSI_OFFSETS_ERROR: 'AEC_RSI_OFFSETS_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def aec_version():
    """Return libaec library version string."""
    return (
        f'libaec {AEC_VERSION_MAJOR}.{AEC_VERSION_MINOR}.{AEC_VERSION_PATCH}'
    )


def aec_check(const uint8_t[::1] data not None, /):
    """Return whether data is AEC encoded or None if unknown."""
    return None if data.shape[0] > 0 else False


def aec_encode(
    data,
    /,
    *,
    bitspersample=None,
    flags=None,
    blocksize=None,
    rsi=None,
    out=None,
):
    """Return AEC encoded data.

    Does not work well with RGB contig samples.

    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        ssize_t byteswritten
        int ret = AEC_OK
        unsigned int flags_ = 0
        unsigned int bits_per_sample = 8
        unsigned int block_size = _default_value(blocksize, 8, 8, 64)
        unsigned int rsi_ = _default_value(rsi, 2, 1, 4096)
        aec_stream strm
        bint initialized = False

    if data is out:
        raise ValueError('cannot transform in-place')

    if flags is None:
        flags_ = AEC_DATA_PREPROCESS
    else:
        flags_ = flags

    try:
        src = data  # common case: contiguous bytes
        bits_per_sample = 8
    except Exception:
        view = memoryview(data)
        if view.contiguous:
            src = view.cast('B')  # view as bytes
        else:
            src = view.tobytes()  # copy
        if view.format.lstrip('@=<>!') in {
            'b', 'h', 'i', 'l', 'q', 'n', 'f', 'd', 'e'
        }:
            flags_ |= AEC_DATA_SIGNED
        bits_per_sample = <unsigned int> (view.itemsize * 8)

    srcsize = src.shape[0]

    if bitspersample is not None:
        if bitspersample > bits_per_sample > 8:
            raise ValueError(
                f'{bitspersample=} > itemsize*8={bits_per_sample!r}'
            )
        bits_per_sample = bitspersample

    if bits_per_sample > 32:
        raise ValueError(f'invalid {bits_per_sample=}')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            # https://gitlab.dkrz.de/k202009/libaec/-/tree/master#output
            dstsize = srcsize + (srcsize // 64) * 3 + 3 + 256 + 1
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.shape[0]

    try:
        with nogil:
            memset(<void*> &strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char*> src._data
            strm.avail_in = srcsize
            strm.next_out = <unsigned char*> dst._data
            strm.avail_out = dstsize
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi_
            strm.flags = flags_

            ret = aec_encode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_encode_init', ret)
            initialized = True

            ret = aec_encode_c(&strm, AEC_FLUSH)
            if ret != AEC_OK:
                raise AecError('aec_encode', ret)

            byteswritten = <ssize_t> strm.total_out
            if strm.total_in != <size_t> srcsize:
                raise ValueError('output buffer too small')

            # AEC_STREAM_ERROR if the final flush did not fit in output
            initialized = False
            ret = aec_encode_end(&strm)
            if ret != AEC_OK:
                raise ValueError('output buffer too small')
    finally:
        if initialized:
            ret = aec_encode_end(&strm)
        # if ret != AEC_OK:
        #     raise AecError('aec_encode_end', ret)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def aec_decode(
    data,
    /,
    *,
    bitspersample=None,
    flags=None,
    blocksize=None,
    rsi=None,
    out=None,
):
    """Return decoded AEC data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.shape[0]
        ssize_t dstsize
        ssize_t byteswritten
        int ret = AEC_OK
        unsigned int flags_ = 0
        unsigned int bits_per_sample = 8
        unsigned int block_size = _default_value(blocksize, 8, 8, 64)
        unsigned int rsi_ = _default_value(rsi, 2, 1, 4096)
        aec_stream strm
        bint initialized = False

    if data is out:
        raise ValueError('cannot transform in-place')

    if flags is None:
        flags_ = AEC_DATA_PREPROCESS
    else:
        flags_ = flags

    if isinstance(out, numpy.ndarray):
        if not numpy.PyArray_ISCONTIGUOUS(out):
            # TODO: handle this
            raise ValueError('output is not contiguous')
        if bitspersample is None:
            bitspersample = out.itemsize * 8
        elif bitspersample > out.itemsize * 8:
            raise ValueError('invalid bitspersample')
        if out.dtype.kind == 'i':
            flags_ |= AEC_DATA_SIGNED

    if bitspersample is not None:
        bits_per_sample = bitspersample

    if bits_per_sample > 32:
        raise ValueError('invalid bits_per_sample')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            return _aec_decode(
                src, outtype, bits_per_sample, block_size, rsi_, flags_
            )
        out = _create_output(outtype, dstsize)

    dst = _inplace_input(out) if outgiven else out
    dstsize = dst.shape[0]

    try:
        with nogil:
            memset(<void*> &strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char*> src._data
            strm.avail_in = srcsize
            strm.next_out = <unsigned char*> dst._data
            strm.avail_out = dstsize
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi_
            strm.flags = flags_

            ret = aec_decode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_decode_init', ret)
            initialized = True

            ret = aec_decode_c(&strm, AEC_FLUSH)
            if ret != AEC_OK:
                raise AecError('aec_decode', ret)

            byteswritten = <ssize_t> strm.total_out
            if strm.total_in != <size_t> srcsize:
                raise ValueError('output buffer too small')
    finally:
        if initialized:
            ret = aec_decode_end(&strm)
        # if ret != AEC_OK:
        #     raise AecError('aec_decode_end', ret)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


cdef _aec_decode(
    const uint8_t[::1] src,
    outtype,
    unsigned int bits_per_sample,
    unsigned int block_size,
    unsigned int rsi,
    unsigned int flags,
):
    """Decode AEC stream of unknown decoded size into growing buffer."""
    cdef:
        size_t srcsize = <size_t> src.shape[0]
        size_t size = _align_size_t(srcsize * 8)
        size_t total_in, total_out
        uint8_t* buffer = NULL
        uint8_t* tmp
        aec_stream strm
        int ret = AEC_OK
        bint initialized = False

    try:
        with nogil:
            buffer = <uint8_t*> malloc(size)
            if buffer == NULL:
                raise MemoryError('failed to allocate output buffer')

            memset(<void*> &strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char*> src._data
            strm.avail_in = srcsize
            strm.next_out = <unsigned char*> buffer
            strm.avail_out = size
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi
            strm.flags = flags

            ret = aec_decode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_decode_init', ret)
            initialized = True

            while True:
                total_in = strm.total_in
                total_out = strm.total_out
                ret = aec_decode_c(&strm, AEC_FLUSH)
                # AEC_MEM_ERROR: output ended within a sample
                if ret != AEC_OK and ret != AEC_MEM_ERROR:
                    raise AecError('aec_decode', ret)
                if (
                    ret == AEC_OK
                    and strm.avail_in == 0
                    and strm.avail_out > 0
                ):
                    break
                if (
                    strm.total_in == total_in
                    and strm.total_out == total_out
                    and strm.avail_out > 0
                ):
                    raise AecError('aec_decode', AEC_STREAM_ERROR)
                # grow geometrically and resume decoding
                tmp = <uint8_t*> realloc(<void*> buffer, size * 2)
                if tmp == NULL:
                    raise MemoryError('failed to reallocate output buffer')
                buffer = tmp
                strm.next_out = <unsigned char*> (buffer + strm.total_out)
                strm.avail_out = size * 2 - strm.total_out
                size *= 2

        out = _create_output(
            outtype, <ssize_t> strm.total_out, <const char*> buffer
        )
    finally:
        if initialized:
            aec_decode_end(&strm)
        free(buffer)

    return out
