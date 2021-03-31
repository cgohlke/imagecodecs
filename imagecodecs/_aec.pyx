# imagecodecs/_aec.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019-2021, Christoph Gohlke
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

"""AEC codec for the imagecodecs package."""

__version__ = '2020.3.31'

include '_shared.pxi'

from libaec cimport *


class AEC:
    """AEC Constants."""

    DATA_SIGNED = AEC_DATA_SIGNED
    DATA_3BYTE = AEC_DATA_3BYTE
    DATA_PREPROCESS = AEC_DATA_PREPROCESS
    RESTRICTED = AEC_RESTRICTED
    PAD_RSI = AEC_PAD_RSI
    NOT_ENFORCE = AEC_NOT_ENFORCE


class AecError(RuntimeError):
    """AEC Exceptions."""

    def __init__(self, func, err):
        msg = {
            AEC_OK: 'AEC_OK',
            AEC_CONF_ERROR: 'AEC_CONF_ERROR',
            AEC_STREAM_ERROR: 'AEC_STREAM_ERROR',
            AEC_DATA_ERROR: 'AEC_DATA_ERROR',
            AEC_MEM_ERROR: 'AEC_MEM_ERROR',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def aec_version():
    """Return libaec library version string."""
    return 'libaec 1.0.4'


def aec_check(data):
    """Return True if data likely contains AEC data."""


def aec_encode(
    data,
    level=None,
    bitspersample=None,
    flags=None,
    blocksize=None,
    rsi=None,
    out=None
):
    """Compress AEC.

    Does not work well with RGB contig samples.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t byteswritten
        int ret = AEC_OK
        unsigned int flags_ = 0
        unsigned int bits_per_sample = 8
        unsigned int block_size = _default_value(blocksize, 8, 8, 64)
        unsigned int rsi_ = _default_value(rsi, 2, 1, 4096)
        aec_stream strm

    if data is out:
        raise ValueError('cannot encode in-place')

    if flags is None:
        flags_ = AEC_DATA_PREPROCESS
    else:
        flags_ = flags

    if isinstance(data, numpy.ndarray):
        if bitspersample is None:
            bitspersample = data.itemsize * 8
        elif bitspersample > data.itemsize * 8:
            raise ValueError('invalid bitspersample')
        if data.dtype.char == 'i':
            flags_ |= AEC_DATA_SIGNED

    if bitspersample:
        bits_per_sample = bitspersample

    if bits_per_sample > 32:
        raise ValueError('invalid bits_per_sample')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 2  # ? TODO: use dynamic destination buffer
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    try:
        with nogil:
            memset(&strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char*> &src[0]
            strm.avail_in = srcsize
            strm.next_out = <unsigned char*> &dst[0]
            strm.avail_out = dstsize
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi_
            strm.flags = flags_

            ret = aec_encode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_encode_init', ret)

            ret = aec_encode_c(&strm, AEC_FLUSH)
            if ret != AEC_OK:
                raise AecError('aec_encode', ret)

            byteswritten = <ssize_t> strm.total_out
            if strm.total_in != <size_t> srcsize:
                raise ValueError('output buffer too small')
    finally:
        ret = aec_encode_end(&strm)
        # if ret != AEC_OK:
        #     raise AecError('aec_encode_end', ret)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)


def aec_decode(
    data,
    bitspersample=None,
    flags=None,
    blocksize=None,
    rsi=None,
    out=None
):
    """Decompress AEC.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t byteswritten
        int ret = AEC_OK
        unsigned int flags_ = 0
        unsigned int bits_per_sample = 8
        unsigned int block_size = _default_value(blocksize, 8, 8, 64)
        unsigned int rsi_ = _default_value(rsi, 2, 1, 4096)
        aec_stream strm

    if data is out:
        raise ValueError('cannot decode in-place')

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
        if out.dtype.char == 'i':
            flags_ |= AEC_DATA_SIGNED

    if bitspersample:
        bits_per_sample = bitspersample

    if bits_per_sample > 32:
        raise ValueError('invalid bits_per_sample')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 8  # ? TODO: use dynamic destination buffer
        out = _create_output(outtype, dstsize)

    try:
        dst = out
    except ValueError:
        dst = numpy.ravel(out).view('uint8')
    dstsize = <int> dst.size

    try:
        with nogil:
            memset(&strm, 0, sizeof(aec_stream))
            strm.next_in = <unsigned char*> &src[0]
            strm.avail_in = srcsize
            strm.next_out = <unsigned char*> &dst[0]
            strm.avail_out = dstsize
            strm.bits_per_sample = bits_per_sample
            strm.block_size = block_size
            strm.rsi = rsi_
            strm.flags = flags_

            ret = aec_decode_init(&strm)
            if ret != AEC_OK:
                raise AecError('aec_decode_init', ret)

            ret = aec_decode_c(&strm, AEC_FLUSH)
            if ret != AEC_OK:
                raise AecError('aec_decode', ret)

            byteswritten = <ssize_t> strm.total_out
            if strm.total_in != <size_t> srcsize:
                raise ValueError('output buffer too small')
    finally:
        ret = aec_decode_end(&strm)
        # if ret != AEC_OK:
        #     raise AecError('aec_decode_end', ret)

    del dst
    return _return_output(out, dstsize, byteswritten, outgiven)
