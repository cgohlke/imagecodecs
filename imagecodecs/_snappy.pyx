# imagecodecs/_snappy.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2021, Christoph Gohlke
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

"""Snappy codec for the imagecodecs package."""

__version__ = '2020.12.22'

include '_shared.pxi'

from snappy cimport *


class SNAPPY:
    """Snappy Constants."""


class SnappyError(RuntimeError):
    """Snappy Exceptions."""

    def __init__(self, func, err):
        msg = {
            SNAPPY_OK: 'SNAPPY_OK',
            SNAPPY_INVALID_INPUT: 'SNAPPY_INVALID_INPUT',
            SNAPPY_BUFFER_TOO_SMALL: 'SNAPPY_BUFFER_TOO_SMALL',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def snappy_version():
    """Return Snappy library version string."""
    return 'snappy 1.1.7'


def snappy_check(arg):
    """Return True if data likely contains Snappy data."""


def snappy_encode(data, level=None, out=None):
    """Encode Snappy.

    """
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        size_t output_length = snappy_max_compressed_length(<size_t> srcsize)
        snappy_status ret
        char* buffer = NULL

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        # override any provided output size
        if dstsize < 0:
            dstsize = <ssize_t> output_length
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if <size_t> dstsize < output_length:
        # snappy_compress requires at least (32+len(data)+len(data)/6) bytes
        with nogil:
            buffer = <char*> malloc(output_length)
            if buffer == NULL:
                raise MemoryError('failed to allocate buffer')
            ret = snappy_compress(
                <const char*> &src[0],
                <size_t> srcsize,
                buffer,
                &output_length
            )
            if ret != SNAPPY_OK:
                free(buffer)
                raise SnappyError('snappy_compress', ret)
            if <size_t> dstsize < output_length:
                free(buffer)
                raise SnappyError('snappy_compress', SNAPPY_BUFFER_TOO_SMALL)
            memcpy(<void*> &dst[0], buffer, output_length)
            free(buffer)
    else:
        with nogil:
            output_length = <size_t> dstsize
            ret = snappy_compress(
                <const char*> &src[0],
                <size_t> srcsize,
                <char*> &dst[0],
                &output_length
            )
        if ret != SNAPPY_OK:
            raise SnappyError('snappy_compress', ret)

    del dst
    return _return_output(out, dstsize, output_length, outgiven)


def snappy_decode(data, out=None):
    """Decode Snappy.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        ssize_t srcsize = src.size
        snappy_status ret
        size_t output_length
        size_t result

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            ret = snappy_uncompressed_length(
                <const char*> &src[0],
                <size_t> srcsize,
                &result
            )
            if ret != SNAPPY_OK:
                raise SnappyError('snappy_uncompressed_length', ret)
            dstsize = <ssize_t> result
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    output_length = <size_t> dstsize

    with nogil:
        ret = snappy_uncompress(
            <const char*> &src[0],
            <size_t> srcsize,
            <char*> &dst[0],
            &output_length
        )
    if ret != SNAPPY_OK:
        raise SnappyError('snappy_uncompress', ret)

    del dst
    return _return_output(out, dstsize, output_length, outgiven)
