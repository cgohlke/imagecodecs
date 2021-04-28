# imagecodecs/_brunsli.pyx
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

"""Brunsli codec for the imagecodecs package.

Brunsli works as a transcoder converting between JPEG and JPEG XL.

Brunsli 0.1 is not compatible with the final JPEG XL specification.

"""

__version__ = '2021.4.28'

include '_shared.pxi'

from brunsli cimport *

from . import jpeg8_decode, jpeg8_encode


class BRUNSLI:
    """Brunsli Constants."""


class BrunsliError(RuntimeError):
    """Brunsli Exceptions."""

    def __init__(self, func, err):
        # msg = {
        #     BRUNSLI_OK: 'BRUNSLI_OK',
        #     BRUNSLI_NON_REPRESENTABLE: 'BRUNSLI_NON_REPRESENTABLE',
        #     BRUNSLI_MEMORY_ERROR: 'BRUNSLI_MEMORY_ERROR',
        #     BRUNSLI_INVALID_PARAM: 'BRUNSLI_INVALID_PARAM',
        #     BRUNSLI_COMPRESSION_ERROR: 'BRUNSLI_COMPRESSION_ERROR',
        #     BRUNSLI_INVALID_BRN: 'BRUNSLI_INVALID_BRN',
        #     BRUNSLI_DECOMPRESSION_ERROR: 'BRUNSLI_DECOMPRESSION_ERROR',
        #     BRUNSLI_NOT_ENOUGH_DATA: 'BRUNSLI_NOT_ENOUGH_DATA',
        # }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {err}'
        super().__init__(msg)


def brunsli_version():
    """Return Brunsli library version string."""
    # TODO: use version from header when available
    return 'brunsli 0.1'


def brunsli_check(data):
    """Return True if data likely contains a Brunsli image."""


def brunsli_encode(
    data,
    level=None,
    colorspace=None,
    outcolorspace=None,
    subsampling=None,
    optimize=None,
    smoothing=None,
    out=None
):
    """Return Brunsli encoded image from numpy array.

    """
    cdef:
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t[::1] src
        char* dstptr = NULL
        ssize_t dstsize
        size_t srcsize
        brunsli_sink_t* sink = NULL
        int ret

    if data is out:
        raise ValueError('cannot encode in-place')

    if isinstance(data, numpy.ndarray):
        src = jpeg8_encode(
            data,
            level=level,
            colorspace=colorspace,
            outcolorspace=outcolorspace,
            subsampling=subsampling,
            optimize=optimize,
            smoothing=smoothing
        )
    else:
        # existing JPEG stream
        src = data
    srcsize = src.size

    out, dstsize, outgiven, outtype = _parse_output(out)

    sink = brunsli_sink_new(srcsize // 2)
    try:
        with nogil:
            ret = EncodeBrunsli(srcsize, &src[0], sink, brunsli_sink_write)
        if ret != 1:
            raise BrunsliError('EncodeBrunsli', ret)
        if out is None:
            if dstsize < 0:
                dstsize = sink.byteswritten
                dstptr = <char*> sink.data
            out = _create_output(outtype, dstsize, dstptr)
        if dstptr == NULL:
            dst = out
            dstsize = dst.size
            if <size_t> dstsize < sink.byteswritten:
                raise ValueError('output too small')
            memcpy(<void*> &dst[0], <void*> sink.data, sink.byteswritten)
            del dst
            out = _return_output(out, dstsize, sink.byteswritten, outgiven)
    finally:
        brunsli_sink_del(sink)
    return out


def brunsli_decode(
    data,
    index=None,
    colorspace=None,
    outcolorspace=None,
    asjpeg=False,
    out=None
):
    """Return numpy array from Brunsli encoded image.
    """
    cdef:
        const uint8_t[::1] src = data
        size_t srcsize = <size_t> src.size
        brunsli_sink_t* sink = brunsli_sink_new(srcsize * 2)
        int ret

    try:
        with nogil:
            ret = DecodeBrunsli(srcsize, &src[0], sink, brunsli_sink_write)
        if ret != 1:
            raise BrunsliError('DecodeBrunsli', ret)
        out = jpeg8_decode(
            <uint8_t[:sink.byteswritten]> sink.data,
            index=index,
            colorspace=colorspace,
            outcolorspace=outcolorspace,
            out=out
        )
    finally:
        brunsli_sink_del(sink)
    return out


ctypedef struct brunsli_sink_t:
    uint8_t* data
    size_t size
    size_t offset
    size_t byteswritten


cdef brunsli_sink_t* brunsli_sink_new(size_t size):
    """Return new Brunsli sink."""
    cdef:
        brunsli_sink_t* sink = <brunsli_sink_t*> malloc(sizeof(brunsli_sink_t))

    if sink == NULL:
        raise MemoryError('failed to allocate brunsli_sink')
    sink.byteswritten = 0
    sink.size = size
    sink.offset = 0
    sink.data = <uint8_t*> malloc(size)
    if sink.data == NULL:
        free(sink)
        raise MemoryError('failed to allocate brunsli_sink.data')
    return sink


cdef brunsli_sink_del(brunsli_sink_t* sink):
    """Free Brunsli sink."""
    if sink != NULL:
        free(sink.data)
        free(sink)


cdef size_t brunsli_sink_write(
    void* brunsli_sink_ptr,
    const uint8_t* src,
    size_t size
) nogil:
    """Brunsli callback function for writing to sink."""
    cdef:
        uint8_t* tmp
        size_t newsize
        brunsli_sink_t* sink = <brunsli_sink_t*> brunsli_sink_ptr

    if sink == NULL or size == 0 or sink.offset > sink.size:
        return 0
    if sink.offset + size > sink.size:
        # output stream too small; realloc
        newsize = sink.offset + size
        if newsize <= <size_t> (<double> sink.size * 1.25):
            # moderate upsize: overallocate
            newsize = newsize + newsize // 4
            newsize = (((newsize - 1) // 4096) + 1) * 4096
        tmp = <uint8_t*> realloc(<void*> sink.data, newsize)
        if tmp == NULL:
            return 0
        sink.data = tmp
        sink.size = newsize
    memcpy(<void*> &(sink.data[sink.offset]), <const void*> src, size)
    sink.offset += size
    if sink.offset > sink.byteswritten:
        sink.byteswritten = sink.offset
    return size
