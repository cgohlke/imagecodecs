# imagecodecs/_ccitt.pyx
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

"""Codecs for the imagecodecs package using the ccitt.c library.

- CCITTRLE (CCITT Modified Huffman RLE, TIFF Compression 2)
- CCITTFAX3 (CCITT Group 3 Fax, T.4, TIFF Compression 3)
- CCITTFAX4 (CCITT Group 4 Fax, T.6, TIFF Compression 4)

"""

include '_shared.pxi'

from ccitt cimport *

ccitt_lut_init()


class CCITT:
    """CCITT codec constants."""

    available = True


class CcittError(RuntimeError):
    """CCITT codec exceptions."""

    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            CCITT_OK: 'CCITT_OK',
            CCITT_ERROR: 'CCITT_ERROR',
            CCITT_MEMORY_ERROR: 'CCITT_MEMORY_ERROR',
            CCITT_RUNTIME_ERROR: 'CCITT_RUNTIME_ERROR',
            CCITT_VALUE_ERROR: 'CCITT_VALUE_ERROR',
            CCITT_INPUT_CORRUPT: 'CCITT_INPUT_CORRUPT',
            CCITT_OUTPUT_TOO_SMALL: 'CCITT_OUTPUT_TOO_SMALL',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def ccitt_version():
    """Return ccitt library version string."""
    return f'ccitt {CCITT_VERSION.decode()}'


def ccitt_check(data, /):
    """Return whether data is encoded or None if unknown."""


# CCITTRLE ####################################################################

CCITTRLE = CCITT
CcittrleError = CcittError
ccittrle_version = ccitt_version
ccittrle_check = ccitt_check


def ccittrle_encode(
    data,
    /,
    level=None,
    *,
    axis=None,
    out=None,
):
    """Return CCITTRLE encoded data."""
    raise NotImplementedError('ccittrle_encode')


def ccittrle_decode(
    data,
    /,
    ssize_t height=0,
    ssize_t width=0,
    *,
    out=None,
):
    """Return decoded CCITTRLE data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t dstsize
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if out is None:
        if width <= 0:
            raise ValueError('width required if out is not given')
        if height <= 0:
            with nogil:
                dstsize = ccitt_rle_decode_size(&src[0], srcsize, width)
            if dstsize < 0:
                raise CcittrleError('ccitt_rle_decode_size', dstsize)
            height = dstsize // width
    else:
        if width <= 0:
            width = out.shape[out.ndim - 1]
        height = out.shape[0]

    out = _create_array(
        out, (height, width), numpy.uint8, strides=None, zero=True
    )
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = ccitt_rle_decode(
            &src[0],
            srcsize,
            <uint8_t*> dst.data,
            dstsize,
            width
        )
    if ret < 0:
        raise CcittrleError('ccitt_rle_decode', ret)

    return out


# CCITTFAX3 ###################################################################

CCITTFAX3 = CCITT
Ccittfax3Error = CcittError
ccittfax3_version = ccitt_version
ccittfax3_check = ccitt_check


def ccittfax3_encode(
    data,
    /,
    level=None,
    *,
    axis=None,
    out=None,
):
    """Return CCITTFAX3 encoded data."""
    raise NotImplementedError('ccittfax3_encode')


def ccittfax3_decode(
    data,
    /,
    ssize_t height=0,
    ssize_t width=0,
    *,
    int t4options=0,
    out=None,
):
    """Return decoded CCITTFAX3 data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t dstsize
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if out is None:
        if width <= 0:
            raise ValueError('width required if out is not given')
        if height <= 0:
            with nogil:
                dstsize = ccitt_fax3_decode_size(
                    &src[0], srcsize, width, t4options
                )
            if dstsize < 0:
                raise Ccittfax3Error('ccitt_fax3_decode_size', dstsize)
            height = dstsize // width
    else:
        if width <= 0:
            width = out.shape[out.ndim - 1]
        height = out.shape[0]

    out = _create_array(
        out, (height, width), numpy.uint8, strides=None, zero=True
    )
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = ccitt_fax3_decode(
            &src[0],
            srcsize,
            <uint8_t*> dst.data,
            dstsize,
            width,
            t4options
        )
    if ret < 0:
        raise Ccittfax3Error('ccitt_fax3_decode', ret)

    return out


# CCITTFAX4 ###################################################################

CCITTFAX4 = CCITT
Ccittfax4Error = CcittError
ccittfax4_version = ccitt_version
ccittfax4_check = ccitt_check


def ccittfax4_encode(
    data,
    /,
    level=None,
    *,
    axis=None,
    out=None,
):
    """Return CCITTFAX4 encoded data."""
    raise NotImplementedError('ccittfax4_encode')


def ccittfax4_decode(
    data,
    /,
    ssize_t height=0,
    ssize_t width=0,
    *,
    out=None,
):
    """Return decoded CCITTFAX4 data."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t dstsize
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if out is None:
        if width <= 0:
            raise ValueError('width required if out is not given')
        if height <= 0:
            with nogil:
                dstsize = ccitt_fax4_decode_size(
                    &src[0], srcsize, width
                )
            if dstsize < 0:
                raise Ccittfax4Error('ccitt_fax4_decode_size', dstsize)
            height = dstsize // width
    else:
        if width <= 0:
            width = out.shape[out.ndim - 1]
        height = out.shape[0]

    out = _create_array(
        out, (height, width), numpy.uint8, strides=None, zero=True
    )
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = ccitt_fax4_decode(
            &src[0],
            srcsize,
            <uint8_t*> dst.data,
            dstsize,
            width
        )
    if ret < 0:
        raise Ccittfax4Error('ccitt_fax4_decode', ret)

    return out
