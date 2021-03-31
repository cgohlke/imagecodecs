# imagecodecs/_jpegsof3.pyx
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

"""JPEG SOF3 codec for the imagecodecs package.

The "JPEG Lossless, Nonhierarchical, First Order Prediction" format is
described at <http://www.w3.org/Graphics/JPEG/itu-t81.pdf>.
The format is identified by a Start of Frame (SOF) code 0xC3.

"""

__version__ = '2020.12.22'

include '_shared.pxi'

from jpegsof3 cimport *


class JPEGSOF3:
    """JPEG SOF3 Constants."""


class Jpegsof3Error(RuntimeError):
    """JPEG SOF3 Exceptions."""

    def __init__(self, err):
        msg = {
            JPEGSOF3_INVALID_OUTPUT:
                'output array is too small',
            JPEGSOF3_INVALID_SIGNATURE:
                'JPEG signature 0xFFD8FF not found',
            JPEGSOF3_INVALID_HEADER_TAG:
                'header tag must begin with 0xFF',
            JPEGSOF3_SEGMENT_GT_IMAGE:
                'segment larger than image',
            JPEGSOF3_INVALID_ITU_T81:
                'not a lossless (sequential) JPEG image (SoF must be 0xC3)',
            JPEGSOF3_INVALID_BIT_DEPTH:
                'data must be 2..16 bit, 1..4 frames',
            JPEGSOF3_TABLE_CORRUPTED:
                'Huffman table corrupted',
            JPEGSOF3_TABLE_SIZE_CORRUPTED:
                'Huffman size array corrupted',
            JPEGSOF3_INVALID_RESTART_SEGMENTS:
                'unsupported Restart Segments',
            JPEGSOF3_NO_TABLE:
                'no Huffman tables',
        }.get(err, f'unknown error {err!r}')
        msg = f'decode_jpegsof3 returned {msg!r}'
        super().__init__(msg)


def jpegsof3_version():
    """Return JPEG SOF3 library version string."""
    return 'jpegsof3 ' + JPEGSOF3_VERSION.decode()


def jpegsof3_check(data):
    """Return True if data likely contains a JPEG SOF3 image."""


def jpegsof3_encode(*args, **kwargs):
    """Return JPEG SOF3 image from numpy array."""
    raise NotImplementedError('jpegsof3_encode')


def jpegsof3_decode(data, index=None, out=None):
    """Decode JPEG SOF3 image to numpy array.

    Beware, the input data must be writable and is modified in-place!

    RGB images are returned as non-contiguous arrays as samples are decoded
    into separate frames first (RRGGBB).

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = _writable_input(data)
        ssize_t srcsize = src.size
        ssize_t dstsize
        int dimX, dimY, bits, frames
        int ret = JPEGSOF3_OK

    if data is out:
        raise ValueError('cannot decode in-place')

    with nogil:
        ret = decode_jpegsof3(
            <unsigned char*> &src[0],
            srcsize,
            NULL,
            0,
            &dimX,
            &dimY,
            &bits,
            &frames
        )
    if ret != JPEGSOF3_OK:
        raise Jpegsof3Error(ret)

    if frames > 1:
        shape = frames, dimY, dimX
    else:
        shape = dimY, dimX

    if bits > 8:
        dtype = numpy.uint16
    else:
        dtype = numpy.uint8

    out = _create_array(out, shape, dtype)
    dst = out
    dstsize = dst.nbytes

    with nogil:
        ret = decode_jpegsof3(
            <unsigned char*> &src[0],
            srcsize,
            <unsigned char*> dst.data,
            dstsize,
            &dimX,
            &dimY,
            &bits,
            &frames
        )
    if ret != JPEGSOF3_OK:
        raise Jpegsof3Error(ret)

    if frames > 1:
        out = numpy.moveaxis(out, 0, -1)

    return out
