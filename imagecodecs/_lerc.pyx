# imagecodecs/_lerc.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2020-2021, Christoph Gohlke
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

"""LERC (Limited Error Raster Compression) codec for the imagecodecs package.

"""

__version__ = '2021.2.26'

include '_shared.pxi'

from lerc cimport *


class LERC:
    """LERC Constants."""


class LercError(RuntimeError):
    """LERC Exceptions."""

    def __init__(self, func, err):
        msg = {
            Ok: 'Ok',
            Failed: 'Failed',
            WrongParam: 'WrongParam',
            BufferTooSmall: 'BufferTooSmall',
            NaN: 'NaN',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def lerc_version():
    """Return LERC library version string."""
    return 'lerc 2.2'


def lerc_check(const uint8_t[::1] data):
    """Return True if data likely contains LERC data."""
    cdef:
        bytes sig = bytes(data[:9])

    return sig[:5] == b'Lerc2' or sig[:9] == b'CntZImage'


def lerc_encode(
    data,
    level=None,
    mask=None,
    version=None,
    planarconfig=None,
    out=None
):
    """Compress LERC.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t dstsize
        unsigned char* pValidBytes = NULL
        lerc_status ret
        unsigned int nBytesWritten
        unsigned int dataType
        int nDim = 1
        int nCols = 1
        int nRows = 1
        int nBands = 1
        int iversion = 4 if version is None else version
        double maxZErr = _default_value(level, 0.0, 0.0, None)
        unsigned int blobSize
        int ndim = src.ndim

    if data is out:
        raise ValueError('cannot encode in-place')

    if src.dtype == numpy.uint8:
        dataType = dt_uchar
    elif src.dtype == numpy.uint16:
        dataType = dt_ushort
    elif src.dtype == numpy.int32:
        dataType = dt_int
    elif src.dtype == numpy.float32:
        dataType = dt_float
    elif src.dtype == numpy.float64:
        dataType = dt_double
    elif src.dtype == numpy.int8:
        dataType = dt_char
    elif src.dtype == numpy.int16:
        dataType = dt_short
    elif src.dtype == numpy.uint32:
        dataType = dt_uint
    else:
        raise ValueError('data type not supported by LERC')

    if ndim == 2:
        nRows = <int> src.shape[0]
        nCols = <int> src.shape[1]
    elif ndim == 3:
        if planarconfig is None or planarconfig in ('contig', 'CONTIG', 1):
            nRows = <int> src.shape[0]
            nCols = <int> src.shape[1]
            nDim = <int> src.shape[2]
        else:
            nBands = <int> src.shape[0]
            nRows = <int> src.shape[1]
            nCols = <int> src.shape[2]
    elif ndim == 4:
        nBands = <int> src.shape[0]
        nRows = <int> src.shape[1]
        nCols = <int> src.shape[2]
        nDim = <int> src.shape[3]
    elif ndim == 1:
        nCols = <int> src.shape[0]
    else:
        raise ValueError('data shape not supported by LERC')

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            ret = lerc_computeCompressedSizeForVersion(
                <const void*> src.data,
                iversion,
                dataType,
                nDim,
                nCols,
                nRows,
                nBands,
                pValidBytes,
                maxZErr,
                &blobSize
            )
            if ret != 0:
                raise LercError('lerc_computeCompressedSizeForVersion', ret)
            dstsize = <ssize_t> blobSize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes
    blobSize = <unsigned int> dstsize

    with nogil:
        ret = lerc_encodeForVersion(
            <const void*> src.data,
            iversion,
            dataType,
            nDim,
            nCols,
            nRows,
            nBands,
            pValidBytes,
            maxZErr,
            <unsigned char*> &dst[0],
            blobSize,
            &nBytesWritten
        )
    if ret != 0:
        raise LercError('lerc_encodeForVersion', ret)

    del dst
    return _return_output(out, dstsize, <ssize_t> nBytesWritten, outgiven)


def lerc_decode(data, index=None, mask=None, out=None):
    """Decompress LERC.

    """
    cdef:
        numpy.ndarray dst
        numpy.ndarray valid
        const uint8_t[::1] src = data
        const uint8_t[::1] header
        ssize_t srcsize = src.size
        unsigned int[8] infoArray
        double[3] dataRangeArray
        unsigned char* pValidBytes = NULL
        lerc_status ret
        int version
        unsigned int dataType
        int nDim
        int nCols
        int nRows
        int nBands
        int nValidPixels
        unsigned int blobSize

    if data is out:
        raise ValueError('cannot decode in-place')

    if bytes(src[:9]) == b'CntZImage' and hasattr(data, 'write_byte'):
        # Lerc1 decoder segfaults if data is not writable
        # raises TypeError: mmap can't modify a readonly memory map
        data[0] = data[0]

    ret = lerc_getBlobInfo(
        <const unsigned char*> &src[0],
        <unsigned int> srcsize,
        &infoArray[0],
        &dataRangeArray[0],
        8,
        3
    )
    if ret != 0:
        raise LercError('lerc_getBlobInfo', ret)

    version = infoArray[0]
    dataType = infoArray[1]
    nDim = infoArray[2]
    nCols = infoArray[3]
    nRows = infoArray[4]
    nBands = infoArray[5]
    nValidPixels = infoArray[6]
    blobSize = infoArray[7]

    if srcsize < <ssize_t> blobSize:
        raise RuntimeError('incomplete blob')

    if dataType == dt_char:
        dtype = numpy.int8
    elif dataType == dt_uchar:
        dtype = numpy.uint8
    elif dataType == dt_short:
        dtype = numpy.int16
    elif dataType == dt_ushort:
        dtype = numpy.uint16
    elif dataType == dt_int:
        dtype = numpy.int32
    elif dataType == dt_uint:
        dtype = numpy.uint32
    elif dataType == dt_float:
        dtype = numpy.float32
    elif dataType == dt_double:
        dtype = numpy.float64
    else:
        raise RuntimeError('invalid data type')

    if nBands > 1:
        if nDim > 1:
            shape = nBands, nRows, nCols, nDim
        else:
            shape = nBands, nRows, nCols
    elif nDim > 1:
        shape = nRows, nCols, nDim
    else:
        shape = nRows, nCols

    out = _create_array(out, shape, dtype, None, nValidPixels != nRows * nCols)
    dst = out

    if not (mask is None or mask is False):
        if mask is True:
            mask = None
        mask = _create_array(mask, (nRows, nCols), numpy.bool8)
        valid = mask
        pValidBytes = <unsigned char*> valid.data

    with nogil:
        ret = lerc_decode_c(
            <const unsigned char*> &src[0],
            blobSize,
            pValidBytes,
            nDim,
            nCols,
            nRows,
            nBands,
            dataType,
            <void*> dst.data
        )
    if ret != 0:
        raise LercError('lerc_decode', ret)

    if pValidBytes != NULL:
        return out, mask

    return out
