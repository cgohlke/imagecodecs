# -*- coding: utf-8 -*-
# _jpegls.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018, Christoph Gohlke
# Copyright (c) 2018, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
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

"""JPEG-LS codec for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:Version: 2018.12.12

"""

__version__ = '2018.12.12'

_CHARLS_VERSION = '2.0.0'

import numpy

cimport cython
cimport numpy

from cpython.bytearray cimport PyByteArray_FromStringAndSize
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
from libc.string cimport memset

numpy.import_array()


# JPEG LS #####################################################################

cdef extern from 'charls.h':

    ctypedef enum CharlsApiResultType:
        CHARLS_API_RESULT_OK
        CHARLS_API_RESULT_INVALID_JLS_PARAMETERS
        CHARLS_API_RESULT_PARAMETER_VALUE_NOT_SUPPORTED
        CHARLS_API_RESULT_UNCOMPRESSED_BUFFER_TOO_SMALL
        CHARLS_API_RESULT_COMPRESSED_BUFFER_TOO_SMALL
        CHARLS_API_RESULT_INVALID_COMPRESSED_DATA
        CHARLS_API_RESULT_TOO_MUCH_COMPRESSED_DATA
        CHARLS_API_RESULT_IMAGE_TYPE_NOT_SUPPORTED
        CHARLS_API_RESULT_UNSUPPORTED_BIT_DEPTH_FOR_TRANSFORM
        CHARLS_API_RESULT_UNSUPPORTED_COLOR_TRANSFORM
        CHARLS_API_RESULT_UNSUPPORTED_ENCODING
        CHARLS_API_RESULT_UNKNOWN_JPEG_MARKER
        CHARLS_API_RESULT_MISSING_JPEG_MARKER_START
        CHARLS_API_RESULT_UNSPECIFIED_FAILURE
        CHARLS_API_RESULT_UNEXPECTED_FAILURE

    ctypedef enum CharlsInterleaveModeType:
        CHARLS_IM_NONE
        CHARLS_IM_LINE
        CHARLS_IM_SAMPLE

    cdef struct JlsCustomParameters:
        pass

    cdef struct JfifParameters:
        pass

    cdef struct JlsParameters:
        int width
        int height
        int bitsPerSample
        int stride
        int components
        int allowedLossyError
        CharlsInterleaveModeType interleaveMode
        # CharlsColorTransformationType colorTransformation
        char outputBgr
        # JlsCustomParameters custom
        # struct JfifParameters jfif

    cdef struct JlsRect:
        pass

    CharlsApiResultType JpegLsEncode(void* destination,
                                     size_t destinationLength,
                                     size_t* bytesWritten,
                                     const void* source,
                                     size_t sourceLength,
                                     const JlsParameters* params,
                                     char* errorMessage) nogil

    CharlsApiResultType JpegLsReadHeader(const void* compressedData,
                                         size_t compressedLength,
                                         JlsParameters* params,
                                         char* errorMessage) nogil

    CharlsApiResultType JpegLsDecode(void* destination,
                                     size_t destinationLength,
                                     const void* source,
                                     size_t sourceLength,
                                     const JlsParameters* params,
                                     char* errorMessage) nogil

    CharlsApiResultType JpegLsDecodeRect(void* uncompressedData,
                                         size_t uncompressedLength,
                                         const void* compressedData,
                                         size_t compressedLength,
                                         JlsRect rect,
                                         JlsParameters* params,
                                         char* errorMessage) nogil

    CharlsApiResultType JpegLsVerifyEncode(const void* uncompressedData,
                                           size_t uncompressedLength,
                                           const void* compressedData,
                                           size_t compressedLength,
                                           char* errorMessage) nogil


class JpegLsError(RuntimeError):
    """JPEG-LS Exceptions."""
    def __init__(self, func, err):
        msg = {
            CHARLS_API_RESULT_OK:
                'OK',
            CHARLS_API_RESULT_INVALID_JLS_PARAMETERS:
                'InvalidJlsParameters',
            CHARLS_API_RESULT_PARAMETER_VALUE_NOT_SUPPORTED:
                'ParameterValueNotSupported',
            CHARLS_API_RESULT_UNCOMPRESSED_BUFFER_TOO_SMALL:
                'UncompressedBufferTooSmall',
            CHARLS_API_RESULT_COMPRESSED_BUFFER_TOO_SMALL:
                'CompressedBufferTooSmall',
            CHARLS_API_RESULT_INVALID_COMPRESSED_DATA:
                'InvalidCompressedData',
            CHARLS_API_RESULT_TOO_MUCH_COMPRESSED_DATA:
                'TooMuchCompressedData',
            CHARLS_API_RESULT_IMAGE_TYPE_NOT_SUPPORTED:
                'ImageTypeNotSupported',
            CHARLS_API_RESULT_UNSUPPORTED_BIT_DEPTH_FOR_TRANSFORM:
                'UnsupportedBitDepthForTransform',
            CHARLS_API_RESULT_UNSUPPORTED_COLOR_TRANSFORM:
                'UnsupportedColorTransform',
            CHARLS_API_RESULT_UNSUPPORTED_ENCODING:
                'UnsupportedEncoding',
            CHARLS_API_RESULT_UNKNOWN_JPEG_MARKER:
                'UnknownJPEGMarker',
            CHARLS_API_RESULT_MISSING_JPEG_MARKER_START:
                'MissingJPEGMarkerStart',
            CHARLS_API_RESULT_UNSPECIFIED_FAILURE:
                'UnspecifiedFailure',
            CHARLS_API_RESULT_UNEXPECTED_FAILURE:
                'UnexpectedFailure',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def jpegls_encode(data, level=None, out=None):
    """Return JPEG-LS image from numpy array.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst
        size_t byteswritten
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        CharlsApiResultType ret = CHARLS_API_RESULT_OK
        JlsParameters params
        int allowedlossyerror = _default_level(level, 0, 0, 9)

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (data.dtype in (numpy.uint8, numpy.uint16)
            and data.ndim in (2, 3)
            and numpy.PyArray_ISCONTIGUOUS(data)):
        raise ValueError('invalid input shape, strides, or dtype')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize + 2048
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size * dst.itemsize

    memset(&params, 0, sizeof(JlsParameters))
    params.height = src.shape[0]
    params.width = src.shape[1]
    params.bitsPerSample = src.itemsize * 8
    params.allowedLossyError = allowedlossyerror

    if src.ndim == 2 or src.shape[2] == 1:
        params.components = 1
        params.interleaveMode = CHARLS_IM_NONE
    elif src.shape[2] == 3:
        params.components = 3
        params.interleaveMode = CHARLS_IM_SAMPLE
    elif src.shape[2] == 4:
        params.components = 4
        params.interleaveMode = CHARLS_IM_LINE
    else:
        raise ValueError('invalid shape')

    with nogil:
        ret = JpegLsEncode(<void*>&dst[0], <size_t>dstsize, &byteswritten,
                           <const void*>src.data, <size_t>srcsize,
                           <const JlsParameters*>&params, NULL)

    if ret != CHARLS_API_RESULT_OK:
        raise JpegLsError('JpegLsEncode', ret)

    if byteswritten < dstsize:
        if out_given:
            out = memoryview(out)[:byteswritten]
        else:
            out = out[:byteswritten]

    return out


def jpegls_decode(data, out=None):
    """Return numpy array from JPEG LS image.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        int itemsize = 0
        CharlsApiResultType ret = CHARLS_API_RESULT_OK
        JlsParameters params

    if data is out:
        raise ValueError('cannot decode in-place')

    ret = JpegLsReadHeader(&src[0], <size_t>srcsize, &params, NULL)
    if ret != CHARLS_API_RESULT_OK:
        raise JpegLsError('JpegLsReadHeader', ret)

    if params.bitsPerSample <= 8:
        dtype = numpy.uint8
        itemsize = 1
    elif params.bitsPerSample <= 16:
        dtype = numpy.uint16
        itemsize = 2
    else:
        raise ValueError(
            'JpegLs bitsPerSample not supported: %i' % params.bitsPerSample)

    if params.components == 1:
        shape = params.height, params.width
        shape_ = params.height, params.stride // itemsize
        strides = params.stride, itemsize
    elif params.interleaveMode == CHARLS_IM_NONE:
        # planar
        shape = params.components, params.height, params.width
        shape_ = params.components, params.height, params.stride // itemsize
        strides = params.stride * params.height, params.stride, itemsize
    else:
        # contig
        # params.interleaveMode == CHARLS_IM_SAMPLE or CHARLS_IM_LINE
        shape = params.height, params.width, params.components
        shape_ = (params.height,
                 params.stride // (itemsize * params.components),
                 params.components)
        strides = params.stride, itemsize * params.components, itemsize

    out = _create_array(out, shape_, dtype, strides=strides)
    dst = out
    dstsize = dst.size * dst.itemsize

    with nogil:
        ret = JpegLsDecode(<void *>dst.data, <size_t>dstsize,
                           <const void *>&src[0], <size_t>srcsize,
                           &params, NULL)

    if ret != CHARLS_API_RESULT_OK:
        raise JpegLsError('JpegLsDecode', ret)

    if shape != shape_:
        # TODO: test this
        if params.interleaveMode == CHARLS_IM_NONE:
            out = out[:, :, :shape[2]]
        else:
            out = out[:, :shape[1]]

    if params.components > 1 and params.interleaveMode == CHARLS_IM_NONE:
        out = numpy.moveaxis(out, 0, -1)

    return out


###############################################################################

cdef _create_array(out, shape, dtype, strides=None):
    """Return numpy array of shape and dtype from output argument."""
    if out is None:
        out = numpy.empty(shape, dtype)
    elif isinstance(out, numpy.ndarray):
        if out.shape != shape:
            raise ValueError('invalid output shape')
        if out.itemsize != numpy.dtype(dtype).itemsize:
            raise ValueError('invalid output dtype')
        if strides is not None:
            for i, j in zip(strides, out.strides):
                if i is not None and i != j:
                    raise ValueError('invalid output strides')
        elif not numpy.PyArray_ISCONTIGUOUS(out):
            raise ValueError('output is not contiguous')
    else:
        dstsize = 1
        for i in shape:
            dstsize *= i
        out = numpy.frombuffer(out, dtype, dstsize)
        out.shape = shape
    return out


cdef _parse_output(out, ssize_t out_size=-1, out_given=False, out_type=bytes):
    """Return out, out_size, out_given, out_type from output argument."""
    if out is None:
        pass
    elif out is bytes:
        out = None
        out_type = bytes
    elif out is bytearray:
        out = None
        out_type = bytearray
    elif isinstance(out, int):
        out_size = out
        out = None
    else:
        # out_size = len(out)
        # out_type = type(out)
        out_given = True
    return out, out_size, out_given, out_type


def _default_level(level, default, smallest, largest):
    """Return compression level in range."""
    if level is None:
        level = default
    if largest is not None:
        level = min(level, largest)
    if smallest is not None:
        level = max(level, smallest)
    return level
