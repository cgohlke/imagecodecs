# -*- coding: utf-8 -*-
# _imagecodecs_lite.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2008-2019, Christoph Gohlke
# Copyright (c) 2008-2019, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
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

"""Subset of image transformation, compression, and decompression codecs.

Imagecodecs-lite is a Python library that provides block-oriented, in-memory
buffer transformation, compression, and/or decompression functions for
LZW, PackBits, Delta, XOR Delta, Packed Integers, Floating Point Predictor,
and Bitorder reversal.

Imagecodecs-lite is a subset of the `imagecodecs
<https://pypi.org/project/imagecodecs/>`_ library, which provides additional
codecs for Zlib DEFLATE, ZStandard, Blosc, LZMA, BZ2, LZ4, LZF, AEC, ZFP,
PNG, WebP, JPEG 8-bit, JPEG 12-bit, JPEG SOF3, JPEG LS, JPEG 2000, and JPEG XR.

Unlike imagecodecs, imagecodecs-lite does not depend on external third-party
C libraries and is therefore simple to build from source code.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: BSD 3-Clause

:Version: 2019.12.3

Requirements
------------
This release has been tested with the following requirements and dependencies
(other versions may work):

* `CPython 2.7.15, 3.5.4, 3.6.8, 3.7.5, 3.8.0 64-bit <https://www.python.org>`_
* `Numpy 1.16.5 <https://www.numpy.org>`_
* `Cython 0.29.14 <https://cython.org>`_

Notes
-----
The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 2.7, 3.5, and 32-bit are deprecated.

Build instructions for manylinux and macOS courtesy of Grzegorz Bokota.

Revisions
---------
2019.12.3
    Release manylinux and macOS wheels.
2019.4.20
    Fix setup requirements.
2019.2.22
    Initial release based on imagecodecs 2019.2.22.

"""

__version__ = '2019.12.3'


import io
import numbers
import numpy

cimport numpy
cimport cython

from cpython.bytearray cimport PyByteArray_FromStringAndSize
from cpython.bytes cimport PyBytes_FromStringAndSize

from libc.math cimport ceil
from libc.stdint cimport int8_t, uint8_t, int64_t, uint64_t

cdef extern from 'numpy/arrayobject.h':
    int NPY_VERSION
    int NPY_FEATURE_VERSION

numpy.import_array()


###############################################################################

cdef extern from 'imagecodecs.h':

    char* ICD_VERSION
    char ICD_BOC
    int ICD_OK
    int ICD_ERROR
    int ICD_MEMORY_ERROR
    int ICD_RUNTIME_ERROR
    int ICD_NOTIMPLEMENTED_ERROR
    int ICD_VALUE_ERROR
    int ICD_LZW_INVALID
    int ICD_LZW_NOTIMPLEMENTED
    int ICD_LZW_BUFFER_TOO_SMALL
    int ICD_LZW_TABLE_TOO_SMALL


class IcdError(RuntimeError):
    """Imagecodec Exceptions."""
    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            ICD_OK: 'ICD_OK',
            ICD_ERROR: 'ICD_ERROR',
            ICD_MEMORY_ERROR: 'ICD_MEMORY_ERROR',
            ICD_RUNTIME_ERROR: 'ICD_RUNTIME_ERROR',
            ICD_NOTIMPLEMENTED_ERROR: 'ICD_NOTIMPLEMENTED_ERROR',
            ICD_VALUE_ERROR: 'ICD_VALUE_ERROR',
            ICD_LZW_INVALID: 'ICD_LZW_INVALID',
            ICD_LZW_NOTIMPLEMENTED: 'ICD_LZW_NOTIMPLEMENTED',
            ICD_LZW_BUFFER_TOO_SMALL: 'ICD_LZW_BUFFER_TOO_SMALL',
            ICD_LZW_TABLE_TOO_SMALL: 'ICD_LZW_TABLE_TOO_SMALL',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


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
    elif isinstance(out, numbers.Integral):
        out_size = out
        out = None
    else:
        # out_size = len(out)
        # out_type = type(out)
        out_given = True
    return out, out_size, out_given, out_type


cdef _parse_input(data):
    """Return bytes memoryview to input argument."""
    cdef const uint8_t[::1] src
    try:
        src = data
    except ValueError:
        src = numpy.ravel(data, 'K').view(numpy.uint8)
    return src


def version(astype=None):
    """Return detailed version information."""
    versions = (
        ('imagecodecs_lite', __version__),
        ('icd', _ICD_VERSION),
        ('numpy_abi', '0x%X.%i' % (NPY_VERSION, NPY_FEATURE_VERSION)),
        ('numpy', numpy.__version__),
        ('cython', cython.__version__),
        )
    if astype is str or astype is None:
        return ', '.join('%s-%s' % (k, v) for k, v in versions)
    elif astype is dict:
        return dict(versions)
    else:
        return versions


_ICD_VERSION = ICD_VERSION.decode('utf-8')


# No Operation ################################################################

def none_decode(data, *args, **kwargs):
    """Decode NOP."""
    return data


def none_encode(data, *args, **kwargs):
    """Encode NOP."""
    return data


# Numpy #######################################################################

def numpy_decode(data, index=0, out=None, **kwargs):
    """Decode NPY and NPZ."""
    with io.BytesIO(data) as fh:
        out = numpy.load(fh, **kwargs)
        if hasattr(out, 'files'):
            try:
                index = out.files[index]
            except Exception:
                pass
            out = out[index]
    return out


def numpy_encode(data, level=None, out=None, **kwargs):
    """Encode NPY and NPZ."""
    with io.BytesIO() as fh:
        if level:
            numpy.savez_compressed(fh, data, **kwargs)
        else:
            numpy.save(fh, data, **kwargs)
        fh.seek(0)
        out = fh.read()
    return out


# Delta #######################################################################

cdef extern from 'imagecodecs.h':

    ssize_t icd_delta(
        void *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void *dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const int decode) nogil


cdef _delta(data, int axis, out, int decode):
    """Decode or encode Delta."""
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t ret = 0
        void* srcptr = NULL
        void* dstptr = NULL
        numpy.flatiter srciter
        numpy.flatiter dstiter
        ssize_t itemsize

    if isinstance(data, numpy.ndarray):
        if data.dtype.kind not in 'fiu':
            raise ValueError('not an integer or floating-point array')

        if out is None:
            out = numpy.empty_like(data)
        else:
            if not isinstance(out, numpy.ndarray):
                raise ValueError('output is not a numpy array')
            if (data.shape != out.shape or
                    data.dtype.itemsize != out.dtype.itemsize):
                raise ValueError('output is not compatible with data array')

        if axis < 0:
            axis = data.ndim + axis
        if axis > data.ndim:
            raise ValueError('invalid axis')

        srciter = numpy.PyArray_IterAllButAxis(data, &axis)
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis)
        srcsize = data.shape[axis]
        dstsize = out.shape[axis]
        srcstride = data.strides[axis]
        dststride = out.strides[axis]
        itemsize = data.dtype.itemsize

        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = numpy.PyArray_ITER_DATA(srciter)
                dstptr = numpy.PyArray_ITER_DATA(dstiter)
                ret = icd_delta(<void *>srcptr, srcsize, srcstride,
                                <void *>dstptr, dstsize, dststride,
                                itemsize, decode)
                if ret < 0:
                    break
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        if ret < 0:
            raise IcdError('icd_delta', ret)

        return out

    src = data
    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = src.size
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size
    srcsize = src.size
    srcstride = 1
    dststride = 1
    itemsize = 1

    with nogil:
        ret = icd_delta(<void *>&src[0], srcsize, srcstride,
                        <void *>&dst[0], dstsize, dststride,
                        itemsize, decode)
    if ret < 0:
        raise IcdError('icd_delta', ret)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


def delta_decode(data, axis=-1, out=None):
    """Decode differencing.

    Same as numpy.cumsum

    """
    return _delta(data, axis=axis, out=out, decode=True)


def delta_encode(data, axis=-1, out=None):
    """Encode differencing.

    """
    return _delta(data, axis=axis, out=out, decode=False)


# XOR Delta ###################################################################

cdef extern from 'imagecodecs.h':

    ssize_t icd_xor(
        void *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void *dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const int decode) nogil


cdef _xor(data, int axis, out, int decode):
    """Decode or encode XOR."""
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t ret = 0
        void* srcptr = NULL
        void* dstptr = NULL
        numpy.flatiter srciter
        numpy.flatiter dstiter
        ssize_t itemsize

    if isinstance(data, numpy.ndarray):
        if data.dtype.kind not in 'fiu':
            raise ValueError('not an integer or floating-point array')

        if out is None:
            out = numpy.empty_like(data)
        else:
            if not isinstance(out, numpy.ndarray):
                raise ValueError('output is not a numpy array')
            if data.shape != out.shape or data.dtype != out.dtype:
                raise ValueError('output is not compatible with data array')

        if axis < 0:
            axis = data.ndim + axis
        if axis > data.ndim:
            raise ValueError('invalid axis')

        srciter = numpy.PyArray_IterAllButAxis(data, &axis)
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis)
        srcsize = data.shape[axis]
        dstsize = out.shape[axis]
        srcstride = data.strides[axis]
        dststride = out.strides[axis]

        itemsize = data.dtype.itemsize

        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = numpy.PyArray_ITER_DATA(srciter)
                dstptr = numpy.PyArray_ITER_DATA(dstiter)
                ret = icd_xor(<void *>srcptr, srcsize, srcstride,
                              <void *>dstptr, dstsize, dststride,
                              itemsize, decode)
                if ret < 0:
                    break
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        if ret < 0:
            raise IcdError('icd_xor', ret)

        return out

    src = data
    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = src.size
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size
    srcsize = src.size
    srcstride = 1
    dststride = 1
    itemsize = 1

    with nogil:
        ret = icd_xor(<void *>&src[0], srcsize, srcstride,
                      <void *>&dst[0], dstsize, dststride,
                      itemsize, decode)
    if ret < 0:
        raise IcdError('icd_xor', ret)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


def xor_decode(data, axis=-1, out=None):
    """Decode XOR.

    """
    return _xor(data, axis=axis, out=out, decode=True)


def xor_encode(data, axis=-1, out=None):
    """Encode XOR.

    """
    return _xor(data, axis=axis, out=out, decode=False)


# Floating Point Predictor ####################################################

# TIFF Technical Note 3. April 8, 2005.

cdef extern from 'imagecodecs.h':

    ssize_t icd_floatpred(
        void *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        void *dst,
        const ssize_t dstsize,
        const ssize_t dststride,
        const ssize_t itemsize,
        const ssize_t samples,
        const char byteorder,
        const int decode) nogil


cdef _floatpred(data, int axis, out, int decode):
    """Encode or decode Floating Point Predictor."""
    cdef:
        void* srcptr = NULL
        void* dstptr = NULL
        numpy.flatiter srciter
        numpy.flatiter dstiter
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t itemsize
        ssize_t samples
        ssize_t ret = 0
        char byteorder

    if not isinstance(data, numpy.ndarray) or data.dtype.kind != 'f':
        raise ValueError('not a floating-point numpy array')

    if out is None or out is data:
        out = numpy.empty_like(data)
    else:
        if not isinstance(out, numpy.ndarray):
            raise ValueError('output is not a numpy array')
        if out.shape != data.shape or out.itemsize != data.itemsize:
            raise ValueError('output is not compatible with data array')

    ndim = data.ndim
    axis = axis % ndim
    if ndim < 1 or ndim - axis > 2:
        raise ValueError('invalid axis')

    samples = data.shape[axis+1] if ndim - axis == 2 else 1

    src = data.view()
    src.shape = data.shape[:axis] + (-1,)
    dst = out.view()
    dst.shape = src.shape

    if src.dtype.byteorder == '=':
        byteorder = ICD_BOC
    else:
        byteorder = <char>ord(src.dtype.byteorder)

    srciter = numpy.PyArray_IterAllButAxis(src, &axis)
    dstiter = numpy.PyArray_IterAllButAxis(dst, &axis)
    itemsize = src.dtype.itemsize
    srcsize = src.shape[axis] * itemsize
    dstsize = dst.shape[axis] * itemsize
    srcstride = src.strides[axis]
    dststride = dst.strides[axis]
    if decode != 0 and srcstride != itemsize:
        raise ValueError('data not contiguous on dimensions >= axis')
    elif decode == 0 and dststride != itemsize:
        raise ValueError('output not contiguous on dimensions >= axis')

    with nogil:
        while numpy.PyArray_ITER_NOTDONE(srciter):
            srcptr = numpy.PyArray_ITER_DATA(srciter)
            dstptr = numpy.PyArray_ITER_DATA(dstiter)
            ret = icd_floatpred(<void *>srcptr, srcsize, srcstride,
                                <void *>dstptr, dstsize, dststride,
                                itemsize, samples, byteorder, decode)
            if ret < 0:
                break
            numpy.PyArray_ITER_NEXT(srciter)
            numpy.PyArray_ITER_NEXT(dstiter)
    if ret < 0:
        raise IcdError('icd_floatpred', ret)

    return out


def floatpred_encode(data, axis=-1, out=None):
    """Encode Floating Point Predictor.

    The output array should not be treated as floating-point numbers but as an
    encoded byte sequence viewed as a numpy array with shape and dtype of the
    input data.

    """
    return _floatpred(data, axis=axis, out=out, decode=False)


def floatpred_decode(data, axis=-1, out=None):
    """Decode Floating Point Predictor.

    The data array is not really an array of floating-point numbers but an
    encoded byte sequence viewed as a numpy array of requested output shape
    and dtype.

    """
    return _floatpred(data, axis=axis, out=out, decode=True)


# BitOrder Reversal ###########################################################

cdef extern from 'imagecodecs.h':

    ssize_t icd_bitorder(
        uint8_t *src,
        const ssize_t srcsize,
        const ssize_t srcstride,
        const ssize_t itemsize,
        uint8_t *dst,
        const ssize_t dstsize,
        const ssize_t dststride) nogil


def bitorder_decode(data, out=None):
    """"Reverse bits in each byte of bytes, bytearray or numpy array.

    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] dst
        uint8_t* srcptr = NULL
        uint8_t* dstptr = NULL
        ssize_t srcsize = 0
        ssize_t dstsize = 0
        ssize_t srcstride = 1
        ssize_t dststride = 1
        ssize_t itemsize = 1
        numpy.flatiter srciter
        numpy.flatiter dstiter
        int axis = -1

    if isinstance(data, numpy.ndarray):
        itemsize = data.dtype.itemsize
        if data is out:
            # in-place
            if not numpy.PyArray_ISWRITEABLE(data):
                raise ValueError('data is not writable')

            if numpy.PyArray_ISCONTIGUOUS(data):
                srcptr = <uint8_t *>numpy.PyArray_DATA(data)
                srcsize = data.size * itemsize
                srcstride = itemsize
                with nogil:
                    icd_bitorder(<uint8_t *>srcptr, srcsize, srcstride,
                                 itemsize,
                                 <uint8_t *>dstptr, dstsize, dststride)
                return data

            srciter = numpy.PyArray_IterAllButAxis(data, &axis)
            srcsize = data.shape[axis] * itemsize
            srcstride = data.strides[axis]
            with nogil:
                while numpy.PyArray_ITER_NOTDONE(srciter):
                    srcptr = <uint8_t *>numpy.PyArray_ITER_DATA(srciter)
                    icd_bitorder(<uint8_t *>srcptr, srcsize, srcstride,
                                 itemsize,
                                 <uint8_t *>dstptr, dstsize, dststride)
                    numpy.PyArray_ITER_NEXT(srciter)
            return data

        if out is None:
            out = numpy.empty_like(data)
        else:
            if not isinstance(out, numpy.ndarray):
                raise ValueError('output is not a numpy array')
            if data.shape != out.shape or itemsize != out.dtype.itemsize:
                raise ValueError('output is not compatible with data array')
        srciter = numpy.PyArray_IterAllButAxis(data, &axis)
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis)
        srcsize = data.shape[axis] * itemsize
        dstsize = out.shape[axis] * itemsize
        srcstride = data.strides[axis]
        dststride = out.strides[axis]
        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = <uint8_t *>numpy.PyArray_ITER_DATA(srciter)
                dstptr = <uint8_t *>numpy.PyArray_ITER_DATA(dstiter)
                icd_bitorder(<uint8_t *>srcptr, srcsize, srcstride,
                             itemsize,
                             <uint8_t *>dstptr, dstsize, dststride)
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        return out

    # contiguous byte buffers: bytes or bytearray
    src = data
    srcsize = src.size
    if data is out:
        # in-place
        with nogil:
            icd_bitorder(<uint8_t *>&src[0], srcsize, 1, 1,
                         <uint8_t *>&src[0], srcsize, 1)
        return data

    if out is None:
        out = PyByteArray_FromStringAndSize(NULL, srcsize)
    dst = out
    dstsize = dst.size
    with nogil:
        icd_bitorder(<uint8_t *>&src[0], srcsize, 1, 1,
                     <uint8_t *>&dst[0], dstsize, 1)
    return out


bitorder_encode = bitorder_decode


# PackBits ####################################################################

cdef extern from 'imagecodecs.h':

    ssize_t icd_packbits_size(
        const uint8_t *src,
        const ssize_t srcsize) nogil

    ssize_t icd_packbits_decode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize) nogil

    ssize_t icd_packbits_encode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize) nogil


def packbits_encode(data, level=None, out=None):
    """Compress PackBits.

    """
    cdef:
        numpy.flatiter srciter
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t* srcptr
        const uint8_t* dstptr
        ssize_t srcsize
        ssize_t dstsize
        ssize_t ret = 0
        bint isarray = False
        int axis = 0

    if isinstance(data, numpy.ndarray):
        if data.itemsize != 1:
            raise ValueError('data is not a byte array')
        if data.ndim != 1:
            isarray = True
            axis = data.ndim - 1
        if data.strides[axis] != 1:
            raise ValueError('data array is not contiguous along last axis')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstsize < 0:
            if isarray:
                srcsize = data.shape[axis]
                dstsize = data.size // srcsize * (srcsize + srcsize // 128 + 2)
            else:
                srcsize = len(data)
                dstsize = srcsize + srcsize // 128 + 2
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size

    if isarray:
        srciter = numpy.PyArray_IterAllButAxis(data, &axis)
        srcsize = data.shape[axis]
        dstptr = &dst[0]
        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = <uint8_t*>numpy.PyArray_ITER_DATA(srciter)
                ret = icd_packbits_encode(srcptr, srcsize,
                                          <uint8_t *>dstptr, dstsize)
                if ret < 0:
                    break
                dstptr = dstptr + ret
                dstsize -= ret
                if dstsize <= 0:
                    break
                numpy.PyArray_ITER_NEXT(srciter)
        if ret >= 0:
            ret = dstptr - &dst[0]
    else:
        src = _parse_input(data)
        srcsize = src.size
        with nogil:
            ret = icd_packbits_encode(&src[0], srcsize,
                                      <uint8_t *>&dst[0], dstsize)

    if ret < 0:
        raise IcdError('icd_packbits_encode', ret)
    if ret < dst.size:
        if out_given and not isinstance(out, numpy.ndarray):
            out = memoryview(out)[:ret]
        else:
            out = out[:ret]
    return out


def packbits_decode(data, out=None):
    """Decompress PackBits.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret = 0

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstsize < 0:
            with nogil:
                dstsize = icd_packbits_size(&src[0], srcsize)
            if dstsize < 0:
                raise IcdError('icd_packbits_size', dstsize)
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = icd_packbits_decode(&src[0], srcsize,
                                  <uint8_t *>&dst[0], dstsize)
    if ret < 0:
        raise IcdError('icd_packbits_decode', ret)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


# Packed Integers #############################################################

cdef extern from 'imagecodecs.h':

    int SSIZE_MAX

    ssize_t icd_packints_decode(
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize,
        const int numbits) nogil

    void icd_swapbytes(
        void *src,
        const ssize_t srcsize,
        const ssize_t itemsize) nogil


def packints_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: PackInts encoding
    raise NotImplementedError('packints_encode')


def packints_decode(data, dtype, int numbits, ssize_t runlen=0, out=None):
    """Unpack groups of bits in byte sequence into numpy array."""
    cdef:
        const uint8_t[::1] src = data
        uint8_t* srcptr = <uint8_t*>&src[0]
        uint8_t* dstptr = NULL
        ssize_t srcsize = src.size
        ssize_t dstsize = 0
        ssize_t bytesize
        ssize_t itemsize
        ssize_t skipbits, i
        ssize_t ret = 0

    if numbits < 1 or (numbits > 32 and numbits != 64):
        raise ValueError('numbits out of range')

    bytesize = <ssize_t>ceil(numbits / 8.0)
    itemsize = bytesize if bytesize < 3 else (8 if bytesize > 4 else 4)

    if srcsize > <ssize_t>SSIZE_MAX / itemsize:
        raise ValueError('data size out of range')

    dtype = numpy.dtype(dtype)
    if dtype.itemsize != itemsize:
        raise ValueError('dtype.itemsize does not fit numbits')

    if runlen == 0:
        runlen = <ssize_t>((<uint64_t>srcsize * 8) / <uint64_t>numbits)

    skipbits = <ssize_t>((<uint64_t>runlen * <uint64_t>numbits) % 8)
    if skipbits > 0:
        skipbits = 8 - skipbits

    dstsize = <ssize_t>(<uint64_t>runlen * <uint64_t>numbits
                        + <uint64_t>skipbits)
    if dstsize > 0:
        dstsize = <ssize_t>(<uint64_t>runlen * ((<uint64_t>srcsize * 8)
                            / <uint64_t>dstsize))

    if out is None or out is data:
        out = numpy.empty(dstsize, dtype)
    else:
        if out.dtype != dtype or out.size < dstsize:
            raise ValueError('invalid output size or dtype')
        if not numpy.PyArray_ISCONTIGUOUS(out):
            raise ValueError('output array is not contiguous')
    if dstsize == 0:
        return out

    dstptr = <uint8_t *>numpy.PyArray_DATA(out)
    srcsize = <ssize_t>((<uint64_t>runlen * <uint64_t>numbits
                        + <uint64_t>skipbits) / 8)

    with nogil:
        # work around "Converting to Python object not allowed without gil"
        # for i in range(0, dstsize, runlen):
        for i from 0 <= i < dstsize by runlen:
            ret = icd_packints_decode(<const uint8_t*>srcptr, srcsize,
                                      dstptr, runlen, numbits)
            if ret < 0:
                break
            srcptr += srcsize
            dstptr += runlen * itemsize

    if ret < 0:
        raise IcdError('icd_packints_decode', ret)

    if not dtype.isnative and numbits % 8:
        itemsize = dtype.itemsize
        dstptr = <uint8_t*>numpy.PyArray_DATA(out)
        with nogil:
            icd_swapbytes(<void *>dstptr, dstsize, itemsize)

    return out


# LZW #########################################################################

cdef extern from 'imagecodecs.h':

    ctypedef struct icd_lzw_handle_t:
        pass

    icd_lzw_handle_t *icd_lzw_new(ssize_t buffersize) nogil

    void icd_lzw_del(icd_lzw_handle_t *handle) nogil

    ssize_t icd_lzw_decode_size(
        icd_lzw_handle_t *handle,
        const uint8_t *src,
        const ssize_t srcsize) nogil

    ssize_t icd_lzw_decode(
        icd_lzw_handle_t *handle,
        const uint8_t *src,
        const ssize_t srcsize,
        uint8_t *dst,
        const ssize_t dstsize) nogil


def lzw_decode(data, buffersize=0, out=None):
    """Decompress LZW.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret = 0
        icd_lzw_handle_t *handle = NULL

    out, dstsize, out_given, out_type = _parse_output(out)

    handle = icd_lzw_new(buffersize)
    if handle == NULL:
        raise IcdError('icd_lzw_new', None)
    try:
        if out is None or out is data:
            if dstsize < 0:
                with nogil:
                    dstsize = icd_lzw_decode_size(handle, &src[0], srcsize)
                if dstsize < 0:
                    raise IcdError('icd_lzw_decode_size', dstsize)
            if out_type is bytes:
                out = PyBytes_FromStringAndSize(NULL, dstsize)
            else:
                out = PyByteArray_FromStringAndSize(NULL, dstsize)

        dst = out
        dstsize = dst.size

        with nogil:
            ret = icd_lzw_decode(handle, &src[0], srcsize,
                                 <uint8_t*>&dst[0], dstsize)
        if ret < 0:
            raise IcdError('icd_lzw_decode', ret)
    finally:
        icd_lzw_del(handle)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


def lzw_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: LZW encoding
    raise NotImplementedError('lzw_encode')


###############################################################################

# TODO: LZW encode
# TODO: Integer resize; magic kernel
# TODO: Dtype conversion/quantizations
# TODO: Scale Offset
