# -*- coding: utf-8 -*-
# _imagecodecs.pyx
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2008-2018, Christoph Gohlke
# Copyright (c) 2008-2018, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Image transformation, compression, and decompression codecs.

The imagecodecs package provides various block-oriented, in-memory
buffer transformation, compression, and decompression functions
for use in the tiffile, czifile, and other Python imaging modules.

Decode and/or encode functions are currently implemented for Zlib DEFLATE,
ZStandard, LZMA, BZ2, LZ4, LZW, LZF, PNG, WebP, JPEG, JPEG 12-bit, JPEG 2000,
JPEG XR, PackBits, Packed Integers, Delta, XOR Delta, Floating Point Predictor,
and Bitorder reversal.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:Version: 2018.10.10

Requirements
------------
* `CPython 2.7 or 3.5+ <https://www.python.org>`_
* `Numpy 1.14 <https://www.numpy.org>`_
* `Cython 0.28 <http://cython.org/>`_
* `zlib 1.2.11 <https://github.com/madler/zlib/>`_
* `lz4 1.8.2 <https://github.com/lz4/lz4/>`_
* `zstd 1.3.5 <https://github.com/facebook/zstd/>`_
* `bzip2 1.0.6 <http://www.bzip.org/>`_
* `xz lzma 5.2.4 <https://github.com/xz-mirror/xz/>`_
* `libpng 1.6.35 <https://github.com/glennrp/libpng/>`_
* `libwebp 1.0 <https://github.com/webmproject/libwebp/>`_
* `liblzf 3.6 <http://oldhome.schmorp.de/marc/liblzf.html>`_
* `libjpeg-turbo 2.0 <https://libjpeg-turbo.org/>`_
* `openjpeg 2.3 <http://www.openjpeg.org/>`_
* `jxrlib 0.2.0 <https://github.com/glencoesoftware/jxrlib/>`_
  with `patch <https://www.lfd.uci.edu/~gohlke/code/
  jxrlib_CreateDecoderFromBytes.diff.html>`_
* A Python distutils compatible C compiler

Notes
-----
Imagecodecs is currently developed, built, and tested on Windows only.

The API is not stable yet and might change between revisions.

Works on little-endian platforms only.

Python 2.7 and 3.4 are deprecated.

Other Python packages providing imaging or compression codecs:

* `numcodecs <https://github.com/zarr-developers/numcodecs>`_
* `python-blosc <https://github.com/Blosc/python-blosc>`_
* `Python zlib <https://docs.python.org/3/library/zlib.html>`_
* `Python bz2 <https://docs.python.org/3/library/bz2.html>`_
* `Python lzma <https://docs.python.org/3/library/lzma.html>`_ and
  `backports.lzma <https://github.com/peterjc/backports.lzma>`_
* `python-snappy <https://github.com/andrix/python-snappy>`_
* `python-brotli <https://github.com/google/brotli/tree/master/python>`_
* `python-lzo <https://bitbucket.org/james_taylor/python-lzo-static>`_
* `python-lz4 <https://github.com/python-lz4/python-lz4>`_
* `python-zstd <https://github.com/sergey-dryabzhinsky/python-zstd>`_
* `python-lzw <https://github.com/joeatwork/python-lzw>`_
* `python-lzf <https://github.com/teepark/python-lzf>`_
* `packbits <https://github.com/psd-tools/packbits>`_

Revisions
---------
2018.10.10
    Add PNG codecs.
    Add option to specify output colorspace in JPEG decoder.
    Fix Delta codec for floating point numbers.
    Fix XOR Delta codecs.
2018.9.30
    Add LZF codecs.
2018.9.22
    Add WebP codecs.
2018.8.29
    Pass 396 tests.
    Add PackBits encoder.
2018.8.22
    Add link library version information.
    Add option to specify size of LZW buffer.
    Add JPEG 2000 decoder.
    Add XOR Delta codec.
2018.8.16
    Link to libjpeg-turbo.
    Support Python 2.7 and Visual Studio 2008.
2018.8.10
    Initial alpha release.
    Add LZW, PackBits, PackInts and FloatPred decoders from tifffile.c module.
    Add JPEG and JXR decoders from czifile.pyx module.

"""

__version__ = '2018.10.10'

import numpy

cimport numpy
cimport cython

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING
from cython.operator cimport dereference as deref

from libc.math cimport ceil
from libc.string cimport memset, memcpy
from libc.stdlib cimport malloc, free
from libc.setjmp cimport setjmp, longjmp, jmp_buf
from libc.stdint cimport (int8_t, uint8_t, int16_t, uint16_t,
                          int32_t, uint32_t, int64_t, uint64_t, UINT64_MAX)

cdef extern from 'Python.h':
    object PyByteArray_FromStringAndSize(const char *string, Py_ssize_t len)


numpy.import_array()


###############################################################################

cdef extern from 'imagecodecs.h':
    char* ICD_VERSION
    char ICD_BOC
    int ICD_OK = 0
    int ICD_ERROR = -1
    int ICD_MEMORY_ERROR = -2
    int ICD_RUNTIME_ERROR = -3
    int ICD_NOTIMPLEMENTED_ERROR = -4
    int ICD_VALUE_ERROR = -5
    int ICD_LZW_INVALID = -10
    int ICD_LZW_NOTIMPLEMENTED = -11
    int ICD_LZW_BUFFER_INSUFFICIENT = -12

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
            ICD_LZW_BUFFER_INSUFFICIENT: 'ICD_LZW_BUFFER_INSUFFICIENT',
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
    elif isinstance(out, int):
        out_size = out
        out = None
    else:
        # out_size = len(out)
        # out_type = type(out)
        out_given = True
    return out, out_size, out_given, out_type


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


cdef _parse_input(data):
    """Return bytes memoryview to input argument."""
    cdef const uint8_t[::1] src
    try:
        src = data
    except ValueError:
        src = numpy.ravel(data, 'K').view(numpy.uint8)
    return src


def _default_level(level, default, smallest, largest):
    """Return compression level in range."""
    if level is None:
        level = default
    if largest is not None:
        level = min(level, largest)
    if smallest is not None:
        level = max(level, smallest)
    return level


def version(astype=str):
    """Return detailed version information."""
    jpeg_turbo_version = str(LIBJPEG_TURBO_VERSION_NUMBER)
    versions = (
        ('imagecodecs', __version__),
        ('icd', ICD_VERSION.decode('utf-8')),
        ('zlib', '%i.%i.%i' % (ZLIB_VER_MAJOR, ZLIB_VER_MINOR,
                               ZLIB_VER_REVISION)),
        ('lzma', '%i.%i.%i' % (LZMA_VERSION_MAJOR, LZMA_VERSION_MINOR,
                               LZMA_VERSION_PATCH)),
        ('zstd', '%i.%i.%i' % (ZSTD_VERSION_MAJOR, ZSTD_VERSION_MINOR,
                               ZSTD_VERSION_RELEASE)),
        ('lz4', '%i.%i.%i' % (LZ4_VERSION_MAJOR, LZ4_VERSION_MINOR,
                              LZ4_VERSION_RELEASE)),
        ('bz2', str(BZ2_bzlibVersion().decode('utf-8')).split(',')[0]),
        ('lzf', hex(LZF_VERSION)),
        ('png', PNG_LIBPNG_VER_STRING.decode('utf-8')),
        ('webp', hex(WebPGetDecoderVersion())),
        ('jpeg', '%.1f' % (JPEG_LIB_VERSION / 10.0)),
        ('jpeg_turbo', '%i.%i.%i' % (int(jpeg_turbo_version[:1]),
                                     int(jpeg_turbo_version[2:3]),
                                     int(jpeg_turbo_version[4:5]))),
        ('opj', opj_version().decode('utf-8')),
        ('jxr', hex(WMP_SDK_VERSION)),
        )
    if astype is str:
        return ', '.join('%s-%s' % (k, v) for k, v in versions)
    elif astype is dict:
        return dict(versions)
    else:
        return versions


# No Operation ################################################################

def none_decode(data, axis=None, out=None):
    """Decode NOP."""
    return data


def none_encode(data, level=None, axis=None, out=None):
    """Encode NOP."""
    return data


# Delta #######################################################################

cdef extern from 'imagecodecs.h':
    ssize_t icd_delta(void *src,
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
        const uint8_t[::1] dst
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t ret
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

        srciter = numpy.PyArray_IterAllButAxis(data, &axis);
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis);
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
    ssize_t icd_xor(void *src,
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
        const uint8_t[::1] dst
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t ret
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

        srciter = numpy.PyArray_IterAllButAxis(data, &axis);
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis);
        srcsize = data.shape[axis]
        dstsize = out.shape[axis]
        srcstride = data.strides[axis]
        dststride = out.strides[axis]

        itemsize = data.dtype.itemsize

        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = numpy.PyArray_ITER_DATA(srciter)
                dstptr = numpy.PyArray_ITER_DATA(dstiter)
                ret = icd_xor(srcptr, srcsize, srcstride,
                              dstptr, dstsize, dststride,
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
        ret = icd_xor(&src[0], srcsize, srcstride,
                      &dst[0], dstsize, dststride,
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
    ssize_t icd_floatpred(void *src,
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
        ssize_t ret
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
    ssize_t icd_bitorder(uint8_t *src,
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
                    icd_bitorder(srcptr, srcsize, srcstride,
                                 itemsize,
                                 dstptr, dstsize, dststride)
                return data

            srciter = numpy.PyArray_IterAllButAxis(data, &axis);
            srcsize = data.shape[axis] * itemsize
            srcstride = data.strides[axis]
            with nogil:
                while numpy.PyArray_ITER_NOTDONE(srciter):
                    srcptr = <uint8_t *>numpy.PyArray_ITER_DATA(srciter)
                    icd_bitorder(srcptr, srcsize, srcstride,
                                 itemsize,
                                 dstptr, dstsize, dststride)
                    numpy.PyArray_ITER_NEXT(srciter)
            return data

        if out is None:
            out = numpy.empty_like(data)
        else:
            if not isinstance(out, numpy.ndarray):
                raise ValueError('output is not a numpy array')
            if data.shape != out.shape or itemsize != out.dtype.itemsize:
                raise ValueError('output is not compatible with data array')
        srciter = numpy.PyArray_IterAllButAxis(data, &axis);
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis);
        srcsize = data.shape[axis] * itemsize
        dstsize = out.shape[axis] * itemsize
        srcstride = data.strides[axis]
        dststride = out.strides[axis]
        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = <uint8_t *>numpy.PyArray_ITER_DATA(srciter)
                dstptr = <uint8_t *>numpy.PyArray_ITER_DATA(dstiter)
                icd_bitorder(srcptr, srcsize, srcstride,
                             itemsize,
                             dstptr, dstsize, dststride)
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        return out

    # contiguous byte buffers: bytes or bytearray
    src = data
    srcsize = src.size
    if data is out:
        # in-place
        with nogil:
            icd_bitorder(&src[0], srcsize, 1, 1, &src[0], srcsize, 1)
        return data

    if out is None:
        out = PyByteArray_FromStringAndSize(NULL, srcsize)
    dst = out
    dstsize = dst.size
    with nogil:
        icd_bitorder(&src[0], srcsize, 1, 1, &dst[0], dstsize, 1)
    return out


bitorder_encode = bitorder_decode


# PackBits ####################################################################

cdef extern from 'imagecodecs.h':
    ssize_t icd_packbits_size(const uint8_t *src,
                              const ssize_t srcsize) nogil

    ssize_t icd_packbits_decode(const uint8_t *src,
                                const ssize_t srcsize,
                                uint8_t *dst,
                                const ssize_t dstsize) nogil

    ssize_t icd_packbits_encode(const uint8_t *src,
                                const ssize_t srcsize,
                                uint8_t *dst,
                                const ssize_t dstsize) nogil


def packbits_encode(data, level=None, out=None):
    """Compress PackBits.

    """
    cdef:
        numpy.flatiter srciter
        const uint8_t[::1] src
        const uint8_t[::1] dst
        const uint8_t* srcptr
        const uint8_t* dstptr
        ssize_t srcsize
        ssize_t dstsize
        ssize_t ret,
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
                ret = icd_packbits_encode(srcptr, srcsize, dstptr, dstsize)
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
            ret = icd_packbits_encode(&src[0], srcsize, &dst[0], dstsize)

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
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret

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
        ret = icd_packbits_decode(&src[0], srcsize, &dst[0], dstsize)
    if ret < 0:
        raise IcdError('icd_packbits_decode', ret)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


# Packed Integers #############################################################

cdef extern from 'imagecodecs.h':
    int SSIZE_MAX

    ssize_t icd_packints_decode(const uint8_t *src,
                                const ssize_t srcsize,
                                uint8_t *dst,
                                const ssize_t dstsize,
                                const int numbits) nogil

    void icd_swapbytes(void *src,
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
        uint8_t* srcptr = &src[0]
        uint8_t* dstptr = NULL
        ssize_t srcsize = src.size
        ssize_t dstsize = 0
        ssize_t bytesize
        ssize_t itemsize
        ssize_t skipbits, i, ret

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
            ret = icd_packints_decode(srcptr, srcsize,
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

    ssize_t icd_lzw_size(icd_lzw_handle_t *handle,
                         const uint8_t *src,
                         const ssize_t srcsize) nogil

    ssize_t icd_lzw_decode(icd_lzw_handle_t *handle,
                           const uint8_t *src,
                           const ssize_t srcsize,
                           uint8_t *dst,
                           const ssize_t dstsize) nogil


def lzw_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: LZW encoding
    raise NotImplementedError('lzw_encode')


def lzw_decode(data, buffersize=0, out=None):
    """Decompress LZW.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret
        icd_lzw_handle_t *handle = NULL

    out, dstsize, out_given, out_type = _parse_output(out)

    handle = icd_lzw_new(buffersize)
    if handle == NULL:
        raise IcdError('icd_lzw_new', None)
    try:
        if out is None or out is data:
            if dstsize < 0:
                with nogil:
                    dstsize = icd_lzw_size(handle, &src[0], srcsize)
                if dstsize < 0:
                    raise IcdError('icd_lzw_size', dstsize)
            if out_type is bytes:
                out = PyBytes_FromStringAndSize(NULL, dstsize)
            else:
                out = PyByteArray_FromStringAndSize(NULL, dstsize)

        dst = out
        dstsize = dst.size

        with nogil:
            ret = icd_lzw_decode(handle, &src[0], srcsize, &dst[0], dstsize)
        if ret < 0:
            raise IcdError('icd_lzw_decode', ret)
    finally:
        icd_lzw_del(handle)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


# Zlib DEFLATE ################################################################

cdef extern from 'zlib.h':
    int ZLIB_VER_MAJOR
    int ZLIB_VER_MINOR
    int ZLIB_VER_REVISION
    int Z_OK
    int Z_MEM_ERROR
    int Z_BUF_ERROR
    int Z_DATA_ERROR
    int Z_STREAM_ERROR

    ctypedef unsigned char Bytef
    ctypedef unsigned long uLong
    ctypedef unsigned long uLongf
    ctypedef unsigned int uInt

    uLong crc32(uLong crc, const Bytef *buf, uInt len) nogil

    int uncompress2(Bytef *dst,
                    uLongf *dstLen,
                    const Bytef *src,
                    uLong *srcLen) nogil

    int compress2(Bytef *dst,
                  uLongf *dstLen,
                  const Bytef *src,
                  uLong srcLen,
                  int level) nogil


class ZlibError(RuntimeError):
    """Zlib Exceptions."""
    def __init__(self, func, err):
        msg = {
            Z_OK: 'Z_OK',
            Z_MEM_ERROR: 'Z_MEM_ERROR',
            Z_BUF_ERROR: 'Z_BUF_ERROR',
            Z_DATA_ERROR: 'Z_DATA_ERROR',
            Z_STREAM_ERROR: 'Z_STREAM_ERROR',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def zlib_encode(data, level=None, out=None):
    """Compress Zlib DEFLATE.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)  # TODO: non-contiguous
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret
        int compresslevel = _default_level(level, 6, 0, 9)

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's zlib module
        import zlib
        return zlib.compress(data, compresslevel)

    if out is None or out is data:
        if dstsize < 0:
            raise ValueError('invalid output')
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <unsigned long>dstsize
    srclen = <unsigned long>srcsize

    with nogil:
        ret = compress2(&dst[0], &dstlen, &src[0], srclen, compresslevel)
    if ret != Z_OK:
        raise ZlibError('compress2', ret)

    if dstlen < dstsize:
        out = memoryview(out)[:dstlen] if out_given else out[:dstlen]

    return out


def zlib_decode(data, out=None):
    """Decompress Zlib DEFLATE.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        unsigned long srclen, dstlen
        int ret

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's zlib module
        import zlib
        return zlib.decompress(data)

    if out is None or out is data:
        if dstsize < 0:
            raise ValueError('invalid output size')
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size
    dstlen = <unsigned long>dstsize
    srclen = <unsigned long>srcsize

    with nogil:
        ret = uncompress2(&dst[0], &dstlen, &src[0], &srclen)
    if ret != Z_OK:
        raise ZlibError('uncompress2', ret)

    if dstlen < dstsize:
        out = memoryview(out)[:dstlen] if out_given else out[:dstlen]

    return out


def zlib_crc32(data):
    """Return cyclic redundancy checksum CRC-32 of data."""
    cdef:
        const uint8_t[::1] src = _parse_input(data)  # TODO: non-contiguous
        uInt srcsize = <uInt>src.size
        uLong crc = 0
    with nogil:
        crc = crc32(crc, NULL, 0)
        crc = crc32(crc, <Bytef *>&src[0], srcsize)
    return int(crc)


# ZStandard ###################################################################

cdef extern from 'zstd.h':
    int ZSTD_VERSION_MAJOR
    int ZSTD_VERSION_MINOR
    int ZSTD_VERSION_RELEASE
    int ZSTD_CONTENTSIZE_UNKNOWN
    int ZSTD_CONTENTSIZE_ERROR

    unsigned int ZSTD_isError(size_t code) nogil
    size_t ZSTD_compressBound(size_t srcSize) nogil
    const char* ZSTD_getErrorName(size_t code) nogil
    uint64_t ZSTD_getFrameContentSize(const void *src, size_t srcSize) nogil

    size_t ZSTD_decompress(void* dst,
                           size_t dstCapacity,
                           const void* src,
                           size_t compressedSize) nogil

    size_t ZSTD_compress(void* dst,
                         size_t dstCapacity,
                         const void* src,
                         size_t srcSize,
                         int compressionLevel) nogil


class ZstdError(RuntimeError):
    """ZStandard Exceptions."""
    def __init__(self, func, msg='', err=0):
        cdef char* errmsg
        if msg:
            RuntimeError.__init__(self, "%s returned '%s'" % (func, msg))
        else:
            errmsg = ZSTD_getErrorName(err)
            RuntimeError.__init__(
                self, u"%s returned '%s'" % (func, errmsg.decode('utf-8')))


def zstd_encode(data, level=None, out=None):
    """Compress ZStandard.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst
        size_t srcsize = src.size
        size_t dstsize
        ssize_t dstlen
        size_t ret
        char* errmsg
        int compresslevel = _default_level(level, 5, 1, 22)

    out, dstlen, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstlen < 0:
            dstlen = <ssize_t>ZSTD_compressBound(srcsize)
            if dstlen < 0:
                raise ZstdError('ZSTD_compressBound', '%i' % dstlen)
        if dstlen < 64:
            dstlen = 64
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstlen)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstlen)

    dst = out
    dstsize = <size_t>dst.size

    with nogil:
        ret = ZSTD_compress(<void *>&dst[0], dstsize,
                            <void *>&src[0], srcsize, compresslevel)
    if ZSTD_isError(ret):
        raise ZstdError('ZSTD_compress', err=ret)
    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


def zstd_decode(data, out=None):
    """Decompress ZStandard.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        size_t srcsize = <size_t>src.size
        size_t dstsize
        ssize_t dstlen
        size_t ret
        uint64_t cntsize
        char* errmsg

    out, dstlen, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstlen < 0:
            cntsize = ZSTD_getFrameContentSize(<void *>&src[0], srcsize)
            if cntsize == ZSTD_CONTENTSIZE_UNKNOWN:
                raise ZstdError('ZSTD_getFrameContentSize',
                                'ZSTD_CONTENTSIZE_UNKNOWN')
            if cntsize == ZSTD_CONTENTSIZE_ERROR:
                raise ZstdError('ZSTD_getFrameContentSize',
                                'ZSTD_CONTENTSIZE_ERROR')
            dstlen = <ssize_t>cntsize
            if dstlen < 0:
                raise ZstdError('ZSTD_getFrameContentSize', '%i' % dstlen)

        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstlen)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstlen)

    dst = out
    dstsize = <size_t>dst.size

    with nogil:
        ret = ZSTD_decompress(<void *>&dst[0], dstsize,
                              <void *>&src[0], srcsize)
    if ZSTD_isError(ret):
        raise ZstdError('ZSTD_decompress', err=ret)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


# LZ4 #########################################################################

cdef extern from 'lz4.h':
    int LZ4_VERSION_MAJOR
    int LZ4_VERSION_MINOR
    int LZ4_VERSION_RELEASE
    int LZ4_MAX_INPUT_SIZE

    int LZ4_compressBound(int isize) nogil

    int LZ4_compress_fast(const char* src,
                          char* dst,
                          int srcSize,
                          int dstCapacity,
                          int acceleration) nogil

    int LZ4_decompress_safe(const char* src,
                            char* dst,
                            int compressedSize,
                            int dstCapacity) nogil


def lz4_encode(data, level=None, header=False, out=None):
    """Compress LZ4.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst
        int srcsize = src.size
        int dstsize
        int offset = 4 if header else 0
        int ret
        uint8_t *pdst
        int acceleration = _default_level(level, 1, 1, 1000)

    if src.size > LZ4_MAX_INPUT_SIZE:
        raise ValueError('data too large')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstsize < 0:
            dstsize = LZ4_compressBound(srcsize) + offset
            if dstsize < 0:
                raise RuntimeError('LZ4_compressBound returned %i' % dstsize)
        if dstsize < offset:
            dstsize = offset
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = <int>dst.size - offset

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_compress_fast(<char *>&src[0], <char *>&dst[offset],
                                srcsize, dstsize, acceleration)
    if ret <= 0:
        raise RuntimeError('LZ4_compress_fast returned %i' % ret)

    if header:
        pdst = <uint8_t *>&dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    if ret < dstsize:
        out = memoryview(out)[:ret+offset] if out_given else out[:ret+offset]

    return out


def lz4_decode(data, header=False, out=None):
    """Decompress LZ4.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        int srcsize = <int>src.size
        int dstsize
        int offset = 4 if header else 0
        int ret

    if src.size > 2**31:
        raise ValueError('data too large')

    out, dstsize, out_given, out_type = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None or out is data:
        if dstsize < 0:
            dstsize = max(24, 24 + 255 * (srcsize - offset - 10))  # ugh
            if dstsize < 0:
                raise RuntimeError('invalid output size %i' % dstsize)
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = <int>dst.size

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = LZ4_decompress_safe(<char *>&src[offset], <char *>&dst[0],
                                  srcsize-offset, dstsize)
    if ret < 0:
        raise RuntimeError('LZ4_decompress_safe returned %i' % ret)

    if ret < dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


# LZF #########################################################################

cdef extern from 'lzf.h':
    int LZF_VERSION

    unsigned int lzf_compress(const void *const in_data,
                              unsigned int in_len,
                              void *out_data,
                              unsigned int out_len) nogil

    unsigned int lzf_decompress(const void *const in_data,
                                unsigned int in_len,
                                void *out_data,
                                unsigned int out_len) nogil


def lzf_encode(data, level=None, header=False, out=None):
    """Compress LZF.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst
        int srcsize = <int>src.size
        int dstsize
        unsigned int ret
        uint8_t *pdst
        int offset = 4 if header else 0

    if src.size > 2**31:
        raise ValueError('data too large')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstsize < 0:
            # dstsize = ((srcsize * 33) >> 5 ) + 1 + offset
            dstsize = srcsize + srcsize // 20 + 32
        else:
            dstsize += 1  # bug in liblzf ?
        if dstsize < offset:
            dstsize = offset
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = <int>dst.size - offset

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = lzf_compress(<void *>&src[0], <unsigned int>srcsize,
                           <void *>&dst[offset], <unsigned int>dstsize)
    if ret == 0:
        raise RuntimeError('lzf_compress returned 0')

    if header:
        pdst = <uint8_t *>&dst[0]
        pdst[0] = srcsize & 255
        pdst[1] = (srcsize >> 8) & 255
        pdst[2] = (srcsize >> 16) & 255
        pdst[3] = (srcsize >> 24) & 255

    if ret < <unsigned int>dstsize:
        out = memoryview(out)[:ret+offset] if out_given else out[:ret+offset]

    return out


def lzf_decode(data, header=False, out=None):
    """Decompress LZF.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        int dstsize
        int srcsize = <unsigned int>src.size
        unsigned int ret
        int offset = 4 if header else 0

    if src.size > 2**31:
        raise ValueError('data too large')

    out, dstsize, out_given, out_type = _parse_output(out)

    if header and dstsize < 0:
        if srcsize < offset:
            raise ValueError('invalid data size')
        dstsize = src[0] | (src[1] << 8) | (src[2] << 16) | (src[3] << 24)

    if out is None or out is data:
        if dstsize < 0:
            dstsize = srcsize
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = <int>dst.size

    if dst.size > 2**31:
        raise ValueError('output too large')

    with nogil:
        ret = lzf_decompress(<void *>&src[offset], srcsize-offset,
                             <void *>&dst[0], dstsize)
    if ret == 0:
        raise RuntimeError('lzf_decompress returned %i' % ret)

    if ret < <unsigned int>dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


# LZMA ########################################################################

cdef extern from 'lzma.h':
    int LZMA_VERSION_MAJOR
    int LZMA_VERSION_MINOR
    int LZMA_VERSION_PATCH
    int LZMA_CONCATENATED
    int LZMA_STREAM_HEADER_SIZE

    ctypedef uint64_t lzma_vli

    ctypedef struct lzma_stream_flags:
        uint32_t version
        lzma_vli backward_size

    ctypedef struct lzma_index:
        pass

    ctypedef struct lzma_allocator:
        pass

    ctypedef struct lzma_internal:
        pass

    ctypedef enum lzma_reserved_enum:
        LZMA_RESERVED_ENUM

    ctypedef enum lzma_check:
        LZMA_CHECK_NONE = 0
        LZMA_CHECK_CRC32 = 1
        LZMA_CHECK_CRC64 = 4
        LZMA_CHECK_SHA256 = 10

    ctypedef struct lzma_stream:
        uint8_t *next_in
        size_t avail_in
        uint64_t total_in
        uint8_t *next_out
        size_t avail_out
        uint64_t total_out
        lzma_allocator *allocator
        lzma_internal *internal
        void *reserved_ptr1
        void *reserved_ptr2
        void *reserved_ptr3
        void *reserved_ptr4
        uint64_t reserved_int1
        uint64_t reserved_int2
        size_t reserved_int3
        size_t reserved_int4
        lzma_reserved_enum reserved_enum1
        lzma_reserved_enum reserved_enum2

    ctypedef enum lzma_action:
        LZMA_RUN,
        LZMA_SYNC_FLUSH,
        LZMA_FULL_FLUSH,
        LZMA_FULL_BARRIER,
        LZMA_FINISH

    ctypedef enum lzma_ret:
        LZMA_OK,
        LZMA_STREAM_END,
        LZMA_NO_CHECK,
        LZMA_UNSUPPORTED_CHECK,
        LZMA_GET_CHECK,
        LZMA_MEM_ERROR,
        LZMA_MEMLIMIT_ERROR,
        LZMA_FORMAT_ERROR,
        LZMA_OPTIONS_ERROR,
        LZMA_DATA_ERROR,
        LZMA_BUF_ERROR,
        LZMA_PROG_ERROR

    lzma_ret lzma_easy_encoder(lzma_stream *strm,
                               uint32_t preset,
                               lzma_check check) nogil

    lzma_ret lzma_stream_decoder(lzma_stream *strm,
                                 uint64_t memlimit,
                                 uint32_t flags) nogil

    lzma_ret lzma_stream_footer_decode(lzma_stream_flags *options,
                                       const uint8_t *in_) nogil

    lzma_ret lzma_index_buffer_decode(lzma_index **i,
                                      uint64_t *memlimit,
                                      const lzma_allocator *allocator,
                                      const uint8_t *in_,
                                      size_t *in_pos,
                                      size_t in_size) nogil

    lzma_ret lzma_code(lzma_stream *strm, lzma_action action) nogil
    void lzma_end(lzma_stream *strm) nogil
    size_t lzma_stream_buffer_bound(size_t uncompressed_size) nogil
    lzma_vli lzma_index_uncompressed_size(const lzma_index *i) nogil
    lzma_index * lzma_index_init(const lzma_allocator *allocator) nogil
    void lzma_index_end(lzma_index *i, const lzma_allocator *allocator) nogil


class LzmaError(RuntimeError):
    """LZMA Exceptions."""
    def __init__(self, func, err):
        msg = {
            LZMA_OK: 'LZMA_OK',
            LZMA_STREAM_END: 'LZMA_STREAM_END',
            LZMA_NO_CHECK: 'LZMA_NO_CHECK',
            LZMA_UNSUPPORTED_CHECK: 'LZMA_UNSUPPORTED_CHECK',
            LZMA_GET_CHECK: 'LZMA_GET_CHECK',
            LZMA_MEM_ERROR: 'LZMA_MEM_ERROR',
            LZMA_MEMLIMIT_ERROR: 'LZMA_MEMLIMIT_ERROR',
            LZMA_FORMAT_ERROR: 'LZMA_FORMAT_ERROR',
            LZMA_OPTIONS_ERROR: 'LZMA_OPTIONS_ERROR',
            LZMA_DATA_ERROR: 'LZMA_DATA_ERROR',
            LZMA_BUF_ERROR: 'LZMA_BUF_ERROR',
            LZMA_PROG_ERROR: 'LZMA_PROG_ERROR',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def _lzma_uncompressed_size(const uint8_t[::1] data, ssize_t size):
    """Return size of decompressed LZMA data."""
    cdef:
        lzma_ret ret
        lzma_index *index
        lzma_stream_flags options
        lzma_vli usize = 0
        uint64_t memlimit = UINT64_MAX
        ssize_t offset
        size_t pos = 0

    if size < LZMA_STREAM_HEADER_SIZE:
        raise ValueError('invalid LZMA data')
    try:
        index = lzma_index_init(NULL)
        offset = size - LZMA_STREAM_HEADER_SIZE
        ret = lzma_stream_footer_decode(&options, &data[offset])
        if ret != LZMA_OK:
            raise LzmaError('lzma_stream_footer_decode', ret)
        offset -= options.backward_size
        ret = lzma_index_buffer_decode(&index, &memlimit, NULL,
                                       &data[offset], &pos,
                                       options.backward_size)
        if ret != LZMA_OK or pos != options.backward_size:
            raise LzmaError('lzma_index_buffer_decode', ret)
        usize = lzma_index_uncompressed_size(index)
    finally:
        lzma_index_end(index, NULL)
    return <ssize_t>usize


def lzma_decode(data, out=None):
    """Decompress LZMA.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen
        lzma_ret ret
        lzma_stream strm

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstsize < 0:
            dstsize = _lzma_uncompressed_size(src, srcsize)
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(lzma_stream))
    ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED)
    if ret != LZMA_OK:
        raise LzmaError('lzma_stream_decoder', ret)

    try:
        with nogil:
            strm.next_in = &src[0]
            strm.avail_in = <size_t>srcsize
            strm.next_out = &dst[0]
            strm.avail_out = <size_t>dstsize
            ret = lzma_code(&strm, LZMA_RUN)
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret != LZMA_OK and ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    if dstlen < dstsize:
        out = memoryview(out)[:dstlen] if out_given else out[:dstlen]

    return out


def lzma_encode(data, level=None, out=None):
    """Compress LZMA.

    """
    cdef:
        const uint8_t[::1] src = _parse_input(data)
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen
        uint32_t preset = _default_level(level, 6, 0, 9)
        lzma_stream strm
        lzma_ret ret

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None or out is data:
        if dstsize < 0:
            dstsize = lzma_stream_buffer_bound(srcsize)
            if dstsize == 0:
                raise RuntimeError('lzma_stream_buffer_bound returned 0')
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(lzma_stream))
    ret = lzma_easy_encoder(&strm, preset, LZMA_CHECK_CRC64)
    if ret != LZMA_OK:
        raise LzmaError('lzma_easy_encoder', ret)

    try:
        with nogil:
            strm.next_in = &src[0]
            strm.avail_in = <size_t>srcsize
            strm.next_out = &dst[0]
            strm.avail_out = <size_t>dstsize
            ret = lzma_code(&strm, LZMA_RUN)
            if ret == LZMA_OK or ret == LZMA_STREAM_END:
                ret = lzma_code(&strm, LZMA_FINISH)
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret != LZMA_STREAM_END:
            raise LzmaError('lzma_code', ret)
    finally:
        lzma_end(&strm)

    if dstlen < dstsize:
        out = memoryview(out)[:dstlen] if out_given else out[:dstlen]

    return out


# BZ2 #########################################################################

cdef extern from 'bzlib.h':
    int BZ_RUN
    int BZ_FLUSH
    int BZ_FINISH

    int BZ_OK
    int BZ_RUN_OK
    int BZ_FLUSH_OK
    int BZ_FINISH_OK
    int BZ_STREAM_END
    int BZ_SEQUENCE_ERROR
    int BZ_PARAM_ERROR
    int BZ_MEM_ERROR
    int BZ_DATA_ERROR
    int BZ_DATA_ERROR_MAGIC
    int BZ_IO_ERROR
    int BZ_UNEXPECTED_EOF
    int BZ_OUTBUFF_FULL
    int BZ_CONFIG_ERROR

    ctypedef struct bz_stream:
        char *next_in
        unsigned int avail_in
        unsigned int total_in_lo32
        unsigned int total_in_hi32
        char *next_out
        unsigned int avail_out
        unsigned int total_out_lo32
        unsigned int total_out_hi32
        void *state
        void *(*bzalloc)(void *,int,int)
        void (*bzfree)(void *,void *)
        void *opaque

    int BZ2_bzCompressInit(bz_stream* strm,
                           int blockSize100k,
                           int verbosity,
                           int workFactor) nogil

    int BZ2_bzCompress(bz_stream* strm, int action) nogil
    int BZ2_bzCompressEnd(bz_stream* strm) nogil
    int BZ2_bzDecompressInit(bz_stream *strm, int verbosity, int small) nogil
    int BZ2_bzDecompress(bz_stream* strm) nogil
    int BZ2_bzDecompressEnd(bz_stream *strm) nogil
    char* BZ2_bzlibVersion() nogil


class Bz2Error(RuntimeError):
    """BZ2 Exceptions."""
    def __init__(self, func, err):
        msg = {
            BZ_OK: 'BZ_OK',
            BZ_RUN_OK: 'BZ_RUN_OK',
            BZ_FLUSH_OK: 'BZ_FLUSH_OK',
            BZ_FINISH_OK: 'BZ_FINISH_OK',
            BZ_STREAM_END: 'BZ_STREAM_END',
            BZ_SEQUENCE_ERROR: 'BZ_SEQUENCE_ERROR',
            BZ_PARAM_ERROR: 'BZ_PARAM_ERROR',
            BZ_MEM_ERROR: 'BZ_MEM_ERROR',
            BZ_DATA_ERROR: 'BZ_DATA_ERROR',
            BZ_DATA_ERROR_MAGIC: 'BZ_DATA_ERROR_MAGIC',
            BZ_IO_ERROR: 'BZ_IO_ERROR',
            BZ_UNEXPECTED_EOF: 'BZ_UNEXPECTED_EOF',
            BZ_OUTBUFF_FULL: 'BZ_OUTBUFF_FULL',
            BZ_CONFIG_ERROR: 'BZ_CONFIG_ERROR',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def bz2_encode(data, level=None, out=None):
    """Compress BZ2.

    """
    cdef:
        const uint8_t[::1] src =  _parse_input(data)
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen = 0
        int ret
        bz_stream strm
        int compresslevel = _default_level(level, 9, 1, 9)

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's bz2 module
        import bz2
        return bz2.compress(data, compresslevel)

    if out is None or out is data:
        if dstsize < 0:
            raise ValueError('invalid output')
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(bz_stream))
    ret = BZ2_bzCompressInit(&strm, compresslevel, 0, 0)
    if ret != BZ_OK:
        raise Bz2Error('BZ2_bzCompressInit', ret)

    try:
        with nogil:
            strm.next_in = <char *>&src[0]
            strm.avail_in = <unsigned int>srcsize
            strm.next_out = <char *>&dst[0]
            strm.avail_out = <unsigned int>dstsize
            # while True
            ret = BZ2_bzCompress(&strm, BZ_FINISH)
            #    if ret == BZ_STREAM_END:
            #        break
            #    elif ret != BZ_OK:
            #        break
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzCompress', ret)
    finally:
        ret = BZ2_bzCompressEnd(&strm)

    if dstlen < dstsize:
        out = memoryview(out)[:dstlen] if out_given else out[:dstlen]

    return out


def bz2_decode(data, out=None):
    """Decompress BZ2.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t dstlen = 0
        int ret
        bz_stream strm

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None and dstsize < 0:
        # use Python's bz2 module
        import bz2
        return bz2.decompress(data)

    if out is None or out is data:
        if dstsize < 0:
            raise ValueError('invalid output')
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size

    memset(&strm, 0, sizeof(bz_stream))
    ret = BZ2_bzDecompressInit(&strm, 0, 0)
    if ret != BZ_OK:
        raise Bz2Error('BZ2_bzDecompressInit', ret)

    try:
        with nogil:
            strm.next_in = <char *>&src[0]
            strm.avail_in = <unsigned int>srcsize
            strm.next_out = <char *>&dst[0]
            strm.avail_out = <unsigned int>dstsize
            ret = BZ2_bzDecompress(&strm)
            dstlen = dstsize - <ssize_t>strm.avail_out
        if ret == BZ_OK:
            pass  # output buffer too small
        elif ret != BZ_STREAM_END:
            raise Bz2Error('BZ2_bzDecompress', ret)
    finally:
        ret = BZ2_bzDecompressEnd(&strm)

    if dstlen < dstsize:
        out = memoryview(out)[:dstlen] if out_given else out[:dstlen]

    return out


# JPEG XR #####################################################################

cdef extern from 'windowsmediaphoto.h':
    int WMP_errSuccess
    int WMP_errFail
    int WMP_errNotYetImplemented
    int WMP_errAbstractMethod
    int WMP_errOutOfMemory
    int WMP_errFileIO
    int WMP_errBufferOverflow
    int WMP_errInvalidParameter
    int WMP_errInvalidArgument
    int WMP_errUnsupportedFormat
    int WMP_errIncorrectCodecVersion
    int WMP_errIndexNotFound
    int WMP_errOutOfSequence
    int WMP_errNotInitialized
    int WMP_errMustBeMultipleOf16LinesUntilLastCall
    int WMP_errPlanarAlphaBandedEncRequiresTempFile
    int WMP_errAlphaModeCannotBeTranscoded
    int WMP_errIncorrectCodecSubVersion

    ctypedef long ERR
    ctypedef int I32
    ctypedef int PixelI
    ctypedef unsigned char U8
    ctypedef unsigned int U32


cdef extern from 'guiddef.h':
    ctypedef struct GUID:
        pass

    int IsEqualGUID(GUID*, GUID*)


cdef extern from 'JXRGlue.h':
    int WMP_SDK_VERSION
    int PK_SDK_VERSION

    ctypedef GUID PKPixelFormatGUID

    GUID GUID_PKPixelFormat8bppGray
    GUID GUID_PKPixelFormat16bppGray
    GUID GUID_PKPixelFormat32bppGrayFloat
    GUID GUID_PKPixelFormat24bppBGR
    GUID GUID_PKPixelFormat24bppRGB
    GUID GUID_PKPixelFormat48bppRGB
    GUID GUID_PKPixelFormat128bppRGBFloat
    GUID GUID_PKPixelFormat32bppRGBA
    GUID GUID_PKPixelFormat32bppBGRA
    GUID GUID_PKPixelFormat64bppRGBA
    GUID GUID_PKPixelFormat128bppRGBAFloat

    ctypedef struct PKFactory:
        pass

    ctypedef struct PKCodecFactory:
        pass

    ctypedef struct PKImageDecode:
        pass

    ctypedef struct PKImageEncode:
        pass

    ctypedef struct PKFormatConverter:
        pass

    ctypedef struct PKRect:
        I32 X, Y, Width, Height

    ERR PKCreateCodecFactory(PKCodecFactory**, U32) nogil
    ERR PKCreateCodecFactory_Release(PKCodecFactory**) nogil
    ERR PKCodecFactory_CreateFormatConverter(PKFormatConverter**) nogil
    ERR PKImageDecode_GetSize(PKImageDecode*, I32*, I32*) nogil
    ERR PKImageDecode_Release(PKImageDecode**) nogil
    ERR PKImageDecode_GetPixelFormat(PKImageDecode*, PKPixelFormatGUID*) nogil
    ERR PKFormatConverter_Release(PKFormatConverter**) nogil

    ERR PKCodecFactory_CreateDecoderFromBytes(void*,
                                              size_t,
                                              PKImageDecode**) nogil

    ERR PKFormatConverter_Initialize(PKFormatConverter*,
                                     PKImageDecode*,
                                     char*,
                                     PKPixelFormatGUID) nogil

    ERR PKFormatConverter_Copy(PKFormatConverter*,
                               const PKRect*,
                               U8*,
                               U32) nogil

    ERR PKFormatConverter_Convert(PKFormatConverter*,
                                  const PKRect*,
                                  U8*,
                                  U32) nogil


class WmpError(RuntimeError):
    """WMP Exceptions."""
    def __init__(self, func, err):
        msg = {
            WMP_errFail: 'WMP_errFail',
            WMP_errNotYetImplemented: 'WMP_errNotYetImplemented',
            WMP_errAbstractMethod: 'WMP_errAbstractMethod',
            WMP_errOutOfMemory: 'WMP_errOutOfMemory',
            WMP_errFileIO: 'WMP_errFileIO',
            WMP_errBufferOverflow: 'WMP_errBufferOverflow',
            WMP_errInvalidParameter: 'WMP_errInvalidParameter',
            WMP_errInvalidArgument: 'WMP_errInvalidArgument',
            WMP_errUnsupportedFormat: 'WMP_errUnsupportedFormat',
            WMP_errIncorrectCodecVersion: 'WMP_errIncorrectCodecVersion',
            WMP_errIndexNotFound: 'WMP_errIndexNotFound',
            WMP_errOutOfSequence: 'WMP_errOutOfSequence',
            WMP_errNotInitialized: 'WMP_errNotInitialized',
            WMP_errAlphaModeCannotBeTranscoded:
                'WMP_errAlphaModeCannotBeTranscoded',
            WMP_errIncorrectCodecSubVersion:
                'WMP_errIncorrectCodecSubVersion',
            WMP_errMustBeMultipleOf16LinesUntilLastCall:
                'WMP_errMustBeMultipleOf16LinesUntilLastCall',
            WMP_errPlanarAlphaBandedEncRequiresTempFile:
                'WMP_errPlanarAlphaBandedEncRequiresTempFile',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def jxr_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: JXR encoding
    raise NotImplementedError('jxr_encode')


def jxr_decode(data, out=None):
    """Decode JPEG XR image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        PKImageDecode* decoder = NULL
        PKFormatConverter* converter = NULL
        PKPixelFormatGUID pixel_format
        PKRect rect
        I32 width
        I32 height
        U32 stride
        ERR err
        ssize_t dstsize

    if data is out:
        raise ValueError('cannot decode in-place')

    try:
        err = PKCodecFactory_CreateDecoderFromBytes(&src[0], src.size,
                                                    &decoder)
        if err:
            raise WmpError('PKCodecFactory_CreateDecoderFromBytes', err)

        err = PKImageDecode_GetSize(decoder, &width, &height)
        if err:
            raise WmpError('PKImageDecode_GetSize', err)

        err = PKImageDecode_GetPixelFormat(decoder, &pixel_format)
        if err:
            raise WmpError('PKImageDecode_GetPixelFormat', err)

        if IsEqualGUID(&pixel_format, &GUID_PKPixelFormat8bppGray):
            dtype = numpy.uint8
            samples = 1
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat16bppGray):
            dtype = numpy.uint16
            samples = 1
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat32bppGrayFloat):
            dtype = numpy.float32
            samples = 1
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat24bppBGR):
            dtype = numpy.uint8
            samples = 3
            pixel_format = GUID_PKPixelFormat24bppRGB
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat24bppRGB):
            dtype = numpy.uint8
            samples = 3
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat48bppRGB):
            dtype = numpy.uint16
            samples = 3
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat128bppRGBFloat):
            dtype = numpy.float32
            samples = 3
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat32bppBGRA):
            dtype = numpy.uint8
            samples = 4
            pixel_format = GUID_PKPixelFormat32bppRGBA
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat32bppRGBA):
            dtype = numpy.uint8
            samples = 4
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat64bppRGBA):
            dtype = numpy.uint16
            samples = 4
        elif IsEqualGUID(&pixel_format, &GUID_PKPixelFormat128bppRGBAFloat):
            dtype = numpy.float32
            samples = 4
        else:
            raise ValueError('unknown pixel format')

        err = PKCodecFactory_CreateFormatConverter(&converter)
        if err:
            raise WmpError('PKCodecFactory_CreateFormatConverter', err)

        err = PKFormatConverter_Initialize(converter, decoder, NULL,
                                           pixel_format)
        if err:
            raise WmpError('PKFormatConverter_Initialize', err)

        shape = height, width
        if samples > 1:
            shape += samples,

        out = _create_array(out, shape, dtype)
        dst = out
        rect.X = 0
        rect.Y = 0
        rect.Width = <I32>dst.shape[1]
        rect.Height = <I32>dst.shape[0]
        stride = <U32>dst.strides[0]
        dstsize = dst.size * dst.itemsize

        # TODO: check alignment issues
        with nogil:
            memset(<void *>dst.data, 0, dstsize)
            err = PKFormatConverter_Copy(converter, &rect, <U8*>dst.data,
                                         stride)
        if err:
            raise WmpError('PKFormatConverter_Copy', err)

    finally:
        if converter != NULL:
            PKFormatConverter_Release(&converter)
        if decoder != NULL:
            PKImageDecode_Release(&decoder)

    return out


# PNG #########################################################################

cdef extern from 'png.h':

    char* PNG_LIBPNG_VER_STRING
    int PNG_COLOR_TYPE_GRAY
    int PNG_COLOR_TYPE_PALETTE
    int PNG_COLOR_TYPE_RGB
    int PNG_COLOR_TYPE_RGB_ALPHA
    int PNG_COLOR_TYPE_GRAY_ALPHA
    int PNG_INTERLACE_NONE
    int PNG_COMPRESSION_TYPE_DEFAULT
    int PNG_FILTER_TYPE_DEFAULT
    int PNG_ZBUF_SIZE

    ctypedef struct png_struct:
        pass

    ctypedef struct png_info:
        pass

    ctypedef size_t png_size_t
    ctypedef unsigned int png_uint_32
    ctypedef unsigned char* png_bytep
    ctypedef unsigned char* png_const_bytep
    ctypedef unsigned char** png_bytepp
    ctypedef char* png_charp
    ctypedef char* png_const_charp
    ctypedef void* png_voidp
    ctypedef png_struct* png_structp
    ctypedef png_struct* png_structrp
    ctypedef png_struct* png_const_structp
    ctypedef png_struct* png_const_structrp
    ctypedef png_struct** png_structpp
    ctypedef png_info* png_infop
    ctypedef png_info* png_inforp
    ctypedef png_info* png_const_infop
    ctypedef png_info* png_const_inforp
    ctypedef png_info** png_infopp
    ctypedef void(*png_error_ptr)(png_structp, png_const_charp)
    ctypedef void(*png_rw_ptr)(png_structp, png_bytep, size_t)
    ctypedef void(*png_flush_ptr)(png_structp)
    ctypedef void(*png_read_status_ptr)(png_structp, png_uint_32, int)
    ctypedef void(*png_write_status_ptr)(png_structp, png_uint_32, int)

    int png_sig_cmp(png_const_bytep sig,
                    size_t start,
                    size_t num_to_check) nogil

    void png_set_sig_bytes(png_structrp png_ptr, int num_bytes) nogil

    png_uint_32 png_get_IHDR(png_const_structrp png_ptr,
                             png_const_inforp info_ptr,
                             png_uint_32 *width,
                             png_uint_32 *height,
                             int *bit_depth,
                             int *color_type,
                             int *interlace_method,
                             int *compression_method,
                             int *filter_method) nogil

    void png_set_IHDR(png_const_structrp png_ptr,
                      png_inforp info_ptr,
                      png_uint_32 width,
                      png_uint_32 height,
                      int bit_depth,
                      int color_type,
                      int interlace_method,
                      int compression_method,
                      int filter_method) nogil

    void png_read_row(png_structrp png_ptr,
                      png_bytep row,
                      png_bytep display_row) nogil

    void png_write_row(png_structrp png_ptr, png_const_bytep row) nogil
    void png_read_image(png_structrp png_ptr, png_bytepp image) nogil
    void png_write_image(png_structrp png_ptr, png_bytepp image) nogil
    png_infop png_create_info_struct(const png_const_structrp png_ptr) nogil

    png_structp png_create_write_struct(png_const_charp user_png_ver,
                                        png_voidp error_ptr,
                                        png_error_ptr error_fn,
                                        png_error_ptr warn_fn) nogil

    png_structp png_create_read_struct(png_const_charp user_png_ver,
                                       png_voidp error_ptr,
                                       png_error_ptr error_fn,
                                       png_error_ptr warn_fn) nogil

    void png_destroy_write_struct(png_structpp png_ptr_ptr,
                                  png_infopp info_ptr_ptr) nogil

    void png_destroy_read_struct(png_structpp png_ptr_ptr,
                                 png_infopp info_ptr_ptr,
                                 png_infopp end_info_ptr_ptr) nogil

    void png_set_write_fn(png_structrp png_ptr,
                          png_voidp io_ptr,
                          png_rw_ptr write_data_fn,
                          png_flush_ptr output_flush_fn) nogil

    void png_set_read_fn(png_structrp png_ptr,
                         png_voidp io_ptr,
                         png_rw_ptr read_data_fn) nogil

    png_voidp png_get_io_ptr(png_const_structrp png_ptr) nogil
    void png_set_palette_to_rgb(png_structrp png_ptr) nogil
    void png_set_expand_gray_1_2_4_to_8(png_structrp png_ptr) nogil
    void png_read_info(png_structrp png_ptr, png_inforp info_ptr) nogil
    void png_write_info(png_structrp png_ptr, png_const_inforp info_ptr) nogil
    void png_write_end(png_structrp png_ptr, png_inforp info_ptr) nogil
    void png_read_update_info(png_structrp png_ptr, png_inforp info_ptr) nogil
    void png_set_expand_16(png_structrp png_ptr) nogil
    void png_set_swap(png_structrp png_ptr) nogil
    void png_set_compression_level(png_structrp png_ptr, int level) nogil


cdef void png_error_callback(png_structp png_ptr,
                             png_const_charp msg) with gil:
    raise RuntimeError(msg.decode('utf8').strip())


cdef void png_warn_callback(png_structp png_ptr,
                            png_const_charp msg) with gil:
    import warnings
    warnings.warn('PNG %s' % msg.decode('utf8').strip())


ctypedef struct png_memstream_t:
    png_bytep data
    png_size_t size
    png_size_t offset


cdef void png_read_data_fn(png_structp png_ptr,
                           png_bytep dst,
                           png_size_t size) nogil:
    """PNG read callback function."""
    cdef png_memstream_t* memstream = <png_memstream_t*>png_get_io_ptr(png_ptr)
    if memstream == NULL:
        return
    if memstream.offset >= memstream.size:
        return
    if size > memstream.size - memstream.offset:
        # size = memstream.size - memstream.offset
        with gil:
            raise RuntimeError('PNG input stream too small %i' % memstream.size)
    memcpy(dst, &(memstream.data[memstream.offset]), size)
    memstream.offset += size


cdef void png_write_data_fn(png_structp png_ptr,
                            png_bytep src,
                            png_size_t size) nogil:
    """PNG write callback function."""
    cdef png_memstream_t* memstream = <png_memstream_t*>png_get_io_ptr(png_ptr)
    if memstream == NULL:
        return
    if memstream.offset >= memstream.size:
        return
    if size > memstream.size - memstream.offset:
        # size = memstream.size - memstream.offset
        with gil:
            raise RuntimeError(
                'PNG output stream too small %i' % memstream.size)
    memcpy(&(memstream.data[memstream.offset]), src, size)
    memstream.offset += size


cdef void png_output_flush_fn(png_structp png_ptr) nogil:
    """PNG flush callback function."""
    pass


cdef ssize_t png_size_max(ssize_t size):
    """Return upper bound size of PNG stream from uncompressed image size."""
    size += ((size + 7) >> 3) + ((size + 63) >> 6) + 11  # ZLIB compression
    size += 12 * (size / PNG_ZBUF_SIZE + 1)  # IDAT
    size += 8 + 25 + 16 + 44 + 12  # sig IHDR gAMA cHRM IEND
    return size


def png_decode(data, out=None):
    """Decode PNG image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size * src.itemsize
        int samples = 0
        png_memstream_t memstream
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_uint_32 ret
        png_uint_32 width = 0
        png_uint_32 height = 0
        png_uint_32 row
        int bit_depth = 0
        int color_type = -1
        png_bytepp image = NULL  # row pointers
        png_bytep rowptr
        ssize_t rowstride

    if data is out:
        raise ValueError('cannot decode in-place')

    if png_sig_cmp(&src[0], 0, 8) != 0:
        raise ValueError('not a PNG image')

    try:
        with nogil:
            memstream.data = <png_bytep>&src[0]
            memstream.size = srcsize
            memstream.offset = 8

            png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,
                                             png_error_callback,
                                             png_warn_callback)
            if png_ptr == NULL:
                with gil:
                    raise RuntimeError('png_create_read_struct returned NULL')

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                with gil:
                    raise RuntimeError('png_create_info_struct returned NULL')

            png_set_read_fn(png_ptr, <png_voidp>&memstream, png_read_data_fn)
            png_set_sig_bytes(png_ptr, 8)
            png_read_info(png_ptr, info_ptr)
            ret = png_get_IHDR(png_ptr, info_ptr,
                               &width, &height, &bit_depth, &color_type,
                               NULL, NULL, NULL)
            if ret != 1:
                with gil:
                    raise RuntimeError('png_get_IHDR returned %i' % ret)

            if bit_depth > 8:
                png_set_swap(png_ptr)

            if color_type == PNG_COLOR_TYPE_GRAY:
                samples = 1
                if bit_depth < 8:
                    png_set_expand_gray_1_2_4_to_8(png_ptr)
            elif color_type == PNG_COLOR_TYPE_GRAY_ALPHA:
                samples = 2
            elif color_type == PNG_COLOR_TYPE_RGB:
                samples = 3
            elif color_type == PNG_COLOR_TYPE_PALETTE:
                samples = 3
                png_set_palette_to_rgb(png_ptr)
            elif color_type == PNG_COLOR_TYPE_RGB_ALPHA:
                samples = 4
            else:
                with gil:
                    raise ValueError(
                        'PNG color type not supported % i' % color_type)

        dtype = numpy.dtype('u%i' % (1 if bit_depth//8 < 2 else 2))
        if samples > 1:
            shape = int(height), int(width), int(samples)
            strides = None, shape[2] * dtype.itemsize, dtype.itemsize
        else:
            shape = int(height), int(width)
            strides = None, dtype.itemsize

        out = _create_array(out, shape, dtype, strides)
        dst = out
        rowptr = <png_bytep>&dst.data[0]
        rowstride = dst.strides[0]

        with nogil:
            image = <png_bytepp>malloc(sizeof(png_bytep) * height)
            if image == NULL:
                with gil:
                    raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                image[row] = <png_bytep>rowptr
                rowptr += rowstride
            png_read_update_info(png_ptr, info_ptr)
            png_read_image(png_ptr, image)

    finally:
        if image != NULL:
            free(image)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_read_struct(&png_ptr, &info_ptr, NULL)
        elif png_ptr != NULL:
            png_destroy_read_struct(&png_ptr, NULL, NULL)

    return out


def png_encode(data, level=None, out=None):
    """Encode numpy array to PNG image.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        ssize_t rowstride = src.strides[0]
        png_bytep rowptr = <png_bytep>&src.data[0]
        int color_type
        int bit_depth = src.itemsize * 8
        int samples = <int>src.shape[2] if src.ndim == 3 else 1
        int compresslevel = _default_level(level, 5, 0, 10)
        png_memstream_t memstream
        png_structp png_ptr = NULL
        png_infop info_ptr = NULL
        png_bytepp image = NULL  # row pointers
        png_uint_32 width = <png_uint_32>src.shape[1]
        png_uint_32 height = <png_uint_32>src.shape[0]
        png_uint_32 row

    if not (data.dtype in ('uint8', 'uint16')
            and data.ndim in (2, 3)
            and data.shape[0] < 2**31-1
            and data.shape[1] < 2**31-1
            and samples <= 4
            and data.strides[data.ndim-1] == data.itemsize
            and (data.ndim == 2 or data.strides[1] == samples*data.itemsize)):
        raise ValueError('invalid input shape, strides, or dtype')

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = png_size_max(srcsize)
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size * dst.itemsize

    try:
        with nogil:

            memstream.data = &dst[0]
            memstream.size = <png_size_t>dstsize
            memstream.offset = 0

            if samples == 1:
                color_type = PNG_COLOR_TYPE_GRAY
            elif samples == 2:
                color_type = PNG_COLOR_TYPE_GRAY_ALPHA
            elif samples == 3:
                color_type = PNG_COLOR_TYPE_RGB
            elif samples == 4:
                color_type = PNG_COLOR_TYPE_RGB_ALPHA
            else:
                with gil:
                    raise ValueError('PNG color type not supported')

            png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL,
                                              png_error_callback,
                                              png_warn_callback)
            if png_ptr == NULL:
                with gil:
                    raise RuntimeError('png_create_write_struct returned NULL')

            png_set_write_fn(png_ptr, <png_voidp>&memstream,
                             png_write_data_fn, png_output_flush_fn)

            info_ptr = png_create_info_struct(png_ptr)
            if info_ptr == NULL:
                with gil:
                    raise RuntimeError('png_create_info_struct returned NULL')

            png_set_IHDR(png_ptr, info_ptr,
                         width, height, bit_depth, color_type,
                         PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                         PNG_FILTER_TYPE_DEFAULT)

            png_write_info(png_ptr, info_ptr)
            png_set_compression_level(png_ptr, compresslevel)
            if bit_depth > 8:
                png_set_swap(png_ptr)

            image = <png_bytepp>malloc(sizeof(png_bytep) * height)
            if image == NULL:
                with gil:
                    raise MemoryError('failed to allocate row pointers')
            for row in range(height):
                image[row] = rowptr
                rowptr += rowstride

            png_write_image(png_ptr, image)
            png_write_end(png_ptr, info_ptr)

    finally:
        if image != NULL:
            free(image)
        if png_ptr != NULL and info_ptr != NULL:
            png_destroy_write_struct(&png_ptr, &info_ptr)
        elif png_ptr != NULL:
            png_destroy_write_struct(&png_ptr, NULL)

    if memstream.offset < <png_size_t>dstsize:
        if out_given:
            out = memoryview(out)[:memstream.offset]
        else:
            out = out[:memstream.offset]

    return out


# WebP ##################################################################

cdef extern from 'webp/decode.h':

    ctypedef enum VP8StatusCode:
        VP8_STATUS_OK = 0,
        VP8_STATUS_OUT_OF_MEMORY,
        VP8_STATUS_INVALID_PARAM,
        VP8_STATUS_BITSTREAM_ERROR,
        VP8_STATUS_UNSUPPORTED_FEATURE,
        VP8_STATUS_SUSPENDED,
        VP8_STATUS_USER_ABORT,
        VP8_STATUS_NOT_ENOUGH_DATA

    ctypedef struct WebPBitstreamFeatures:
        int width
        int height
        int has_alpha
        int has_animation
        int format
        uint32_t[5] pad

    int WebPGetDecoderVersion() nogil

    int WebPGetInfo(const uint8_t* data,
                    size_t data_size,
                    int* width,
                    int* height) nogil

    VP8StatusCode WebPGetFeatures(const uint8_t* data,
                                  size_t data_size,
                                  WebPBitstreamFeatures* features) nogil

    uint8_t* WebPDecodeRGBAInto(const uint8_t* data,
                                size_t data_size,
                                uint8_t* output_buffer,
                                size_t output_buffer_size,
                                int output_stride) nogil

    uint8_t* WebPDecodeRGBInto(const uint8_t* data,
                               size_t data_size,
                               uint8_t* output_buffer,
                               size_t output_buffer_size,
                               int output_stride) nogil

    uint8_t* WebPDecodeYUVInto(const uint8_t* data,
                               size_t data_size,
                               uint8_t* luma,
                               size_t luma_size,
                               int luma_stride,
                               uint8_t* u,
                               size_t u_size,
                               int u_stride,
                               uint8_t* v,
                               size_t v_size,
                               int v_stride) nogil

cdef extern from 'webp/encode.h':

    int WEBP_MAX_DIMENSION

    int WebPGetEncoderVersion() nogil
    void WebPFree(void* ptr) nogil

    size_t WebPEncodeRGB(const uint8_t* rgb,
                         int width,
                         int height,
                         int stride,
                         float quality_factor,
                         uint8_t** output) nogil

    size_t WebPEncodeRGBA(const uint8_t* rgba,
                          int width,
                          int height,
                          int stride,
                          float quality_factor,
                          uint8_t** output) nogil

    size_t WebPEncodeLosslessRGB(const uint8_t* rgb,
                                 int width,
                                 int height,
                                 int stride,
                                 uint8_t** output) nogil

    size_t WebPEncodeLosslessRGBA(const uint8_t* rgba,
                                  int width,
                                  int height,
                                  int stride,
                                  uint8_t** output) nogil


class WebpError(RuntimeError):
    """WebP Exceptions."""
    def __init__(self, func, err):
        msg = {
            VP8_STATUS_OK: 'VP8_STATUS_OK',
            VP8_STATUS_OUT_OF_MEMORY: 'VP8_STATUS_OUT_OF_MEMORY',
            VP8_STATUS_INVALID_PARAM: 'VP8_STATUS_INVALID_PARAM',
            VP8_STATUS_BITSTREAM_ERROR: 'VP8_STATUS_BITSTREAM_ERROR',
            VP8_STATUS_UNSUPPORTED_FEATURE: 'VP8_STATUS_UNSUPPORTED_FEATURE',
            VP8_STATUS_SUSPENDED: 'VP8_STATUS_SUSPENDED',
            VP8_STATUS_USER_ABORT: 'VP8_STATUS_USER_ABORT',
            VP8_STATUS_NOT_ENOUGH_DATA: 'VP8_STATUS_NOT_ENOUGH_DATA',
            }.get(err, 'unknown error % i' % err)
        msg = '%s returned %s' % (func, msg)
        RuntimeError.__init__(self, msg)


def webp_encode(data, level=None, out=None):
    """Encode numpy array to WebP image.

    """
    cdef:
        const uint8_t[:, :, :] src = data
        const uint8_t[::1] dst
        uint8_t* srcptr = &src[0, 0, 0]
        uint8_t* output
        ssize_t dstsize
        size_t ret
        int width, height, stride
        float quality_factor = _default_level(level, 75.0, -1.0, 100.0)
        int lossless = quality_factor < 0.0
        int rgba = data.shape[2] == 4

    if not (data.ndim == 3
            and data.shape[0] < WEBP_MAX_DIMENSION
            and data.shape[1] < WEBP_MAX_DIMENSION
            and data.shape[2] in (3, 4)
            and data.strides[2] == 1
            and data.strides[1] in (3, 4)
            and data.strides[0] >= data.strides[1] * data.strides[2]
            and data.dtype == 'uint8'):
        raise ValueError('invalid input shape, strides, or dtype')

    height, width = data.shape[:2]
    stride = data.strides[0]

    with nogil:
        if lossless:
            if rgba:
                ret = WebPEncodeLosslessRGBA(
                    srcptr, width, height, stride, &output)
            else:
                ret = WebPEncodeLosslessRGB(
                    srcptr, width, height, stride, &output)
        elif rgba:
            ret = WebPEncodeRGBA(
                srcptr, width, height, stride, quality_factor, &output)
        else:
            ret = WebPEncodeRGB(
                srcptr, width, height, stride, quality_factor, &output)

    if ret <= 0:
        raise RuntimeError('WebPEncode returned 0')

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, out_given, out_type = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = ret
        if out_type is bytes:
            out = PyBytes_FromStringAndSize(NULL, dstsize)
        else:
            out = PyByteArray_FromStringAndSize(NULL, dstsize)

    dst = out
    dstsize = dst.size
    if <size_t>dstsize < ret:
        raise ValueError('output too small')

    with nogil:
        memcpy(&dst[0], output, ret)
        WebPFree(<void*> output)

    if ret < <size_t>dstsize:
        out = memoryview(out)[:ret] if out_given else out[:ret]

    return out


def webp_decode(data, out=None):
    """Decode WebP image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t dstsize
        int output_stride
        size_t data_size
        WebPBitstreamFeatures features
        int ret
        uint8_t* pout

    if data is out:
        raise ValueError('cannot decode in-place')

    ret = <int>WebPGetFeatures(&src[0], <size_t>srcsize, &features)
    if ret != VP8_STATUS_OK:
        raise WebpError('WebPGetFeatures', ret)

    if features.has_alpha:
        shape = features.height, features.width, 4
    else:
        shape = features.height, features.width, 3

    out = _create_array(out, shape, numpy.uint8, strides=(None, shape[2], 1))
    dst = out
    dstsize = dst.shape[0] * dst.strides[0]
    output_stride = <int>dst.strides[0]

    with nogil:
        if features.has_alpha:
            pout = WebPDecodeRGBAInto(&src[0],
                                      <size_t> srcsize,
                                      <uint8_t*> dst.data,
                                      <size_t> dstsize,
                                      output_stride)
        else:
            pout = WebPDecodeRGBInto(&src[0],
                                     <size_t> srcsize,
                                     <uint8_t*> dst.data,
                                     <size_t> dstsize,
                                     output_stride)
    if pout == NULL:
        raise RuntimeError('WebPDecodeRGBAInto returned NULL')

    return out


# JPEG 8-bit ##################################################################

cdef extern from 'jpeglib.h':
    int JPEG_LIB_VERSION
    int LIBJPEG_TURBO_VERSION_NUMBER

    ctypedef void noreturn_t
    ctypedef int boolean
    ctypedef unsigned int JDIMENSION
    ctypedef unsigned char JSAMPLE
    ctypedef JSAMPLE* JSAMPROW
    ctypedef JSAMPROW* JSAMPARRAY

    ctypedef enum J_COLOR_SPACE:
        JCS_UNKNOWN,
        JCS_GRAYSCALE,
        JCS_RGB,
        JCS_YCbCr,
        JCS_CMYK,
        JCS_YCCK

    ctypedef enum J_DITHER_MODE:
        JDITHER_NONE,
        JDITHER_ORDERED,
        JDITHER_FS

    ctypedef enum J_DCT_METHOD:
        JDCT_ISLOW,
        JDCT_IFAST,
        JDCT_FLOAT

    struct jpeg_source_mgr:
        pass

    struct jpeg_error_mgr:
        int msg_code
        char** jpeg_message_table
        noreturn_t error_exit(jpeg_common_struct*)
        void output_message(jpeg_common_struct*)

    struct jpeg_common_struct:
        jpeg_error_mgr* err

    struct jpeg_decompress_struct:
        jpeg_error_mgr* err
        void* client_data
        jpeg_source_mgr* src
        JDIMENSION image_width
        JDIMENSION image_height
        JDIMENSION output_width
        JDIMENSION output_height
        JDIMENSION output_scanline
        J_COLOR_SPACE jpeg_color_space
        J_COLOR_SPACE out_color_space
        J_DCT_METHOD dct_method
        J_DITHER_MODE dither_mode
        boolean buffered_image
        boolean raw_data_out
        boolean do_fancy_upsampling
        boolean do_block_smoothing
        boolean quantize_colors
        boolean two_pass_quantize
        unsigned int scale_num
        unsigned int scale_denom
        int num_components
        int out_color_components
        int output_components
        int rec_outbuf_height
        int desired_number_of_colors
        int actual_number_of_colors
        int data_precision
        double output_gamma

    jpeg_error_mgr* jpeg_std_error(jpeg_error_mgr*) nogil
    void jpeg_create_decompress(jpeg_decompress_struct*) nogil
    void jpeg_destroy_decompress(jpeg_decompress_struct*) nogil
    int jpeg_read_header(jpeg_decompress_struct*, boolean) nogil
    boolean jpeg_start_decompress(jpeg_decompress_struct*) nogil
    boolean jpeg_finish_decompress(jpeg_decompress_struct*) nogil

    JDIMENSION jpeg_read_scanlines(jpeg_decompress_struct*,
                                   JSAMPARRAY,
                                   JDIMENSION) nogil

    void jpeg_mem_src(jpeg_decompress_struct*,
                      unsigned char*,
                      unsigned long) nogil


ctypedef struct my_error_mgr:
    jpeg_error_mgr pub
    jmp_buf setjmp_buffer


cdef void my_error_exit(jpeg_common_struct* cinfo):
    cdef my_error_mgr* error = <my_error_mgr*> deref(cinfo).err
    longjmp(deref(error).setjmp_buffer, 1)


cdef void my_output_message(jpeg_common_struct* cinfo):
    pass


class Jpeg8Error(RuntimeError):
    """JPEG Exceptions."""
    pass


def jpeg8_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: JPEG 8-bit encoding
    raise NotImplementedError('jpeg8_encode')


def jpeg8_decode(data, tables=None, colorspace=None, outcolorspace=None,
                 out=None):
    """Decode JPEG 8-bit image to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        const uint8_t[::1] tables_
        ssize_t dstsize
        int numlines
        my_error_mgr err
        jpeg_decompress_struct cinfo
        JSAMPROW samples

    if data is out:
        raise ValueError('cannot decode in-place')

    cinfo.err = jpeg_std_error(&err.pub)
    err.pub.error_exit = my_error_exit
    err.pub.output_message = my_output_message
    if setjmp(err.setjmp_buffer):
        jpeg_destroy_decompress(&cinfo)
        raise Jpeg8Error(
            err.pub.jpeg_message_table[err.pub.msg_code].decode('utf-8'))

    jpeg_create_decompress(&cinfo)
    cinfo.do_fancy_upsampling = True

    if tables is not None:
        tables_ = tables
        jpeg_mem_src(&cinfo, &tables_[0], tables_.size)
        jpeg_read_header(&cinfo, 0)

    jpeg_mem_src(&cinfo, &src[0], src.size)
    jpeg_read_header(&cinfo, 1)

    if colorspace is None:
        pass
    elif colorspace == 'RGB':
        cinfo.jpeg_color_space = JCS_RGB
        cinfo.out_color_space = JCS_RGB
    elif colorspace == 'YCBCR':
        cinfo.jpeg_color_space = JCS_YCbCr
        cinfo.out_color_space = JCS_YCbCr
    elif colorspace in ('MINISBLACK', 'MINISWHITE', 'GRAYSCALE'):
        cinfo.jpeg_color_space = JCS_GRAYSCALE
        cinfo.out_color_space = JCS_GRAYSCALE
    elif colorspace == 'CMYK':
        cinfo.jpeg_color_space = JCS_CMYK
        cinfo.out_color_space = JCS_CMYK
    elif colorspace == 'YCCK':
        cinfo.jpeg_color_space = JCS_YCCK
        cinfo.out_color_space = JCS_YCCK

    if outcolorspace is None:
        pass
    elif outcolorspace == 'RGB':
        cinfo.out_color_space = JCS_RGB
    elif outcolorspace == 'YCBCR':
        cinfo.out_color_space = JCS_YCbCr
    elif outcolorspace in ('MINISBLACK', 'MINISWHITE', 'GRAYSCALE'):
        cinfo.out_color_space = JCS_GRAYSCALE
    elif outcolorspace == 'CMYK':
        cinfo.out_color_space = JCS_CMYK
    elif outcolorspace == 'YCCK':
        cinfo.out_color_space = JCS_YCCK

    jpeg_start_decompress(&cinfo)

    shape = cinfo.output_height, cinfo.output_width
    if cinfo.output_components > 1:
        shape += cinfo.output_components,

    out = _create_array(out, shape, numpy.uint8)
    dst = out
    dstsize = dst.size * dst.itemsize

    with nogil:
        memset(<void *>dst.data, 0, dstsize)
        samples = <JSAMPROW>dst.data
        while cinfo.output_scanline < cinfo.output_height:
            numlines = jpeg_read_scanlines(&cinfo, <JSAMPARRAY> &samples, 1)
            samples += numlines * cinfo.output_width * cinfo.output_components
        jpeg_finish_decompress(&cinfo)
        jpeg_destroy_decompress(&cinfo)

    return out


# JPEG 12-bit #################################################################

# JPEG 12-bit codecs are implemented in a separate extension module
# due to header and link conflicts with JPEG 8-bit.

try:
    from ._jpeg12 import jpeg12_decode, jpeg12_encode
except ImportError:

    def jpeg12_decode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpeg12_decode')

    def jpeg12_encode(*args, **kwargs):
        """Not implemented."""
        raise NotImplementedError('jpeg12_encode')


# JPEG Wrapper ################################################################

def jpeg_decode(data, bitspersample, tables=None, colorspace=None,
                outcolorspace=None, out=None):
    """Decode JPEG.

    """
    if bitspersample == 8:
        return jpeg8_decode(data, tables, colorspace, outcolorspace, out)
    if bitspersample == 12:
        return jpeg12_decode(data, tables, colorspace, outcolorspace, out)
    raise NotImplementedError('JPEG %i-bit not supported' % bitspersample)


def jpeg_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: JPEG encoding
    raise NotImplementedError('jpeg_encode')


# JPEG 2000 ###################################################################

cdef extern from 'openjpeg.h':
    int OPJ_FALSE = 0
    int OPJ_TRUE = 1

    ctypedef int OPJ_BOOL
    ctypedef char OPJ_CHAR
    ctypedef float OPJ_FLOAT32
    ctypedef double OPJ_FLOAT64
    ctypedef unsigned char OPJ_BYTE
    ctypedef int8_t OPJ_INT8
    ctypedef uint8_t OPJ_UINT8
    ctypedef int16_t OPJ_INT16
    ctypedef uint16_t OPJ_UINT16
    ctypedef int32_t OPJ_INT32
    ctypedef uint32_t OPJ_UINT32
    ctypedef int64_t OPJ_INT64
    ctypedef uint64_t OPJ_UINT64
    ctypedef int64_t OPJ_OFF_T
    ctypedef size_t OPJ_SIZE_T

    ctypedef enum OPJ_CODEC_FORMAT:
        OPJ_CODEC_UNKNOWN = -1
        OPJ_CODEC_J2K = 0
        OPJ_CODEC_JPT = 1
        OPJ_CODEC_JP2 = 2
        OPJ_CODEC_JPP = 3
        OPJ_CODEC_JPX = 4

    ctypedef enum OPJ_COLOR_SPACE:
        OPJ_CLRSPC_UNKNOWN = -1
        OPJ_CLRSPC_UNSPECIFIED = 0
        OPJ_CLRSPC_SRGB = 1
        OPJ_CLRSPC_GRAY = 2
        OPJ_CLRSPC_SYCC = 3
        OPJ_CLRSPC_EYCC = 4
        OPJ_CLRSPC_CMYK = 5

    ctypedef struct opj_codec_t:
        pass

    ctypedef struct opj_stream_t:
        pass

    ctypedef struct opj_image_cmptparm_t:
        pass

    ctypedef struct opj_dparameters_t:
        OPJ_UINT32 cp_reduce
        OPJ_UINT32 cp_layer
        # char infile[OPJ_PATH_LEN]
        # char outfile[OPJ_PATH_LEN]
        int decod_format
        int cod_format
        OPJ_UINT32 DA_x0
        OPJ_UINT32 DA_x1
        OPJ_UINT32 DA_y0
        OPJ_UINT32 DA_y1
        OPJ_BOOL m_verbose
        OPJ_UINT32 tile_index
        OPJ_UINT32 nb_tile_to_decode
        OPJ_BOOL jpwl_correct
        int jpwl_exp_comps
        int jpwl_max_tiles
        unsigned int flags

    ctypedef struct opj_image_comp_t:
        OPJ_UINT32 dx
        OPJ_UINT32 dy
        OPJ_UINT32 w
        OPJ_UINT32 h
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 prec
        OPJ_UINT32 bpp
        OPJ_UINT32 sgnd
        OPJ_UINT32 resno_decoded
        OPJ_UINT32 factor
        OPJ_INT32* data
        OPJ_UINT16 alpha

    ctypedef struct opj_image_t:
        OPJ_UINT32 x0
        OPJ_UINT32 y0
        OPJ_UINT32 x1
        OPJ_UINT32 y1
        OPJ_UINT32 numcomps
        OPJ_COLOR_SPACE color_space
        opj_image_comp_t* comps
        OPJ_BYTE* icc_profile_buf
        OPJ_UINT32 icc_profile_len

    ctypedef OPJ_SIZE_T(* opj_stream_read_fn)(void*, OPJ_SIZE_T, void*)
    ctypedef OPJ_SIZE_T(* opj_stream_write_fn)(void*, OPJ_SIZE_T, void*)
    ctypedef OPJ_OFF_T(* opj_stream_skip_fn)(OPJ_OFF_T, void*)
    ctypedef OPJ_BOOL(* opj_stream_seek_fn)(OPJ_OFF_T, void*)
    ctypedef void(* opj_stream_free_user_data_fn)(void*)
    ctypedef void(*opj_msg_callback)(const char *msg, void *client_data)

    opj_stream_t* opj_stream_default_create(OPJ_BOOL p_is_input) nogil
    opj_codec_t* opj_create_decompress(OPJ_CODEC_FORMAT format) nogil
    void opj_destroy_codec(opj_codec_t* p_codec) nogil
    void opj_set_default_decoder_parameters(opj_dparameters_t *params) nogil
    void opj_image_destroy(opj_image_t* image) nogil
    void* opj_image_data_alloc(OPJ_SIZE_T size) nogil
    void opj_image_data_free(void* ptr) nogil
    void opj_stream_destroy(opj_stream_t* p_stream) nogil
    void color_sycc_to_rgb(opj_image_t* img) nogil
    void color_apply_icc_profile(opj_image_t* image) nogil
    void color_cielab_to_rgb(opj_image_t* image) nogil
    void color_cmyk_to_rgb(opj_image_t* image) nogil
    void color_esycc_to_rgb(opj_image_t* image) nogil
    char* opj_version() nogil

    OPJ_BOOL opj_end_decompress(opj_codec_t *p_codec,
                                opj_stream_t *p_stream) nogil

    OPJ_BOOL opj_setup_decoder(opj_codec_t *p_codec,
                               opj_dparameters_t *params) nogil

    OPJ_BOOL opj_codec_set_threads(opj_codec_t *p_codec,
                                   int num_threads) nogil

    OPJ_BOOL opj_read_header(opj_stream_t *p_stream,
                             opj_codec_t *p_codec,
                             opj_image_t **p_image) nogil

    OPJ_BOOL opj_set_decode_area(opj_codec_t *p_codec,
                                 opj_image_t* p_image,
                                 OPJ_INT32 p_start_x,
                                 OPJ_INT32 p_start_y,
                                 OPJ_INT32 p_end_x,
                                 OPJ_INT32 p_end_y) nogil

    OPJ_BOOL opj_set_info_handler(opj_codec_t * p_codec,
                                  opj_msg_callback p_callback,
                                  void * p_user_data) nogil

    OPJ_BOOL opj_set_warning_handler(opj_codec_t * p_codec,
                                     opj_msg_callback p_callback,
                                     void * p_user_data) nogil

    OPJ_BOOL opj_set_error_handler(opj_codec_t * p_codec,
                                   opj_msg_callback p_callback,
                                   void * p_user_data) nogil

    OPJ_BOOL opj_decode(opj_codec_t *p_decompressor,
                        opj_stream_t *p_stream,
                        opj_image_t *p_image) nogil

    opj_image_t* opj_image_create(OPJ_UINT32 numcmpts,
                                  opj_image_cmptparm_t* cmptparms,
                                  OPJ_COLOR_SPACE clrspc) nogil

    void opj_stream_set_read_function(opj_stream_t* p_stream,
                                      opj_stream_read_fn p_func) nogil

    void opj_stream_set_write_function(opj_stream_t* p_stream,
                                       opj_stream_write_fn p_func) nogil

    void opj_stream_set_seek_function(opj_stream_t* p_stream,
                                      opj_stream_seek_fn p_func) nogil

    void opj_stream_set_skip_function(opj_stream_t* p_stream,
                                      opj_stream_skip_fn p_func) nogil

    void opj_stream_set_user_data(opj_stream_t* p_stream,
                                  void* p_data,
                                  opj_stream_free_user_data_fn p_func) nogil

    void opj_stream_set_user_data_length(opj_stream_t* p_stream,
                                         OPJ_UINT64 data_length) nogil


ctypedef struct opj_memstream_t:
    OPJ_UINT8* data
    OPJ_UINT64 size
    OPJ_UINT64 offset


cdef OPJ_SIZE_T opj_mem_read(void* dst, OPJ_SIZE_T size, void* data) nogil:
    """opj_stream_set_read_function."""
    cdef:
        opj_memstream_t* memstream = <opj_memstream_t*>data
        OPJ_SIZE_T count = size
    if memstream.offset >= memstream.size:
        return <OPJ_SIZE_T>-1
    if size > memstream.size - memstream.offset:
        count = memstream.size - memstream.offset
    memcpy(dst, &(memstream.data[memstream.offset]), count)
    memstream.offset += count
    return count


cdef OPJ_SIZE_T opj_mem_write(void* dst, OPJ_SIZE_T size, void* data) nogil:
    """opj_stream_set_write_function."""
    cdef:
        opj_memstream_t* memstream = <opj_memstream_t*>data
        OPJ_SIZE_T count = size
    if memstream.offset >= memstream.size:
        return <OPJ_SIZE_T>-1
    if size > memstream.size - memstream.offset:
        count = memstream.size - memstream.offset
    memcpy(&(memstream.data[memstream.offset]), dst, count)
    memstream.offset += count
    return count


cdef OPJ_BOOL opj_mem_seek(OPJ_OFF_T size, void* data) nogil:
    """opj_stream_set_seek_function."""
    cdef opj_memstream_t* memstream = <opj_memstream_t*>data
    if size < 0 or size >= <OPJ_OFF_T>memstream.size:
        return OPJ_FALSE
    memstream.offset = <OPJ_SIZE_T>size
    return OPJ_TRUE


cdef OPJ_OFF_T opj_mem_skip(OPJ_OFF_T size, void* data) nogil:
    """opj_stream_set_skip_function."""
    cdef:
        opj_memstream_t* memstream = <opj_memstream_t*>data
        OPJ_SIZE_T count
    if size < 0:
        return -1
    count = <OPJ_SIZE_T>size
    if count > memstream.size - memstream.offset:
       count = memstream.size - memstream.offset
    memstream.offset += count
    return count


cdef void opj_mem_nop(void* data) nogil:
    """opj_stream_set_user_data."""
    pass


cdef opj_stream_t* opj_memstream_create(opj_memstream_t* memstream,
                                        OPJ_BOOL isinput) nogil:
    """Return OPJ stream using memory as input or output."""

    cdef opj_stream_t* stream = opj_stream_default_create(isinput)
    if stream == NULL:
        return NULL
    if isinput:
        opj_stream_set_read_function(stream, <opj_stream_read_fn>opj_mem_read)
    else:
        opj_stream_set_write_function(stream,
                                      <opj_stream_write_fn>opj_mem_write)
    opj_stream_set_seek_function(stream, <opj_stream_seek_fn>opj_mem_seek)
    opj_stream_set_skip_function(stream, <opj_stream_skip_fn>opj_mem_skip)
    opj_stream_set_user_data(stream, memstream,
                             <opj_stream_free_user_data_fn>opj_mem_nop)
    opj_stream_set_user_data_length(stream, memstream.size)
    return stream


class J2KError(RuntimeError):
    """OpenJPEG Exceptions."""
    def __init__(self, msg):
        RuntimeError.__init__(self, 'J2K %s' % msg)


cdef void j2k_error_callback(char* msg, void* client_data) with gil:
    raise J2KError(msg.decode('utf8').strip())


cdef void j2k_warning_callback(char* msg, void* client_data) with gil:
    import warnings
    warnings.warn('J2K %s' % msg.decode('utf8').strip())


cdef void j2k_info_callback(char* msg, void* client_data) with gil:
    print('J2K %s' % msg.decode('utf8').strip())


def j2k_encode(*args, **kwargs):
    """Not implemented."""
    # TODO: JPEG 2000 encoding
    raise NotImplementedError('j2k_encode')


def j2k_decode(data, verbose=0, out=None):
    """Decode JPEG 2000 J2K image to numpy array."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        int32_t* band
        uint32_t* u4
        uint16_t* u2
        uint8_t* u1
        int32_t* i4
        int16_t* i2
        int8_t* i1
        ssize_t dstsize
        ssize_t itemsize
        opj_memstream_t memstream
        opj_codec_t* codec = NULL
        opj_image_t* image = NULL
        opj_stream_t* stream = NULL
        opj_image_comp_t* comp = NULL
        opj_dparameters_t parameters
        OPJ_BOOL ret
        OPJ_CODEC_FORMAT codecformat
        OPJ_UINT32 signed, prec, width, height, samples
        ssize_t i, j
        int verbosity = verbose

    if data is out:
        raise ValueError('cannot decode in-place')

    signature = bytearray(src[:12])
    if signature[:4] == b'\xff\x4f\xff\x51':
        codecformat = OPJ_CODEC_J2K
    elif (signature == b'\x00\x00\x00\x0c\x6a\x50\x20\x20\x0d\x0a\x87\x0a' or
          signature[:4] == b'\x0d\x0a\x87\x0a'):
        codecformat = OPJ_CODEC_JP2
    else:
        raise J2KError('not a J2K or JP2 data stream')

    try:
        memstream.data = <OPJ_UINT8*>&src[0]
        memstream.size = src.size
        memstream.offset = 0

        with nogil:
            stream = opj_memstream_create(&memstream, OPJ_TRUE)
            if stream == NULL:
                with gil:
                    raise J2KError('opj_memstream_create failed')

            codec = opj_create_decompress(codecformat)
            if codec == NULL:
                with gil:
                    raise J2KError('opj_create_decompress failed')

            if verbosity > 0:
                opj_set_error_handler(
                    codec, <opj_msg_callback>j2k_error_callback, NULL)
                if verbosity > 1:
                    opj_set_warning_handler(
                        codec, <opj_msg_callback>j2k_warning_callback, NULL)
                if verbosity > 2:
                    opj_set_info_handler(
                        codec, <opj_msg_callback>j2k_info_callback, NULL)

            opj_set_default_decoder_parameters(&parameters)

            ret = opj_setup_decoder(codec, &parameters)
            if ret == OPJ_FALSE:
                with gil:
                    raise J2KError('opj_setup_decoder failed')

            ret = opj_read_header(stream, codec, &image)
            if ret == OPJ_FALSE:
                with gil:
                    raise J2KError('opj_read_header failed')

            ret = opj_set_decode_area(codec, image,
                                      <OPJ_INT32>parameters.DA_x0,
                                      <OPJ_INT32>parameters.DA_y0,
                                      <OPJ_INT32>parameters.DA_x1,
                                      <OPJ_INT32>parameters.DA_y1)
            if ret == OPJ_FALSE:
                with gil:
                    raise J2KError('opj_set_decode_area failed')

            # with nogil:
            ret = opj_decode(codec, stream, image)
            if ret != OPJ_FALSE:
                ret = opj_end_decompress(codec, stream)
            if ret == OPJ_FALSE:
                with gil:
                    raise J2KError('opj_decode or opj_end_decompress failed')

            # TODO: handle subsampling
            # TODO: color space functions are not part of openjp2
            #
            # if (image.color_space != OPJ_CLRSPC_SYCC
            #        and image.numcomps == 3
            #        and image.comps[0].dx == image.comps[0].dy
            #        and image.comps[1].dx != 1):
            #     image.color_space = OPJ_CLRSPC_SYCC
            # elif image.numcomps <= 2:
            #     image.color_space = OPJ_CLRSPC_GRAY
            # if image.color_space == OPJ_CLRSPC_SYCC:
            #     color_sycc_to_rgb(image)
            # if image.icc_profile_buf:
            #     color_apply_icc_profile(image)
            #     free(image.icc_profile_buf)
            #     image.icc_profile_buf = NULL
            #     image.icc_profile_len = 0

            comp = &image.comps[0]
            signed = comp.sgnd
            prec = comp.prec
            height = comp.h * comp.dy
            width = comp.w * comp.dx
            samples = image.numcomps
            itemsize = (prec + 7) // 8

            for i in range(samples):
                comp = &image.comps[i]
                if comp.sgnd != signed or comp.prec != prec:
                    with gil:
                        raise J2KError('components dtype mismatch')
                if comp.w != width or comp.h != height:
                    with gil:
                        raise J2KError('subsampling not supported')
            if itemsize == 3:
                itemsize = 4
            elif itemsize < 1 or itemsize > 4:
                with gil:
                    raise J2KError('unsupported itemsize %i' % int(itemsize))

        dtype = ('i' if signed else 'u') + ('%i' % itemsize)
        if samples > 1:
            shape = int(height), int(width), int(samples)
        else:
            shape = int(height), int(width)
        out = _create_array(out, shape, dtype)
        dst = out
        dstsize = dst.size * itemsize

        with nogil:
            # memset(<void *>dst.data, 0, dstsize)
            if itemsize == 1:
                if signed:
                    for i in range(samples):
                        i1 = <int8_t*>dst.data + i
                        band = <int32_t*>image.comps[i].data
                        for j in range(height * width):
                            i1[j * samples] = <int8_t>band[j]
                else:
                    for i in range(samples):
                        u1 = <uint8_t*>dst.data + i
                        band = <int32_t*>image.comps[i].data
                        for j in range(height * width):
                            u1[j * samples] = <uint8_t>band[j]
            elif itemsize == 2:
                if signed:
                    for i in range(samples):
                        i2 = <int16_t*>dst.data + i
                        band = <int32_t*>image.comps[i].data
                        for j in range(height * width):
                            i2[j * samples] = <int16_t>band[j]
                else:
                    for i in range(samples):
                        u2 = <uint16_t*>dst.data + i
                        band = <int32_t*>image.comps[i].data
                        for j in range(height * width):
                            u2[j * samples] = <uint16_t>band[j]
            elif itemsize == 4:
                if signed:
                    for i in range(samples):
                        i4 = <int32_t*>dst.data + i
                        band = <int32_t*>image.comps[i].data
                        for j in range(height * width):
                            i4[j * samples] = <int32_t>band[j]
                else:
                    for i in range(samples):
                        u4 = <uint32_t*>dst.data + i
                        band = <int32_t*>image.comps[i].data
                        for j in range(height * width):
                            u4[j * samples] = <uint32_t>band[j]

    finally:
        if stream != NULL:
            opj_stream_destroy(stream)
        if codec != NULL:
            opj_destroy_codec(codec)
        if image != NULL:
            opj_image_destroy(image)

    return out


###############################################################################

# TODO: Chroma Subsampling
# TODO: Integer Resize
# TODO: Dtype conversion
# TODO: Scale Offset
# TODO: CCITT and JBIG
# TODO: SZIP via libaec
# TODO: BMP
