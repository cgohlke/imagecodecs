# -*- coding: utf-8 -*-
# _zfp.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2019, Christoph Gohlke
# Copyright (c) 2019, The Regents of the University of California
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

"""ZFP codec for the imagecodecs package.

:Author:
  `Christoph Gohlke <https://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics. University of California, Irvine

:License: 3-clause BSD

:Version: 2019.2.2

"""

__version__ = '2019.2.2'

import numbers
import numpy

cimport cython
cimport numpy

from cpython.bytearray cimport PyByteArray_FromStringAndSize
from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
from libc.string cimport memset

numpy.import_array()


# ZFP #########################################################################

cdef extern from 'bitstream.h':

    ctypedef struct bitstream:
        pass

    bitstream* stream_open(void* buffer, size_t bytes) nogil
    void stream_close(bitstream* stream) nogil


cdef extern from 'zfp.h':

    ctypedef unsigned int uint

    char*  ZFP_VERSION_STRING

    int ZFP_HEADER_MAGIC
    int ZFP_HEADER_META
    int ZFP_HEADER_MODE
    int ZFP_HEADER_FULL
    int ZFP_MIN_BITS
    int ZFP_MAX_BITS
    int ZFP_MAX_PREC
    int ZFP_MIN_EXP

    ctypedef enum zfp_exec_policy:
        zfp_exec_serial
        zfp_exec_omp
        zfp_exec_cuda

    ctypedef enum zfp_type:
        zfp_type_none
        zfp_type_int32
        zfp_type_int64
        zfp_type_float
        zfp_type_double

    ctypedef enum zfp_mode:
        zfp_mode_null
        zfp_mode_expert
        zfp_mode_fixed_rate
        zfp_mode_fixed_precision
        zfp_mode_fixed_accuracy

    ctypedef struct zfp_exec_params_omp:
        uint threads
        uint chunk_size

    ctypedef union zfp_exec_params:
        zfp_exec_params_omp omp

    ctypedef struct zfp_execution:
        zfp_exec_policy policy
        zfp_exec_params params

    ctypedef struct zfp_stream:
        uint minbits
        uint maxbits
        uint maxprec
        int minexp
        bitstream* stream
        zfp_execution zexec 'exec'

    ctypedef struct zfp_field:
        zfp_type dtype 'type'
        uint nx, ny, nz, nw
        int sx, sy, sz, sw
        void* data

    zfp_stream* zfp_stream_open(zfp_stream*) nogil
    void zfp_stream_close(zfp_stream*) nogil
    void zfp_stream_rewind(zfp_stream*) nogil
    void zfp_stream_set_bit_stream(zfp_stream*, bitstream*) nogil
    size_t zfp_stream_flush(zfp_stream*) nogil
    size_t zfp_write_header(zfp_stream*, const zfp_field*, uint mask) nogil
    size_t zfp_read_header(zfp_stream*, zfp_field*, uint mask) nogil
    size_t zfp_stream_maximum_size(const zfp_stream*, const zfp_field*) nogil
    size_t zfp_stream_compressed_size(const zfp_stream*) nogil
    size_t zfp_compress(zfp_stream*, const zfp_field*) nogil
    size_t zfp_decompress(zfp_stream*, zfp_field*) nogil
    int zfp_stream_set_execution(zfp_stream*, zfp_exec_policy) nogil
    uint zfp_stream_set_precision(zfp_stream*, uint precision) nogil
    double zfp_stream_set_accuracy(zfp_stream*, double tolerance) nogil
    double zfp_stream_set_rate(zfp_stream*,
                               double rate,
                               zfp_type type,
                               uint dims,
                               int wra) nogil
    int zfp_stream_set_params(zfp_stream*,
                              uint minbits,
                              uint maxbits,
                              uint maxprec,
                              int minexp) nogil

    zfp_field* zfp_field_alloc() nogil
    void zfp_field_free(zfp_field*) nogil
    zfp_type zfp_field_set_type(zfp_field*, zfp_type type) nogil
    void zfp_field_set_pointer(zfp_field*, void* pointer) nogil
    void zfp_field_set_size_1d(zfp_field*, uint) nogil
    void zfp_field_set_size_2d(zfp_field*, uint, uint) nogil
    void zfp_field_set_size_3d(zfp_field*, uint, uint, uint) nogil
    void zfp_field_set_size_4d(zfp_field*, uint, uint, uint, uint) nogil
    void zfp_field_set_stride_1d(zfp_field*, int) nogil
    void zfp_field_set_stride_2d(zfp_field*, int, int) nogil
    void zfp_field_set_stride_3d(zfp_field*, int, int, int) nogil
    void zfp_field_set_stride_4d(zfp_field*, int, int, int, int) nogil


_ZFP_VERSION = ZFP_VERSION_STRING


def zfp_encode(data, level=None, mode=None, execution=None, header=True,
               out=None):
    """Compress numpy array to ZFP stream.

    """
    cdef:
        numpy.ndarray src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        size_t byteswritten
        ssize_t dstsize
        ssize_t srcsize = src.size * src.itemsize
        bitstream* stream = NULL
        zfp_stream* zfp = NULL
        zfp_field* field = NULL
        zfp_type ztype
        zfp_mode zmode
        zfp_exec_policy zexec
        uint ndim = data.ndim
        ssize_t itemsize = data.itemsize
        uint precision
        uint minbits, maxbits, maxprec, minexp
        uint nx, ny, nz, nw
        int sx, sy, sz, sw
        int ret
        double tolerance, rate
        int bheader = header

    if data is out:
        raise ValueError('cannot encode in-place')

    if data.dtype == numpy.int32:
        ztype = zfp_type_int32
    elif data.dtype == numpy.int64:
        ztype = zfp_type_int64
    elif data.dtype == numpy.float32:
        ztype = zfp_type_float
    elif data.dtype == numpy.float64:
        ztype = zfp_type_double
    else:
        raise ValueError('data type not supported by ZFP')

    if ndim == 1:
        nx = <uint>data.shape[0]
        sx = <int>(data.strides[0] / itemsize)
    elif ndim == 2:
        ny = <uint>data.shape[0]
        nx = <uint>data.shape[1]
        sy = <int>(data.strides[0] / itemsize)
        sx = <int>(data.strides[1] / itemsize)
    elif ndim == 3:
        nz = <uint>data.shape[0]
        ny = <uint>data.shape[1]
        nx = <uint>data.shape[2]
        sz = <int>(data.strides[0] / itemsize)
        sy = <int>(data.strides[1] / itemsize)
        sx = <int>(data.strides[2] / itemsize)
    elif ndim == 4:
        nw = <uint>data.shape[0]
        nz = <uint>data.shape[1]
        ny = <uint>data.shape[2]
        nx = <uint>data.shape[3]
        sw = <int>(data.strides[0] / itemsize)
        sz = <int>(data.strides[1] / itemsize)
        sy = <int>(data.strides[2] / itemsize)
        sx = <int>(data.strides[3] / itemsize)
    else:
        raise ValueError('data shape not supported by ZFP')

    if mode is None:
        zmode = zfp_mode_null
    elif mode in (zfp_mode_fixed_precision, 'p', 'precision'):
        zmode = zfp_mode_fixed_precision
        precision = _default_level(level, ZFP_MAX_PREC, 0, ZFP_MAX_PREC)
    elif mode in (zfp_mode_fixed_rate, 'r', 'rate'):
        zmode = zfp_mode_fixed_rate
        rate = level
    elif mode in (zfp_mode_fixed_accuracy, 'a', 'accuracy'):
        zmode = zfp_mode_fixed_accuracy
        tolerance = level
    elif mode in (zfp_mode_expert, 'c', 'expert'):
        zmode = zfp_mode_expert
        minbits, maxbits, maxprec, minexp = level
    else:
        raise ValueError('invalid ZFP mode')

    if execution is None or execution == 'serial':
        zexec = zfp_exec_serial
    elif execution == 'omp':
        zexec = zfp_exec_omp
    elif execution == 'cuda':
        zexec = zfp_exec_cuda
    else:
        raise ValueError('invalid ZFP execution policy')

    try:
        zfp = zfp_stream_open(NULL)
        if zfp == NULL:
            raise RuntimeError('zfp_stream_open failed')

        field = zfp_field_alloc()
        if field == NULL:
            raise RuntimeError('zfp_field_alloc failed')

        ztype = zfp_field_set_type(field, ztype)
        if ztype == zfp_type_none:
            raise RuntimeError('zfp_field_set_type failed')

        zfp_field_set_pointer(field, <void*>src.data)

        if ndim == 1:
            zfp_field_set_size_1d(field, nx)
            zfp_field_set_stride_1d(field, sx)
        elif ndim == 2:
            zfp_field_set_size_2d(field, nx, ny)
            zfp_field_set_stride_2d(field, sx, sy)
        elif ndim == 3:
            zfp_field_set_size_3d(field, nx, ny, nz)
            zfp_field_set_stride_3d(field, sx, sy, sz)
        elif ndim == 4:
            zfp_field_set_size_4d(field, nx, ny, nz, nw)
            zfp_field_set_stride_4d(field, sx, sy, sz, sw)

        if zmode == zfp_mode_fixed_precision:
            precision = zfp_stream_set_precision(zfp, precision)
        elif zmode == zfp_mode_fixed_rate:
            rate = zfp_stream_set_rate(zfp, rate, ztype, ndim, 0)
        elif zmode == zfp_mode_fixed_accuracy:
            tolerance = zfp_stream_set_accuracy(zfp, tolerance)
        elif zmode == zfp_mode_expert:
            ret = zfp_stream_set_params(zfp, minbits, maxbits, maxprec, minexp)
            if ret == 0:
                raise RuntimeError('zfp_stream_set_params failed')

        out, dstsize, out_given, out_type = _parse_output(out)
        if out is None:
            if dstsize < 0:
                dstsize = zfp_stream_maximum_size(zfp, field)
            if out_type is bytes:
                out = PyBytes_FromStringAndSize(NULL, dstsize)
            else:
                out = PyByteArray_FromStringAndSize(NULL, dstsize)

        dst = out
        dstsize = dst.size * dst.itemsize

        with nogil:
            stream = stream_open(<void*>&dst[0], dstsize)
            if stream == NULL:
                raise RuntimeError('stream_open failed')

            zfp_stream_set_bit_stream(zfp, stream)

            ret = zfp_stream_set_execution(zfp, zexec)
            if ret == 0:
                raise RuntimeError('zfp_stream_set_execution failed')

            if bheader != 0:
                byteswritten = zfp_write_header(zfp, field, ZFP_HEADER_FULL)
                if byteswritten == 0:
                    raise RuntimeError('zfp_write_header failed')

            byteswritten = zfp_compress(zfp, field)
            if byteswritten == 0:
                raise RuntimeError('zfp_compress failed')

    finally:
        if field != NULL:
            zfp_field_free(field)
        if zfp != NULL:
            zfp_stream_close(zfp)
        if stream != NULL:
            stream_close(stream)

    if <ssize_t>byteswritten < dstsize:
        if out_given:
            out = memoryview(out)[:byteswritten]
        else:
            out = out[:byteswritten]

    return out


def zfp_decode(data, shape=None, dtype=None, out=None):
    """Decompress ZFP stream to numpy array.

    """
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        zfp_stream* zfp = NULL
        bitstream* stream = NULL
        zfp_field* field = NULL
        zfp_type ztype
        ssize_t ndim
        size_t size
        uint nx, ny, nz, nw
        int ret

    if data is out:
        raise ValueError('cannot decode in-place')

    if dtype is None:
        ztype = zfp_type_none
    elif dtype == numpy.int32:
        ztype = zfp_type_int32
    elif dtype == numpy.int64:
        ztype = zfp_type_int64
    elif dtype == numpy.float32:
        ztype = zfp_type_float
    elif dtype == numpy.float64:
        ztype = zfp_type_double
    else:
        raise ValueError('dtype not supported by ZFP')

    ndim = -1 if shape is None else len(shape)
    if ndim == -1:
        pass
    elif ndim == 1:
        nx = <uint>shape[0]
    elif ndim == 2:
        nx = <uint>shape[1]
        ny = <uint>shape[0]
    elif ndim == 3:
        nx = <uint>shape[2]
        ny = <uint>shape[1]
        nz = <uint>shape[0]
    elif ndim == 4:
        nx = <uint>shape[3]
        ny = <uint>shape[2]
        nz = <uint>shape[1]
        nw = <uint>shape[0]
    else:
        raise ValueError('shape not supported by ZFP')

    # TODO: enable execution mode when supported
    # zfp_exec_policy zexec
    # if execution is None or execution == 'serial':
    #     zexec = zfp_exec_serial
    # elif execution == 'omp':
    #     zexec = zfp_exec_omp
    # elif execution == 'cuda':
    #     zexec = zfp_exec_cuda
    # else:
    #    raise ValueError('invalid ZFP execution policy')

    try:
        zfp = zfp_stream_open(NULL)
        if zfp == NULL:
            raise RuntimeError('zfp_stream_open failed')

        field = zfp_field_alloc()
        if field == NULL:
            raise RuntimeError('zfp_field_alloc failed')

        stream = stream_open(<void*>&src[0], srcsize)
        if stream == NULL:
            raise RuntimeError('stream_open failed')

        zfp_stream_set_bit_stream(zfp, stream)

        # ret = zfp_stream_set_execution(zfp, zexec)
        # if ret == 0:
        #     raise RuntimeError('zfp_stream_set_execution failed')

        if ztype == zfp_type_none or ndim == -1:
            size = zfp_read_header(zfp, field, ZFP_HEADER_FULL)
            if size == 0:
                raise RuntimeError('zfp_read_header failed')

        if ztype == zfp_type_none:
            ztype = field.dtype
            if ztype == zfp_type_float:
                dtype = numpy.float32
            elif ztype == zfp_type_double:
                dtype = numpy.float64
            elif ztype == zfp_type_int32:
                dtype = numpy.int32
            elif ztype == zfp_type_int64:
                dtype = numpy.int64
            else:
                raise RuntimeError('invalid zfp_field type')
        else:
            ztype = zfp_field_set_type(field, ztype)
            if ztype == zfp_type_none:
                raise RuntimeError('zfp_field_set_type failed')

        if ndim == -1:
            if field.nx == 0:
                raise RuntimeError('invalid zfp_field nx')
            elif field.ny == 0:
                shape = field.nx,
            elif field.nz == 0:
                shape = field.ny, field.nx
            elif field.nw == 0:
                shape = field.nz, field.ny, field.nx
            else:
                shape = field.nw, field.nz, field.ny, field.nx
        elif ndim == 1:
            zfp_field_set_size_1d(field, nx)
        elif ndim == 2:
            zfp_field_set_size_2d(field, nx, ny)
        elif ndim == 3:
            zfp_field_set_size_3d(field, nx, ny, nz)
        elif ndim == 4:
            zfp_field_set_size_4d(field, nx, ny, nz, nw)

        out = _create_array(out, shape, dtype)
        dst = out

        with nogil:
            zfp_field_set_pointer(field, <void *>dst.data)
            size = zfp_decompress(zfp, field)

        if size == 0:
            raise RuntimeError('zfp_decompress failed')

    finally:
        if field != NULL:
            zfp_field_free(field)
        if zfp != NULL:
            zfp_stream_close(zfp)
        if stream != NULL:
            stream_close(stream)

    return out


###############################################################################

cdef _create_array(out, shape, dtype, strides=None):
    """Return numpy array of shape and dtype from output argument."""
    if out is None or isinstance(out, numbers.Integral):
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
    elif isinstance(out, numbers.Integral):
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
