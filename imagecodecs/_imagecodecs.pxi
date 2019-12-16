# -*- coding: utf-8 -*-
# _imagecodecs.pxi
# cython: language_level = 3

# Copyright (c) 2018-2019, Christoph Gohlke
# This source code is distributed under the 3-clause BSD license.

# Cython include file for shared functions.

import numbers
import numpy

cimport numpy
cimport cython

from cpython.bytes cimport (
    PyBytes_FromStringAndSize, PyBytes_Check, _PyBytes_Resize
)

from cpython.bytearray cimport (
    PyByteArray_FromStringAndSize, PyByteArray_Check, PyByteArray_Resize
)

from libc.stdint cimport uint8_t

from numpy cimport PyArray_Check

cdef extern from 'numpy/arrayobject.h':
    int NPY_VERSION
    int NPY_FEATURE_VERSION

numpy.import_array()


cdef _parse_output(out, ssize_t outsize=-1, outgiven=False, outtype=bytes):
    """Return out, outsize, outgiven, outtype from output argument.

    """
    if out is None:
        # create new bytes output
        return out, outsize, outgiven, outtype
    if out is bytes:
        # create new bytes output
        out = None
        outtype = bytes
    elif out is bytearray:
        # create new bytearray output
        out = None
        outtype = bytearray
    elif isinstance(out, numbers.Integral):
        # create new bytes output of expected length
        outsize = out
        out = None
    else:
        # use provided output buffer
        # outsize = len(out)
        # outtype = type(out)
        outgiven = True
    return out, outsize, outgiven, outtype


cdef object _create_output(object out, ssize_t size, const char* string=NULL):
    """Return new bytes or bytesarray of length size.

    Copy content of `string`, if provided, to new object.
    Return NULL on failure.

    """
    if PyBytes_Check(out):
        return PyBytes_FromStringAndSize(string, size)
    return PyByteArray_FromStringAndSize(string, size)


cdef object _return_output(object out, ssize_t size, ssize_t used, outgiven):
    """Return a memoryview, slice, or copy of `out` of length `used`.

    If `used >= size`, return `out` unchanged.

    If `out` was provided by the user, return a memoryview or a ndarray slice
    of length `used` of `out`.
    The memoryview will be read-only if `out` is bytes.

    Else if `out` is a bytesarray, return `out` resized to length `used`.

    Else, `out` is bytes, return a copy of `out` of length `used`.

    """
    if used >= size:
        return out
    if not outgiven:
        if PyByteArray_Check(out):
            # resize bytearray
            if PyByteArray_Resize(out, used) < 0:
                raise MemoryError('PyByteArray_Resize failed')
        else:
            # copy of bytes
            # TODO: use _PyBytes_Resize?
            out = out[:used]
    elif PyArray_Check(out):
        # slice of user provided numpy array
        out = out[:used]
    else:
        # memoryview of user provided bytes or bytearray
        out = memoryview(out)[:used]
    return out


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


cdef const uint8_t[::1] _parse_input(data):
    """Return contiguous 1D bytes memoryview to input argument."""
    cdef const uint8_t[::1] src
    try:
        src = data
    except ValueError:
        src = numpy.ravel(data, 'K').view(numpy.uint8)
    return src


cdef _default_value(value, default, smallest, largest):
    """Return default value or value in range."""
    if value is None:
        value = default
    if largest is not None:
        value = min(value, largest)
    if smallest is not None:
        value = max(value, smallest)
    return value


def _add_globals(**kwargs):
    """Add kwargs to the global symbol table.

    This is a hack to add constants defined in C to the module namespace.

    """
    globals().update(**kwargs)
