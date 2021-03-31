# imagecodecs/_shared.pyx
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

"""Shared functions for imagecodecs extensions."""

import numbers

import numpy

cimport numpy
cimport cython

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_Check

from cpython.bytearray cimport (
    PyByteArray_FromStringAndSize, PyByteArray_Check, PyByteArray_Resize
)

from libc.stdint cimport uint8_t


numpy.import_array()


cdef _parse_output(out, ssize_t outsize=-1, outgiven=False, outtype=bytes):
    """Return out, outsize, outgiven, outtype from output argument.

    """
    if out is None:
        # create new bytes output
        return out, outsize, bool(outgiven), outtype
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
    elif isinstance(out, bytes):
        raise TypeError("'bytes' object does not support item assignment")
    else:
        # use provided output buffer
        # outsize = len(out)
        # outtype = type(out)
        outgiven = True
    return out, outsize, bool(outgiven), outtype


cdef object _create_output(object out, ssize_t size, const char* string=NULL):
    """Return new bytes or bytesarray of length size.

    Copy content of 'string', if provided, to new object.
    Return NULL on failure.

    """
    IF IS_PYPY:
        # PyPy can not modify the content of bytes
        pass
    ELSE:
        if out == bytes or PyBytes_Check(out):
            return PyBytes_FromStringAndSize(string, size)
    return PyByteArray_FromStringAndSize(string, size)


cdef object _return_output(object out, ssize_t size, ssize_t used, outgiven):
    """Return a memoryview, slice, or copy of 'out' of length 'used'.

    If 'used >= size', return 'out' unchanged.

    If 'out' was provided by the user, return a memoryview or a ndarray slice
    of length 'used' of 'out'. The memoryview will be read-only if 'out' is
    bytes.

    Else if 'out' is a bytesarray, return 'out' resized to length 'used'.

    Else, 'out' is bytes, return a copy of 'out' of length 'used'.

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
    elif numpy.PyArray_Check(out):
        # slice of user numpy array
        out = out[:used]
    else:
        # memoryview of user bytes or bytearray
        out = memoryview(out)[:used]
    return out


cdef _create_array(out, shape, dtype, strides=None, zero=False):
    """Return numpy array of shape and dtype from output argument."""
    if out is None or isinstance(out, numbers.Integral):
        if zero:
            out = numpy.zeros(shape, dtype)
        else:
            out = numpy.empty(shape, dtype)
    elif isinstance(out, numpy.ndarray):
        if out.shape != shape:
            raise ValueError(f'invalid output shape {out.shape!r} {shape!r}')
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


cdef const uint8_t[::1] _readable_input(data):
    """Return readable, contiguous 1D bytes memoryview of data.

    Make copy if necessary.

    """
    cdef:
        const uint8_t[::1] src

    try:
        src = data
    except Exception:
        # not contiguous
        try:
            # numpy array
            # src = numpy.ravel(data, 'K').view(numpy.uint8)
            src = data.reshape(-1).view(numpy.uint8)
        except Exception:
            # buffer protocol
            src = data.tobytes()
    return src


cdef const uint8_t[::1] _writable_input(data):
    """Return writable, contiguous 1D bytes memoryview to data.

    Make copy if necessary.

    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] writable

    if isinstance(data, bytes):
        src = data
        return src

    try:
        writable = data
        src = writable
    except Exception:
        # not writable or not contiguous
        if hasattr(data, 'read'):
            # mmap
            src = data.read()
        else:
            try:
                # numpy array
                src_writable = data.reshape(-1).view(numpy.uint8)
                src = writable
            except Exception:
                # buffer protocol
                src = data.tobytes()

    return src


cdef const uint8_t[::1] _inplace_input(data):
    """Return writable, contiguous 1D bytes memoryview to data.

    Fail if input is not writable and contiguous.

    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] writable

    if isinstance(data, bytes):
        # disallow writing to bytes
        if isinstance(data, bytes):
            raise TypeError("'bytes' object does not support item assignment")
        # src = data
        # return src

    try:
        writable = data
        src = writable
    except Exception:
        # not writable or not contiguous
        view = memoryview(data)
        if not view.contiguous:
            raise ValueError('input data is not writable and contiguous')
        src_writable = view.cast('B')
        src = writable

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


def _set_attributes(cls=None, name=None, **kwargs):
    """Add kwargs to the specified class or the global symbol table.

    This is a hack to add constants defined in C to module and class
    namespaces.

    """
    if cls is None:
        globals().update(**kwargs)
    else:
        for name, value in kwargs.items():
            setattr(cls, name, value)
        if name is not None:
            globals()[name] = cls


def _log_warning(msg, *args, **kwargs):
    """Logs a message with level WARNING."""
    import logging

    logging.getLogger('imagecodecs').warning(msg, *args, **kwargs)
