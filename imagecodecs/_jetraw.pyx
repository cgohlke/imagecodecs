# imagecodecs/_jetraw.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: freethreading_compatible = True

# Copyright (c) 2022-2026, Christoph Gohlke
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

"""JETRAW codec for the imagecodecs package.

This implementation is based on the discontinued version 23.03.16.4
Jetraw SDK by the Dotphoton.

"""

include '_shared.pxi'

import sys

cimport jetraw

import cython

from cpython.mem cimport PyMem_Free


cdef extern from 'Python.h':
    jetraw.CHARTYPE* PyUnicode_AsWideCharString(object, Py_ssize_t*)

cdef cython.pymutex global_lock


class JETRAW:
    """JETRAW codec constants."""

    available = True


class JetrawError(RuntimeError):
    """JETRAW codec exceptions."""

    def __init__(self, func, int status):
        cdef:
            const char* msg = jetraw.dp_status_description(
                <jetraw.dp_status> status
            )

        super().__init__(f'{func} returned {msg.decode()!r}')


def jetraw_version():
    """Return Jetraw library version string."""
    return 'jetraw ' + jetraw.jetraw_version().decode()


def jetraw_check(const uint8_t[::1] data, /):
    """Return whether data is JETRAW encoded image or None if unknown."""


def jetraw_init(
    parameters=None,
    *,
    verbose=None,
):
    """Initialize JETRAW codec.

    Load preparation parameters and set verbosity level.

    """
    cdef:
        char* charp = NULL
        jetraw.CHARTYPE* wcharp = NULL
        int verbose_
        jetraw.dp_status status = jetraw.dp_success

    if verbose is not None:
        verbose_ = verbose
        with nogil:
            with global_lock:
                jetraw.dpcore_set_loglevel(verbose_)

    if parameters is None:
        with nogil:
            with global_lock:
                jetraw.dpcore_init()
    elif sys.platform == 'win32':
        # TODO: conditional compilation
        wcharp = PyUnicode_AsWideCharString(parameters, NULL)
        if wcharp == NULL:
            raise ValueError('PyUnicode_AsWideCharString returned NULL')
        with nogil:
            with global_lock:
                status = jetraw.dpcore_load_parameters(wcharp)
        if status != jetraw.dp_success:
            raise JetrawError('dpcore_load_parameters', status)
        PyMem_Free(<void*> wcharp)
    else:
        bytestr = parameters.encode()
        charp = bytestr
        with nogil:
            with global_lock:
                status = jetraw.dpcore_load_parameters(charp)
        if status != jetraw.dp_success:
            raise JetrawError('dpcore_load_parameters', status)


def jetraw_encode(
    data,
    /,
    identifier,
    *,
    errorbound=None,
    out=None,
):
    """Return JETRAW encoded image.

    The Jetraw codec is only viable for encoding whole frames from a few
    supported scientific cameras.

    """
    cdef:
        numpy.ndarray src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        int32_t pdstlen
        uint32_t height, width
        char* identifier_ = NULL
        float error_bound = _default_value(errorbound, 1.0, 1e-6, 1e3)
        jetraw.dp_status status = jetraw.dp_success

    src = numpy.atleast_2d(numpy.array(data, copy=True).squeeze())
    srcsize = src.size

    if not (
        src.ndim == 2
        and srcsize <= INT32_MAX
        and src.dtype == numpy.uint16
    ):
        raise ValueError('invalid data shape or dtype')

    height = <uint32_t> src.shape[0]
    width = <uint32_t> src.shape[1]

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = min(max(src.nbytes, <ssize_t> 1024), <ssize_t> INT32_MAX)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.nbytes
    if dstsize > INT32_MAX:
        raise RuntimeError('output too large')
    pdstlen = <int32_t> dstsize

    identifier = identifier.encode()
    identifier_ = identifier

    with nogil:
        status = jetraw.dpcore_embed_meta(
            <uint16_t*> src.data,
            <int32_t> srcsize,
            <const char*> identifier_,
            error_bound
        )
        if status != jetraw.dp_success:
            raise JetrawError('dpcore_embed_meta', status)

        status = jetraw.jetraw_encode(
            <const uint16_t*> src.data,
            width,
            height,
            <char*> &dst[0],
            &pdstlen
        )
        if status != jetraw.dp_success:
            raise JetrawError('jetraw_encode', status)

    del dst
    return _return_output(out, dstsize, pdstlen, outgiven)


def jetraw_decode(
    data,
    /,
    *,
    out=None,
):
    """Return decoded JETRAW image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.nbytes
        ssize_t dstsize
        jetraw.dp_status status = jetraw.dp_success

    if data is out:
        raise ValueError('cannot decode in-place')

    if out is None:
        raise ValueError('invalid output buffer')

    dst = out
    dstsize = dst.size
    if dstsize > INT32_MAX or dst.dtype != numpy.uint16:
        raise ValueError('invalid output shape or dtype')

    if srcsize > INT32_MAX:
        raise ValueError('input too large')

    with nogil:
        status = jetraw.jetraw_decode(
            <const char*> &src[0],
            <int32_t> srcsize,
            <uint16_t*> dst.data,
            <int32_t> dstsize
        )
        if status != jetraw.dp_success:
            raise JetrawError('jetraw_decode', status)

    return out
