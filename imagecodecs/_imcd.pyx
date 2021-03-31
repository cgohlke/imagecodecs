# imagecodecs/_imcd.pyx
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

"""Codecs for the imagecodecs package using the imcd.c library."""

__version__ = '2021.2.26'

include '_shared.pxi'

from imcd cimport *

from libc.math cimport ceil

from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.bytearray cimport PyByteArray_FromStringAndSize

cdef extern from 'numpy/arrayobject.h':
    int NPY_VERSION
    int NPY_FEATURE_VERSION


class IMCD:
    """Imcd Constants."""


class ImcdError(RuntimeError):
    """Imcd Exceptions."""

    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            IMCD_OK: 'IMCD_OK',
            IMCD_ERROR: 'IMCD_ERROR',
            IMCD_MEMORY_ERROR: 'IMCD_MEMORY_ERROR',
            IMCD_RUNTIME_ERROR: 'IMCD_RUNTIME_ERROR',
            IMCD_NOTIMPLEMENTED_ERROR: 'IMCD_NOTIMPLEMENTED_ERROR',
            IMCD_VALUE_ERROR: 'IMCD_VALUE_ERROR',
            IMCD_LZW_INVALID: 'IMCD_LZW_INVALID',
            IMCD_LZW_NOTIMPLEMENTED: 'IMCD_LZW_NOTIMPLEMENTED',
            IMCD_LZW_BUFFER_TOO_SMALL: 'IMCD_LZW_BUFFER_TOO_SMALL',
            IMCD_LZW_TABLE_TOO_SMALL: 'IMCD_LZW_TABLE_TOO_SMALL',
            IMCD_LZW_CORRUPT: 'IMCD_LZW_CORRUPT',
        }.get(err, f'unknown error {err!r}')
        msg = f'{func} returned {msg}'
        super().__init__(msg)


def imcd_version():
    """Return imcd library version string."""
    return f'imcd {IMCD_VERSION.decode()}'


def cython_version():
    """Return Cython version string."""
    return f'cython {cython.__version__}'


def numpy_abi_version():
    """Return numpy ABI version string."""
    return f'numpy_abi 0x{NPY_VERSION:X}.{NPY_FEATURE_VERSION}'


def imcd_check(arg):
    """Return True if data likely contains IMCD data."""


# Delta #######################################################################

DELTA = IMCD
DeltaError = ImcdError
delta_version = imcd_version
delta_check = imcd_check


def delta_encode(data, axis=-1, dist=1, out=None):
    """Encode differencing.

    """
    return _delta(data, axis=axis, dist=dist, out=out, decode=False)


def delta_decode(data, axis=-1, dist=1, out=None):
    """Decode differencing.

    Same as numpy.cumsum

    """
    return _delta(data, axis=axis, dist=dist, out=out, decode=True)


cdef _delta(data, int axis, ssize_t dist, out, int decode):
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

    if dist != 1:
        raise NotImplementedError(f'dist {dist} not implemented')

    if isinstance(data, numpy.ndarray):
        if data.dtype.kind not in 'fiu':
            raise ValueError('not an integer or floating-point array')

        if out is None:
            out = numpy.empty_like(data)
        elif not isinstance(out, numpy.ndarray):
            raise ValueError('output is not a numpy array')
        elif out.shape != data.shape or out.itemsize != data.itemsize:
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
                ret = imcd_delta(
                    <void*> srcptr,
                    srcsize,
                    srcstride,
                    <void*> dstptr,
                    dstsize,
                    dststride,
                    itemsize,
                    decode
                )
                if ret < 0:
                    break
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        if ret < 0:
            raise DeltaError('imcd_delta', ret)

        return out

    src = _readable_input(data)
    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = src.size
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    srcsize = src.size
    srcstride = 1
    dststride = 1
    itemsize = 1

    with nogil:
        ret = imcd_delta(
            <void*> &src[0],
            srcsize,
            srcstride,
            <void*> &dst[0],
            dstsize,
            dststride,
            itemsize,
            decode
        )
    if ret < 0:
        raise DeltaError('imcd_delta', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


# XOR Delta ###################################################################

XOR = IMCD
XorError = ImcdError
xor_version = imcd_version
xor_check = imcd_check


def xor_encode(data, axis=-1, out=None):
    """Encode XOR.

    """
    return _xor(data, axis=axis, out=out, decode=False)


def xor_decode(data, axis=-1, out=None):
    """Decode XOR.

    """
    return _xor(data, axis=axis, out=out, decode=True)


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
        elif not isinstance(out, numpy.ndarray):
            raise ValueError('output is not a numpy array')
        elif out.shape != data.shape or out.itemsize != data.itemsize:
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
                ret = imcd_xor(
                    <void*> srcptr,
                    srcsize,
                    srcstride,
                    <void*> dstptr,
                    dstsize,
                    dststride,
                    itemsize,
                    decode
                )
                if ret < 0:
                    break
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        if ret < 0:
            raise XorError('imcd_xor', ret)

        return out

    src = _readable_input(data)
    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = src.size
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size
    srcsize = src.size
    srcstride = 1
    dststride = 1
    itemsize = 1

    with nogil:
        ret = imcd_xor(
            <void*> &src[0],
            srcsize,
            srcstride,
            <void*> &dst[0],
            dstsize,
            dststride,
            itemsize,
            decode
        )
    if ret < 0:
        raise XorError('imcd_xor', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


# Floating Point Predictor ####################################################

# TIFF Technical Note 3. April 8, 2005.

FLOATPRED = IMCD
FloatpredError = ImcdError
floatpred_version = imcd_version
floatpred_check = imcd_check


def floatpred_encode(data, axis=-1, dist=1, out=None):
    """Encode Floating Point Predictor.

    The output array should not be treated as floating-point numbers but as an
    encoded byte sequence viewed as a numpy array with shape and dtype of the
    input data.

    """
    return _floatpred(data, axis=axis, dist=dist, out=out, decode=False)


def floatpred_decode(data, axis=-1, dist=1, out=None):
    """Decode Floating Point Predictor.

    The data array is not really an array of floating-point numbers but an
    encoded byte sequence viewed as a numpy array of requested output shape
    and dtype.

    """
    return _floatpred(data, axis=axis, dist=dist, out=out, decode=True)


cdef _floatpred(data, int axis, ssize_t dist, out, int decode):
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

    # this needs to pass silently in tifffile
    # if data is out:
    #     raise ValueError('cannot decode in-place')
    # TODO: support in-place decoding

    if out is None or data is out:
        out = numpy.empty_like(data)
    elif not isinstance(out, numpy.ndarray):
        raise ValueError('output is not a numpy array')
    elif out.shape != data.shape or out.itemsize != data.itemsize:
        raise ValueError('output is not compatible with data array')

    ndim = data.ndim
    axis = axis % ndim
    if ndim < 1 or ndim - axis > 2:
        raise ValueError('invalid axis')

    samples = data.shape[axis+1] if ndim - axis == 2 else 1

    if dist != 1 and dist != 2 and dist != 4:
        raise NotImplementedError(f'dist {dist} not implemented')

    samples *= dist

    src = data.view()
    src.shape = data.shape[:axis] + (-1,)
    dst = out.view()
    dst.shape = src.shape

    if src.dtype.byteorder == '=':
        byteorder = IMCD_BOC
    else:
        byteorder = <char> ord(src.dtype.byteorder)

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
            ret = imcd_floatpred(
                <void*> srcptr,
                srcsize,
                srcstride,
                <void*> dstptr,
                dstsize,
                dststride,
                itemsize,
                samples,
                byteorder,
                decode
            )
            if ret < 0:
                break
            numpy.PyArray_ITER_NEXT(srciter)
            numpy.PyArray_ITER_NEXT(dstiter)
    if ret < 0:
        raise FloatpredError('imcd_floatpred', ret)

    return out


# BitOrder Reversal ###########################################################

BITORDER = IMCD
BitorderError = ImcdError
bitorder_version = imcd_version
bitorder_check = imcd_check


def bitorder_encode(data, out=None):
    """"Reverse bits in each byte of bytes, bytearray or numpy array.

    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        uint8_t* srcptr = NULL
        uint8_t* dstptr = NULL
        ssize_t srcsize = 0
        ssize_t dstsize = 0
        ssize_t srcstride = 1
        ssize_t dststride = 1
        ssize_t itemsize = 1
        ssize_t ret = 0
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
                srcptr = <uint8_t*> numpy.PyArray_DATA(data)
                srcsize = data.size * itemsize
                srcstride = itemsize
                with nogil:
                    ret = imcd_bitorder(
                        <uint8_t*> srcptr,
                        srcsize,
                        srcstride,
                        itemsize,
                        <uint8_t*> dstptr,
                        dstsize,
                        dststride
                    )
                    if ret < 0:
                        raise BitorderError('imcd_bitorder', ret)
                return data

            srciter = numpy.PyArray_IterAllButAxis(data, &axis)
            srcsize = data.shape[axis] * itemsize
            srcstride = data.strides[axis]
            with nogil:
                while numpy.PyArray_ITER_NOTDONE(srciter):
                    srcptr = <uint8_t*> numpy.PyArray_ITER_DATA(srciter)
                    ret = imcd_bitorder(
                        <uint8_t*> srcptr,
                        srcsize,
                        srcstride,
                        itemsize,
                        <uint8_t*> dstptr,
                        dstsize,
                        dststride
                    )
                    if ret < 0:
                        raise BitorderError('imcd_bitorder', ret)
                    numpy.PyArray_ITER_NEXT(srciter)
            return data

        if out is None:
            out = numpy.empty_like(data)
        elif not isinstance(out, numpy.ndarray):
            raise ValueError('output is not a numpy array')
        elif data.shape != out.shape or itemsize != out.dtype.itemsize:
            raise ValueError('output is not compatible with data array')
        srciter = numpy.PyArray_IterAllButAxis(data, &axis)
        dstiter = numpy.PyArray_IterAllButAxis(out, &axis)
        srcsize = data.shape[axis] * itemsize
        dstsize = out.shape[axis] * itemsize
        srcstride = data.strides[axis]
        dststride = out.strides[axis]
        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = <uint8_t*> numpy.PyArray_ITER_DATA(srciter)
                dstptr = <uint8_t*> numpy.PyArray_ITER_DATA(dstiter)
                ret = imcd_bitorder(
                    <uint8_t*> srcptr,
                    srcsize,
                    srcstride,
                    itemsize,
                    <uint8_t*> dstptr,
                    dstsize,
                    dststride
                )
                if ret < 0:
                    raise BitorderError('imcd_bitorder', ret)
                numpy.PyArray_ITER_NEXT(srciter)
                numpy.PyArray_ITER_NEXT(dstiter)
        return out

    # contiguous byte buffers: bytes or bytearray

    if data is out:
        # in-place
        src = _inplace_input(data)
        srcsize = src.size
        with nogil:
            ret = imcd_bitorder(
                <uint8_t*> &src[0],
                srcsize,
                1,
                1,
                <uint8_t*> &src[0],
                srcsize,
                1
            )
        if ret < 0:
            raise BitorderError('imcd_bitorder', ret)
        return data

    src = _readable_input(data)
    srcsize = src.size
    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            dstsize = srcsize
        out = _create_output(outtype, dstsize)
    dst = out
    dstsize = dst.size

    with nogil:
        ret = imcd_bitorder(
            <uint8_t*> &src[0],
            srcsize,
            1,
            1,
            <uint8_t*> &dst[0],
            dstsize,
            1
        )
    if ret < 0:
        raise BitorderError('imcd_bitorder', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


bitorder_decode = bitorder_encode


# PackBits ####################################################################

PACKBITS = IMCD
PackbitsError = ImcdError
packbits_version = imcd_version
packbits_check = imcd_check


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

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            if isarray:
                srcsize = data.shape[axis]
                dstsize = data.size // srcsize * (srcsize + srcsize // 128 + 2)
            else:
                srcsize = len(data)
                dstsize = srcsize + srcsize // 128 + 2
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if isarray:
        srciter = numpy.PyArray_IterAllButAxis(data, &axis)
        srcsize = data.shape[axis]
        dstptr = &dst[0]
        with nogil:
            while numpy.PyArray_ITER_NOTDONE(srciter):
                srcptr = <uint8_t*> numpy.PyArray_ITER_DATA(srciter)
                ret = imcd_packbits_encode(
                    srcptr,
                    srcsize,
                    <uint8_t*> dstptr,
                    dstsize
                )
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
        src = _readable_input(data)
        srcsize = src.size
        with nogil:
            ret = imcd_packbits_encode(
                &src[0],
                srcsize,
                <uint8_t*> &dst[0],
                dstsize
            )
    if ret < 0:
        raise PackbitsError('imcd_packbits_encode', ret)

    dstsize = dst.size
    del dst
    return _return_output(out, dstsize, ret, outgiven)


def packbits_decode(data, out=None):
    """Decompress PackBits.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            with nogil:
                dstsize = imcd_packbits_size(&src[0], srcsize)
            if dstsize < 0:
                raise PackbitsError('imcd_packbits_size', dstsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = imcd_packbits_decode(
            &src[0],
            srcsize,
            <uint8_t*> &dst[0],
            dstsize
        )
    if ret < 0:
        raise PackbitsError('imcd_packbits_decode', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


# Packed Integers #############################################################

PACKINTS = IMCD
PackintsError = ImcdError
packints_version = imcd_version
packints_check = imcd_check


def packints_encode(data, int bitspersample, int axis=-1, out=None):
    """Pack integers."""

    raise NotImplementedError('packints_encode')


def packints_decode(
    data, dtype, int bitspersample, ssize_t runlen=0, out=None
):
    """Unpack groups of bits in byte sequence into numpy array."""
    cdef:
        const uint8_t[::1] src = data
        uint8_t* srcptr = <uint8_t*> &src[0]
        uint8_t* dstptr = NULL
        ssize_t srcsize = src.size
        ssize_t dstsize = 0
        ssize_t bytesize
        ssize_t itemsize
        ssize_t skipbits, i
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if bitspersample < 1 or (bitspersample > 32 and bitspersample != 64):
        raise ValueError('bitspersample out of range')

    bytesize = <ssize_t> ceil(bitspersample / 8.0)
    itemsize = bytesize if bytesize < 3 else (8 if bytesize > 4 else 4)

    if srcsize > <ssize_t> SSIZE_MAX / itemsize:
        raise ValueError('data size out of range')

    dtype = numpy.dtype(dtype)
    if dtype.itemsize != itemsize:
        raise ValueError('dtype.itemsize does not fit bitspersample')

    if runlen == 0:
        runlen = <ssize_t> (
            (<uint64_t> srcsize * 8) / <uint64_t> bitspersample
        )

    skipbits = <ssize_t> ((<uint64_t> runlen * <uint64_t> bitspersample) % 8)
    if skipbits > 0:
        skipbits = 8 - skipbits

    dstsize = <ssize_t> (
        <uint64_t> runlen * <uint64_t> bitspersample + <uint64_t> skipbits
    )
    if dstsize > 0:
        dstsize = <ssize_t> (
            <uint64_t> runlen * ((<uint64_t> srcsize * 8) / <uint64_t> dstsize)
        )

    if out is None:
        out = numpy.empty(dstsize, dtype)
    elif (
        not isinstance(data, numpy.ndarray)
        or out.dtype != dtype
        or out.size < dstsize
    ):
        raise ValueError('invalid output type, size, or dtype')
    elif not numpy.PyArray_ISCONTIGUOUS(out):
        raise ValueError('output array is not contiguous')
    if dstsize == 0:
        return out

    dstptr = <uint8_t*> numpy.PyArray_DATA(out)
    srcsize = <ssize_t> (
        (<uint64_t> runlen * <uint64_t> bitspersample + <uint64_t> skipbits)
        / 8
    )

    with nogil:
        # work around "Converting to Python object not allowed without gil"
        # for i in range(0, dstsize, runlen):
        for i from 0 <= i < dstsize by runlen:
            ret = imcd_packints_decode(
                <const uint8_t*> srcptr,
                srcsize,
                dstptr,
                runlen,
                bitspersample
            )
            if ret < 0:
                break
            srcptr += srcsize
            dstptr += runlen * itemsize

    if ret < 0:
        raise PackintsError('imcd_packints_decode', ret)

    if not dtype.isnative and bitspersample % 8:
        itemsize = dtype.itemsize
        dstptr = <uint8_t*> numpy.PyArray_DATA(out)
        with nogil:
            imcd_swapbytes(<void*> dstptr, dstsize, itemsize)

    return out


# 24-bit Floating Point #######################################################

# Adobe Photoshop(r) TIFF Technical Note 3. April 8, 2005

Float24Error = ImcdError
float24_version = imcd_version
float24_check = imcd_check


class FLOAT24:
    """Float24 Constants."""

    ROUND_TONEAREST = FE_TONEAREST
    ROUND_UPWARD = FE_UPWARD
    ROUND_DOWNWARD = FE_DOWNWARD
    ROUND_TOWARDZERO = FE_TOWARDZERO


def float24_encode(data, byteorder=None, rounding=None, out=None):
    """Return byte sequence of float24 from numpy.float32 array.

    """
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        ssize_t ret = 0
        char boc
        int feround = FE_TONEAREST if rounding is None else rounding

    if data is out:
        raise ValueError('cannot encode in-place')

    if not (isinstance(data, numpy.ndarray) and data.dtype == numpy.float32):
        raise ValueError('not a numpy.float32 array with native byte order')

    srcsize = data.size

    if byteorder is None or byteorder == '=':
        boc = IMCD_BOC
    elif byteorder == '<':
        boc = b'<'
    elif byteorder == '>':
        boc = b'>'
    else:
        raise ValueError('invalid byteorder')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = srcsize * 3
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    if dst.size < srcsize * 3:
        raise ValueError('output buffer too short')

    src = _readable_input(data)  # TODO: use numpy iterator?

    with nogil:
        ret = imcd_float24_encode(
            &src[0],
            srcsize * 4,
            <uint8_t*> &dst[0],
            boc,
            feround
        )

    if ret < 0:
        raise Float24Error('imcd_float24_encode', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


def float24_decode(data, byteorder=None, out=None):
    """Return numpy.float32 array from byte sequence of float24.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t* srcptr = &src[0]
        uint8_t* dstptr = NULL
        ssize_t srcsize = src.size
        ssize_t ret = 0
        char boc

    if data is out:
        raise ValueError('cannot decode in-place')

    if srcsize % 3 != 0:
        raise ValueError('data size not a multiple of 3')

    if byteorder is None or byteorder == '=':
        boc = IMCD_BOC
    elif byteorder == '<':
        boc = b'<'
    elif byteorder == '>':
        boc = b'>'
    else:
        raise ValueError('invalid byteorder')

    if out is None:
        out = numpy.empty(srcsize // 3, numpy.float32)
    elif (
        not isinstance(data, numpy.ndarray)
        or out.dtype != numpy.float32  # must be native
        or out.size < srcsize
    ):
        raise ValueError('invalid output type, size, or dtype')
    elif not numpy.PyArray_ISCONTIGUOUS(out):
        raise ValueError('output array is not contiguous')
    if srcsize == 0:
        return out

    dstptr = <uint8_t*> numpy.PyArray_DATA(out)

    with nogil:
        ret = imcd_float24_decode(
            srcptr,
            srcsize,
            dstptr,
            boc
        )

    if ret < 0:
        raise Float24Error('imcd_float24_decode', ret)
    return out


# LZW #########################################################################

LZW = IMCD
LzwError = ImcdError
lzw_version = imcd_version


def lzw_check(const uint8_t[::1] data):
    """Return True if data likely contains LZW data."""
    return bool(imcd_lzw_check(&data[0], data.size))


def lzw_encode(*args, **kwargs):
    """Compress LZW."""
    raise NotImplementedError('lzw_encode')


def lzw_decode(data, buffersize=0, out=None):
    """Decompress LZW.

    """
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret = 0
        imcd_lzw_handle_t* handle = NULL

    if data is out:
        raise ValueError('cannot decode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    handle = imcd_lzw_new(buffersize)
    if handle == NULL:
        raise LzwError('imcd_lzw_new', None)
    try:
        if out is None:
            if dstsize < 0:
                with nogil:
                    dstsize = imcd_lzw_decode_size(handle, &src[0], srcsize)
                if dstsize < 0:
                    raise LzwError('imcd_lzw_decode_size', dstsize)
            out = _create_output(outtype, dstsize)

        dst = out
        dstsize = dst.size

        with nogil:
            ret = imcd_lzw_decode(
                handle,
                &src[0],
                srcsize,
                <uint8_t*> &dst[0],
                dstsize
            )
        if ret < 0:
            raise LzwError('imcd_lzw_decode', ret)
    finally:
        imcd_lzw_del(handle)

    del dst
    return _return_output(out, dstsize, ret, outgiven)
