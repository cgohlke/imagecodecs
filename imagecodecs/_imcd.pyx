# imagecodecs/_imcd.pyx
# distutils: language = c
# cython: language_level = 3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

# Copyright (c) 2018-2025, Christoph Gohlke
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

include '_shared.pxi'

from imcd cimport *

from libc.math cimport ceil

cdef extern from 'numpy/arrayobject.h':
    int NPY_VERSION
    int NPY_FEATURE_VERSION


class IMCD:
    """IMCD codec constants."""

    available = True


class ImcdError(RuntimeError):
    """IMCD codec exceptions."""

    def __init__(self, func, err):
        msg = {
            None: 'NULL',
            IMCD_OK: 'IMCD_OK',
            IMCD_ERROR: 'IMCD_ERROR',
            IMCD_MEMORY_ERROR: 'IMCD_MEMORY_ERROR',
            IMCD_RUNTIME_ERROR: 'IMCD_RUNTIME_ERROR',
            IMCD_NOTIMPLEMENTED_ERROR: 'IMCD_NOTIMPLEMENTED_ERROR',
            IMCD_VALUE_ERROR: 'IMCD_VALUE_ERROR',
            IMCD_INPUT_CORRUPT: 'IMCD_INPUT_CORRUPT',
            IMCD_OUTPUT_TOO_SMALL: 'IMCD_OUTPUT_TOO_SMALL',
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
    """Return Numpy ABI version string."""
    return f'numpy_abi 0x{NPY_VERSION:X}.{NPY_FEATURE_VERSION}'


def imcd_check(arg):
    """Return whether data is encoded."""


# Delta #######################################################################

DELTA = IMCD
DeltaError = ImcdError
delta_version = imcd_version
delta_check = imcd_check


def delta_encode(data, axis=-1, dist=1, out=None):
    """Return DELTA encoded data.

    Preserve byteorder.

    """
    return _delta(data, axis=axis, dist=dist, out=out, decode=False)


def delta_decode(data, axis=-1, dist=1, out=None):
    """Return decoded DELTA data.

    Same as numpy.cumsum. Preserve byteorder.

    """
    return _delta(data, axis=axis, dist=dist, out=out, decode=True)


cdef _delta(data, int axis, ssize_t dist, out, int decode):
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t itemsize
        ssize_t ret = 0
        void* srcptr = NULL
        void* dstptr = NULL
        numpy.flatiter srciter
        numpy.flatiter dstiter
        bint isnative = True

    if dist != 1:
        raise NotImplementedError(f'{dist=} not implemented')  # TODO

    if isinstance(data, numpy.ndarray):
        if data.dtype.kind not in 'fiu':
            raise ValueError('not an integer or floating-point array')

        isnative = data.dtype.isnative

        out = _create_array(
            out, data.shape, data.dtype, strides=None, zero=False, contig=False
        )
        if out.shape != data.shape or out.dtype != data.dtype:
            raise ValueError('output is not compatible with data array')

        if not isnative:
            # imcd_delta requires native byteorder
            data = data.byteswap()
            data = data.view(data.dtype.newbyteorder())

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

        if not isnative:
            try:
                out = out.byteswap(True)
            except ValueError:  # read-only out, for example, out=data
                out = out.byteswap()

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
    """Return XOR encoded data."""
    return _xor(data, axis=axis, out=out, decode=False)


def xor_decode(data, axis=-1, out=None):
    """Return decoded XOR data."""
    return _xor(data, axis=axis, out=out, decode=True)


cdef _xor(data, int axis, out, int decode):
    cdef:
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize
        ssize_t dstsize
        ssize_t srcstride
        ssize_t dststride
        ssize_t itemsize
        ssize_t ret = 0
        void* srcptr = NULL
        void* dstptr = NULL
        numpy.flatiter srciter
        numpy.flatiter dstiter

    if isinstance(data, numpy.ndarray):
        if data.dtype.kind not in 'fiu':
            raise ValueError('not an integer or floating-point array')

        out = _create_array(
            out, data.shape, data.dtype, strides=None, zero=False, contig=False
        )
        if out.shape != data.shape or out.dtype != data.dtype:
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


# ByteShuffle Filter ##########################################################

# Separately pack bytes of values

BYTESHUFFLE = IMCD
ByteshuffleError = ImcdError
byteshuffle_version = imcd_version
byteshuffle_check = imcd_check


def byteshuffle_encode(
    data,
    axis=-1,
    dist=1,
    delta=False,
    reorder=False,
    out=None
):
    """Return byte-shuffled data.

    The output array should not be treated as numbers but as an encoded byte
    sequence viewed as a numpy array with shape and dtype of the input data.

    """
    return _byteshuffle(
        data,
        axis=axis,
        dist=dist,
        delta=delta,
        reorder=reorder,
        decode=False,
        out=out
    )


def byteshuffle_decode(
    data,
    axis=-1,
    dist=1,
    delta=False,
    reorder=False,
    out=None
):
    """Return un-byte-shuffled data.

    The input array is not really an array of numbers but an encoded byte
    sequence viewed as a numpy array of requested output shape and dtype.

    """
    return _byteshuffle(
        data,
        axis=axis,
        dist=dist,
        delta=delta,
        reorder=reorder,
        decode=True,
        out=out
    )


cdef _byteshuffle(
    data, int axis, ssize_t dist, bint delta, bint reorder, bint decode, out
):
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

    if not isinstance(data, numpy.ndarray):
        raise ValueError('not a numpy array')

    # this needs to pass silently in tifffile
    # if data is out:
    #     raise ValueError('cannot decode in-place')
    # TODO: support in-place decoding
    if data is out:
        out = None

    out = _create_array(
        out, data.shape, data.dtype, strides=None, zero=False, contig=False
    )
    if out.shape != data.shape or out.dtype != data.dtype:
        raise ValueError('output is not compatible with data array')

    ndim = data.ndim
    axis = axis % ndim
    if ndim < 1 or ndim - axis > 2:
        raise ValueError(f'invalid {axis=}')

    samples = data.shape[axis+1] if ndim - axis == 2 else 1

    if dist != 1 and dist != 2 and dist != 4:
        raise NotImplementedError(f'{dist=} not implemented')

    samples *= dist

    src = data.view()
    src.shape = data.shape[:axis] + (-1,)
    dst = out.view()
    dst.shape = src.shape

    if not reorder:
        byteorder = b'>'  # use order of bytes in data
    elif src.dtype.byteorder == '=':
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
    if decode and srcstride != itemsize:
        raise ValueError('data not contiguous on dimensions >= axis')
    elif decode and dststride != itemsize:
        raise ValueError('output not contiguous on dimensions >= axis')

    with nogil:
        while numpy.PyArray_ITER_NOTDONE(srciter):
            srcptr = numpy.PyArray_ITER_DATA(srciter)
            dstptr = numpy.PyArray_ITER_DATA(dstiter)
            ret = imcd_byteshuffle(
                <void*> srcptr,
                srcsize,
                srcstride,
                <void*> dstptr,
                dstsize,
                dststride,
                itemsize,
                samples,
                byteorder,
                delta,
                decode,
            )
            if ret < 0:
                break
            numpy.PyArray_ITER_NEXT(srciter)
            numpy.PyArray_ITER_NEXT(dstiter)
    if ret < 0:
        raise ByteshuffleError('imcd_byteshuffle', ret)

    return out


# Floating Point Predictor ####################################################

# TIFF Technical Note 3. April 8, 2005.

FLOATPRED = IMCD
FloatpredError = ImcdError
floatpred_version = imcd_version
floatpred_check = imcd_check


def floatpred_encode(data, axis=-1, dist=1, out=None):
    """Return floating-point predicted array.

    The output array should not be treated as floating-point numbers but as an
    encoded byte sequence viewed as a numpy array with shape and dtype of the
    input data.

    """
    return _byteshuffle(
        data,
        axis=axis,
        dist=dist,
        delta=True,
        reorder=True,
        decode=False,
        out=out
    )


def floatpred_decode(data, axis=-1, dist=1, out=None):
    """Return un-predicted floating-point array.

    The input array is not really an array of floating-point numbers but an
    encoded byte sequence viewed as a numpy array of requested output shape
    and dtype.

    """
    return _byteshuffle(
        data,
        axis=axis,
        dist=dist,
        delta=True,
        reorder=True,
        decode=True,
        out=out
    )


# BitOrder Reversal ###########################################################

BITORDER = IMCD
BitorderError = ImcdError
bitorder_version = imcd_version
bitorder_check = imcd_check


def bitorder_encode(data, out=None):
    """Return data with reversed bit-order in each byte."""
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

        out = _create_array(
            out, data.shape, data.dtype, strides=None, zero=False, contig=False
        )
        if out.shape != data.shape or out.dtype != data.dtype:
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


def packbits_encode(data, axis=None, out=None):
    """Return PACKBITS encoded data."""
    cdef:
        numpy.flatiter srciter
        const uint8_t[::1] src
        const uint8_t[::1] dst  # must be const to write to bytes
        const uint8_t* srcptr
        const uint8_t* dstptr
        ssize_t srcsize
        ssize_t dstsize
        ssize_t ret = 0
        bint isarray = isinstance(data, numpy.ndarray)
        int axis_ = 0

    if data is out:
        raise ValueError('cannot decode in-place')

    if isarray:
        data = numpy.ascontiguousarray(data)
        if axis is None:
            axis = data.ndim - 1
        elif axis < 0:
            axis = data.ndim + axis
        if axis >= data.ndim:
            raise ValueError('invalid axis')
        if axis < data.ndim - 1:
            # merge trailing dimensions
            data = numpy.reshape(data, data.shape[:axis] + (-1,))
        axis_ = axis
        if data.strides[axis_] != data.itemsize:
            raise ValueError(
                f'data array is not contiguous along axis {axis_}'
            )

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            if isarray:
                srcsize = data.shape[axis_] * data.itemsize
                if srcsize > 0:
                    dstsize = (
                        data.nbytes // srcsize
                        * imcd_packbits_encode_size(srcsize)
                    )
                else:
                    dstsize = 0
            else:
                dstsize = imcd_packbits_encode_size(len(data))
        out = _create_output(outtype, dstsize)

    dst = out  # must be contiguous bytes
    dstsize = dst.size

    if isarray and data.ndim > 1:
        srciter = numpy.PyArray_IterAllButAxis(data, &axis_)
        srcsize = data.shape[axis_] * data.itemsize
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
    """Return decoded PACKBITS data."""
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
                dstsize = imcd_packbits_decode_size(&src[0], srcsize)
            if dstsize < 0:
                raise PackbitsError('imcd_packbits_decode_size', dstsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = imcd_packbits_decode(
            &src[0],
            srcsize,
            <uint8_t*> &dst[0],
            dstsize,
            1
        )
    if ret < 0:
        raise PackbitsError('imcd_packbits_decode', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


# DICOM RLE ###################################################################

# https://dicom.nema.org
# /medical/Dicom/current/output/chtml/part05/sect_8.2.2.html

DICOMRLE = IMCD
DicomrleError = ImcdError
dicomrle_version = imcd_version

ctypedef struct dicomrle_header:
    uint32_t segments
    uint32_t offset[15]


def dicomrle_check(const uint8_t[::1] data):
    """Return whether data is DICOMRLE encoded data."""
    cdef:
        ssize_t segment
        dicomrle_header header

    if data.size < 64:
        return False
    memcpy(&header, &data[0], 64)
    if header.segments == 0 or header.segments > 15 or header.offset[0] != 64:
        return False
    for segment in range(header.segments):
        if header.offset[segment] == 0 or header.offset[segment] >= data.size:
            return False
    return True


def dicomrle_encode(data, out=None):
    """Return DICOMRLE encoded data."""
    raise NotImplementedError('dicomrle_encode')


def dicomrle_decode(data, dtype, out=None):
    """Return decoded DICOMRLE data."""
    cdef:
        const uint8_t[::1] src = data
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t itemsize, dstsize, decoded_size, size, ret, segment, offset
        bint byteswap = 0
        dicomrle_header header

    if data is out:
        raise ValueError('cannot decode in-place')
    if srcsize < 64:
        raise ValueError(f'invalid DICOM RLE size {srcsize} < 64')

    memcpy(&header, &src[0], 64)
    if header.segments == 0 or header.segments > 15 or header.offset[0] != 64:
        raise ValueError(f'invalid DICOM RLE {header.segments=}')

    dtype = numpy.dtype(dtype)
    itemsize = dtype.itemsize
    byteswap = itemsize > 1 and dtype.byteorder != '>'
    if header.segments % itemsize != 0:
        raise ValueError(f'{itemsize=} does not match {header.segments=}')

    out, dstsize, outgiven, outtype = _parse_output(out)
    if out is None:
        if dstsize < 0:
            with nogil:
                dstsize = 0
                segment = 0
                while segment < header.segments:
                    if header.segments == segment + 1:
                        size = srcsize - <ssize_t> header.offset[segment]
                    else:
                        size = (
                            <ssize_t> header.offset[segment + 1]
                            - <ssize_t> header.offset[segment]
                        )
                    if size <= 0:
                        raise ValueError(f'invalid {size=} of {segment=}')
                    ret = imcd_packbits_decode_size(
                        &src[header.offset[segment]], size
                    )
                    if ret < 0:
                        raise DicomrleError('imcd_packbits_decode_size', ret)
                    # all decoded segments in a sample have the same size
                    dstsize += itemsize * ret
                    segment += itemsize
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        decoded_size = 0
        offset = 0
        segment = 0
        while segment < header.segments:
            if header.segments == segment + 1:
                size = srcsize - <ssize_t> header.offset[segment]
            else:
                size = (
                    <ssize_t> header.offset[segment + 1]
                    - <ssize_t> header.offset[segment]
                )
            if size <= 0:
                raise ValueError(f'invalid {segment=} {size=}')
            ret = imcd_packbits_decode(
                &src[header.offset[segment]],
                size,
                <uint8_t*> &dst[offset],
                max(0, dstsize - offset),
                itemsize
            )
            if ret < 0:
                raise DicomrleError('imcd_packbits_decode', ret)
            decoded_size += ret
            offset += 1
            segment += 1
            if segment % itemsize == 0:
                offset = decoded_size

        if byteswap:
            imcd_swapbytes(<void*> &dst[0], decoded_size // itemsize, itemsize)

    del dst
    return _return_output(out, dstsize, decoded_size, outgiven)


# CCITTRLE ####################################################################

CCITTRLE = IMCD
CcittrleError = ImcdError
ccittrle_version = imcd_version
ccittrle_check = imcd_check


def ccittrle_encode(data, level=None, axis=None, out=None):
    """Return CCITTRLE encoded data."""
    raise NotImplementedError('ccittrle_encode')


def ccittrle_decode(data, out=None):
    """Return decoded CCITTRLE data."""
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
                dstsize = imcd_ccittrle_decode_size(&src[0], srcsize)
            if dstsize < 0:
                raise CcittrleError('imcd_ccittrle_decode_size', dstsize)
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = imcd_ccittrle_decode(
            &src[0],
            srcsize,
            <uint8_t*> &dst[0],
            dstsize
        )
    if ret < 0:
        raise CcittrleError('imcd_ccittrle_decode', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)


# Packed Integers #############################################################

PACKINTS = IMCD
PackintsError = ImcdError
packints_version = imcd_version
packints_check = imcd_check


def packints_encode(
    data, int bitspersample, int axis=-1, out=None
):
    """Return packed integers (not implemented)."""

    raise NotImplementedError('packints_encode')


def packints_decode(
    data, dtype, int bitspersample, ssize_t runlen=0, out=None
):
    """Return unpacked integers.

    Unpack groups of bits in byte sequence into numpy array.

    """
    cdef:
        const uint8_t[::1] src = data
        uint8_t* srcptr = <uint8_t*> &src[0]
        uint8_t* dstptr = NULL
        ssize_t srcsize = src.size
        ssize_t dstsize = 0
        ssize_t bytesize, itemsize, skipbits, i
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


# MONO12P #####################################################################

MONO12P = IMCD
Mono12pError = ImcdError
mono12p_version = imcd_version
mono12p_check = imcd_check


def mono12p_encode(
    data, msfirst=False, int axis=-1, out=None
):
    """Return MONO12 packed integers (not implemented)."""

    raise NotImplementedError('packints_encode')


def mono12p_decode(
    data, msfirst=False, ssize_t runlen=0, out=None
):
    """Return unpacked MONO12p integers (not implemented)."""

    raise NotImplementedError('mono12p_decode')


# 24-bit Floating Point #######################################################

# Adobe Photoshop(r) TIFF Technical Note 3. April 8, 2005

Float24Error = ImcdError
float24_version = imcd_version
float24_check = imcd_check


class FLOAT24:
    """FLOAT24 codec constants."""

    available = True

    class ROUND(enum.IntEnum):
        """FLOAT24 codec rounding types."""

        TONEAREST = FE_TONEAREST
        UPWARD = FE_UPWARD
        DOWNWARD = FE_DOWNWARD
        TOWARDZERO = FE_TOWARDZERO


def float24_encode(
    data, byteorder=None, rounding=None, out=None
):
    """Return FLOAT24 encoded array."""
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

    data = numpy.asarray(data)
    if not data.dtype == numpy.float32:
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
    """Return decoded FLOAT24 array."""
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
        not isinstance(out, numpy.ndarray)
        or out.dtype != numpy.float32  # must be native
        or out.size < srcsize // 3
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


# EER #########################################################################

# EER file format documentation 3.0. Section 4. by M. Leichsenring. Mar 2023

EER = IMCD
EerError = ImcdError
eer_version = imcd_version


def eer_check(const uint8_t[::1] data):
    """Return whether data is EER encoded."""
    return None


def eer_decode(
    data,
    shape,
    int rlebits,
    int horzbits,
    int vertbits,
    bint superres=False,
    out=None
):
    """Return decoded EER image."""
    cdef:
        numpy.ndarray dst
        const uint8_t[::1] src = data
        ssize_t srcsize = src.size
        ssize_t ret = 0
        ssize_t height = shape[0]
        ssize_t width = shape[1]
        uint8_t* dstptr

    if data is out:
        raise ValueError('cannot decode in-place')

    if not (
        1 < rlebits < 15
        and 0 < horzbits < 5
        and 0 < vertbits < 5
        and 8 < rlebits + horzbits + vertbits < 17
    ):
        raise ValueError(
            f'compression scheme {rlebits}_{horzbits}_{vertbits} not supported'
        )

    if (
        superres
        and (
            height % (2 ** <uint32_t> vertbits)
            or width % (2 ** <uint32_t> horzbits)
        )
    ):
        raise ValueError('shape not compatible with superresolution')

    out = _create_array(out, shape, numpy.bool_, strides=None, zero=True)
    dst = out
    dstptr = <uint8_t*> dst.data

    with nogil:
        ret = imcd_eer_decode(
            &src[0],
            srcsize,
            dstptr,
            height,
            width,
            rlebits,
            horzbits,
            vertbits,
            superres
        )
    if ret < 0:
        raise EerError('imcd_eer_decode', ret)

    return out


def eer_encode(data, out=None):
    """Return EER encoded image (not implemented)."""
    raise NotImplementedError('eer_encode')


# LZW #########################################################################

# TIFF Revision 6, Section 13, June 3, 1992

LZW = IMCD
LzwError = ImcdError
lzw_version = imcd_version


def lzw_check(const uint8_t[::1] data):
    """Return whether data is LZW encoded."""
    return bool(imcd_lzw_check(&data[0], data.size))


def lzw_decode(data, buffersize=0, out=None):
    """Return decoded LZW data."""
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


def lzw_encode(data, out=None):
    """Return LZW encoded data."""
    cdef:
        const uint8_t[::1] src = _readable_input(data)
        const uint8_t[::1] dst  # must be const to write to bytes
        ssize_t srcsize = src.size
        ssize_t dstsize
        ssize_t ret = 0

    if data is out:
        raise ValueError('cannot encode in-place')

    out, dstsize, outgiven, outtype = _parse_output(out)

    if out is None:
        if dstsize < 0:
            dstsize = imcd_lzw_encode_size(srcsize)
            if dstsize < 0:
                raise LzwError(f'imcd_lzw_encode_size returned {dstsize}')
        out = _create_output(outtype, dstsize)

    dst = out
    dstsize = dst.size

    with nogil:
        ret = imcd_lzw_encode(
            <const uint8_t*> &src[0], srcsize, <uint8_t*> &dst[0], dstsize
        )
    if ret < 0:
        raise LzwError('imcd_lzw_encode', ret)

    del dst
    return _return_output(out, dstsize, ret, outgiven)
